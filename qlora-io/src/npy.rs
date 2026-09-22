//! `.npy` format: header parsing, dtype conversion, order handling.
//!
//! Format reference: magic `\x93NUMPY`, major/minor bytes, then a header
//! length (`u16` LE for v1.0, `u32` LE for v2.0) followed by an ASCII dict
//! like `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }`.

use crate::error::IoError;

/// Parsed `.npy` payload: C-order `f32` data plus shape.
#[derive(Debug, Clone, PartialEq)]
pub struct RawArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Read a single `.npy` buffer.
pub fn read_npy_bytes(bytes: &[u8]) -> Result<RawArray, IoError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(IoError::Malformed("bad npy magic".to_string()));
    }
    let (major, minor) = (bytes[6], bytes[7]);
    let (hlen, preamble) = match (major, minor) {
        (1, 0) => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        (2, 0) => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        _ => {
            return Err(IoError::Unsupported(format!(
                "npy version {major}.{minor} (supports 1.0 and 2.0)"
            )));
        }
    };
    let hstart = preamble;
    let hend = hstart
        .checked_add(hlen)
        .ok_or_else(|| IoError::Malformed("npy header length overflow".to_string()))?;
    if bytes.len() < hend {
        return Err(IoError::Malformed("truncated npy header".to_string()));
    }
    let header = std::str::from_utf8(&bytes[hstart..hend])
        .map_err(|_| IoError::Malformed("npy header is not ASCII".to_string()))?;
    let (descr, fortran, shape) = parse_header(header)?;

    let (itemsize, convert) = dtype_converter(&descr)?;
    let numel = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| IoError::Malformed("npy shape overflow".to_string()))?;
    let want = numel
        .checked_mul(itemsize)
        .ok_or_else(|| IoError::Malformed("npy data size overflow".to_string()))?;
    let data_bytes = &bytes[hend..];
    if data_bytes.len() != want {
        return Err(IoError::Malformed(format!(
            "npy data length {} != shape elements {numel} * itemsize {itemsize}",
            data_bytes.len()
        )));
    }
    let mut data = Vec::with_capacity(numel);
    for chunk in data_bytes.chunks_exact(itemsize) {
        data.push(convert(chunk));
    }
    if fortran {
        data = fortran_to_c(&data, &shape);
    }
    Ok(RawArray { shape, data })
}

/// Minimal parser for the npy header dict. Handles single/double quotes,
/// `True`/`False`, `(d0, d1, ...)`, `(n,)` and `()`.
fn parse_header(header: &str) -> Result<(String, bool, Vec<usize>), IoError> {
    let bad = || IoError::Malformed(format!("cannot parse npy header: {header:?}"));
    let descr = parse_quoted_value(header, "descr").ok_or_else(bad)?;
    let fortran_raw = parse_bare_value(header, "fortran_order").ok_or_else(bad)?;
    let fortran = match fortran_raw {
        "True" => true,
        "False" => false,
        _ => return Err(bad()),
    };
    let shape_raw = parse_bare_value(header, "shape").ok_or_else(bad)?;
    let shape = parse_shape(shape_raw).ok_or_else(bad)?;
    Ok((descr, fortran, shape))
}

/// Find `'key': <quoted string>` and return the string contents.
fn parse_quoted_value(header: &str, key: &str) -> Option<String> {
    let key_pos = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))?;
    let after_key = &header[key_pos + key.len() + 2..];
    let colon = after_key.find(':')?;
    let mut chars = after_key[colon + 1..].trim_start().chars();
    let quote = chars.next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = chars.as_str();
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

/// Find `'key': <bare value up to the top-level , or }>` and return it
/// trimmed. Parenthesis depth is tracked so `(2, 3, 4)` survives intact.
fn parse_bare_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let key_pos = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))?;
    let after_key = &header[key_pos + key.len() + 2..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let mut depth = 0usize;
    let mut end = None;
    for (i, ch) in rest.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = depth.saturating_sub(1),
            ',' | '}' if depth == 0 => {
                end = Some(i);
                break;
            }
            _ => {}
        }
    }
    Some(rest[..end?].trim())
}

fn parse_shape(raw: &str) -> Option<Vec<usize>> {
    let raw = raw.trim();
    if !raw.starts_with('(') || !raw.ends_with(')') {
        return None;
    }
    let inner = raw[1..raw.len() - 1].trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().ok())
        .collect()
}

type Converter = fn(&[u8]) -> f32;

/// Map a numpy `descr` to `(itemsize, converter to f32)`.
///
/// Supports `<`/`>` endian numerics, `|u1`, `|b1`. Rejects complex,
/// strings, void, object, datetime.
fn dtype_converter(descr: &str) -> Result<(usize, Converter), IoError> {
    let unsupported = || IoError::Unsupported(format!("dtype {descr:?}"));
    let b = descr.as_bytes();
    if b.len() < 3 {
        return Err(unsupported());
    }
    let (endian, kind, size): (u8, u8, usize) =
        (b[0], b[1], descr[2..].parse().map_err(|_| unsupported())?);
    if endian != b'<' && endian != b'>' && endian != b'|' {
        return Err(unsupported());
    }
    let le = endian != b'>';
    if endian == b'|' && kind != b'u' && kind != b'b' {
        return Err(unsupported());
    }
    let conv: Converter = match (kind, size) {
        (b'f', 8) => {
            if le {
                |c| f64::from_le_bytes(c.try_into().unwrap()) as f32
            } else {
                |c| f64::from_be_bytes(c.try_into().unwrap()) as f32
            }
        }
        (b'f', 4) => {
            if le {
                |c| f32::from_le_bytes(c.try_into().unwrap())
            } else {
                |c| f32::from_be_bytes(c.try_into().unwrap())
            }
        }
        (b'f', 2) => {
            if le {
                |c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))
            } else {
                |c| f16_to_f32(u16::from_be_bytes(c.try_into().unwrap()))
            }
        }
        (b'i', 8) => {
            if le {
                |c| i64::from_le_bytes(c.try_into().unwrap()) as f32
            } else {
                |c| i64::from_be_bytes(c.try_into().unwrap()) as f32
            }
        }
        (b'i', 4) => {
            if le {
                |c| i32::from_le_bytes(c.try_into().unwrap()) as f32
            } else {
                |c| i32::from_be_bytes(c.try_into().unwrap()) as f32
            }
        }
        (b'i', 2) => {
            if le {
                |c| i16::from_le_bytes(c.try_into().unwrap()) as f32
            } else {
                |c| i16::from_be_bytes(c.try_into().unwrap()) as f32
            }
        }
        (b'i', 1) => |c| c[0] as i8 as f32,
        (b'u', 8) => {
            if le {
                |c| u64::from_le_bytes(c.try_into().unwrap()) as f32
            } else {
                |c| u64::from_be_bytes(c.try_into().unwrap()) as f32
            }
        }
        (b'u', 4) => {
            if le {
                |c| u32::from_le_bytes(c.try_into().unwrap()) as f32
            } else {
                |c| u32::from_be_bytes(c.try_into().unwrap()) as f32
            }
        }
        (b'u', 2) => {
            if le {
                |c| u16::from_le_bytes(c.try_into().unwrap()) as f32
            } else {
                |c| u16::from_be_bytes(c.try_into().unwrap()) as f32
            }
        }
        (b'u', 1) => |c| c[0] as f32,
        (b'b', 1) => |c| (c[0] != 0) as u8 as f32,
        _ => return Err(unsupported()),
    };
    let itemsize = match (kind, size) {
        (b'f', 8) | (b'i', 8) | (b'u', 8) => 8,
        (b'f', 4) | (b'i', 4) | (b'u', 4) => 4,
        (b'f', 2) | (b'i', 2) | (b'u', 2) => 2,
        (b'i', 1) | (b'u', 1) | (b'b', 1) => 1,
        _ => return Err(unsupported()),
    };
    Ok((itemsize, conv))
}

/// IEEE-754 binary16 to `f32` (handles subnormals, inf, NaN).
fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 1;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let bits: u32 = if e == 0 {
        if m == 0 {
            (s as u32) << 31
        } else {
            let mut mm = m;
            let mut exp = -14i32;
            while mm & 0x400 == 0 {
                mm <<= 1;
                exp -= 1;
            }
            mm &= 0x3ff;
            ((s as u32) << 31) | (((exp + 127) as u32) << 23) | ((mm as u32) << 13)
        }
    } else if e == 31 {
        ((s as u32) << 31) | (0xff << 23) | ((m as u32) << 13)
    } else {
        ((s as u32) << 31) | ((e as u32 + 112) << 23) | ((m as u32) << 13)
    };
    f32::from_bits(bits)
}

/// Convert Fortran-order flat data to C-order (general N-D).
fn fortran_to_c(data: &[f32], shape: &[usize]) -> Vec<f32> {
    let nd = shape.len();
    let mut fstr = vec![1usize; nd];
    for i in 1..nd {
        fstr[i] = fstr[i - 1] * shape[i - 1];
    }
    let n: usize = shape.iter().product();
    let mut out = vec![0.0f32; n];
    let mut idx = vec![0usize; nd];
    for o in out.iter_mut() {
        let mut foff = 0;
        for d in 0..nd {
            foff += idx[d] * fstr[d];
        }
        *o = data[foff];
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Build a v1.0 `.npy` buffer (C-order `f32`, descr `<f4`).
pub fn write_npy_bytes(shape: &[usize], data: &[f32]) -> Vec<u8> {
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
    dict.push('\n');
    // Pad like numpy: preamble (10 bytes) + header is a multiple of 64.
    let pad = (64 - (10 + dict.len()) % 64) % 64;
    let mut header = dict.into_bytes();
    // Insert spaces before the trailing newline.
    let nl = header.pop().unwrap();
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(nl);
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
        // Smallest subnormal: 2^-24.
        assert!((f16_to_f32(0x0001) - 5.9604645e-8).abs() < 1e-14);
        assert!((f16_to_f32(0x3555) - 0.33325195).abs() < 1e-7);
    }

    #[test]
    fn write_then_read_roundtrip() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        let bytes = write_npy_bytes(&[2, 3, 4], &data);
        // Preamble + header is 64-aligned like numpy's.
        assert_eq!((bytes.len() - data.len() * 4) % 64, 0);
        let back = read_npy_bytes(&bytes).unwrap();
        assert_eq!(back.shape, vec![2, 3, 4]);
        assert_eq!(back.data, data);
    }

    #[test]
    fn header_parse_edge_cases() {
        let (d, f, s) =
            parse_header("{'descr': '<f8', 'fortran_order': True, 'shape': (5,), }").unwrap();
        assert_eq!((d, f, s), ("<f8".to_string(), true, vec![5]));
        let (_, _, s) =
            parse_header("{'descr': '|u1', 'fortran_order': False, 'shape': (), }").unwrap();
        assert_eq!(s, Vec::<usize>::new());
    }
}
