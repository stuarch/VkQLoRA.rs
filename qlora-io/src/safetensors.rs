//! `.safetensors` reader (subset for model weights).
//!
//! Layout: `u64` LE header length, JSON header, then raw little-endian
//! tensor data. Header entries look like
//! `"weight": {"dtype": "BF16", "shape": [576, 49152],
//! "data_offsets": [0, 56623104]}`.
//!
//! Supported dtypes (converted to `f32`): `F32`, `F16`, `BF16`, `I32`,
//! `I64`, `U8`, `BOOL`. Anything else is [`IoError::Unsupported`].
//! The JSON parser below is intentionally minimal — it handles exactly the
//! flat `{name: {dtype, shape, data_offsets}}` schema safetensors writes.

use crate::error::IoError;
use crate::npz::NamedArray;

#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// Read all tensors from a `.safetensors` buffer (converted to `f32`).
pub fn read_safetensors(bytes: &[u8]) -> Result<Vec<NamedArray>, IoError> {
    if bytes.len() < 8 {
        return Err(IoError::Malformed(
            "file smaller than safetensors header".to_string(),
        ));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if hlen > bytes.len() - 8 {
        return Err(IoError::Malformed(
            "safetensors header length exceeds file".to_string(),
        ));
    }
    let header = std::str::from_utf8(&bytes[8..8 + hlen])
        .map_err(|_| IoError::Malformed("safetensors header is not UTF-8".to_string()))?;
    let metas = parse_header(header)?;
    let data = &bytes[8 + hlen..];

    let mut out = Vec::with_capacity(metas.len());
    for (name, m) in metas {
        if m.end < m.start || m.end > data.len() {
            return Err(IoError::Malformed(format!(
                "tensor {name:?}: offsets [{}, {}) outside data (len {})",
                m.start,
                m.end,
                data.len()
            )));
        }
        let (itemsize, convert) = safetensors_dtype(&m.dtype)?;
        let numel: usize = m.shape.iter().product();
        if m.end - m.start != numel * itemsize {
            return Err(IoError::Malformed(format!(
                "tensor {name:?}: byte range {} != elements {numel} * itemsize {itemsize}",
                m.end - m.start
            )));
        }
        let raw = &data[m.start..m.end];
        let mut v = Vec::with_capacity(numel);
        for chunk in raw.chunks_exact(itemsize) {
            v.push(convert(chunk));
        }
        out.push(NamedArray {
            name,
            shape: m.shape,
            data: v,
        });
    }
    Ok(out)
}

type Converter = fn(&[u8]) -> f32;

fn safetensors_dtype(dtype: &str) -> Result<(usize, Converter), IoError> {
    let unsupported = || IoError::Unsupported(format!("safetensors dtype {dtype:?}"));
    match dtype {
        "F32" => Ok((4, |c| f32::from_le_bytes(c.try_into().unwrap()))),
        "F16" => Ok((2, |c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))),
        "BF16" => Ok((2, |c| {
            f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)
        })),
        "I32" => Ok((4, |c| i32::from_le_bytes(c.try_into().unwrap()) as f32)),
        "I64" => Ok((8, |c| i64::from_le_bytes(c.try_into().unwrap()) as f32)),
        "U8" => Ok((1, |c| c[0] as f32)),
        "BOOL" => Ok((1, |c| (c[0] != 0) as u8 as f32)),
        _ => Err(unsupported()),
    }
}

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

/// Minimal JSON parser for the safetensors header schema.
///
/// Returns `(name, meta)` pairs in document order. Handles objects, arrays
/// of numbers, and double-quoted strings with simple escapes.
fn parse_header(header: &str) -> Result<Vec<(String, TensorMeta)>, IoError> {
    let bad = |msg: &str| IoError::Malformed(format!("safetensors header: {msg}"));
    let b = header.as_bytes();
    let skip_ws = |pos: &mut usize| {
        while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') {
            *pos += 1;
        }
    };
    let parse_string = |pos: &mut usize| -> Result<String, IoError> {
        if b.get(*pos) != Some(&b'"') {
            return Err(bad("expected string"));
        }
        *pos += 1;
        let mut s = String::new();
        loop {
            match b.get(*pos) {
                None => return Err(bad("unterminated string")),
                Some(b'"') => {
                    *pos += 1;
                    return Ok(s);
                }
                Some(b'\\') => {
                    *pos += 1;
                    match b.get(*pos) {
                        Some(b'"') => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'/') => s.push('/'),
                        Some(b'n') => s.push('\n'),
                        Some(b't') => s.push('\t'),
                        Some(b'u') => {
                            // \uXXXX — only needed for exotic tensor names.
                            if *pos + 4 >= b.len() {
                                return Err(bad("bad unicode escape"));
                            }
                            let hex = std::str::from_utf8(&b[*pos + 1..*pos + 5])
                                .map_err(|_| bad("bad unicode escape"))?;
                            let cp = u32::from_str_radix(hex, 16)
                                .map_err(|_| bad("bad unicode escape"))?;
                            s.push(char::from_u32(cp).ok_or_else(|| bad("bad unicode escape"))?);
                            *pos += 4;
                        }
                        _ => return Err(bad("bad escape")),
                    }
                    *pos += 1;
                }
                Some(&ch) => {
                    s.push(ch as char);
                    *pos += 1;
                }
            }
        }
    };
    let expect = |pos: &mut usize, ch: u8| -> Result<(), IoError> {
        skip_ws(pos);
        if b.get(*pos) != Some(&ch) {
            return Err(bad("unexpected character"));
        }
        *pos += 1;
        Ok(())
    };
    let parse_nums = |pos: &mut usize| -> Result<Vec<usize>, IoError> {
        expect(pos, b'[')?;
        let mut v = Vec::new();
        loop {
            skip_ws(pos);
            if b.get(*pos) == Some(&b']') {
                *pos += 1;
                return Ok(v);
            }
            let start = *pos;
            while *pos < b.len() && (b[*pos].is_ascii_digit()) {
                *pos += 1;
            }
            if start == *pos {
                return Err(bad("expected number"));
            }
            v.push(
                std::str::from_utf8(&b[start..*pos])
                    .unwrap()
                    .parse()
                    .map_err(|_| bad("bad number"))?,
            );
            skip_ws(pos);
            match b.get(*pos) {
                Some(b',') => *pos += 1,
                Some(b']') => continue,
                _ => return Err(bad("expected , or ]")),
            }
        }
    };

    let mut pos = 0;
    expect(&mut pos, b'{')?;
    let mut out = Vec::new();
    loop {
        skip_ws(&mut pos);
        if b.get(pos) == Some(&b'}') {
            pos += 1;
            break;
        }
        let name = parse_string(&mut pos)?;
        expect(&mut pos, b':')?;
        if name == "__metadata__" {
            // Skip the metadata object (balanced braces).
            expect(&mut pos, b'{')?;
            let mut depth = 1;
            while depth > 0 {
                match b.get(pos) {
                    None => return Err(bad("unterminated metadata")),
                    Some(b'{') => depth += 1,
                    Some(b'}') => depth -= 1,
                    Some(b'"') => {
                        parse_string(&mut pos)?;
                        continue;
                    }
                    _ => {}
                }
                pos += 1;
            }
        } else {
            expect(&mut pos, b'{')?;
            let mut dtype = None;
            let mut shape = None;
            let mut offsets = None;
            loop {
                skip_ws(&mut pos);
                if b.get(pos) == Some(&b'}') {
                    pos += 1;
                    break;
                }
                let key = parse_string(&mut pos)?;
                expect(&mut pos, b':')?;
                skip_ws(&mut pos);
                match key.as_str() {
                    "dtype" => dtype = Some(parse_string(&mut pos)?),
                    "shape" => shape = Some(parse_nums(&mut pos)?),
                    "data_offsets" => offsets = Some(parse_nums(&mut pos)?),
                    _ => return Err(bad("unexpected tensor field")),
                }
                skip_ws(&mut pos);
                match b.get(pos) {
                    Some(b',') => pos += 1,
                    Some(b'}') => continue,
                    _ => return Err(bad("expected , or }")),
                }
            }
            let (dtype, shape, offsets) = match (dtype, shape, offsets) {
                (Some(d), Some(s), Some(o)) => (d, s, o),
                _ => return Err(bad("tensor entry missing dtype/shape/data_offsets")),
            };
            if offsets.len() != 2 {
                return Err(bad("data_offsets must have 2 elements"));
            }
            out.push((
                name,
                TensorMeta {
                    dtype,
                    shape,
                    start: offsets[0],
                    end: offsets[1],
                },
            ));
        }
        skip_ws(&mut pos);
        match b.get(pos) {
            Some(b',') => pos += 1,
            Some(b'}') => continue,
            _ => return Err(bad("expected , or }")),
        }
    }
    skip_ws(&mut pos);
    if pos != b.len() {
        return Err(bad("trailing data after header object"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic() -> Vec<u8> {
        // {"a": F32[2], "b": BF16[3]} with data.
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[3],"data_offsets":[8,14]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&1.5f32.to_le_bytes());
        out.extend_from_slice(&(-2.0f32).to_le_bytes());
        for &v in &[1.0f32, -0.5, 0.0] {
            out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        out
    }

    #[test]
    fn synthetic_header_roundtrip() {
        let arrays = read_safetensors(&synthetic()).unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].name, "a");
        assert_eq!(arrays[0].shape, vec![2]);
        assert_eq!(arrays[0].data, vec![1.5, -2.0]);
        assert_eq!(arrays[1].name, "b");
        assert_eq!(arrays[1].data, vec![1.0, -0.5, 0.0]);
    }

    #[test]
    fn rejects_garbage_and_bad_offsets() {
        assert!(read_safetensors(&[]).is_err());
        assert!(read_safetensors(&[1u8; 100]).is_err());
        let mut bad = synthetic();
        bad[8] = b'X'; // corrupt JSON
        assert!(read_safetensors(&bad).is_err());
    }
}
