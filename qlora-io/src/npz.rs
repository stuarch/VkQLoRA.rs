//! `.npz`: minimal ZIP reader (stored entries only) and writer.
//!
//! `np.savez` writes a plain ZIP with one `.npy` member per array, no
//! compression, no data descriptors, no ZIP64. Anything else is rejected
//! with an actionable message.

use crate::error::IoError;
use crate::npy::{read_npy_bytes, write_npy_bytes, RawArray};

/// One named array from an archive: C-order `f32` data plus shape.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedArray {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NamedArray {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Look up an array by name.
    pub fn find<'a>(arrays: &'a [NamedArray], name: &str) -> Option<&'a NamedArray> {
        arrays.iter().find(|a| a.name == name)
    }
}

/// Read all arrays from an `.npz` buffer, in archive order.
pub fn read_npz(bytes: &[u8]) -> Result<Vec<NamedArray>, IoError> {
    read_npz_impl(bytes)
}

/// Read a single `.npy` buffer (name is empty).
pub fn read_npy(bytes: &[u8]) -> Result<NamedArray, IoError> {
    let RawArray { shape, data } = read_npy_bytes(bytes)?;
    Ok(NamedArray {
        name: String::new(),
        shape,
        data,
    })
}

/// Write `[(name, shape, data)]` as an `.npz` buffer (stored, like
/// `np.savez`). Data is always `f32` (`<f4`).
pub fn write_npz(arrays: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, shape, data) in arrays {
        let member = format!("{name}.npy");
        let payload = write_npy_bytes(shape, data);
        let crc = crc32(&payload);
        let local_off = out.len() as u32;

        // Local file header.
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        out.extend_from_slice(&0u16.to_le_bytes()); // time
        out.extend_from_slice(&0u16.to_le_bytes()); // date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(member.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(member.as_bytes());
        out.extend_from_slice(&payload);

        // Central directory entry.
        central.extend_from_slice(b"PK\x01\x02");
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&20u16.to_le_bytes()); // version needed
        central.extend_from_slice(&0u16.to_le_bytes()); // flags
        central.extend_from_slice(&0u16.to_le_bytes()); // method
        central.extend_from_slice(&0u16.to_le_bytes()); // time
        central.extend_from_slice(&0u16.to_le_bytes()); // date
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        central.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        central.extend_from_slice(&(member.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra
        central.extend_from_slice(&0u16.to_le_bytes()); // comment
        central.extend_from_slice(&0u16.to_le_bytes()); // disk
        central.extend_from_slice(&0u16.to_le_bytes()); // int attrs
        central.extend_from_slice(&0u32.to_le_bytes()); // ext attrs
        central.extend_from_slice(&local_off.to_le_bytes());
        central.extend_from_slice(member.as_bytes());
    }
    let cd_off = out.len() as u32;
    let cd_count = arrays.len();
    out.extend_from_slice(&central);
    let cd_size = (out.len() as u32) - cd_off;
    // End of central directory (no comment).
    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes()); // disk
    out.extend_from_slice(&0u16.to_le_bytes()); // cd disk
    out.extend_from_slice(&(cd_count as u16).to_le_bytes());
    out.extend_from_slice(&(cd_count as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len
    out
}

/// ISO-HDLC CRC32 (table built on first use; fine for our sizes).
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB88320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn u16le(b: &[u8]) -> usize {
    u16::from_le_bytes([b[0], b[1]]) as usize
}

fn u32le(b: &[u8]) -> usize {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
}

fn slice<'a>(data: &'a [u8], off: usize, len: usize, what: &str) -> Result<&'a [u8], IoError> {
    let end = off
        .checked_add(len)
        .ok_or_else(|| IoError::Malformed(format!("{what} out of bounds")))?;
    data.get(off..end)
        .ok_or_else(|| IoError::Malformed(format!("truncated {what}")))
}

struct Entry {
    name: String,
    method: usize,
    flags: usize,
    comp_size: usize,
    local_off: usize,
}

fn read_npz_impl(bytes: &[u8]) -> Result<Vec<NamedArray>, IoError> {
    if bytes.len() < 22 {
        return Err(IoError::Malformed("file smaller than ZIP EOCD".to_string()));
    }
    // EOCD: 22 bytes, no comment expected; else scan back (comment <= 64KB).
    let mut eocd = None;
    let tail_off = bytes.len() - 22;
    if bytes[tail_off..tail_off + 4] == *b"PK\x05\x06" && u16le(&bytes[tail_off + 20..]) == 0 {
        eocd = Some(tail_off);
    } else {
        let scan_from = bytes.len().saturating_sub(22 + 65535);
        let mut i = bytes.len() - 22;
        loop {
            if bytes[i..i + 4] == *b"PK\x05\x06" {
                let comment = u16le(&bytes[i + 20..]);
                if i + 22 + comment == bytes.len() {
                    eocd = Some(i);
                    break;
                }
            }
            if i == scan_from {
                break;
            }
            i -= 1;
        }
    }
    let eocd =
        eocd.ok_or_else(|| IoError::Malformed("no ZIP end-of-central-directory".to_string()))?;
    let cd_count = u16le(&bytes[eocd + 10..]);
    let cd_size = u32le(&bytes[eocd + 12..]);
    let cd_off = u32le(&bytes[eocd + 16..]);
    if cd_count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
        return Err(IoError::Unsupported("zip64 archives".to_string()));
    }
    slice(bytes, cd_off, cd_size, "central directory")?;

    let mut entries = Vec::with_capacity(cd_count);
    let mut off = cd_off;
    for _ in 0..cd_count {
        let h = slice(bytes, off, 46, "central header")?;
        if h[0..4] != *b"PK\x01\x02" {
            return Err(IoError::Malformed("bad central header magic".to_string()));
        }
        let flags = u16le(&h[8..]);
        let method = u16le(&h[10..]);
        let comp_size = u32le(&h[20..]);
        let name_len = u16le(&h[28..]);
        let extra_len = u16le(&h[30..]);
        let comment_len = u16le(&h[32..]);
        let local_off = u32le(&h[42..]);
        if comp_size == 0xFFFF_FFFF || local_off == 0xFFFF_FFFF {
            return Err(IoError::Unsupported("zip64 archives".to_string()));
        }
        let name_bytes = slice(bytes, off + 46, name_len, "entry name")?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| IoError::Malformed("entry name is not UTF-8".to_string()))?
            .to_string();
        entries.push(Entry {
            name,
            method,
            flags,
            comp_size,
            local_off,
        });
        off += 46 + name_len + extra_len + comment_len;
    }

    let mut arrays = Vec::new();
    for e in entries {
        if e.name.ends_with('/') {
            continue; // directory entry
        }
        if e.flags & 0x01 != 0 {
            return Err(IoError::Unsupported(format!(
                "encrypted entry {:?}",
                e.name
            )));
        }
        if e.flags & 0x08 != 0 {
            return Err(IoError::Unsupported(format!(
                "entry {:?} uses a data descriptor (not written by numpy)",
                e.name
            )));
        }
        if e.method != 0 {
            return Err(IoError::Unsupported(format!(
                "entry {:?} uses compression method {} (np.savez_compressed output); resave with np.savez (uncompressed)",
                e.name, e.method
            )));
        }
        let lh = slice(bytes, e.local_off, 30, "local header")?;
        if lh[0..4] != *b"PK\x03\x04" {
            return Err(IoError::Malformed(format!(
                "bad local header for {:?}",
                e.name
            )));
        }
        let lh_name_len = u16le(&lh[26..]);
        let lh_extra_len = u16le(&lh[28..]);
        let data_off = e.local_off + 30 + lh_name_len + lh_extra_len;
        let payload = slice(bytes, data_off, e.comp_size, "entry data")?;
        let RawArray { shape, data } = read_npy_bytes(payload).map_err(|err| match err {
            IoError::Malformed(m) => IoError::Malformed(format!("{:?}: {m}", e.name)),
            other => other,
        })?;
        let name = e.name.strip_suffix(".npy").unwrap_or(&e.name).to_string();
        arrays.push(NamedArray { name, shape, data });
    }
    Ok(arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn empty_archive_roundtrip() {
        let bytes = write_npz(&[]);
        let back = read_npz(&bytes).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn garbage_is_malformed_not_panic() {
        assert!(read_npz(&[]).is_err());
        assert!(read_npz(&[0u8; 100]).is_err());
        assert!(read_npz(b"PK\x05\x06not-a-zip-at-all-padded....").is_err());
    }
}
