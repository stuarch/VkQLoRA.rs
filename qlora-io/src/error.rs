//! Error type for weight IO.

use std::fmt;

/// Error type for `.npy` / `.npz` reading and writing.
#[derive(Debug, Clone, PartialEq)]
pub enum IoError {
    /// File is truncated or structurally invalid.
    Malformed(String),
    /// Valid file, but outside the supported subset (dtype, compression...).
    Unsupported(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Malformed(msg) => write!(f, "malformed input: {msg}"),
            IoError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}
