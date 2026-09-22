//! Error type shared by all qlora crates.

use std::fmt;

/// Error type for QLoRA operations.
#[derive(Debug, Clone, PartialEq)]
pub enum QloraError {
    /// Tensor shapes do not match (includes a human-readable message).
    ShapeMismatch(String),
    /// Invalid configuration value (includes a human-readable message).
    InvalidConfig(String),
    /// GPU backend failure or unavailable feature.
    Gpu(String),
}

impl fmt::Display for QloraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QloraError::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            QloraError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            QloraError::Gpu(msg) => write!(f, "gpu error: {msg}"),
        }
    }
}

impl std::error::Error for QloraError {}
