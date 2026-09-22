//! `qlora-tokenizer`: BPE tokenizer loaded from `tokenizer.json`.
//!
//! Supports the GPT-2-style family (SmolLM): `BPE` model,
//! `Sequence[Digits(individual_digits), ByteLevel]` pre-tokenizer,
//! `ByteLevel` decoder, no normalizer. Anything else is rejected at load.
//!
//! # Example
//!
//! ```no_run
//! use qlora_tokenizer::Tokenizer;
//!
//! let bytes = std::fs::read("models/SmolLM-135M/tokenizer.json").unwrap();
//! let tok = Tokenizer::from_json(&bytes).unwrap();
//! let ids = tok.encode("Hello, world!").unwrap();
//! assert_eq!(tok.decode(&ids).unwrap(), "Hello, world!");
//! ```

pub mod bpe;
pub mod json;

pub use bpe::{Error, Tokenizer};
