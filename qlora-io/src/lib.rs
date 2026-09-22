//! `qlora-io`: dependency-free NumPy `.npy` / `.npz` weight interchange.
//!
//! Role in the project: `.npz` is the safe, simple interchange format
//! between the Python/torch world and the Rust/WGPU side. `.pt` files are
//! handled at the boundary by `tools/pt2npz.py` (torch required) because
//! pickle is executable code and does not belong in a Rust parser.
//!
//! Supported subset (deliberate, with explicit errors outside it):
//!
//! * `.npy` versions 1.0 and 2.0; dtypes `<f8 <f4 <f2 <i8 <i4 <i2 <i1`
//!   `<u8 <u4 <u2 <u1 |u1 |b1` (plus big-endian `>` variants of the numeric
//!   ones). Everything is converted to `f32`. Complex, strings, void,
//!   object arrays are rejected.
//! * C-order and Fortran-order (converted to C-order on load).
//! * `.npz`: stored (uncompressed) entries only — that is what
//!   `np.savez` writes. `np.savez_compressed` output is rejected with a
//!   message telling the user to resave uncompressed.
//!
//! # Example
//!
//! ```
//! use qlora_io::{read_npz, write_npz};
//!
//! let bytes = write_npz(&[("w", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
//! let arrays = read_npz(&bytes).unwrap();
//! assert_eq!(arrays[0].name, "w");
//! assert_eq!(arrays[0].shape, vec![2, 3]);
//! assert_eq!(arrays[0].data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! ```

pub mod error;
mod npy;
pub mod npz;
pub mod safetensors;

pub use error::IoError;
pub use npy::write_npy_bytes as write_npy;
pub use npz::{read_npy, read_npz, write_npz, NamedArray};
pub use safetensors::read_safetensors;
