//! Reader tests against real numpy output (see `fixtures/`, generated with
//! numpy 2.4.6 in `guix shell -f guix.scm`). Expected values below were
//! extracted from the same files, so any drift means the reader is wrong.

use qlora_io::{read_npy, read_npz, write_npz, NamedArray};
use std::path::PathBuf;

fn fixture(name: &str) -> Vec<u8> {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    std::fs::read(&p).unwrap()
}

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol, "elem {i}: {x} vs {y}");
    }
}

#[test]
fn numpy_f32_c_order() {
    let a = read_npy(&fixture("f32_c.npy")).unwrap();
    assert_eq!(a.shape, vec![3, 4]);
    approx(&a.data[..3], &[0.12573022, -0.13210486, 0.64042264], 1e-7);
    assert_eq!(a.data.len(), 12);
}

#[test]
fn numpy_f64_converts() {
    let a = read_npy(&fixture("f64.npy")).unwrap();
    assert_eq!(a.shape, vec![2, 5]);
    approx(&a.data[..2], &[-2.3250308, -0.21879166], 1e-6);
}

#[test]
fn numpy_int_and_bool() {
    let a = read_npy(&fixture("i32.npy")).unwrap();
    assert_eq!(a.shape, vec![3, 4]);
    assert_eq!(&a.data[..4], &[0.0, 1.0, 2.0, 3.0]);
    let b = read_npy(&fixture("bool.npy")).unwrap();
    assert_eq!(b.shape, vec![2, 2]);
    assert_eq!(&b.data, &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
#[allow(clippy::excessive_precision)] // f16 values need 11 significant digits
fn numpy_f16_exact() {
    // f16 values are exactly representable in f32.
    let a = read_npy(&fixture("f16.npy")).unwrap();
    assert_eq!(a.shape, vec![2, 3]);
    assert_eq!(&a.data[..3], &[-0.6650390625, 0.3515625, 0.9033203125]);
}

#[test]
fn numpy_fortran_order_transposed_to_c() {
    let a = read_npy(&fixture("f32_fortran.npy")).unwrap();
    assert_eq!(a.shape, vec![3, 4]);
    // C-order view of the same matrix.
    approx(&a.data[..3], &[-0.45772582, 0.22019513, -1.0096182], 1e-7);
    assert_eq!(a.data.len(), 12);
}

#[test]
fn numpy_big_endian() {
    let a = read_npy(&fixture("be_f32.npy")).unwrap();
    assert_eq!(a.shape, vec![2, 2]);
    approx(&a.data[..2], &[1.0039616, -0.61790705], 1e-7);
}

#[test]
fn numpy_multi_array_npz() {
    let arrays = read_npz(&fixture("multi.npz")).unwrap();
    assert_eq!(arrays.len(), 3);
    assert_eq!(arrays[0].name, "w");
    assert_eq!(arrays[1].name, "b");
    assert_eq!(arrays[2].name, "steps");
    let w = NamedArray::find(&arrays, "w").unwrap();
    assert_eq!(w.shape, vec![4, 8]);
    approx(&w.data[..3], &[-1.2590655, 1.5139238, 1.3458754], 1e-7);
    let b = NamedArray::find(&arrays, "b").unwrap();
    assert!(b.data.iter().all(|&v| v == 0.0));
    let steps = NamedArray::find(&arrays, "steps").unwrap();
    assert_eq!(steps.shape, vec![2, 3]);
    assert_eq!(&steps.data, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn compressed_npz_rejected_with_actionable_message() {
    let err = read_npz(&fixture("compressed.npz")).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("savez"),
        "error should tell the user to resave uncompressed, got: {msg}"
    );
}

#[test]
fn our_writer_roundtrips_through_real_numpy() {
    // This test only checks our own roundtrip; the numpy direction is
    // verified by hand in the shell (see README): np.load must read our
    // files, which constrains header layout and CRC.
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let bytes = write_npz(&[("w", &[4, 6], &data), ("b", &[6], &data[..6])]);
    let back = read_npz(&bytes).unwrap();
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].data, data);
    assert_eq!(back[1].data, data[..6]);
}
