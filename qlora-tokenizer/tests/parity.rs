//! Parity with HuggingFace `transformers` on SmolLM-135M's tokenizer.
//!
//! Expected vectors were produced by
//! `AutoTokenizer.from_pretrained(models/SmolLM-135M)` inside
//! `guix shell -f guix.scm`. Regenerate after any pre-tokenizer change:
//! see the dump procedure in git history (examples/dump.rs).

use qlora_tokenizer::Tokenizer;
use std::path::PathBuf;

fn tokenizer() -> Tokenizer {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crate dir -> workspace root
    p.push("models");
    p.push("SmolLM-135M");
    p.push("tokenizer.json");
    let bytes = std::fs::read(&p).unwrap();
    Tokenizer::from_json(&bytes).unwrap()
}

fn check(text: &str, expected: &[u32]) {
    let tok = tokenizer();
    let ids = tok.encode(text).unwrap();
    assert_eq!(ids, expected, "encode mismatch for {text:?}");
    // Roundtrip must restore the exact text (ByteLevel decoder).
    assert_eq!(tok.decode(&ids).unwrap(), text);
}

#[test]
fn parity_basic() {
    check("Hello, world!", &[19556, 28, 905, 17]);
    check("a", &[81]);
    check("", &[]);
}

#[test]
fn parity_spaces_digits_code() {
    check(
        "The quick brown fox jumps over 13 lazy dogs.",
        &[
            504, 2365, 6354, 16438, 27003, 690, 216, 33, 35, 23790, 5046, 30,
        ],
    );
    check(
        "  leading and trailing spaces  ",
        &[216, 2899, 284, 35079, 5600, 256],
    );
    check(
        "line1\nline2\n\nline3",
        &[1311, 33, 198, 1311, 34, 198, 198, 1311, 35],
    );
    check(
        "123 4567 3.14159",
        &[
            33, 34, 35, 216, 36, 37, 38, 39, 216, 35, 30, 33, 36, 33, 37, 41,
        ],
    );
    check(
        "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        &[
            1604, 3987, 24, 94, 727, 1003, 304, 585, 304, 2067, 216, 34, 1745, 3987, 24, 94, 29,
            33, 25, 1232, 3987, 24, 94, 29, 34, 25,
        ],
    );
}

#[test]
fn parity_unicode_and_chat_template() {
    check(
        "中文測試 Under the sea 🌊 emoji!",
        &[
            28589, 29184, 177, 133, 122, 179, 119, 116, 2995, 260, 3426, 15107, 230, 228, 649,
            33777, 17,
        ],
    );
    check(
        "<|im_start|>user\nHi there!<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>",
        &[
            1, 4093, 198, 26843, 665, 17, 2, 198, 1, 520, 9531, 198, 19556, 17, 2,
        ],
    );
}
