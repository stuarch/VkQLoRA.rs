//! BPE encode/decode for the GPT-2-style family used by SmolLM:
//! `Sequence[Digits(individual_digits), ByteLevel]` pre-tokenizer,
//! byte-level BPE model, `ByteLevel` decoder, no normalizer.
//!
//! The loader validates the `tokenizer.json` recipe and refuses anything
//! else — silent mis-tokenization is worse than an error.

use fancy_regex::Regex;
use std::collections::HashMap;

use crate::json::Value;

#[derive(Debug, Clone)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tokenizer: {}", self.0)
    }
}

impl std::error::Error for Error {}

/// GPT-2 byte→unicode table (bytes 0x00-0xFF to chars).
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut kept = Vec::new();
    for b in 0x21u32..=0x7E {
        kept.push(b);
    }
    for b in 0xA1u32..=0xAC {
        kept.push(b);
    }
    for b in 0xAEu32..=0xFF {
        kept.push(b);
    }
    let mut extra = 0x100u32;
    for b in 0u32..256 {
        if kept.contains(&b) {
            table[b as usize] = char::from_u32(b).unwrap();
        } else {
            table[b as usize] = char::from_u32(extra).unwrap();
            extra += 1;
        }
    }
    table
}

/// The GPT-2 pre-tokenizer split pattern (HF `ByteLevel(use_regex=true)`).
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ids: Vec<String>,
    ranks: HashMap<(String, String), usize>,
    /// (content, id), longest-first for greedy matching.
    added: Vec<(String, u32)>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    word_re: Regex,
}

impl Tokenizer {
    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        let v = crate::json::parse(bytes).map_err(|e| Error(e.to_string()))?;
        let fail = |msg: &str| Error(format!("unsupported tokenizer.json recipe: {msg}"));

        let model = v.get("model").ok_or_else(|| fail("no model"))?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
            return Err(fail("model.type != BPE"));
        }
        // Validate (not necessarily use) the pre-tokenizer recipe.
        let pre = v
            .get("pre_tokenizer")
            .ok_or_else(|| fail("no pre_tokenizer"))?;
        if pre.get("type").and_then(|t| t.as_str()) != Some("Sequence") {
            return Err(fail("pre_tokenizer.type != Sequence"));
        }
        let parts = pre
            .get("pretokenizers")
            .and_then(|p| p.as_array())
            .ok_or_else(|| fail("no pretokenizers"))?;
        if parts.len() != 2
            || parts[0].get("type").and_then(|t| t.as_str()) != Some("Digits")
            || parts[0].get("individual_digits").and_then(|t| t.as_bool()) != Some(true)
            || parts[1].get("type").and_then(|t| t.as_str()) != Some("ByteLevel")
        {
            return Err(fail("need Sequence[Digits(individual_digits), ByteLevel]"));
        }
        if v.get("normalizer") != Some(&Value::Null) {
            return Err(fail("normalizer must be null"));
        }

        let vocab_obj = model
            .get("vocab")
            .and_then(|x| match x {
                Value::Object(m) => Some(m),
                _ => None,
            })
            .ok_or_else(|| fail("no model.vocab"))?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0;
        for (tok, id) in vocab_obj {
            let id = match id {
                Value::Number(n) => *n as u32,
                _ => return Err(fail("vocab id not a number")),
            };
            max_id = max_id.max(id);
            vocab.insert(tok.clone(), id);
        }
        let mut ids = vec![String::new(); max_id as usize + 1];
        for (tok, id) in &vocab {
            ids[*id as usize] = tok.clone();
        }

        let mut ranks = HashMap::new();
        if let Some(merges) = model.get("merges").and_then(|m| m.as_array()) {
            for (rank, m) in merges.iter().enumerate() {
                let line = m.as_str().ok_or_else(|| fail("merge not a string"))?;
                let mut it = line.split(' ');
                let (a, b) = match (it.next(), it.next(), it.next()) {
                    (Some(a), Some(b), None) => (a.to_string(), b.to_string()),
                    _ => return Err(fail("bad merge line")),
                };
                ranks.insert((a, b), rank);
            }
        }

        let mut added = Vec::new();
        if let Some(list) = v.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in list {
                let id = tok
                    .get("id")
                    .and_then(|x| match x {
                        Value::Number(n) => Some(*n as u32),
                        _ => None,
                    })
                    .ok_or_else(|| fail("added token without numeric id"))?;
                let content = tok
                    .get("content")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| fail("added token without content"))?;
                added.push((content.to_string(), id));
            }
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(*c, b as u8);
        }
        let word_re = Regex::new(GPT2_PATTERN).map_err(|e| Error(format!("regex: {e}")))?;

        Ok(Self {
            vocab,
            ids,
            ranks,
            added,
            byte_encoder,
            byte_decoder,
            word_re,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.ids.len()
    }

    /// Tokenize text to ids (special added tokens split first, never merged).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for chunk in self.split_added(text) {
            match chunk {
                Chunk::Special(id) => out.push(id),
                Chunk::Text(s) => {
                    for piece in split_digits(s) {
                        for m in self.word_re.find_iter(piece) {
                            let m = m.map_err(|e| Error(format!("regex: {e}")))?;
                            out.extend(self.bpe_word(m.as_str())?);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Detokenize ids (ByteLevel decoder).
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let mut text = String::new();
        for &id in ids {
            text.push_str(
                self.ids
                    .get(id as usize)
                    .ok_or_else(|| Error(format!("unknown id {id}")))?,
            );
        }
        let mut bytes = Vec::with_capacity(text.len());
        for ch in text.chars() {
            bytes.push(
                *self
                    .byte_decoder
                    .get(&ch)
                    .ok_or_else(|| Error(format!("undecodable char {ch:?}")))?,
            );
        }
        String::from_utf8(bytes).map_err(|e| Error(format!("decode utf-8: {e}")))
    }

    /// Split off special added tokens (longest match at each position).
    fn split_added<'a>(&self, text: &'a str) -> Vec<Chunk<'a>> {
        let mut chunks = Vec::new();
        let mut i = 0;
        let mut start = 0;
        while i < text.len() {
            let mut hit: Option<(&str, u32)> = None;
            for (content, id) in &self.added {
                if text[i..].starts_with(content) {
                    hit = Some((content, *id));
                    break; // `added` is longest-first
                }
            }
            if let Some((content, id)) = hit {
                if i > start {
                    chunks.push(Chunk::Text(&text[start..i]));
                }
                chunks.push(Chunk::Special(id));
                i += content.len();
                start = i;
            } else {
                // Advance one char (keep char boundary).
                let ch = text[i..].chars().next().unwrap();
                i += ch.len_utf8();
            }
        }
        if start < text.len() {
            chunks.push(Chunk::Text(&text[start..]));
        }
        chunks
    }

    /// BPE-merge one pre-tokenizer word to ids.
    fn bpe_word(&self, word: &str) -> Result<Vec<u32>, Error> {
        // Map UTF-8 bytes through the byte-level alphabet.
        let mut symbols: Vec<String> = Vec::new();
        for &b in word.as_bytes() {
            symbols.push(self.byte_encoder[b as usize].to_string());
        }
        if symbols.is_empty() {
            return Ok(Vec::new());
        }
        if symbols.len() == 1 {
            return Ok(vec![self.lookup(&symbols[0])?]);
        }
        loop {
            // Lowest-rank adjacent pair.
            let mut best: Option<(usize, usize)> = None; // (rank, pos)
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.map(|(r, _)| rank < r).unwrap_or(true) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, pos)) = best else { break };
            let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
            symbols.splice(pos..pos + 2, [merged]);
            if symbols.len() == 1 {
                break;
            }
        }
        symbols.iter().map(|s| self.lookup(s)).collect()
    }

    fn lookup(&self, token: &str) -> Result<u32, Error> {
        self.vocab
            .get(token)
            .copied()
            .ok_or_else(|| Error(format!("token not in vocab: {token:?}")))
    }
}

enum Chunk<'a> {
    Special(u32),
    Text(&'a str),
}

/// `Digits(individual_digits=true)`: "ab12" -> ["ab", "1", "2"].
///
/// Splits on ASCII digits (the HF reference uses unicode numerics; ASCII
/// covers model text in practice and is pinned by parity tests).
fn split_digits(s: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut in_digits = false;
    let mut first = true;
    for (i, ch) in s.char_indices() {
        let is_digit = ch.is_ascii_digit();
        if first {
            in_digits = is_digit;
            first = false;
            if is_digit {
                // Digit run always splits per-character; handled below.
            }
            continue;
        }
        if is_digit {
            if !in_digits {
                pieces.push(&s[start..i]);
                start = i;
                in_digits = true;
            } else {
                // individual digits: cut before this digit.
                pieces.push(&s[start..i]);
                start = i;
            }
        } else if in_digits {
            pieces.push(&s[start..i]);
            start = i;
            in_digits = false;
        }
    }
    pieces.push(&s[start..]);
    pieces.retain(|p| !p.is_empty());
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_split() {
        assert_eq!(split_digits("ab12"), vec!["ab", "1", "2"]);
        assert_eq!(split_digits("123"), vec!["1", "2", "3"]);
        assert_eq!(split_digits("abc"), vec!["abc"]);
        assert_eq!(split_digits("a1b2"), vec!["a", "1", "b", "2"]);
    }
}
