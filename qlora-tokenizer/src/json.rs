//! Minimal owned JSON parser (objects, arrays, strings with `\u` including
//! surrogate pairs, numbers, bool, null). Enough for `tokenizer.json`.

use std::collections::HashMap;

/// An owned JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "json: {}", self.0)
    }
}

impl std::error::Error for Error {}

struct Parser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> Error {
        Error(format!("{msg} at byte {}", self.pos))
    }

    fn skip_ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), Error> {
        self.skip_ws();
        if self.b.get(self.pos) != Some(&ch) {
            return Err(self.err("unexpected character"));
        }
        self.pos += 1;
        Ok(())
    }

    fn value(&mut self) -> Result<Value, Error> {
        self.skip_ws();
        match self.b.get(self.pos) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(self.err("unexpected value")),
        }
    }

    fn literal(&mut self, word: &str, v: Value) -> Result<Value, Error> {
        if self.b[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(v)
        } else {
            Err(self.err("bad literal"))
        }
    }

    fn number(&mut self) -> Result<Value, Error> {
        let start = self.pos;
        while self.pos < self.b.len()
            && matches!(
                self.b[self.pos],
                b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
            )
        {
            self.pos += 1;
        }
        let s =
            std::str::from_utf8(&self.b[start..self.pos]).map_err(|_| self.err("bad number"))?;
        s.parse::<f64>()
            .map(Value::Number)
            .map_err(|_| self.err("bad number"))
    }

    fn hex4(&mut self) -> Result<u32, Error> {
        if self.pos + 4 > self.b.len() {
            return Err(self.err("bad unicode escape"));
        }
        let s = std::str::from_utf8(&self.b[self.pos..self.pos + 4])
            .map_err(|_| self.err("bad escape"))?;
        self.pos += 4;
        u32::from_str_radix(s, 16).map_err(|_| self.err("bad escape"))
    }

    fn string(&mut self) -> Result<String, Error> {
        // Assumes current byte is the opening quote.
        self.pos += 1;
        let mut s = String::new();
        loop {
            match self.b.get(self.pos) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(s);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.b.get(self.pos) {
                        Some(b'"') => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'/') => s.push('/'),
                        Some(b'b') => s.push('\x08'),
                        Some(b'f') => s.push('\x0c'),
                        Some(b'n') => s.push('\n'),
                        Some(b'r') => s.push('\r'),
                        Some(b't') => s.push('\t'),
                        Some(b'u') => {
                            self.pos += 1;
                            let cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                // High surrogate: expect \uDC00..\uDFFF.
                                if self.b.get(self.pos) == Some(&b'\\')
                                    && self.b.get(self.pos + 1) == Some(&b'u')
                                {
                                    self.pos += 2;
                                    let lo = self.hex4()?;
                                    if !(0xDC00..0xE000).contains(&lo) {
                                        return Err(self.err("bad surrogate pair"));
                                    }
                                    let full = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    s.push(
                                        char::from_u32(full).ok_or_else(|| self.err("bad char"))?,
                                    );
                                } else {
                                    return Err(self.err("lone surrogate"));
                                }
                            } else if (0xDC00..0xE000).contains(&cp) {
                                return Err(self.err("lone surrogate"));
                            } else {
                                s.push(char::from_u32(cp).ok_or_else(|| self.err("bad char"))?);
                            }
                            continue;
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                    self.pos += 1;
                }
                Some(&b0) => {
                    // Raw UTF-8 bytes: decode one char by leading byte
                    // (O(1); validating the whole remainder each time
                    // would be O(n^2) on large files).
                    let len = if b0 < 0x80 {
                        1
                    } else if b0 >> 5 == 0b110 {
                        2
                    } else if b0 >> 4 == 0b1110 {
                        3
                    } else if b0 >> 3 == 0b11110 {
                        4
                    } else {
                        return Err(self.err("bad utf-8"));
                    };
                    if self.pos + len > self.b.len() {
                        return Err(self.err("bad utf-8"));
                    }
                    for k in 1..len {
                        if self.b[self.pos + k] >> 6 != 0b10 {
                            return Err(self.err("bad utf-8"));
                        }
                    }
                    let st = std::str::from_utf8(&self.b[self.pos..self.pos + len])
                        .map_err(|_| self.err("bad utf-8"))?;
                    let ch = st.chars().next().ok_or_else(|| self.err("bad utf-8"))?;
                    s.push(ch);
                    self.pos += len;
                }
            }
        }
    }

    fn array(&mut self) -> Result<Value, Error> {
        self.pos += 1; // [
        let mut v = Vec::new();
        self.skip_ws();
        if self.b.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Value::Array(v));
        }
        loop {
            v.push(self.value()?);
            self.skip_ws();
            match self.b.get(self.pos) {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws();
                    if self.b.get(self.pos) == Some(&b']') {
                        return Err(self.err("trailing comma"));
                    }
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(v));
                }
                _ => return Err(self.err("expected , or ]")),
            }
        }
    }

    fn object(&mut self) -> Result<Value, Error> {
        self.pos += 1; // {
        let mut m = HashMap::new();
        loop {
            self.skip_ws();
            if self.b.get(self.pos) == Some(&b'}') {
                self.pos += 1;
                return Ok(Value::Object(m));
            }
            if self.b.get(self.pos) != Some(&b'"') {
                return Err(self.err("expected string key"));
            }
            let key = self.string()?;
            self.expect(b':')?;
            m.insert(key, self.value()?);
            self.skip_ws();
            match self.b.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => continue,
                _ => return Err(self.err("expected , or }")),
            }
        }
    }
}

/// Parse a JSON document.
pub fn parse(bytes: &[u8]) -> Result<Value, Error> {
    let mut p = Parser { b: bytes, pos: 0 };
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.b.len() {
        return Err(p.err("trailing data"));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        assert_eq!(parse(br#"{"a": [1, -2.5, true, null]}"#).unwrap(), {
            let mut m = HashMap::new();
            m.insert(
                "a".to_string(),
                Value::Array(vec![
                    Value::Number(1.0),
                    Value::Number(-2.5),
                    Value::Bool(true),
                    Value::Null,
                ]),
            );
            Value::Object(m)
        });
        assert!(parse(br#"{"a": }"#).is_err());
        assert!(parse(br#"[1,]"#).is_err());
    }

    #[test]
    fn string_escapes_and_surrogates() {
        let v = parse(br#"{"s": "A\n\u00e9\ud83d\ude00"}"#).unwrap();
        assert_eq!(
            v.get("s").and_then(|v| v.as_str()),
            Some("A\n\u{e9}\u{1f600}")
        );
        assert!(parse(br#"{"s": "\ud83d"}"#).is_err());
    }
}
