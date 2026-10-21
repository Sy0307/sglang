//! The Python values `Qwen3CoderDetector._convert_param_value` builds from a
//! parameter's text — `json.loads`, `ast.literal_eval`, `int()` and `float()`
//! results — and their `json.dumps`.
//!
//! `serde_json::Value` cannot hold what these produce: integers of any size,
//! `NaN`/`Infinity`, tuples, and values `json.dumps` refuses (sets, bytes,
//! complex numbers, `...`). Known gaps: lone surrogates become U+FFFD, `\N{...}`
//! escapes in Python string literals are rejected, and dict keys compare by
//! value without Python's `1 == 1.0 == True` folding.

use crate::py::{py_strip, python_float, write_python_str};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PyVal {
    None,
    Bool(bool),
    /// Canonical decimal digits, `-` for negatives.
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<PyVal>),
    Tuple(Vec<PyVal>),
    /// Insertion order; a repeated key keeps its first position and last value.
    Dict(Vec<(PyVal, PyVal)>),
    /// A set, bytes, complex number or `...`: `json.dumps` raises `TypeError`.
    Opaque,
}

impl PyVal {
    /// `dict[key] = value`.
    pub(crate) fn dict_insert(items: &mut Vec<(PyVal, PyVal)>, key: PyVal, value: PyVal) {
        match items.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => items.push((key, value)),
        }
    }

    fn hashable(&self) -> bool {
        match self {
            PyVal::List(_) | PyVal::Dict(_) => false,
            PyVal::Tuple(items) => items.iter().all(PyVal::hashable),
            _ => true,
        }
    }
}

/// `json.dumps(value, ensure_ascii=ascii)`; `None` where it raises.
pub(crate) fn dumps(value: &PyVal, ascii: bool) -> Option<String> {
    let mut out = String::new();
    write_dumps(value, ascii, &mut out)?;
    Some(out)
}

fn write_dumps(value: &PyVal, ascii: bool, out: &mut String) -> Option<()> {
    match value {
        PyVal::None => out.push_str("null"),
        PyVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        PyVal::Int(digits) => out.push_str(digits),
        PyVal::Float(f) => out.push_str(&python_float(*f)),
        PyVal::Str(s) => write_python_str(s, ascii, out),
        PyVal::List(items) | PyVal::Tuple(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_dumps(item, ascii, out)?;
            }
            out.push(']');
        }
        PyVal::Dict(items) => {
            out.push('{');
            for (i, (key, item)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                // `json.dumps` turns these key types into strings and refuses the rest.
                let key = match key {
                    PyVal::Str(s) => s.clone(),
                    PyVal::Int(d) => d.clone(),
                    PyVal::Float(f) => python_float(*f),
                    PyVal::Bool(b) => (if *b { "true" } else { "false" }).to_owned(),
                    PyVal::None => "null".to_owned(),
                    _ => return None,
                };
                write_python_str(&key, ascii, out);
                out.push_str(": ");
                write_dumps(item, ascii, out)?;
            }
            out.push('}');
        }
        PyVal::Opaque => return None,
    }
    Some(())
}

/// Python's `str.lower()` for the ASCII-only comparisons the detector makes.
pub(crate) fn py_lower(s: &str) -> String {
    s.to_lowercase()
}

// ---------------------------------------------------------------- json.loads

/// `json.loads(text)` (strict mode, `NaN`/`Infinity` accepted); `None` on
/// `JSONDecodeError`.
pub(crate) fn json_loads(text: &str) -> Option<PyVal> {
    let mut p = JsonParser { s: text.as_bytes(), text, i: 0, depth: 0 };
    p.ws();
    let value = p.value()?;
    p.ws();
    (p.i == p.s.len()).then_some(value)
}

struct JsonParser<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
    depth: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.text[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<PyVal> {
        let c = *self.s.get(self.i)?;
        match c {
            b'{' => {
                self.depth += 1;
                if self.depth > 900 {
                    return None;
                }
                self.i += 1;
                self.ws();
                let mut items = Vec::new();
                if self.eat("}") {
                    self.depth -= 1;
                    return Some(PyVal::Dict(items));
                }
                loop {
                    if self.s.get(self.i) != Some(&b'"') {
                        return None;
                    }
                    let key = self.string()?;
                    self.ws();
                    if !self.eat(":") {
                        return None;
                    }
                    self.ws();
                    let value = self.value()?;
                    PyVal::dict_insert(&mut items, PyVal::Str(key), value);
                    self.ws();
                    if self.eat(",") {
                        self.ws();
                        continue;
                    }
                    if self.eat("}") {
                        self.depth -= 1;
                        return Some(PyVal::Dict(items));
                    }
                    return None;
                }
            }
            b'[' => {
                self.depth += 1;
                if self.depth > 900 {
                    return None;
                }
                self.i += 1;
                self.ws();
                let mut items = Vec::new();
                if self.eat("]") {
                    self.depth -= 1;
                    return Some(PyVal::List(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    if self.eat(",") {
                        self.ws();
                        continue;
                    }
                    if self.eat("]") {
                        self.depth -= 1;
                        return Some(PyVal::List(items));
                    }
                    return None;
                }
            }
            b'"' => self.string().map(PyVal::Str),
            b'n' if self.eat("null") => Some(PyVal::None),
            b't' if self.eat("true") => Some(PyVal::Bool(true)),
            b'f' if self.eat("false") => Some(PyVal::Bool(false)),
            b'N' if self.eat("NaN") => Some(PyVal::Float(f64::NAN)),
            b'I' if self.eat("Infinity") => Some(PyVal::Float(f64::INFINITY)),
            b'-' if self.text[self.i..].starts_with("-Infinity") => {
                self.i += "-Infinity".len();
                Some(PyVal::Float(f64::NEG_INFINITY))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    /// `NUMBER_RE`: `(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?`.
    fn number(&mut self) -> Option<PyVal> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let from = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - from
        };
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return None,
        }
        let int_end = self.i;
        let mut is_float = false;
        if self.s.get(self.i) == Some(&b'.') {
            let save = self.i;
            self.i += 1;
            if digits(self) == 0 {
                self.i = save;
            } else {
                is_float = true;
            }
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                self.i = save;
            } else {
                is_float = true;
            }
        }
        let lit = &self.text[start..self.i];
        if is_float {
            return lit.parse::<f64>().ok().map(PyVal::Float);
        }
        canonical_int(&self.text[start..int_end])
    }

    fn string(&mut self) -> Option<String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let rest = &self.text[self.i..];
            let c = rest.chars().next()?;
            self.i += c.len_utf8();
            match c {
                '"' => return Some(out),
                '\\' => {
                    let e = *self.s.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            if (0xd800..0xdc00).contains(&hi) && self.text[self.i..].starts_with("\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    let cp = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                                    continue;
                                }
                                self.i = save;
                            }
                            out.push(char::from_u32(hi).unwrap_or('\u{fffd}'));
                        }
                        _ => return None,
                    }
                }
                c if (c as u32) < 0x20 => return None,
                c => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.text.get(self.i..self.i + 4)?;
        if !h.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }
}

/// Decimal digits (optional `-`, leading zeros allowed) as an `Int`, refusing
/// past CPython's 4300-digit `int`/`str` conversion limit.
fn canonical_int(lit: &str) -> Option<PyVal> {
    let (neg, digits) = match lit.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, lit.strip_prefix('+').unwrap_or(lit)),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) || digits.len() > 4300 {
        return None;
    }
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return Some(PyVal::Int("0".into()));
    }
    Some(PyVal::Int(if neg { format!("-{trimmed}") } else { trimmed.to_owned() }))
}

/// PEP 515 digits: underscores only between digits. Returns the digits without them.
fn underscored_digits(s: &str, radix: u32) -> Option<String> {
    if s.is_empty() || s.starts_with('_') || s.ends_with('_') || s.contains("__") {
        return None;
    }
    let out: String = s.chars().filter(|&c| c != '_').collect();
    out.chars().all(|c| c.is_digit(radix)).then_some(out)
}

// ---------------------------------------------------------------- int(), float()

/// `int(text)`: Python whitespace stripped, an optional sign, decimal digits
/// with PEP 515 underscores.
pub(crate) fn py_int(text: &str) -> Option<PyVal> {
    let s = py_strip(text);
    let (sign, body) = match s.as_bytes().first() {
        Some(b'-') => ("-", &s[1..]),
        Some(b'+') => ("", &s[1..]),
        _ => ("", s),
    };
    let digits = underscored_digits(body, 10)?;
    canonical_int(&format!("{sign}{digits}"))
}

/// `float(text)`.
pub(crate) fn py_float(text: &str) -> Option<f64> {
    let s = py_strip(text);
    let (neg, body) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let lower = body.to_ascii_lowercase();
    let magnitude = match lower.as_str() {
        "inf" | "infinity" => f64::INFINITY,
        "nan" => f64::NAN,
        _ => {
            // digitpart [. [digitpart]] [exp] | . digitpart [exp]
            let (mantissa, exp) = match lower.find('e') {
                Some(e) => (&lower[..e], Some(&lower[e + 1..])),
                None => (lower.as_str(), None),
            };
            let (int_part, frac_part) = match mantissa.split_once('.') {
                Some((i, f)) => (i, Some(f)),
                None => (mantissa, None),
            };
            let int_digits = if int_part.is_empty() { String::new() } else { underscored_digits(int_part, 10)? };
            let frac_digits = match frac_part {
                Some("") | None => String::new(),
                Some(f) => underscored_digits(f, 10)?,
            };
            if int_digits.is_empty() && frac_digits.is_empty() {
                return None;
            }
            let exp_text = match exp {
                None => String::new(),
                Some(e) => {
                    let (esign, edigits) = match e.as_bytes().first() {
                        Some(b'-') => ("-", &e[1..]),
                        Some(b'+') => ("", &e[1..]),
                        _ => ("", e),
                    };
                    format!("e{esign}{}", underscored_digits(edigits, 10)?)
                }
            };
            let canonical = format!(
                "{}.{}{exp_text}",
                if int_digits.is_empty() { "0" } else { &int_digits },
                if frac_digits.is_empty() { "0" } else { &frac_digits }
            );
            canonical.parse::<f64>().ok()?
        }
    };
    Some(if neg { -magnitude } else { magnitude })
}

/// `int(f)` of an integral finite float, exactly.
pub(crate) fn float_to_int(f: f64) -> PyVal {
    if f.abs() < 9.0e15 {
        return PyVal::Int(format!("{}", f as i64));
    }
    let bits = f.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64 - 1075;
    let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    // value = mantissa * 2^exp with exp > 0 here; base-1e9 limbs, little endian.
    let mut limbs: Vec<u64> = vec![mantissa % 1_000_000_000, mantissa / 1_000_000_000];
    for _ in 0..exp {
        let mut carry = 0u64;
        for limb in limbs.iter_mut() {
            let v = *limb * 2 + carry;
            *limb = v % 1_000_000_000;
            carry = v / 1_000_000_000;
        }
        if carry > 0 {
            limbs.push(carry);
        }
    }
    while limbs.len() > 1 && *limbs.last().unwrap() == 0 {
        limbs.pop();
    }
    let mut s = limbs.last().unwrap().to_string();
    for limb in limbs.iter().rev().skip(1) {
        s.push_str(&format!("{limb:09}"));
    }
    PyVal::Int(if f < 0.0 { format!("-{s}") } else { s })
}

// ---------------------------------------------------------------- ast.literal_eval

/// `ast.literal_eval(text)`; `None` where it raises (`ValueError`,
/// `SyntaxError`, `TypeError` for an unhashable key or set item).
pub(crate) fn literal_eval(text: &str) -> Option<PyVal> {
    let text = text.trim_start_matches([' ', '\t']);
    let tokens = tokenize(text)?;
    let mut p = LitParser { t: tokens, i: 0 };
    let value = p.top()?;
    (p.i == p.t.len()).then_some(value)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Op(char),
    Name(String),
    Num(Num),
    Str(StrLit),
}

#[derive(Debug, Clone, PartialEq)]
enum Num {
    Int(PyVal),
    Float(f64),
    Imag,
}

#[derive(Debug, Clone, PartialEq)]
struct StrLit {
    bytes: bool,
    value: String,
}

/// Python's tokenizer for the expression subset `literal_eval` accepts:
/// brackets join lines, comments and `\` continuations are skipped, and a
/// newline outside brackets ends the (single) logical line.
fn tokenize(s: &str) -> Option<Vec<Tok>> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let mut depth = 0i32;
    let mut ended = false; // a NEWLINE token was produced at depth 0
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\u{c}' => i += 1,
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '\\' if chars.get(i + 1) == Some(&'\n') => i += 2,
            '\r' | '\n' => {
                if depth == 0 && !out.is_empty() {
                    ended = true;
                }
                i += 1;
            }
            _ if ended => return None,
            '(' | '[' | '{' => {
                depth += 1;
                out.push(Tok::Op(c));
                i += 1;
            }
            ')' | ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
                out.push(Tok::Op(c));
                i += 1;
            }
            ',' | ':' | '+' | '-' => {
                out.push(Tok::Op(c));
                i += 1;
            }
            '.' if chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()) => {
                let (tok, n) = number(&chars[i..])?;
                out.push(Tok::Num(tok));
                i += n;
            }
            '.' if chars[i..].starts_with(&['.', '.', '.']) => {
                out.push(Tok::Name("...".into()));
                i += 3;
            }
            '0'..='9' => {
                let (tok, n) = number(&chars[i..])?;
                out.push(Tok::Num(tok));
                i += n;
            }
            c if c == '_' || c.is_alphabetic() => {
                let start = i;
                while i < chars.len() && (chars[i] == '_' || chars[i].is_alphanumeric()) {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                let prefix = name.to_ascii_lowercase();
                let quote = chars.get(i).copied();
                if matches!(quote, Some('\'' | '"'))
                    && matches!(prefix.as_str(), "r" | "u" | "b" | "br" | "rb" | "f" | "fr" | "rf")
                {
                    if prefix.contains('f') {
                        return None; // f-strings are not literals
                    }
                    let (lit, n) = string(&chars[i..], prefix.contains('r'), prefix.contains('b'))?;
                    out.push(Tok::Str(lit));
                    i += n;
                } else {
                    out.push(Tok::Name(name));
                }
            }
            '\'' | '"' => {
                let (lit, n) = string(&chars[i..], false, false)?;
                out.push(Tok::Str(lit));
                i += n;
            }
            _ => return None,
        }
    }
    (depth == 0).then_some(out)
}

fn number(c: &[char]) -> Option<(Num, usize)> {
    let text: String = c.iter().collect();
    let lower = text.to_ascii_lowercase();
    let word_end = |from: usize| {
        lower[from..]
            .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .map_or(lower.len(), |n| from + n)
    };
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if lower.starts_with(prefix) {
            let end = word_end(2);
            let mut body = &lower[2..end];
            body = body.strip_prefix('_').unwrap_or(body);
            let digits = underscored_digits(body, radix)?;
            return Some((Num::Int(radix_to_decimal(&digits, radix)?), end));
        }
    }
    // decimal int / float / imaginary
    let bytes = lower.as_bytes();
    let mut i = 0;
    let scan_digits = |i: &mut usize| {
        let from = *i;
        while *i < bytes.len() && (bytes[*i].is_ascii_digit() || bytes[*i] == b'_') {
            *i += 1;
        }
        from
    };
    let int_from = scan_digits(&mut i);
    let int_part = &lower[int_from..i];
    let mut is_float = false;
    let mut frac_part = "";
    if bytes.get(i) == Some(&b'.') {
        is_float = true;
        i += 1;
        let from = scan_digits(&mut i);
        frac_part = &lower[from..i];
    }
    let mut exp_part = String::new();
    if bytes.get(i) == Some(&b'e') {
        let save = i;
        i += 1;
        let mut sign = "";
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            sign = if bytes[i] == b'-' { "-" } else { "" };
            i += 1;
        }
        let from = scan_digits(&mut i);
        if from == i {
            i = save;
        } else {
            exp_part = format!("e{sign}{}", underscored_digits(&lower[from..i], 10)?);
            is_float = true;
        }
    }
    let imag = bytes.get(i) == Some(&b'j');
    if imag {
        i += 1;
    }
    // A name character right after the literal is a syntax error ("1abc").
    if bytes.get(i).is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_') {
        return None;
    }
    let int_digits = if int_part.is_empty() { String::new() } else { underscored_digits(int_part, 10)? };
    let frac_digits = if frac_part.is_empty() { String::new() } else { underscored_digits(frac_part, 10)? };
    if imag {
        return Some((Num::Imag, i));
    }
    if is_float {
        let canonical = format!(
            "{}.{}{exp_part}",
            if int_digits.is_empty() { "0" } else { &int_digits },
            if frac_digits.is_empty() { "0" } else { &frac_digits }
        );
        return Some((Num::Float(canonical.parse().ok()?), i));
    }
    // Decimal literals other than zero may not have leading zeros.
    if int_digits.len() > 1 && int_digits.starts_with('0') && int_digits.bytes().any(|b| b != b'0') {
        return None;
    }
    Some((Num::Int(canonical_int(&int_digits)?), i))
}

fn radix_to_decimal(digits: &str, radix: u32) -> Option<PyVal> {
    let mut limbs: Vec<u64> = vec![0];
    for d in digits.chars() {
        let mut carry = d.to_digit(radix)? as u64;
        for limb in limbs.iter_mut() {
            let v = *limb * radix as u64 + carry;
            *limb = v % 1_000_000_000;
            carry = v / 1_000_000_000;
        }
        while carry > 0 {
            limbs.push(carry % 1_000_000_000);
            carry /= 1_000_000_000;
        }
    }
    let mut s = limbs.last().unwrap().to_string();
    for limb in limbs.iter().rev().skip(1) {
        s.push_str(&format!("{limb:09}"));
    }
    canonical_int(&s)
}

/// One string literal starting at its opening quote; `(literal, chars used)`.
fn string(c: &[char], raw: bool, bytes: bool) -> Option<(StrLit, usize)> {
    let q = c[0];
    let triple = c.len() >= 3 && c[1] == q && c[2] == q;
    let open = if triple { 3 } else { 1 };
    let mut i = open;
    let mut value = String::new();
    loop {
        let ch = *c.get(i)?;
        if ch == q {
            if !triple {
                return Some((StrLit { bytes, value }, i + 1));
            }
            if c.get(i + 1) == Some(&q) && c.get(i + 2) == Some(&q) {
                return Some((StrLit { bytes, value }, i + 3));
            }
            value.push(ch);
            i += 1;
            continue;
        }
        if ch == '\n' && !triple {
            return None;
        }
        if ch == '\\' {
            let next = *c.get(i + 1)?;
            if raw {
                // Raw strings keep the backslash; it still escapes a quote or newline.
                value.push('\\');
                value.push(next);
                i += 2;
                continue;
            }
            i += 2;
            match next {
                '\n' => {}
                '\\' => value.push('\\'),
                '\'' => value.push('\''),
                '"' => value.push('"'),
                'a' => value.push('\u{7}'),
                'b' => value.push('\u{8}'),
                'f' => value.push('\u{c}'),
                'n' => value.push('\n'),
                'r' => value.push('\r'),
                't' => value.push('\t'),
                'v' => value.push('\u{b}'),
                '0'..='7' => {
                    let mut n = next.to_digit(8)?;
                    for _ in 0..2 {
                        match c.get(i).and_then(|d| d.to_digit(8)) {
                            Some(d) => {
                                n = n * 8 + d;
                                i += 1;
                            }
                            None => break,
                        }
                    }
                    value.push(char::from_u32(n).unwrap_or('\u{fffd}'));
                }
                'x' | 'u' | 'U' if !(bytes && next != 'x') => {
                    let width = match next {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let hex: String = c.get(i..i + width)?.iter().collect();
                    if !hex.chars().all(|h| h.is_ascii_hexdigit()) {
                        return None;
                    }
                    i += width;
                    let cp = u32::from_str_radix(&hex, 16).ok()?;
                    if cp > 0x10ffff {
                        return None;
                    }
                    value.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                }
                'N' if !bytes => return None,
                other => {
                    // Unknown escapes keep their backslash (a SyntaxWarning only).
                    value.push('\\');
                    value.push(other);
                }
            }
            continue;
        }
        if bytes && !ch.is_ascii() {
            return None;
        }
        value.push(ch);
        i += 1;
    }
}

struct LitParser {
    t: Vec<Tok>,
    i: usize,
}

impl LitParser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }

    fn op(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Op(c)) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    /// An eval-mode expression: a bare comma list is a tuple.
    fn top(&mut self) -> Option<PyVal> {
        let first = self.expr()?;
        if self.peek() != Some(&Tok::Op(',')) {
            return Some(first);
        }
        let mut items = vec![first];
        while self.op(',') {
            if self.peek().is_none() {
                break;
            }
            items.push(self.expr()?);
        }
        Some(PyVal::Tuple(items))
    }

    /// `literal_eval`'s `_convert`: a signed number, a `num ± complex`, or an atom.
    fn expr(&mut self) -> Option<PyVal> {
        let left = self.signed()?;
        if matches!(self.peek(), Some(Tok::Op('+' | '-'))) {
            // Only `<real> ± <imaginary>` is a literal; anything else raises.
            let is_real = matches!(left, PyVal::Int(_) | PyVal::Float(_));
            self.i += 1;
            let right = self.signed_raw()?;
            return (is_real && right == Num::Imag).then_some(PyVal::Opaque);
        }
        Some(left)
    }

    fn signed(&mut self) -> Option<PyVal> {
        if matches!(self.peek(), Some(Tok::Op('+' | '-'))) {
            return match self.signed_raw()? {
                Num::Int(v) => Some(v),
                Num::Float(f) => Some(PyVal::Float(f)),
                Num::Imag => Some(PyVal::Opaque),
            };
        }
        self.atom()
    }

    /// An optionally signed numeric constant (one sign, as `_convert_signed_num`).
    fn signed_raw(&mut self) -> Option<Num> {
        let neg = match self.peek() {
            Some(Tok::Op('-')) => {
                self.i += 1;
                true
            }
            Some(Tok::Op('+')) => {
                self.i += 1;
                false
            }
            _ => false,
        };
        let Some(Tok::Num(n)) = self.peek().cloned() else {
            return None;
        };
        self.i += 1;
        Some(match (n, neg) {
            (Num::Int(PyVal::Int(d)), true) if d != "0" => Num::Int(PyVal::Int(format!("-{d}"))),
            (Num::Float(f), true) => Num::Float(-f),
            (n, _) => n,
        })
    }

    fn atom(&mut self) -> Option<PyVal> {
        let tok = self.peek()?.clone();
        self.i += 1;
        match tok {
            Tok::Num(Num::Int(v)) => Some(v),
            Tok::Num(Num::Float(f)) => Some(PyVal::Float(f)),
            Tok::Num(Num::Imag) => Some(PyVal::Opaque),
            Tok::Str(first) => {
                let mut lit = first;
                while let Some(Tok::Str(next)) = self.peek().cloned() {
                    if next.bytes != lit.bytes {
                        return None;
                    }
                    lit.value.push_str(&next.value);
                    self.i += 1;
                }
                Some(if lit.bytes { PyVal::Opaque } else { PyVal::Str(lit.value) })
            }
            Tok::Name(name) => match name.as_str() {
                "True" => Some(PyVal::Bool(true)),
                "False" => Some(PyVal::Bool(false)),
                "None" => Some(PyVal::None),
                "..." => Some(PyVal::Opaque),
                _ => None,
            },
            Tok::Op('(') => {
                if self.op(')') {
                    return Some(PyVal::Tuple(Vec::new()));
                }
                let first = self.expr()?;
                if self.op(')') {
                    return Some(first); // parenthesized expression
                }
                let mut items = vec![first];
                while self.op(',') {
                    if self.op(')') {
                        return Some(PyVal::Tuple(items));
                    }
                    items.push(self.expr()?);
                }
                self.op(')').then_some(PyVal::Tuple(items))
            }
            Tok::Op('[') => {
                let mut items = Vec::new();
                loop {
                    if self.op(']') {
                        return Some(PyVal::List(items));
                    }
                    items.push(self.expr()?);
                    if !self.op(',') {
                        return self.op(']').then_some(PyVal::List(items));
                    }
                }
            }
            Tok::Op('{') => {
                if self.op('}') {
                    return Some(PyVal::Dict(Vec::new()));
                }
                let first = self.expr()?;
                if self.op(':') {
                    let mut items = Vec::new();
                    let value = self.expr()?;
                    if !first.hashable() {
                        return None;
                    }
                    PyVal::dict_insert(&mut items, first, value);
                    while self.op(',') {
                        if self.op('}') {
                            return Some(PyVal::Dict(items));
                        }
                        let key = self.expr()?;
                        if !self.op(':') {
                            return None;
                        }
                        let value = self.expr()?;
                        if !key.hashable() {
                            return None;
                        }
                        PyVal::dict_insert(&mut items, key, value);
                    }
                    return self.op('}').then_some(PyVal::Dict(items));
                }
                // A set: its items must be hashable; `json.dumps` refuses it anyway.
                let mut ok = first.hashable();
                while self.op(',') {
                    if self.op('}') {
                        return ok.then_some(PyVal::Opaque);
                    }
                    ok &= self.expr()?.hashable();
                }
                (self.op('}') && ok).then_some(PyVal::Opaque)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(v: &PyVal) -> String {
        dumps(v, false).unwrap_or_else(|| "<TypeError>".into())
    }

    #[test]
    fn json_loads_matches_python() {
        assert_eq!(d(&json_loads(" {\"a\": [1, 2.5, NaN, -Infinity], \"a\": 3} ").unwrap()), "{\"a\": 3}");
        assert_eq!(d(&json_loads("123456789012345678901234567890").unwrap()), "123456789012345678901234567890");
        assert_eq!(d(&json_loads("1e400").unwrap()), "Infinity");
        assert_eq!(d(&json_loads("\"\\ud83d\\ude00\"").unwrap()), "\"\u{1F600}\"");
        assert!(json_loads("01").is_none());
        assert!(json_loads("[1,]").is_none());
        assert!(json_loads("\"a\nb\"").is_none());
        assert!(json_loads("1 2").is_none());
    }

    #[test]
    fn literal_eval_matches_python() {
        assert_eq!(d(&literal_eval("{'a': (1, 2), 'b': [True, None]}").unwrap()), "{\"a\": [1, 2], \"b\": [true, null]}");
        assert_eq!(d(&literal_eval("-5").unwrap()), "-5");
        assert_eq!(d(&literal_eval("0x1F").unwrap()), "31");
        assert_eq!(d(&literal_eval("1_000.5").unwrap()), "1000.5");
        assert_eq!(d(&literal_eval("'a' \"b\"").unwrap()), "\"ab\"");
        assert_eq!(d(&literal_eval("1, 2").unwrap()), "[1, 2]");
        assert_eq!(d(&literal_eval("{1, 2}").unwrap()), "<TypeError>");
        assert_eq!(d(&literal_eval("1+2j").unwrap()), "<TypeError>");
        assert!(literal_eval("true").is_none());
        assert!(literal_eval("--5").is_none());
        assert!(literal_eval("007").is_none());
        assert!(literal_eval("{[1]: 2}").is_none());
        assert!(literal_eval("1\n2").is_none());
    }

    #[test]
    fn int_and_float_match_python() {
        assert_eq!(d(&py_int(" -0_07 ").unwrap()), "-7");
        assert!(py_int("4.0").is_none());
        assert_eq!(py_float(" 1_0.5e1 "), Some(105.0));
        assert!(py_float("1e").is_none());
        assert_eq!(py_float("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(d(&float_to_int(1.2345678901234567e19)), "12345678901234567168");
        assert_eq!(d(&float_to_int(-0.0)), "0");
    }
}
