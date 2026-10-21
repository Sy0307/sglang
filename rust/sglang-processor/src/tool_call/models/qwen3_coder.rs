//! Qwen3-Coder (`Qwen3CoderDetector` in `function_call/qwen3_coder_detector.py`),
//! with the MOSS deployment's overlay of `_convert_param_value`
//! (`v11-parser-patch-v2`): a parameter whose schema `type` is a union of two or
//! more non-null types decodes valid JSON literals of a declared type before
//! falling back to the string branch.
//!
//! Python raises (and drops every call of the output) on a schema property that
//! is not an object, an unhashable union member, or a converted value
//! `json.dumps` refuses; the one-shot parse does the same here, while the
//! streaming parse sends such a value as its raw string instead of failing the
//! stream.

use serde_json::{Map, Value};

use crate::tool_call::pyval::{self, PyVal};
use crate::tool_call::{Parsed, ToolCallItem, ToolDetector, ToolSpec};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_PREFIX: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

pub(super) struct Qwen3CoderDetector {
    tools: Vec<ToolSpec>,
    buffer: String,
    parsed_pos: usize,
    current_tool_id: i64,
    current_tool_param_count: usize,
    json_started: bool,
    is_inside_tool_call: bool,
    current_func_name: Option<String>,
}

/// A conversion Python would raise on.
struct Raised;

impl Qwen3CoderDetector {
    pub(super) fn new(tools: Vec<ToolSpec>) -> Self {
        Self {
            tools,
            buffer: String::new(),
            parsed_pos: 0,
            current_tool_id: -1,
            current_tool_param_count: 0,
            json_started: false,
            is_inside_tool_call: false,
            current_func_name: None,
        }
    }

    /// `_get_arguments_config`.
    fn arguments_config(&self, func_name: &str) -> Map<String, Value> {
        let Some(tool) = self.tools.iter().find(|tool| tool.name == func_name) else {
            tracing::warn!("Tool '{func_name}' is not defined in the tools list.");
            return Map::new();
        };
        match &tool.parameters {
            Some(Value::Object(params)) => {
                let properties = schema_properties(&Value::Object(params.clone()));
                if !properties.is_empty() || params.contains_key("properties") {
                    properties
                } else {
                    params.clone()
                }
            }
            _ => Map::new(),
        }
    }

    /// `detect_and_parse`, as `FunctionCallParser.parse_non_stream` wraps it.
    fn detect_and_parse(&self, text: &str) -> Result<Parsed, Raised> {
        if !text.contains(TOOL_CALL_START) {
            return Ok((text.to_owned(), Vec::new()));
        }
        let mut raw_tool_calls = find_tool_calls(text);
        if raw_tool_calls.is_empty() && text.contains(FUNCTION_PREFIX) {
            raw_tool_calls = vec![text];
        }
        let mut calls = Vec::new();
        let mut tool_idx = 0i64;
        for tool_content in raw_tool_calls {
            for func_body in find_functions(tool_content) {
                let Some(name_end) = func_body.find('>') else {
                    continue;
                };
                let func_name = &func_body[..name_end];
                let params_str = &func_body[name_end + 1..];
                let param_config = self.arguments_config(func_name);
                let mut parsed: Vec<(PyVal, PyVal)> = Vec::new();
                for p_match in find_parameters(params_str) {
                    let Some(p_idx) = p_match.find('>') else {
                        continue;
                    };
                    let p_name = &p_match[..p_idx];
                    let mut p_val = &p_match[p_idx + 1..];
                    p_val = p_val.strip_prefix('\n').unwrap_or(p_val);
                    p_val = p_val.strip_suffix('\n').unwrap_or(p_val);
                    let value = convert_param_value(p_val, p_name, &param_config, func_name)?;
                    PyVal::dict_insert(&mut parsed, PyVal::Str(p_name.to_owned()), value);
                }
                let parameters = pyval::dumps(&PyVal::Dict(parsed), false).ok_or(Raised)?;
                calls.push(ToolCallItem {
                    tool_index: tool_idx,
                    name: func_name.to_owned(),
                    parameters,
                });
                tool_idx += 1;
            }
        }
        let start_idx = text
            .find(TOOL_CALL_START)
            .or_else(|| text.find(FUNCTION_PREFIX));
        let normal_text = match start_idx {
            Some(i) if i > 0 => text[..i].to_owned(),
            _ => String::new(),
        };
        Ok((normal_text, calls))
    }

    /// `parse_streaming_increment`.
    fn parse_streaming_increment(&mut self, new_text: &str) -> Parsed {
        self.buffer.push_str(new_text);
        if self.buffer.is_empty() {
            return Default::default();
        }
        let mut calls = Vec::new();
        let mut normal = String::new();
        loop {
            let current = &self.buffer[self.parsed_pos..];
            if current.is_empty() {
                break;
            }
            if current.starts_with(TOOL_CALL_START) {
                self.parsed_pos += TOOL_CALL_START.len();
                self.is_inside_tool_call = true;
                continue;
            }
            if current.starts_with(FUNCTION_PREFIX) {
                let Some(end_angle) = current.find('>') else {
                    break;
                };
                let func_name = current[FUNCTION_PREFIX.len()..end_angle].to_owned();
                self.current_tool_id += 1;
                self.current_tool_param_count = 0;
                self.json_started = false;
                self.current_func_name = Some(func_name.clone());
                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id,
                    name: func_name,
                    parameters: String::new(),
                });
                self.parsed_pos += end_angle + 1;
                continue;
            }
            if current.starts_with(PARAMETER_PREFIX) {
                let Some(name_end) = current.find('>') else {
                    break;
                };
                let rest = &current[name_end + 1..];
                let candidates = [
                    rest.find(PARAMETER_END).map(|p| (p, PARAMETER_END.len())),
                    rest.find(PARAMETER_PREFIX).map(|p| (p, 0)),
                    rest.find(FUNCTION_END).map(|p| (p, 0)),
                ];
                // `min` over (position, length) keeps the first candidate on a tie,
                // as Python's `min(..., key=lambda x: x[0])` does.
                let Some((end_pos, end_len)) = candidates
                    .into_iter()
                    .flatten()
                    .fold(None, |best: Option<(usize, usize)>, c| match best {
                        Some(b) if b.0 <= c.0 => Some(b),
                        _ => Some(c),
                    })
                else {
                    break;
                };
                let param_name = current[PARAMETER_PREFIX.len()..name_end].to_owned();
                let mut raw_value = &rest[..end_pos];
                raw_value = raw_value.strip_prefix('\n').unwrap_or(raw_value);
                raw_value = raw_value.strip_suffix('\n').unwrap_or(raw_value);
                let raw_value = raw_value.to_owned();
                if !self.json_started {
                    calls.push(argument_delta(self.current_tool_id, "{"));
                    self.json_started = true;
                }
                let func_name = self.current_func_name.clone().unwrap_or_default();
                let param_config = self.arguments_config(&func_name);
                let value = convert_param_value(&raw_value, &param_name, &param_config, &func_name)
                    .ok()
                    .and_then(|v| pyval::dumps(&v, false))
                    .unwrap_or_else(|| {
                        pyval::dumps(&PyVal::Str(raw_value.clone()), false).expect("a string dumps")
                    });
                let key = pyval::dumps(&PyVal::Str(param_name), true).expect("a string dumps");
                let fragment = match self.current_tool_param_count {
                    0 => format!("{key}: {value}"),
                    _ => format!(", {key}: {value}"),
                };
                calls.push(argument_delta(self.current_tool_id, &fragment));
                self.current_tool_param_count += 1;
                self.parsed_pos += name_end + 1 + end_pos + end_len;
                continue;
            }
            if current.starts_with(FUNCTION_END) {
                if !self.json_started {
                    calls.push(argument_delta(self.current_tool_id, "{"));
                    self.json_started = true;
                }
                calls.push(argument_delta(self.current_tool_id, "}"));
                self.parsed_pos += FUNCTION_END.len();
                self.current_func_name = None;
                continue;
            }
            if current.starts_with(TOOL_CALL_END) {
                self.parsed_pos += TOOL_CALL_END.len();
                self.is_inside_tool_call = false;
                continue;
            }
            match current.find('<') {
                None => {
                    if !self.is_inside_tool_call {
                        normal.push_str(current);
                    }
                    self.parsed_pos += current.len();
                }
                Some(0) => {
                    let partial = [
                        TOOL_CALL_START,
                        TOOL_CALL_END,
                        FUNCTION_PREFIX,
                        FUNCTION_END,
                        PARAMETER_PREFIX,
                        PARAMETER_END,
                    ]
                    .iter()
                    .any(|tag| tag.starts_with(current));
                    if partial {
                        break;
                    }
                    if !self.is_inside_tool_call {
                        normal.push('<');
                    }
                    self.parsed_pos += 1;
                }
                Some(i) => {
                    if !self.is_inside_tool_call {
                        normal.push_str(&current[..i]);
                    }
                    self.parsed_pos += i;
                }
            }
        }
        if self.parsed_pos > 0 {
            self.buffer.drain(..self.parsed_pos);
            self.parsed_pos = 0;
        }
        (normal, calls)
    }
}

/// A streamed arguments fragment: Python's `ToolCallItem(name=None, ...)`.
fn argument_delta(tool_index: i64, parameters: &str) -> ToolCallItem {
    ToolCallItem {
        tool_index,
        name: String::new(),
        parameters: parameters.to_owned(),
    }
}

impl ToolDetector for Qwen3CoderDetector {
    fn has_tool_call(&self, text: &str) -> bool {
        text.contains(TOOL_CALL_START)
    }

    fn parse_non_stream(&self, text: &str) -> Parsed {
        let has_tool_call = self.has_tool_call(text);
        match self.detect_and_parse(text) {
            Ok((normal, calls)) if !calls.is_empty() || has_tool_call => (normal, calls),
            Ok(_) => (text.to_owned(), Vec::new()),
            Err(Raised) => (text.to_owned(), Vec::new()),
        }
    }

    fn parse_stream(&mut self, text: &str, _flush: bool) -> Parsed {
        // `finish` is the base detector's no-op: held-back text is dropped.
        self.parse_streaming_increment(text)
    }
}

// ------------------------------------------------------- regex emulation

/// `re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)`.
fn find_tool_calls(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = text[pos..].find(TOOL_CALL_START).map(|i| pos + i) {
        let start = i + TOOL_CALL_START.len();
        let Some(j) = text[start..].find(TOOL_CALL_END).map(|j| start + j) else {
            break;
        };
        out.push(&text[start..j]);
        pos = j + TOOL_CALL_END.len();
    }
    out
}

/// `re.findall(r"<function=(.*?)</function>|<function=(.*)$", content, re.DOTALL)`,
/// each match's `group(1) or group(2)`.
fn find_functions(content: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = content[pos..].find(FUNCTION_PREFIX).map(|i| pos + i) {
        let start = i + FUNCTION_PREFIX.len();
        match content[start..].find(FUNCTION_END).map(|j| start + j) {
            Some(j) => {
                out.push(&content[start..j]);
                pos = j + FUNCTION_END.len();
            }
            None => {
                out.push(&content[start..]);
                break;
            }
        }
    }
    out
}

/// `re.findall(r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
/// params, re.DOTALL)`. Python's `$` also matches just before a final newline.
fn find_parameters(params: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    let dollar = |from: usize| -> usize {
        match params.strip_suffix('\n') {
            Some(_) if params.len() - 1 >= from => params.len() - 1,
            _ => params.len(),
        }
    };
    while let Some(i) = params[pos..].find(PARAMETER_PREFIX).map(|i| pos + i) {
        let start = i + PARAMETER_PREFIX.len();
        let end = [
            params[start..].find(PARAMETER_END).map(|p| start + p),
            params[start..].find(PARAMETER_PREFIX).map(|p| start + p),
            params[start..].find(FUNCTION_END).map(|p| start + p),
            Some(dollar(start)),
        ]
        .into_iter()
        .flatten()
        .min()
        .expect("`$` always matches");
        out.push(&params[start..end]);
        pos = if params[end..].starts_with(PARAMETER_END) {
            end + PARAMETER_END.len()
        } else {
            end
        };
        if pos >= params.len() {
            break;
        }
    }
    out
}

// ------------------------------------------------------- schema helpers

/// `get_schema_properties`.
fn schema_properties(schema: &Value) -> Map<String, Value> {
    let Value::Object(schema) = schema else {
        return Map::new();
    };
    if let Some(Value::Object(properties)) = schema.get("properties") {
        return properties.clone();
    }
    let mut merged = Map::new();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(branches)) = schema.get(keyword) {
            for branch in branches {
                for (key, value) in schema_properties(branch) {
                    merged.entry(key).or_insert(value);
                }
            }
        }
    }
    merged
}

/// `infer_type_from_json_schema`, as `str(...)` would print a non-string result.
fn infer_type(schema: &Value) -> Option<String> {
    let Value::Object(map) = schema else {
        return None;
    };
    if let Some(type_value) = map.get("type") {
        match type_value {
            Value::String(t) => return Some(t.clone()),
            Value::Array(types) if !types.is_empty() => {
                let non_null: Vec<&Value> = types.iter().filter(|t| *t != "null").collect();
                return Some(match non_null.first() {
                    Some(Value::String(t)) => t.clone(),
                    Some(other) => py_str(other),
                    None => "string".to_owned(),
                });
            }
            _ => {}
        }
    }
    if map.contains_key("anyOf") || map.contains_key("oneOf") {
        // `schema.get("anyOf") or schema.get("oneOf")`
        let schemas = match map.get("anyOf") {
            Some(v) if py_truthy(v) => Some(v),
            _ => map.get("oneOf"),
        };
        if let Some(Value::Array(schemas)) = schemas {
            let types: Vec<String> = schemas.iter().filter_map(infer_type).collect();
            if !types.is_empty() {
                let unique: std::collections::HashSet<&String> = types.iter().collect();
                if unique.len() == 1 {
                    return Some(types[0].clone());
                }
                if unique.len() == 2 && types.iter().any(|t| t == "null") {
                    return types.iter().find(|t| *t != "null").cloned();
                }
                if types.iter().any(|t| t == "string") {
                    return Some("string".to_owned());
                }
                return Some(types[0].clone());
            }
        }
    }
    if let Some(Value::Array(values)) = map.get("enum") {
        if values.is_empty() {
            return Some("string".to_owned());
        }
        let kinds: std::collections::HashSet<&str> = values.iter().map(json_kind).collect();
        if kinds.len() == 1 {
            return kinds.into_iter().next().map(str::to_owned);
        }
        return Some("string".to_owned());
    }
    if let Some(Value::Array(schemas)) = map.get("allOf") {
        for sub in schemas {
            if let Some(t) = infer_type(sub).filter(|t| t != "string") {
                return Some(t);
            }
        }
        return Some("string".to_owned());
    }
    if map.contains_key("properties") {
        return Some("object".to_owned());
    }
    if map.contains_key("items") {
        return Some("array".to_owned());
    }
    None
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::Bool(true) => true,
    }
}

/// `str(value)` for a JSON value Python loaded (only reached on odd schemas).
fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The JSON type name of a value `json.loads` produced (MOSS overlay).
fn loaded_kind(value: &PyVal) -> &'static str {
    match value {
        PyVal::None => "null",
        PyVal::Bool(_) => "boolean",
        PyVal::Int(_) => "integer",
        PyVal::Float(_) => "number",
        PyVal::Dict(_) => "object",
        PyVal::List(_) => "array",
        _ => "string",
    }
}

/// `_convert_param_value` with the MOSS union overlay.
fn convert_param_value(
    param_value: &str,
    param_name: &str,
    param_config: &Map<String, Value>,
    func_name: &str,
) -> Result<PyVal, Raised> {
    if pyval::py_lower(param_value) == "null" {
        return Ok(PyVal::None);
    }
    let Some(schema) = param_config.get(param_name) else {
        if !param_config.is_empty() {
            tracing::warn!(
                "Parsed parameter '{param_name}' is not defined in the tool parameters for tool '{func_name}', directly returning the string value."
            );
        }
        return Ok(PyVal::Str(param_value.to_owned()));
    };
    // MOSS: `param_config[param_name].get("type")` (a non-object schema raises).
    let Value::Object(schema_map) = schema else {
        return Err(Raised);
    };
    if let Some(Value::Array(declared)) = schema_map.get("type") {
        if declared.iter().any(|t| t.is_array() || t.is_object()) {
            return Err(Raised); // unhashable in `set(declared)`
        }
        let mut distinct: Vec<&Value> = Vec::new();
        for t in declared.iter().filter(|t| *t != "null") {
            if !distinct.contains(&t) {
                distinct.push(t);
            }
        }
        if distinct.len() > 1 {
            let has = |name: &str| declared.iter().any(|t| t == name);
            if let Some(parsed) = pyval::json_loads(param_value) {
                let kind = loaded_kind(&parsed);
                if has(kind) || (kind == "integer" && has("number")) {
                    return Ok(parsed);
                }
            }
            let stripped = crate::py::py_strip(param_value).to_lowercase();
            if has("boolean") && (stripped == "true" || stripped == "false") {
                tracing::info!("MOSS native union boolean normalized: {}", crate::py::py_strip(param_value));
                return Ok(PyVal::Bool(stripped == "true"));
            }
            if has("string") {
                return Ok(PyVal::Str(param_value.to_owned()));
            }
        }
    }
    let param_type = infer_type(schema)
        .map(|t| crate::py::py_strip(&t).to_lowercase())
        .unwrap_or_else(|| "string".to_owned());
    if matches!(param_type.as_str(), "string" | "str" | "text" | "varchar" | "char" | "enum") {
        return Ok(PyVal::Str(param_value.to_owned()));
    }
    if ["int", "uint", "long", "short", "unsigned"]
        .iter()
        .any(|p| param_type.starts_with(p))
    {
        return Ok(pyval::py_int(param_value).unwrap_or_else(|| {
            tracing::warn!(
                "Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool '{func_name}', degenerating to string."
            );
            PyVal::Str(param_value.to_owned())
        }));
    }
    if param_type.starts_with("num") || param_type.starts_with("float") {
        let maybe_convert = !(param_value.contains('.') || param_value.to_lowercase().contains('e'));
        return Ok(match pyval::py_float(param_value) {
            Some(f) if maybe_convert && f.is_finite() && f.fract() == 0.0 => pyval::float_to_int(f),
            Some(f) => PyVal::Float(f),
            None => {
                tracing::warn!(
                    "Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool '{func_name}', degenerating to string."
                );
                PyVal::Str(param_value.to_owned())
            }
        });
    }
    if matches!(param_type.as_str(), "boolean" | "bool" | "binary") {
        let lower = param_value.to_lowercase();
        return Ok(PyVal::Bool(lower == "true"));
    }
    if matches!(param_type.as_str(), "object" | "array" | "arr")
        || param_type.starts_with("dict")
        || param_type.starts_with("list")
    {
        if let Some(value) = pyval::json_loads(param_value) {
            return Ok(value);
        }
    }
    Ok(pyval::literal_eval(param_value).unwrap_or_else(|| PyVal::Str(param_value.to_owned())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_regex_dollar_matches_before_final_newline() {
        assert_eq!(find_parameters("<parameter=a>\nfoo\n\n"), vec!["a>\nfoo\n"]);
        assert_eq!(find_parameters("<parameter=a>x</parameter><parameter=b>y"), vec!["a>x", "b>y"]);
        assert_eq!(find_parameters("<parameter=a>x<parameter=b>y</function>"), vec!["a>x", "b>y"]);
    }
}
