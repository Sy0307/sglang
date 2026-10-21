//! Parser names SGLang ports from Python, where Dynamo's parsers split output
//! differently. Every other name goes to Dynamo.

mod deepseek_v4;
mod qwen3_coder;

use super::{ToolDetector, ToolSpec};

/// The detector a `--tool-call-parser` name selects, over the request's tools
/// (Qwen3-Coder converts each parameter by its schema type).
pub(crate) fn tool_detector_for(
    tool_parser: &str,
    tools: Vec<ToolSpec>,
) -> Option<Box<dyn ToolDetector>> {
    let names = || tools.iter().map(|tool| tool.name.clone()).collect();
    Some(match tool_parser {
        "deepseekv4" => Box::new(deepseek_v4::DsmlDetector::new(deepseek_v4::DSML_TAGS, names())),
        "deepseekv41" => Box::new(deepseek_v4::DsmlDetector::new(deepseek_v4::DSML_TAGS_V41, names())),
        "qwen3_coder" => Box::new(qwen3_coder::Qwen3CoderDetector::new(tools)),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_call::{ToolCallItem, ToolSpec};
    use serde_json::{Value, json};

    /// `tests/fixtures/tool_parity/*.json`, from `tests/scripts/generate_tool_parity.py`.
    #[test]
    fn tool_fixtures_match_sglang() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tool_parity");
        for entry in std::fs::read_dir(dir).unwrap() {
            let fixture: Value =
                serde_json::from_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap())
                    .unwrap();
            let parser = fixture["parser"].as_str().unwrap();
            let specs: Vec<ToolSpec> = fixture["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|tool| ToolSpec {
                    name: tool["function"]["name"].as_str().unwrap().to_owned(),
                    parameters: tool["function"].get("parameters").cloned(),
                })
                .collect();
            let detector = || tool_detector_for(parser, specs.clone()).unwrap();
            let calls = |calls: Vec<ToolCallItem>| -> Value {
                calls
                    .into_iter()
                    .map(|c| {
                        let name = (!c.name.is_empty() || parser != "qwen3_coder")
                            .then_some(c.name)
                            .map_or(Value::Null, Value::from);
                        json!([c.tool_index, name, c.parameters])
                    })
                    .collect()
            };
            for case in fixture["cases"].as_array().unwrap() {
                let chunks: Vec<&str> = case["chunks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_str().unwrap())
                    .collect();
                let mut streaming = detector();
                let steps: Vec<Value> = chunks
                    .iter()
                    .enumerate()
                    .map(|(i, chunk)| {
                        let (normal, found) = streaming.parse_stream(chunk, i + 1 == chunks.len());
                        json!([normal, calls(found)])
                    })
                    .collect();
                let (normal, found) = detector().parse_non_stream(&chunks.concat());
                // `steps` is null where Python's streaming parser raised.
                let steps = if case["steps"].is_null() { Value::Null } else { Value::from(steps) };
                let got = json!({"steps": steps, "unary": [normal, calls(found)]});
                let want = json!({"steps": case["steps"], "unary": case["unary"]});
                assert_eq!(got, want, "{parser} {:?}", chunks);
            }
        }
    }
}
