//! The `structural_tag` sampling constraint SGLang builds for Qwen3-Coder tool
//! calls (`FunctionCallParser.get_structure_constraint` → xgrammar's
//! `get_model_structural_tag("qwen_3_coder", ...)`), serialized as
//! `json.dumps(tag.model_dump(by_alias=True))`.
//!
//! The server runs a reasoning parser, so SGLang passes `reasoning=False` (the
//! grammar backend owns the think block) and strips the think tokens from the
//! free-text `excludes`: no reasoning prefix, empty `excludes`.

use serde_json::{Value, json};

use crate::py::python_json;

const TOOL_CALL_BEGIN_PREFIX: &str = "<tool_call>\n<function=";
const TOOL_CALL_BEGIN_SUFFIX: &str = ">\n";
const TOOL_CALL_END: &str = "\n</function>\n</tool_call>";
const TOOL_CALL_TRIGGER: &str = "<tool_call>\n<function=";

/// `_get_function_parameters`: a non-strict or schemaless tool is unconstrained (`true`).
fn function_parameters(function: &Value) -> Value {
    if function.get("strict") != Some(&Value::Bool(true)) {
        return Value::Bool(true);
    }
    match function.get("parameters") {
        None | Some(Value::Null) => Value::Bool(true),
        Some(parameters) => parameters.clone(),
    }
}

fn tag(function: &Value) -> Value {
    json!({
        "type": "tag",
        "begin": format!("{TOOL_CALL_BEGIN_PREFIX}{}{TOOL_CALL_BEGIN_SUFFIX}", function["name"].as_str().unwrap_or_default()),
        "content": {
            "type": "json_schema",
            "json_schema": function_parameters(function),
            "style": "qwen_xml",
            "any_order": false,
            "max_whitespace_cnt": null,
        },
        "end": TOOL_CALL_END,
    })
}

/// The constraint for `tools` (request `tools` entries) under `tool_choice`, or
/// `None` where SGLang adds none (`auto` without a strict tool). `Err` where
/// xgrammar raises (a named tool that is not offered).
pub(crate) fn qwen3_coder_constraint(
    tools: &[Value],
    tool_choice: Option<&Value>,
    parallel_tool_calls: bool,
) -> Result<Option<String>, ()> {
    let functions: Vec<&Value> = tools.iter().map(|tool| &tool["function"]).collect();
    let format = match tool_choice {
        Some(Value::Object(choice)) => {
            let name = choice.get("function").and_then(|f| f.get("name"));
            let named: Vec<&&Value> = functions.iter().filter(|f| f.get("name") == name).collect();
            let [function] = named.as_slice() else {
                return Err(());
            };
            tag(function)
        }
        Some(Value::String(choice)) if choice == "required" => {
            if functions.is_empty() {
                return Err(());
            }
            json!({
                "type": "triggered_tags",
                "triggers": [TOOL_CALL_TRIGGER],
                "tags": functions.iter().map(|f| tag(f)).collect::<Vec<_>>(),
                "at_least_one": true,
                "stop_after_first": !parallel_tool_calls,
                "excludes": [],
            })
        }
        // `auto`: constrained only when a tool asks for strict arguments.
        _ => {
            let strict = functions.iter().any(|f| f.get("strict") == Some(&Value::Bool(true)));
            if !strict || functions.is_empty() {
                return Ok(None);
            }
            json!({
                "type": "triggered_tags",
                "triggers": [TOOL_CALL_TRIGGER],
                "tags": functions.iter().map(|f| tag(f)).collect::<Vec<_>>(),
                "at_least_one": false,
                "stop_after_first": !parallel_tool_calls,
                "excludes": [],
            })
        }
    };
    Ok(Some(python_json(&json!({"type": "structural_tag", "format": format}), false)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Value {
        json!({"type": "function", "function": {"name": name, "parameters": {"type": "object"}}})
    }

    /// The strings SGLang sent the scheduler (recorded from the Python server).
    #[test]
    fn matches_python_constraints() {
        let tools = [tool("get_weather"), tool("write_file")];
        let named = json!({"type": "function", "function": {"name": "get_weather"}});
        assert_eq!(
            qwen3_coder_constraint(&tools, Some(&named), true).unwrap().unwrap(),
            "{\"type\": \"structural_tag\", \"format\": {\"type\": \"tag\", \"begin\": \"<tool_call>\\n<function=get_weather>\\n\", \"content\": {\"type\": \"json_schema\", \"json_schema\": true, \"style\": \"qwen_xml\", \"any_order\": false, \"max_whitespace_cnt\": null}, \"end\": \"\\n</function>\\n</tool_call>\"}}"
        );
        assert_eq!(
            qwen3_coder_constraint(&tools, Some(&json!("required")), true).unwrap().unwrap(),
            "{\"type\": \"structural_tag\", \"format\": {\"type\": \"triggered_tags\", \"triggers\": [\"<tool_call>\\n<function=\"], \"tags\": [{\"type\": \"tag\", \"begin\": \"<tool_call>\\n<function=get_weather>\\n\", \"content\": {\"type\": \"json_schema\", \"json_schema\": true, \"style\": \"qwen_xml\", \"any_order\": false, \"max_whitespace_cnt\": null}, \"end\": \"\\n</function>\\n</tool_call>\"}, {\"type\": \"tag\", \"begin\": \"<tool_call>\\n<function=write_file>\\n\", \"content\": {\"type\": \"json_schema\", \"json_schema\": true, \"style\": \"qwen_xml\", \"any_order\": false, \"max_whitespace_cnt\": null}, \"end\": \"\\n</function>\\n</tool_call>\"}], \"at_least_one\": true, \"stop_after_first\": false, \"excludes\": []}}"
        );
        assert_eq!(qwen3_coder_constraint(&tools, Some(&json!("auto")), true), Ok(None));
    }
}
