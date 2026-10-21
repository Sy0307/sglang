//! Qwen3 (`Qwen3Detector` in `parser/reasoning_parser.py`): `<think>` blocks,
//! where an opening `<tool_call>` also ends reasoning, toggled by the
//! `enable_thinking` template kwarg (on unless the request sets it false).

use crate::think::ThinkConfig;

pub(super) const THINK: ThinkConfig = ThinkConfig {
    start: "<think>",
    end: "</think>",
    tool_start: Some("<tool_call>"),
    tool_start_at_line_start: false,
    toggle: ("enable_thinking", true),
};
