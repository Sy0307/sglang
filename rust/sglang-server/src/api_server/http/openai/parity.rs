//! `/v1/chat/completions` through `sglang_processor::openai`, the port of
//! Python's `serving_chat`: the request is lowered to a native `/generate` body,
//! run on this server's own generate path, and its output rebuilt into Python's
//! OpenAI response (sampling resolution, reasoning split, tool calls, chunk
//! shapes, usage). Requests that layer does not reproduce return `None` and
//! go to the Dynamo handler in `chat.rs`.

use std::convert::Infallible;
use std::sync::{Arc, OnceLock};

use axum::body::{Body, Bytes};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use dynamo_protocols::types::CreateChatCompletionRequest;
use sglang_processor::openai::{
    ChatModel, OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, Prompt, Reply, Responder,
    TokenPieces, Unsupported, lower_chat_prompt,
};

use super::{AppState, ChatFormatter, ChatTemplateKwargs};
use crate::api_server::http::native_api::{NativeOutput, run_generate};
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::tokenizer;

/// The served model's facts Python's chat layer reads, resolved at startup.
struct ParityChat {
    settings: OpenAiSettings,
    model: ChatModel,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
}

static PARITY: OnceLock<ParityChat> = OnceLock::new();

/// `tokenizer.convert_ids_to_tokens` bytes for chat logprobs; chat never decodes.
struct Pieces(TokenPieces);

impl OpenAiTokenizer for Pieces {
    fn decode(&self, _ids: &[u32]) -> Option<String> {
        None
    }

    fn byte_level_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.0.byte_level_bytes(id)
    }
}

/// Resolve the chat layer's settings once, when the HTTP app starts.
pub(in crate::api_server::http) fn init(server_args: &ServerArgs) {
    let mc = &server_args.model_config;
    let defaults = &mc.default_sampling_params;
    let mut generation_config = Map::new();
    let mut put = |key: &str, value: Option<Value>| {
        if let Some(value) = value {
            generation_config.insert(key.to_owned(), value);
        }
    };
    put("repetition_penalty", defaults.repetition_penalty.map(Value::from));
    put("temperature", defaults.temperature.map(Value::from));
    put("top_k", defaults.top_k.map(Value::from));
    put("top_p", defaults.top_p.map(Value::from));
    put("min_p", defaults.min_p.map(Value::from));
    let context_length = Some(mc.context_len);
    let tokenizer = tokenizer::resolve_model_file(
        &server_args.tokenizer_path,
        server_args.revision.as_deref(),
        "tokenizer.json",
    )
    .and_then(|path| std::fs::read(path).ok())
    .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
    .map(|json| Arc::new(Pieces(TokenPieces::from_tokenizer_json(&json))) as Arc<dyn OpenAiTokenizer>);
    let settings = OpenAiSettings {
        allow_auto_truncate: server_args.allow_auto_truncate,
        context_length,
        incremental_streaming_output: server_args.incremental_streaming_output,
        reasoning_parser: server_args.reasoning_parser.clone(),
        sampling_defaults: Some("model".into()),
        stream_response_default_include_usage: server_args.stream_response_default_include_usage,
        tool_call_parser: server_args.tool_call_parser.clone(),
        ..Default::default()
    };
    let _ = PARITY.set(ParityChat {
        settings,
        model: ChatModel {
            context_length,
            generation_config,
        },
        tokenizer,
    });
}

/// Serve a chat request through the Python-parity layer, or `None` to leave it
/// to the Dynamo handler.
pub(super) async fn chat(state: &Arc<AppState>, body: &Bytes) -> Option<Response> {
    let parity = PARITY.get()?;
    let formatter = state.chat_formatter.clone()?;
    let render = |request: &Value| render_prompt(&formatter, request).map(Prompt::Text);
    let (lowered, responder) = match lower_chat_prompt(
        body,
        &OpenAiHeaders::default(),
        &parity.settings,
        &parity.model,
        render,
        parity.tokenizer.clone(),
    ) {
        Ok(lowered) => lowered,
        Err(Unsupported(reason)) => {
            tracing::debug!(reason, "chat request left to the Dynamo handler");
            return None;
        }
    };
    let require_reasoning = lowered["require_reasoning"].as_bool().unwrap_or(false);
    let output = match run_generate(state, lowered, true, require_reasoning).await {
        Ok(output) => output,
        Err(response) => return Some(response),
    };
    Some(match output {
        NativeOutput::Unary { status, body } => {
            let reply = match status.is_success() {
                true => responder.unary(&body),
                false => Responder::rejected(&body),
            };
            match reply {
                Some(reply) => reply_response(reply),
                None => (status, [(header::CONTENT_TYPE, "application/json")], body).into_response(),
            }
        }
        NativeOutput::Stream(stream) => stream_response(responder, stream).await,
    })
}

fn reply_response(reply: Reply) -> Response {
    let status = StatusCode::from_u16(reply.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, [(header::CONTENT_TYPE, "application/json")], reply.body).into_response()
}

/// Feed the `/generate` stream to the responder. Its first answer decides the
/// response: a `Reply` replaces the stream (nothing was sent yet), events start
/// an SSE body that carries the rest.
async fn stream_response(
    mut responder: sglang_processor::openai::ChatResponder,
    mut native: futures::stream::BoxStream<'static, String>,
) -> Response {
    let mut first = Vec::new();
    while let Some(data) = native.next().await {
        match responder.stream_data(data.as_bytes()) {
            Err(reply) => return reply_response(reply),
            Ok(events) => {
                let started = !events.is_empty();
                first.extend(events);
                if started || responder.done() {
                    break;
                }
            }
        }
    }
    let body = async_stream::stream! {
        for event in first {
            yield event;
        }
        while !responder.done() {
            let Some(data) = native.next().await else { break };
            match responder.stream_data(data.as_bytes()) {
                Ok(events) => {
                    for event in events {
                        yield event;
                    }
                }
                // A replacement reply can only come before the first event.
                Err(reply) => {
                    yield format!("data: {}\n\n", reply.body);
                    break;
                }
            }
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body.map(Ok::<_, Infallible>)))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[derive(Deserialize)]
struct RenderRequest {
    #[serde(flatten)]
    request: CreateChatCompletionRequest,
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// The prompt Python's `_apply_jinja_template` renders for `request`.
fn render_prompt(formatter: &ChatFormatter, request: &Value) -> Option<String> {
    let parsed: RenderRequest = serde_json::from_value(request.clone()).ok()?;
    formatter
        .render_with_tools(&parsed.request, parsed.chat_template_kwargs.as_ref(), template_tools(request))
        .map_err(|error| tracing::debug!(%error, "chat template render failed"))
        .ok()
}

/// `_process_messages`: the template sees the request's tools as `Tool.model_dump()`,
/// only the named one under a named `tool_choice`, and none under `"none"`.
fn template_tools(request: &Value) -> Option<Vec<Value>> {
    let tools = request.get("tools")?.as_array().filter(|t| !t.is_empty())?;
    let choice = request.get("tool_choice").filter(|c| !c.is_null());
    if choice.is_some_and(|c| c == "none") {
        return None;
    }
    let named = choice.and_then(|c| c.get("function")).and_then(|f| f.get("name"));
    let kept: Vec<Value> = tools
        .iter()
        .filter(|tool| named.is_none_or(|name| tool["function"].get("name") == Some(name)))
        .map(python_tool_dump)
        .collect();
    (!kept.is_empty()).then_some(kept)
}

/// `Tool.model_dump()`: Pydantic's field order and defaults, `function.defer_loading`
/// dropped when unset.
fn python_tool_dump(tool: &Value) -> Value {
    let function = &tool["function"];
    let defer_loading = tool.get("defer_loading").cloned().unwrap_or(Value::Null);
    let mut dumped_function = json!({
        "description": function.get("description").cloned().unwrap_or(Value::Null),
        "name": function.get("name").cloned().unwrap_or(Value::Null),
        "parameters": function.get("parameters").cloned().unwrap_or(Value::Null),
        "strict": function.get("strict").cloned().unwrap_or(Value::Bool(false)),
    });
    let function_defer = match function.get("defer_loading") {
        Some(value) if !value.is_null() => value.clone(),
        _ => defer_loading.clone(),
    };
    if !function_defer.is_null() {
        dumped_function["defer_loading"] = function_defer;
    }
    json!({
        "type": tool.get("type").cloned().unwrap_or_else(|| "function".into()),
        "function": dumped_function,
        "defer_loading": defer_loading,
    })
}
