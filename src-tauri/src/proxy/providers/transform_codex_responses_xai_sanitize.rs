//! xAI (Grok) `Responses` request field sanitization for native Responses
//! upstreams.
//!
//! Codex 0.142+ sends `wire_api="responses"` requests carrying a handful of
//! OpenAI-backend-private fields and tool carriers that xAI's strict
//! `api.x.ai/v1/responses` serde parser rejects (HTTP 400/422). cc-switch's
//! Chat/Anthropic transforms already drop these on the way through, but the
//! *native* Responses passthrough forwards the body verbatim, so we scrub them
//! here.
//!
//! This is a faithful port of sub2api's `patchGrokResponsesBody`
//! (`backend/internal/service/openai_gateway_grok.go`), the production Go
//! gateway that routes Codex → Grok subscriptions. Every transform is a
//! deterministic field removal or structural lift — no semantic rewriting — so
//! the same input always yields the same output and the upstream prompt-cache
//! prefix stays stable across requests. Gated on the xAI OAuth path only (see
//! [`super::codex::provider_needs_responses_namespace_flatten`]), so no other
//! provider is ever touched.
//!
//! Run this *after* namespace flattening: by then Codex's `namespace` tools are
//! already lifted to top-level `function` tools, so the tool-type whitelist
//! below keeps them instead of dropping them.

use std::collections::HashSet;

use serde_json::Value;

/// Codex plugin-private fields removed recursively at any nesting depth.
const RECURSIVE_UNSUPPORTED_FIELDS: &[&str] = &["external_web_access"];

/// Top-level request fields xAI rejects regardless of model.
const TOP_LEVEL_UNSUPPORTED_FIELDS: &[&str] = &["prompt_cache_retention", "safety_identifier"];

/// Top-level sampling fields rejected specifically by grok-4.5.
const GROK_45_UNSUPPORTED_FIELDS: &[&str] = &[
    "presence_penalty",
    "presencePenalty",
    "frequency_penalty",
    "frequencyPenalty",
    "stop",
];

/// Tool `type` values xAI's Responses schema accepts. Sourced from xAI's own
/// serde error enumeration (which is more complete than sub2api's hand-copied
/// list — it includes `image_generation`). Any other `type` is a Codex/OpenAI
/// private carrier (`tool_search`, a stray `namespace`, `custom`, …) that the
/// strict parser would reject, so it is dropped.
const XAI_SUPPORTED_TOOL_TYPES: &[&str] = &[
    "function",
    "web_search",
    "x_search",
    "image_generation",
    "collections_search",
    "file_search",
    "code_execution",
    "code_interpreter",
    "mcp",
    "shell",
];

/// Strip xAI-unsupported fields and tools from a native Codex Responses request
/// body in place. Returns whether anything changed. Deterministic and
/// idempotent: running it twice on the same body changes nothing the second
/// time.
pub(crate) fn sanitize_xai_responses_request(body: &mut Value) -> bool {
    if !body.is_object() {
        return false;
    }

    let mut changed = false;

    // 1. Top-level fields xAI rejects for every model.
    for field in TOP_LEVEL_UNSUPPORTED_FIELDS {
        changed |= remove_top_level_field(body, field);
    }

    // 2. grok-4.5 additionally rejects these sampling knobs.
    if request_targets_grok_45(body) {
        for field in GROK_45_UNSUPPORTED_FIELDS {
            changed |= remove_top_level_field(body, field);
        }
    }

    // 3. Codex plugin-private flags buried at any depth (e.g. inside tools or
    //    tool parameter schemas).
    for field in RECURSIVE_UNSUPPORTED_FIELDS {
        changed |= remove_field_recursive(body, field);
    }

    // 4. Lift the `additional_tools` input carrier (Responses Lite private
    //    shape) up to top-level `tools` so the supported ones survive.
    changed |= promote_additional_tools(body);

    // 5. Drop `content: null` on reasoning input items — xAI's untagged enum
    //    deserializer refuses a present-but-null content field.
    changed |= strip_null_reasoning_content(body);

    // 6. Whitelist the tool types and clean a now-dangling `tool_choice`.
    changed |= filter_unsupported_tools(body);

    // 7. Coerce each surviving `function` tool's `parameters` root schema to an
    //    object type. xAI's strict serde parser rejects tool parameter schemas
    //    whose root is an `anyOf`/`oneOf` union containing a non-object branch
    //    (`[invalid_client_tool_schema] ... root schema is an anyOf/oneOf union
    //    with a non-object branch`), and also rejects a root `type` that is not
    //    `object`. Codex/MCP tools frequently declare `parameters` as a union
    //    (e.g. `{"anyOf":[{"type":"object",...},{"type":"null"}]}`) or omit the
    //    root `type`, so lift those into a single object schema.
    let coerced = coerce_tool_parameters_to_object(body);
    if coerced {
        log::debug!(
            "[Codex] Coerced non-object tool parameter root schemas to object for xAI"
        );
    }
    changed |= coerced;

    changed
}

/// Whether the request's (possibly provider-prefixed) model resolves to
/// grok-4.5. Mirrors sub2api's suffix match: `foo/grok-4.5` counts.
fn request_targets_grok_45(body: &Value) -> bool {
    let Some(model) = body.get("model").and_then(Value::as_str) else {
        return false;
    };
    let mut model = model.trim();
    if let Some(idx) = model.rfind('/') {
        model = model[idx + 1..].trim();
    }
    model.eq_ignore_ascii_case("grok-4.5")
}

fn remove_top_level_field(body: &mut Value, field: &str) -> bool {
    body.as_object_mut()
        .and_then(|obj| obj.remove(field))
        .is_some()
}

/// Delete every occurrence of `field` in the tree, at any depth.
fn remove_field_recursive(value: &mut Value, field: &str) -> bool {
    match value {
        Value::Object(map) => {
            let mut changed = map.remove(field).is_some();
            for child in map.values_mut() {
                changed |= remove_field_recursive(child, field);
            }
            changed
        }
        Value::Array(items) => {
            let mut changed = false;
            for child in items.iter_mut() {
                changed |= remove_field_recursive(child, field);
            }
            changed
        }
        _ => false,
    }
}

fn is_additional_tools_item(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str).map(str::trim) == Some("additional_tools")
}

/// Promote any `additional_tools` carrier items from `input` into top-level
/// `tools`, preserving top-level order and appending carrier tools in order,
/// de-duplicated. The carrier items themselves are removed from `input`.
fn promote_additional_tools(body: &mut Value) -> bool {
    // Clone `input` up front so the later mutable write-back to `body` doesn't
    // collide with the read borrow. Only pays the clone on the rare carrier path.
    let input_items: Vec<Value> = match body.get("input").and_then(Value::as_array) {
        Some(arr) if arr.iter().any(is_additional_tools_item) => arr.clone(),
        _ => return false,
    };

    // Seed merged tools + dedup keys from the existing top-level tools.
    let mut merged: Vec<Value> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            seen.insert(tool_dedup_key(tool));
            merged.push(tool.clone());
        }
    }

    let mut filtered_input: Vec<Value> = Vec::with_capacity(input_items.len());
    let mut promoted = false;
    for item in input_items {
        if is_additional_tools_item(&item) {
            if let Some(carrier_tools) = item.get("tools").and_then(Value::as_array) {
                for tool in carrier_tools {
                    if seen.insert(tool_dedup_key(tool)) {
                        merged.push(tool.clone());
                        promoted = true;
                    }
                }
            }
            continue; // carrier item dropped regardless of dedup outcome
        }
        filtered_input.push(item);
    }

    if let Some(obj) = body.as_object_mut() {
        obj.insert("input".to_string(), Value::Array(filtered_input));
        if promoted {
            obj.insert("tools".to_string(), Value::Array(merged));
        }
    }
    // We reached here only because a carrier existed, so `input` changed.
    true
}

/// Stable dedup key for a tool: `(type, name)`, `(mcp, server_label)`, or the
/// serialized tool as a last resort. Mirrors sub2api's `grokResponsesToolDedupKey`.
fn tool_dedup_key(tool: &Value) -> String {
    let tool_type = tool
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !tool_type.is_empty() {
        if let Some(name) = tool.get("name").and_then(Value::as_str) {
            let name = name.trim();
            if !name.is_empty() {
                return format!("type:{tool_type}\u{0}name:{name}");
            }
        }
        if tool_type == "mcp" {
            if let Some(label) = tool.get("server_label").and_then(Value::as_str) {
                let label = label.trim();
                if !label.is_empty() {
                    return format!("type:mcp\u{0}server_label:{label}");
                }
            }
        }
    }
    format!("json:{tool}")
}

fn strip_null_reasoning_content(body: &mut Value) -> bool {
    let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for item in input.iter_mut() {
        if item.get("type").and_then(Value::as_str).map(str::trim) != Some("reasoning") {
            continue;
        }
        if let Some(obj) = item.as_object_mut() {
            if matches!(obj.get("content"), Some(Value::Null)) {
                obj.remove("content");
                changed = true;
            }
        }
    }
    changed
}

/// Keep only whitelisted tool types and drop a `tool_choice` that now points at
/// a removed or unsupported tool.
fn filter_unsupported_tools(body: &mut Value) -> bool {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return false;
    };
    let original_len = tools.len();
    let filtered: Vec<Value> = tools
        .iter()
        .filter(|tool| {
            let t = tool
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            XAI_SUPPORTED_TOOL_TYPES.contains(&t)
        })
        .cloned()
        .collect();

    let mut changed = false;
    if filtered.len() != original_len {
        if let Some(obj) = body.as_object_mut() {
            if filtered.is_empty() {
                obj.remove("tools");
            } else {
                obj.insert("tools".to_string(), Value::Array(filtered.clone()));
            }
        }
        changed = true;
    }

    if body.get("tool_choice").is_some() && should_drop_tool_choice(body, &filtered) {
        if let Some(obj) = body.as_object_mut() {
            obj.remove("tool_choice");
        }
        changed = true;
    }

    changed
}

/// Whether `tool_choice` should be dropped given the surviving `tools`. String
/// choices (`"auto"`, `"none"`, `"required"`) are always kept; object choices
/// are dropped when they reference an unsupported type or a function name that
/// no longer exists.
fn should_drop_tool_choice(body: &Value, tools: &[Value]) -> bool {
    let Some(tool_choice) = body.get("tool_choice") else {
        return false;
    };
    if tools.is_empty() {
        return true;
    }
    let Some(choice) = tool_choice.as_object() else {
        return false; // "auto"/"none"/"required" string choices stay
    };
    let choice_type = choice
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if choice_type.is_empty() {
        return false;
    }
    if !XAI_SUPPORTED_TOOL_TYPES.contains(&choice_type) {
        return true;
    }
    if choice_type == "function" {
        let choice_name = choice
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| {
                choice
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
            })
            .unwrap_or("")
            .trim();
        if choice_name.is_empty() {
            return false;
        }
        let exists = tools.iter().any(|tool| {
            let t = tool
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| {
                    tool.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                })
                .unwrap_or("")
                .trim();
            t == "function" && name == choice_name
        });
        return !exists;
    }
    false
}

/// Coerce every surviving `function` tool's `parameters` root schema to an
/// object type so xAI's strict tool-schema parser accepts it.
///
/// xAI rejects (`[invalid_client_tool_schema]`) any tool whose `parameters`
/// root is not `{"type":"object",...}` — in particular a root `anyOf`/`oneOf`
/// union that contains a non-object branch (e.g. the common
/// `{"anyOf":[{"type":"object",...},{"type":"null"}]}` shape MCP/Codex tools
/// emit to mark the argument optional). This lifts such unions into a single
/// object schema by merging the `properties`/`required` of every object branch,
/// falling back to a permissive object when no object branch exists. A missing
/// or non-object `parameters` is replaced with a permissive object schema.
///
/// Handles both the flat Responses format (`{"type":"function","parameters":…}`)
/// and the nested Chat format (`{"type":"function","function":{"parameters":…}}`).
/// Also recursively coerces tool declarations embedded in `input` items (e.g.
/// `tool_search_output` carriers), since xAI validates those too. Deterministic
/// and idempotent.
fn coerce_tool_parameters_to_object(body: &mut Value) -> bool {
    let mut changed = false;

    // Top-level `tools` array.
    if let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools.iter_mut() {
            changed |= coerce_tool_entry_parameters(tool);
        }
    }

    // Tool declarations embedded in `input` items (e.g. `tool_search_output`
    // carries a `tools` array with namespace/function children). xAI's strict
    // parser validates these alongside the top-level declarations.
    if let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) {
        for item in input.iter_mut() {
            changed |= coerce_input_item_tools(item);
        }
    }

    changed
}

/// Coerce parameters on a single tool entry, handling both flat and nested
/// (`function` wrapper) formats.
fn coerce_tool_entry_parameters(tool: &mut Value) -> bool {
    let tool_type = tool
        .get("type")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if tool_type != "function" && tool_type != "custom" {
        // Also recurse into carrier types (`namespace`, `mcp`,
        // `tool_search_output`, …) that may embed function children.
        return coerce_nested_tool_children(tool);
    }

    let mut changed = false;
    // Flat format: parameters at top level.
    if let Some(obj) = tool.as_object_mut() {
        if let Some(params) = obj.get_mut("parameters") {
            changed |= coerce_parameter_root_to_object(params);
        }
    }
    // Nested format: parameters inside a `function` wrapper.
    if let Some(func) = tool.get_mut("function").and_then(Value::as_object_mut) {
        if let Some(params) = func.get_mut("parameters") {
            changed |= coerce_parameter_root_to_object(params);
        }
    }
    changed
}

/// Recurse into any `tools`/`children` sub-array of a carrier tool and coerce
/// the function children's parameters.
fn coerce_nested_tool_children(carrier: &mut Value) -> bool {
    let key = if carrier.get("tools").is_some() {
        "tools"
    } else if carrier.get("children").is_some() {
        "children"
    } else {
        return false;
    };
    let Some(children) = carrier.get_mut(key).and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for child in children.iter_mut() {
        changed |= coerce_tool_entry_parameters(child);
    }
    changed
}

/// Coerce tool declarations inside an `input` item (e.g. `tool_search_output`).
fn coerce_input_item_tools(item: &mut Value) -> bool {
    // An input item may carry a `tools` array directly (tool_search_output).
    if let Some(tools) = item.get_mut("tools").and_then(Value::as_array_mut) {
        let mut changed = false;
        for tool in tools.iter_mut() {
            changed |= coerce_tool_entry_parameters(tool);
        }
        if changed {
            return true;
        }
    }
    // Or it may be a carrier with nested children.
    coerce_nested_tool_children(item)
}

/// Normalize a single tool `parameters` schema so its root is an object type.
fn coerce_parameter_root_to_object(params: &mut Value) -> bool {
    // Check for a union at the root FIRST — even if `type: "object"` is also
    // present, xAI's strict parser treats the root as a union and rejects it
    // when any branch is non-object.
    let union_key = ["anyOf", "oneOf", "allOf"]
        .into_iter()
        .find(|k| params.get(*k).is_some());
    if let Some(key) = union_key {
        let mut merged_props: serde_json::Map<String, Value> = serde_json::Map::new();
        let mut merged_required: Vec<String> = Vec::new();
        let mut seen_required: HashSet<String> = HashSet::new();
        if let Some(branches) = params.get(key).and_then(Value::as_array) {
            for branch in branches {
                if branch.get("type").and_then(Value::as_str) != Some("object") {
                    continue;
                }
                if let Some(props) = branch.get("properties").and_then(Value::as_object) {
                    for (k, v) in props {
                        merged_props.insert(k.clone(), v.clone());
                    }
                }
                if let Some(req) = branch.get("required").and_then(Value::as_array) {
                    for r in req.iter() {
                        if let Some(s) = r.as_str() {
                            if seen_required.insert(s.to_string()) {
                                merged_required.push(s.to_string());
                            }
                        }
                    }
                }
            }
        }
        let mut new_schema = serde_json::Map::new();
        new_schema.insert("type".to_string(), Value::String("object".to_string()));
        if let Some(desc) = params.get("description") {
            new_schema.insert("description".to_string(), desc.clone());
        }
        if !merged_props.is_empty() {
            new_schema.insert(
                "properties".to_string(),
                Value::Object(merged_props),
            );
        }
        if !merged_required.is_empty() {
            new_schema.insert(
                "required".to_string(),
                Value::Array(
                    merged_required
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        new_schema.insert(
            "additionalProperties".to_string(),
            Value::Bool(true),
        );
        *params = Value::Object(new_schema);
        return true;
    }

    // Already a proper object schema with no union — nothing to do.
    if params.get("type").and_then(Value::as_str) == Some("object") {
        return false;
    }

    // Object literal missing a `type`, or typed as something other than object:
    // coerce the root to object while preserving any declared `properties`.
    if let Some(obj) = params.as_object_mut() {
        let prev_type = obj.get("type").and_then(Value::as_str).map(str::to_string);
        obj.insert(
            "type".to_string(),
            Value::String("object".to_string()),
        );
        // Keep `properties`/`required` if present; otherwise leave them absent
        // (a parameterless object is valid). Ensure permissiveness when we
        // displaced a non-object type.
        if prev_type.is_some() && prev_type.as_deref() != Some("object") {
            obj.entry("additionalProperties")
                .or_insert(Value::Bool(true));
        }
        return true;
    }

    // Missing/null/non-object `parameters`: substitute a permissive object.
    *params = Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), Value::String("object".to_string())),
        ("properties".to_string(), Value::Object(serde_json::Map::new())),
        (
            "additionalProperties".to_string(),
            Value::Bool(true),
        ),
    ]));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strips_external_web_access_recursively() {
        let mut body = json!({
            "model": "grok-4.5",
            "external_web_access": true,
            "tools": [
                {"type": "function", "name": "f", "external_web_access": true,
                 "parameters": {"type": "object", "q": {"external_web_access": true}}}
            ],
            "metadata": {"external_web_access": false}
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let s = body.to_string();
        assert!(!s.contains("external_web_access"), "left over: {s}");
    }

    #[test]
    fn strips_top_level_unsupported_fields() {
        let mut body = json!({
            "model": "grok-4.5",
            "prompt_cache_retention": "24h",
            "safety_identifier": "abc"
        });
        assert!(sanitize_xai_responses_request(&mut body));
        assert!(body.get("prompt_cache_retention").is_none());
        assert!(body.get("safety_identifier").is_none());
    }

    #[test]
    fn strips_grok_45_only_sampling_fields() {
        let mut body = json!({
            "model": "grok-4.5",
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "stop": ["x"]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        assert!(body.get("presence_penalty").is_none());
        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("stop").is_none());
    }

    #[test]
    fn keeps_sampling_fields_for_non_grok_45() {
        let mut body = json!({
            "model": "grok-4-fast",
            "presence_penalty": 0.1,
            "stop": ["x"]
        });
        // No unsupported fields present, so no change and knobs preserved.
        assert!(!sanitize_xai_responses_request(&mut body));
        assert_eq!(body.get("presence_penalty"), Some(&json!(0.1)));
        assert_eq!(body.get("stop"), Some(&json!(["x"])));
    }

    #[test]
    fn matches_grok_45_with_provider_prefix() {
        let mut body = json!({"model": "xai/grok-4.5", "stop": ["x"]});
        assert!(sanitize_xai_responses_request(&mut body));
        assert!(body.get("stop").is_none());
    }

    #[test]
    fn promotes_additional_tools_dedup() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{"type": "function", "name": "kept"}],
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "additional_tools", "tools": [
                    {"type": "function", "name": "kept"},
                    {"type": "function", "name": "extra"}
                ]}
            ]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        // carrier removed from input
        let input = body.get("input").unwrap().as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert!(input.iter().all(|i| !is_additional_tools_item(i)));
        // extra promoted, kept not duplicated
        let tools = body.get("tools").unwrap().as_array().unwrap();
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t.get("name").and_then(Value::as_str).unwrap())
            .collect();
        assert_eq!(names, vec!["kept", "extra"]);
    }

    #[test]
    fn strips_null_reasoning_content() {
        let mut body = json!({
            "model": "grok-4.5",
            "input": [
                {"type": "reasoning", "content": null, "id": "r1"},
                {"type": "reasoning", "content": [{"text": "keep"}], "id": "r2"}
            ]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let input = body.get("input").unwrap().as_array().unwrap();
        assert!(input[0].get("content").is_none());
        assert!(input[1].get("content").is_some());
    }

    #[test]
    fn filters_unsupported_tool_types() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [
                {"type": "function", "name": "f"},
                {"type": "tool_search"},
                {"type": "custom", "name": "c"},
                {"type": "mcp", "server_label": "s"}
            ]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let types: Vec<&str> = body
            .get("tools")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.get("type").and_then(Value::as_str).unwrap())
            .collect();
        assert_eq!(types, vec!["function", "mcp"]);
    }

    #[test]
    fn drops_dangling_function_tool_choice() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{"type": "tool_search"}],
            "tool_choice": {"type": "function", "name": "gone"}
        });
        assert!(sanitize_xai_responses_request(&mut body));
        // tool_search filtered → no tools → tool_choice dropped
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn keeps_valid_function_tool_choice() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{"type": "function", "name": "run"}],
            "tool_choice": {"type": "function", "name": "run"}
        });
        assert!(!sanitize_xai_responses_request(&mut body));
        assert_eq!(
            body.get("tool_choice").unwrap(),
            &json!({"type": "function", "name": "run"})
        );
    }

    #[test]
    fn keeps_string_tool_choice() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{"type": "function", "name": "run"}],
            "tool_choice": "auto"
        });
        assert!(!sanitize_xai_responses_request(&mut body));
        assert_eq!(body.get("tool_choice").unwrap(), &json!("auto"));
    }

    #[test]
    fn noop_on_clean_request() {
        let mut body = json!({
            "model": "grok-4.5",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "f"}]
        });
        assert!(!sanitize_xai_responses_request(&mut body));
    }

    #[test]
    fn idempotent_second_pass() {
        let mut body = json!({
            "model": "grok-4.5",
            "external_web_access": true,
            "prompt_cache_retention": "24h",
            "tools": [{"type": "function", "name": "f"}, {"type": "tool_search"}]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        // second pass finds nothing left to change
        assert!(!sanitize_xai_responses_request(&mut body));
    }

    #[test]
    fn coerces_anyof_union_parameter_root_to_object() {
        // The exact shape that triggers xAI's
        // `[invalid_client_tool_schema] ... root schema is an anyOf/oneOf union
        // with a non-object branch`: an optional object argument expressed as
        // `anyOf:[object, null]`.
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "mcp__codex_app__automation_update",
                "parameters": {
                    "anyOf": [
                        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                        {"type": "null"}
                    ]
                }
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let params = &body["tools"][0]["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["properties"]["x"]["type"], "string");
        assert_eq!(params["required"], json!(["x"]));
        assert_eq!(params["additionalProperties"], true);
        assert!(params.get("anyOf").is_none());
    }

    #[test]
    fn coerces_oneof_union_merging_multiple_object_branches() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "t",
                "parameters": {
                    "oneOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                        {"type": "object", "properties": {"b": {"type": "number"}}, "required": ["b"]},
                        {"type": "string"}
                    ]
                }
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let props = &body["tools"][0]["parameters"]["properties"];
        assert!(props.get("a").is_some());
        assert!(props.get("b").is_some());
        let required: Vec<&str> = body["tools"][0]["parameters"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"a"));
        assert!(required.contains(&"b"));
        assert_eq!(body["tools"][0]["parameters"]["type"], "object");
    }

    #[test]
    fn coerces_union_with_no_object_branch_to_permissive_object() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "t",
                "parameters": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let params = &body["tools"][0]["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["additionalProperties"], true);
        assert!(params.get("anyOf").is_none());
    }

    #[test]
    fn coerces_non_object_typed_parameters_to_object() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "t",
                "parameters": {"type": "string"}
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let params = &body["tools"][0]["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["additionalProperties"], true);
    }

    #[test]
    fn coerces_null_parameters_to_permissive_object() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "t",
                "parameters": null
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let params = &body["tools"][0]["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["additionalProperties"], true);
    }

    #[test]
    fn leaves_absent_parameters_untouched() {
        // A function tool with no `parameters` field must not gain one — keeps
        // the sanitizer a noop on clean, parameterless tool declarations.
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{"type": "function", "name": "f"}]
        });
        assert!(!sanitize_xai_responses_request(&mut body));
        assert!(body["tools"][0].get("parameters").is_none());
    }

    #[test]
    fn leaves_already_object_parameters_untouched() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
            }]
        });
        assert!(!sanitize_xai_responses_request(&mut body));
        assert_eq!(
            body["tools"][0]["parameters"]["properties"]["x"]["type"],
            "string"
        );
    }

    #[test]
    fn parameter_coercion_is_idempotent() {
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "t",
                "parameters": {"anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}, {"type": "null"}]}
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        assert!(!sanitize_xai_responses_request(&mut body));
    }

    #[test]
    fn coerces_union_parameters_even_with_type_object_present() {
        // Some schemas emit both `type: object` AND `anyOf` at the root.
        // xAI still sees the union and rejects it, so we must coerce.
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "name": "t",
                "parameters": {
                    "type": "object",
                    "anyOf": [
                        {"type": "object", "properties": {"x": {"type": "string"}}},
                        {"type": "null"}
                    ]
                }
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let params = &body["tools"][0]["parameters"];
        assert_eq!(params["type"], "object");
        assert!(params.get("anyOf").is_none());
        assert_eq!(params["properties"]["x"]["type"], "string");
    }

    #[test]
    fn coerces_nested_function_wrapper_parameters() {
        // Chat-style nested format: parameters inside `function`.
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "t",
                    "parameters": {"anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}, {"type": "null"}]}
                }
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let params = &body["tools"][0]["function"]["parameters"];
        assert_eq!(params["type"], "object");
        assert!(params.get("anyOf").is_none());
    }

    #[test]
    fn coerces_tools_nested_in_input_tool_search_output() {
        // Tool declarations inside `input` items (tool_search_output) are also
        // validated by xAI's strict parser.
        let mut body = json!({
            "model": "grok-4.5",
            "tools": [{"type": "function", "name": "top", "parameters": {"type": "object"}}],
            "input": [{
                "type": "tool_search_output",
                "call_id": "c1",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__codex_app__",
                    "tools": [{
                        "type": "function",
                        "name": "automation_update",
                        "parameters": {"anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}, {"type": "null"}]}
                    }]
                }]
            }]
        });
        assert!(sanitize_xai_responses_request(&mut body));
        let params = &body["input"][0]["tools"][0]["tools"][0]["parameters"];
        assert_eq!(params["type"], "object");
        assert!(params.get("anyOf").is_none());
    }
}
