//! Live content tap: subscribe to ChatGPT's WebSocket frames via CDP's Network
//! domain (and the inline-SSE fetch tee), feed payloads to the v1 delta parser.
//!
//! Subscribes to `Network.webSocketFrameReceived` (browser-protocol level, so it
//! catches ChatGPT's socket however it's opened — no fragile page injection),
//! pulls the `conversation-turn-stream` items out of each frame, and feeds their
//! v1 delta payloads to the parser. Emits ("token"|"thinking"|"turn_complete",
//! value) through `on_event`.
//!
//! A turn streams over exactly ONE transport: the WS (when ChatGPT does a
//! stream_handoff) OR the inline /f/conversation SSE body (fresh tabs). We keep a
//! SEPARATE parser per transport so the inactive one's metadata/message-
//! registration ops can't corrupt the active one's token state.

use std::sync::Arc;
use std::sync::Mutex;

use serde_json::Value;

use crate::cdp::CdpClient;
use crate::v1delta::{Event, V1DeltaParser};

/// Callback invoked for every forwarded parser event: `(kind, value)` where
/// `kind` is one of "token" | "thinking" | "turn_complete".
pub type OnEvent = Arc<dyn Fn(&str, &str) + Send + Sync + 'static>;

/// Mutable per-turn state shared with the CDP event handlers. The handlers fire
/// synchronously from the CDP read loop, so the state lives behind a `Mutex`.
///
/// The two parsers are deliberately separate (see module docs): a turn streams
/// over exactly one transport, and the inactive side's message-registration ops
/// would corrupt the active side's token state if they shared a parser.
struct TapState {
    /// WS frames parser.
    parser: V1DeltaParser,
    /// Inline SSE parser.
    sse_parser: V1DeltaParser,
    /// Whether any visible token has been seen this turn (gates turn_complete).
    tokens_seen: bool,
}

impl TapState {
    fn new() -> Self {
        Self {
            parser: V1DeltaParser::default(),
            sse_parser: V1DeltaParser::default(),
            tokens_seen: false,
        }
    }
}

pub struct WsTap {
    cdp: Arc<CdpClient>,
    on_event: OnEvent,
    state: Arc<Mutex<TapState>>,
    started: bool,
}

impl WsTap {
    pub fn new(cdp: Arc<CdpClient>, on_event: OnEvent) -> Self {
        Self {
            cdp,
            on_event,
            state: Arc::new(Mutex::new(TapState::new())),
            started: false,
        }
    }

    /// Enables the Network domain and subscribes to WS frames, then arms the
    /// inline-SSE fetch tee (`Runtime.addBinding("__shadowStream")` +
    /// `Runtime.bindingCalled`). The binding step is best-effort: any failure is
    /// swallowed.
    pub async fn start(&mut self) -> anyhow::Result<()> {
        if self.started {
            return Ok(());
        }
        self.cdp.send("Network.enable", None).await?;

        // WS path: ChatGPT's socket frames at the browser-protocol level.
        {
            let state = Arc::clone(&self.state);
            let on_event = Arc::clone(&self.on_event);
            self.cdp.on(
                "Network.webSocketFrameReceived",
                Arc::new(move |params: Value| {
                    handle_frame(&state, &on_event, &params);
                }),
            );
        }

        // SSE path: fresh tabs stream tokens inline via the /f/conversation
        // text/event-stream body (no WS handoff). The fetch-wrapper tees those
        // `data:` lines to window.__shadowStream -> Runtime.bindingCalled.
        if self
            .cdp
            .send("Runtime.addBinding", Some(serde_json::json!({"name": "__shadowStream"})))
            .await
            .is_ok()
        {
            let state = Arc::clone(&self.state);
            let on_event = Arc::clone(&self.on_event);
            self.cdp.on(
                "Runtime.bindingCalled",
                Arc::new(move |params: Value| {
                    handle_binding(&state, &on_event, &params);
                }),
            );
        }

        self.started = true;
        Ok(())
    }

    /// Fresh parser state for a new turn.
    pub fn reset(&self) {
        let mut state = self.state.lock().expect("WsTap state mutex poisoned");
        state.parser = V1DeltaParser::default();
        state.sse_parser = V1DeltaParser::default();
        state.tokens_seen = false;
    }
}

/// Forward parser events. Suppress a turn_complete that arrives with no tokens
/// this turn — that's the *inactive* transport's spurious complete (e.g. the SSE
/// side of a WS-handoff turn). A real answer always streams visible tokens (incl.
/// the AGENT_STATUS line) before completing.
fn emit(state: &Mutex<TapState>, on_event: &OnEvent, events: Vec<Event>) {
    for ev in events {
        match ev {
            Event::Token(val) => {
                {
                    let mut s = state.lock().expect("WsTap state mutex poisoned");
                    s.tokens_seen = true;
                }
                on_event("token", &val);
            }
            Event::Thinking(val) => {
                on_event("thinking", &val);
            }
            Event::TurnComplete(val) => {
                let tokens_seen = {
                    let s = state.lock().expect("WsTap state mutex poisoned");
                    s.tokens_seen
                };
                if tokens_seen {
                    on_event("turn_complete", &val);
                }
            }
            // Fix wave 2: a typed error event the parser detected (rate-limit /
            // moderation / server error). Forwarded as kind "error" with a JSON
            // payload; the conductor classifies it and fails the turn fast
            // instead of waiting out the stall watchdog.
            Event::StreamError {
                etype,
                code,
                message,
            } => {
                let payload = serde_json::json!({
                    "etype": etype, "code": code, "message": message,
                })
                .to_string();
                on_event("error", &payload);
            }
        }
    }
}

fn handle_binding(state: &Mutex<TapState>, on_event: &OnEvent, params: &Value) {
    if params.get("name").and_then(Value::as_str) != Some("__shadowStream") {
        return;
    }
    let payload = match params.get("payload").and_then(Value::as_str) {
        Some(p) => p,
        None => return,
    };
    // Diagnostics (RUST_LOG=debug): the fetch-wrapper reports the turn-body
    // shape through the same binding with a sentinel prefix. Surface it and stop
    // — it must never reach the SSE parser.
    if let Some(diag) = payload.strip_prefix("__cyrusdiag:") {
        tracing::debug!("[shim] tap/fetch-wrapper {diag}");
        return;
    }
    // Diagnostics (RUST_LOG=debug): the inline-SSE tee fired — log a truncated
    // payload so we can confirm whether fresh-tab turns stream here (and what
    // the delta shape looks like) vs vanishing onto the shared-worker socket.
    tracing::debug!(
        "[shim] tap/sse payload[0..200]={}",
        payload.chars().take(200).collect::<String>()
    );
    // feed the inline-SSE parser; swallow any error.
    let events = {
        let mut s = state.lock().expect("WsTap state mutex poisoned");
        s.sse_parser.feed(payload)
    };
    emit(state, on_event, events);
}

fn handle_frame(state: &Mutex<TapState>, on_event: &OnEvent, params: &Value) {
    let data = params
        .get("response")
        .and_then(|r| r.get("payloadData"))
        .and_then(Value::as_str);
    let data = match data {
        Some(d) if !d.is_empty() => d,
        _ => return,
    };

    let parsed: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Accept either a single object (wrapped to a 1-element list) or a list.
    let arr: Vec<Value> = match parsed {
        Value::Object(_) => vec![parsed],
        Value::Array(items) => items,
        _ => return,
    };

    for m in &arr {
        let m = match m.as_object() {
            Some(obj) => obj,
            None => continue,
        };
        // inner = m["payload"]["payload"] when m["payload"] is a dict.
        let inner = m
            .get("payload")
            .and_then(Value::as_object)
            .and_then(|p| p.get("payload"))
            .and_then(Value::as_object);
        let inner = match inner {
            Some(i) => i,
            None => continue,
        };
        // Diagnostics (RUST_LOG=debug): name every WS frame's inner type so we
        // can tell whether token deltas reach the tap at all, or whether the
        // stream went to the CDP-invisible shared-worker socket (only metadata
        // frames arriving here).
        let inner_type = inner.get("type").and_then(|v| v.as_str());
        tracing::debug!("[shim] tap/ws frame inner.type={inner_type:?}");
        if inner_type != Some("stream-item") {
            continue;
        }
        let enc = match inner.get("encoded_item").and_then(Value::as_str) {
            Some(e) if !e.is_empty() => e,
            _ => continue,
        };
        for line in enc.split('\n') {
            if let Some(rest) = line.strip_prefix("data: ") {
                let events = {
                    let mut s = state.lock().expect("WsTap state mutex poisoned");
                    s.parser.feed(rest)
                };
                emit(state, on_event, events);
            }
        }
    }
}
