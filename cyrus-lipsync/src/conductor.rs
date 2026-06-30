//! ThreadConductor — per-thread-id owner of ONE chatgpt.com tab + all per-turn
//! state. The shim routes each codex request to the conductor for its thread-id,
//! so the main thread and each native subagent thread render natively in codex's
//! TUI without token cross-talk.
//!
//! [`ChatSurface`] (the page-driving surface) and [`ConductorShim`] (the router
//! callbacks) are traits so this file compiles standalone; the page layer and
//! router supply the concrete impls.
//!
//! Two load-bearing invariants:
//!   - Each conductor opens its OWN page socket (one tab = one CdpClient = one
//!     WsTap). A shared socket reintroduces cross-thread token cross-talk.
//!   - Re-delivering a withheld tool result must never re-execute a mutation.
//!     The result is cached against the call signature and the cached value is
//!     replayed through the tool response — never by interrupting generation,
//!     which would derail the model into a summary. A repo mutation
//!     (apply_patch/repo_write/repo_edit) clears the cache so edit→rebuild isn't
//!     read as a loop; `update_plan` is exempt. Firmer wording past
//!     `loop_repeat_threshold` (default 4).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::Mutex;

use crate::config::ShadowConfig;
use crate::responses::completed;
use crate::responses::custom_tool_call_item;
use crate::responses::extract_prompt;
use crate::responses::function_call_item;
use crate::responses::last_tool_output;
use crate::responses::message_item;
use crate::responses::parse_tool_call;
use crate::responses::sse_frame;
use crate::responses::{build_tool_preamble, TurnDriver};

// ===== conductor preambles ===================================================
//
// The text injected once per ChatGPT thread, ahead of the first user task. These
// strings are load-bearing (the connector-load instruction, the "empty result ==
// success" posture, the apply_patch/shell_command/update_plan tool names, and the
// PowerShell guards are what make chatgpt.com behave) — do NOT paraphrase or reflow.

/// `CONDUCTOR_PREAMBLE` — the short hand-rolled fallback (used only when
/// `SHIM_FORWARD_CODEX_PROMPT=0`, kept for A/B and as a safety net).
pub const CONDUCTOR_PREAMBLE_HEAD: &str = concat!(
    "You are an autonomous coding agent operating the user's REAL machine through the ",
    "**repo-agent** MCP connector. Every tool runs for real on the actual machine and ",
    "returns real output — you have no other way to see files, run commands, or change ",
    "anything, and you must never invent or assume a tool's result.\n\n",
    "FIRST — LOAD THE TOOLS: before anything else, make sure the **repo-agent** connector ",
    "and its `repo_*` tools are loaded/enabled in this chat. If they are not active yet, ",
    "load the repo-agent connector now. Do not say you lack access to the tools — you ",
    "have the connector; load it and use it.\n\n",
    "Tools (use the most direct one for each step):\n",
    "- repo_shell({command}) — run a shell command (see the SHELL note below). Your main tool.\n",
    "- repo_read({path}) — read a file (line-numbered). Read before you edit.\n",
    "- repo_grep / repo_glob — search file contents / find files by name.\n",
    "- repo_edit({path, old_string, new_string}) — exact-string edit of an existing file.\n",
    "- repo_write({path, content}) — create or fully overwrite a file.\n",
    "- repo_run({profile}) — run a configured profile (test/build/lint/typecheck).\n\n",
    "How to work:\n",
    "- Go straight at the task with the right tool. Don't pad with exploration — call ",
    "repo_status AT MOST ONCE, and only if you actually need git state.\n",
    "- Read a file before editing it; make minimal, correct edits.\n",
    "- A result labeled `RESULT (BASE64)` is base64-encoded — decode it to read it. If a ",
    "result is empty or a short 'blocked' notice, re-issue the same call once to get the ",
    "`RESULT (BASE64)` version. For long commands use repo_shell({command, background:true}) ",
    "and poll repo_bg_output.\n",
    "- If a result comes back blocked or empty, the call still RAN — do NOT retry it ",
    "identically; recover a different way or move on.\n",
    "- When the task is fully done, reply with a short, plain final answer and NO tool ",
    "call. That message is your result.\n\n",
);
/// [`CONDUCTOR_PREAMBLE_HEAD`]'s tail: the TASK marker. The platform [`shell_rules`]
/// are spliced between them by [`build_conductor_preamble`].
pub const CONDUCTOR_PREAMBLE_TAIL: &str = "TASK:\n";

/// `CONDUCTOR_BRIDGE` — the connector bridge sandwiched ABOVE codex's forwarded
/// real instructions (the default path when `SHIM_FORWARD_CODEX_PROMPT` is on).
pub const CONDUCTOR_BRIDGE: &str = concat!(
    "[connector bridge — read this first]\n",
    "You operate through the **repo-agent** MCP connector, which exposes ALL of your tools in ",
    "THIS chat under their real names: `shell_command` (run ANY command — also how you ",
    "read, search, and list files), `apply_patch` (create/edit/delete files), and ",
    "`update_plan` (your TODO list), plus the `repo_*` tools. These are ALWAYS available — ALL ",
    "of them, EVERY turn. The connector never exposes 'only some' of them, and no tool is ever ",
    "'not made available this turn'.\n",
    "BEFORE YOUR FIRST ACTION, make sure the repo-agent connector is loaded/enabled in this chat. ",
    "If a tool ever looks missing — you can't find `shell_command`, you think you 'only have the ",
    "patch tool', or a call errors with 'no such tool' — the ONLY cause is that the connector ",
    "isn't loaded yet: LOAD it now and call the tool. NEVER tell the user a tool 'wasn't made ",
    "available', that you can 'only edit files but not run commands', or that you lack access — ",
    "you HAVE `shell_command` and can run anything. When unsure whether you can do something, CALL ",
    "the tool and find out; do not narrate that you might not have it.\n",
    "Every tool call runs FOR REAL on the user's actual machine and returns real output. ",
    "You have NO other sandbox: never use a built-in code interpreter, and never invent, ",
    "assume, or guess a tool's result.\n",
    "ENCODED RESULTS: most tool results are plain text. Some come back labeled ",
    "`RESULT (BASE64)` followed by base64 — when you see that label, base64-decode the text ",
    "after it and read the decoded content before reasoning about it. To read a file use ",
    "`repo_read` (or `shell_command`), not `repo_resume` (which resumes a prior session, not ",
    "file contents).\n",
    "If a tool result ever comes back empty or as a short 'blocked'/'safety' notice, the ",
    "call still RAN — just re-issue the SAME call once; the re-issued result comes back as ",
    "`RESULT (BASE64)` (decode it). Don't give up on a tool after one empty result.\n",
    "LONG COMMANDS: for slow work (full builds, large test suites, installs), run it in ",
    "the BACKGROUND — `repo_shell({command, background:true})` returns immediately with an ",
    "id; poll `repo_bg_output({id})` for new output until it finishes. This keeps a slow ",
    "command from timing out the turn.\n",
    "Everything between the markers below is your operating manual — follow it.\n",
    "================ BEGIN OPERATING INSTRUCTIONS ================\n",
);

/// The separator sandwiched BELOW codex's forwarded real instructions (closes the
/// operating manual, re-asserts the connector + the platform shell guards, then
/// opens the task). Split in two so the platform shell rules ([`shell_rules`]) are
/// spliced between by [`build_conductor_preamble`], matching the shell the tools
/// actually run (`powershell` on Windows, `sh` elsewhere). This is the part up to
/// (not including) the SHELL line.
pub const CONDUCTOR_TASK_SEP_HEAD: &str = concat!(
    "\n================ END OPERATING INSTRUCTIONS ================\n\n",
    "Reminder: do everything through the repo-agent connector tools (shell_command, ",
    "apply_patch, update_plan). They execute for real. A result labeled `RESULT (BASE64)` ",
    "is base64-encoded — decode it to read it. If a result is empty or a short 'blocked' ",
    "notice, re-issue the same call once to get the `RESULT (BASE64)` version.\n",
);
/// The task separator after the SHELL line (see [`CONDUCTOR_TASK_SEP_HEAD`]).
pub const CONDUCTOR_TASK_SEP_TAIL: &str = concat!(
    "Now complete the task below.\n\n",
    "# Task\n",
);

/// Platform shell guidance for the model. The repo tools run `powershell.exe` on
/// Windows and `sh -c` elsewhere (chimera's `platform_shell`), so the model has to
/// be told which dialect to write — otherwise on macOS/Linux it writes PowerShell
/// into `sh` and every command fails to parse. Both arms compile on every target
/// (runtime `cfg!`), so a typo in either can't slip through a single-platform build.
pub fn shell_rules() -> &'static str {
    if cfg!(windows) {
        "SHELL = Windows PowerShell 5.1. Write PowerShell, NOT bash — bash-isms throw parser \
errors. Chain with `;` not `&&`/`||`; discard output with `-ErrorAction SilentlyContinue` or \
`$null`, never `2>/dev/null`; list with `Get-ChildItem` not `ls -1`/`find`; read with \
`Get-Content` not `cat`; write with `Set-Content`/`Out-File` not `printf >`; no `&&`, `|| true`, \
`2>&1` on native exes, or `/dev/null`."
    } else {
        "SHELL = POSIX sh (`/bin/sh -c`). Standard Unix shell syntax: chain with `;`, `&&`, `||`; \
redirect with `>`, `>>`, `2>&1`, `2>/dev/null`; the usual `ls`, `cat`, `grep`, `find`, `printf` \
are available. Linux `/bin/sh` may be dash (not bash), so prefer POSIX-portable syntax over \
bash-only extensions."
    }
}

/// The current platform's shell name, for inline tool descriptions.
pub fn shell_word() -> &'static str {
    if cfg!(windows) { "PowerShell" } else { "sh" }
}

/// `THREAD_BIND_DIRECTIVE` — injected at the TOP of a NON-main thread's first
/// message. Reuses chimera's register-first rail: `repo_register` binds THIS
/// conversation to `codex:<thread-id>` so its tool calls route to THIS conductor.
/// Carries a literal `{thread_id}` placeholder filled by [`fill_thread_bind`].
pub const THREAD_BIND_DIRECTIVE: &str = concat!(
    "[thread binding — FIRST ACTION, before any other tool]\n",
    "You are one sub-thread of a larger codex session. Before anything else, call ",
    "`repo_register` with `agent_id` set to EXACTLY this string (copy it verbatim, ",
    "including the `codex:` prefix):\n",
    "    codex:{thread_id}\n",
    "Call it ONCE as your very first action. It binds this conversation to your ",
    "thread so your tool calls run in YOUR thread's context, not another thread's. ",
    "It returns immediately; then carry on with the task below normally.\n\n",
);

/// Substitute `{thread_id}` in [`THREAD_BIND_DIRECTIVE`] — the Python used
/// `THREAD_BIND_DIRECTIVE.format(thread_id=...)`. Only that one token is replaced.
pub fn fill_thread_bind(thread_id: &str) -> String {
    THREAD_BIND_DIRECTIVE.replace("{thread_id}", thread_id)
}

/// The text injected once per ChatGPT thread, ahead of the first user task
/// (`build_conductor_preamble`).
///
/// Default: fuse codex's real `instructions` (forwarded from the request) with the
/// connector bridge. With `SHIM_FORWARD_CODEX_PROMPT=0` (or no usable
/// `instructions`) fall back to the short hand-rolled [`CONDUCTOR_PREAMBLE_HEAD`].
/// Background, tool-less codex subagents: memory consolidation and compaction
/// run whole tasks through `/v1/responses` with NO tools and no need for the
/// connector. They get codex's instructions verbatim — no connector bridge, no
/// thread binding — and run as ephemeral temp-chats outside the project.
pub fn is_headless_kind(kind: Option<&str>) -> bool {
    // codex's `x-codex-turn-metadata.request_kind` uses "memory" and (for the
    // local-compaction turn) "compaction"; older builds used the longer
    // "memory_consolidation" / "compact". Match all of them.
    matches!(
        kind,
        Some("memory") | Some("memory_consolidation") | Some("compact") | Some("compaction")
    )
}

/// codex's per-turn reasoning effort (`body.reasoning.effort`), mapped to the
/// ChatGPT thinking-effort vocabulary (min/standard/extended/max). `None` when
/// codex sends no effort or an unrecognized value — the caller then keeps the
/// launch default already forced on the tab.
pub fn body_reasoning_effort(body: &Value) -> Option<String> {
    let raw = body.get("reasoning")?.get("effort")?.as_str()?;
    crate::provider::resolve_effort(Some(raw))
}

/// codex's per-turn model (`body.model`) — the codex picker's CURRENT choice.
/// It must win over the launch default so switching models mid-session actually
/// re-pins the ChatGPT tab. `None` when a request omits it (the caller then
/// keeps the launch model already forced on the tab).
pub fn body_model(body: &Value) -> Option<String> {
    body.get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// First-turn message for a headless background task: codex's instructions
/// verbatim ahead of the prompt — no connector bridge, no thread binding.
pub fn build_headless_message(body: &Value, prompt: &str) -> String {
    let instr = body
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if instr.is_empty() {
        prompt.to_string()
    } else {
        format!("{instr}\n\n{prompt}")
    }
}

pub fn build_conductor_preamble(body: &Value) -> String {
    let forward = std::env::var("SHIM_FORWARD_CODEX_PROMPT")
        .map(|v| v != "0")
        .unwrap_or(true);
    if forward {
        if let Some(instr) = body.get("instructions").and_then(Value::as_str) {
            let instr = instr.trim();
            if !instr.is_empty() {
                return format!(
                    "{CONDUCTOR_BRIDGE}{instr}{CONDUCTOR_TASK_SEP_HEAD}{}\n{CONDUCTOR_TASK_SEP_TAIL}",
                    shell_rules()
                );
            }
        }
    }
    // Fallback (no forwarded codex instructions): splice the platform shell rules
    // into the hand-rolled preamble before its TASK marker.
    format!("{CONDUCTOR_PREAMBLE_HEAD}{}\n{CONDUCTOR_PREAMBLE_TAIL}", shell_rules())
}

// ===== per-folder project resolution =========================================
//
// codex stamps each request with its workspace cwd in an <environment_context>
// input item (<cwd>PATH</cwd>). We map cwd -> a ChatGPT Project (memory off) so
// every codex session in a repo lands in that repo's project, and — because codex
// subagents inherit the parent's cwd — main + subagents group into the SAME
// project.

/// Pull the workspace cwd from codex's `<environment_context>` input item
/// (`extract_cwd`). Mirrors the Python's `<cwd>\s*(.*?)\s*</cwd>` (case-insensitive,
/// dot-all) scan over each input item's flattened content/text.
pub fn extract_cwd(body: &Value) -> Option<String> {
    let items = body.get("input")?.as_array()?;
    for item in items {
        let obj = item.as_object()?;
        // _content_text(item["content"]) if "content" in item else (item.get("text") or "")
        let txt: String = if obj.contains_key("content") {
            crate::responses::content_text(obj.get("content").unwrap_or(&Value::Null))
        } else {
            obj.get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        if txt.contains("<cwd>") {
            if let Some(cwd) = parse_cwd_tag(&txt) {
                return Some(cwd);
            }
        }
    }
    None
}

/// Extract the inner text of the first `<cwd>...</cwd>` (case-insensitive),
/// trimmed. Returns None when the tag is empty/whitespace, matching the Python
/// guard `if m and m.group(1).strip()`.
fn parse_cwd_tag(txt: &str) -> Option<String> {
    // Case-insensitive locate of the opening/closing tags.
    let lower = txt.to_ascii_lowercase();
    let open = lower.find("<cwd>")?;
    let after = open + "<cwd>".len();
    let close_rel = lower[after..].find("</cwd>")?;
    let inner = txt[after..after + close_rel].trim();
    if inner.is_empty() {
        None
    } else {
        Some(inner.to_string())
    }
}

/// A stable, human-readable project name from a repo path: the leaf folder
/// (`project_name_for_cwd`). Mirrors `re.split(r"[\\/]+", cwd.rstrip("\\/"))[-1]`.
pub fn project_name_for_cwd(cwd: &str) -> String {
    let trimmed = cwd.trim_end_matches(['\\', '/']);
    let leaf = trimmed
        .rsplit(|c| c == '\\' || c == '/')
        .find(|s| !s.is_empty())
        .filter(|s| !s.is_empty())
        .unwrap_or(cwd);
    let leaf = if leaf.is_empty() { cwd } else { leaf };
    format!("codex: {leaf}")
}

// ===== the page surface + router contracts the conductor drives ==============

/// The chat.py `ChatSurface` surface the conductor drives. The page layer
/// (chat.rs, not yet in this scaffold) implements it; the conductor depends only
/// on these behaviours. Method names/semantics mirror chat.py 1:1.
#[async_trait]
pub trait ChatSurface: Send + Sync {
    /// Paste `text` into the composer and send it (chat.inject).
    async fn inject(&self, text: &str) -> anyhow::Result<()>;
    /// Current page state probe: `{generating: bool, hasApprove: bool, ...}`
    /// (chat.state).
    async fn state(&self) -> anyhow::Result<Value>;
    /// Click the write-confirmation card (chat.approve).
    async fn approve(&self) -> anyhow::Result<()>;
    /// Stop the in-flight generation (chat.stop).
    async fn stop(&self) -> anyhow::Result<()>;
    /// Resolve a model slug from a friendly spec (chat.resolve_slug); None ==
    /// account default lane.
    async fn resolve_slug(&self, model: &str) -> anyhow::Result<Option<String>>;
    /// Set the fetch-wrapper overrides for subsequent turns (chat.set_overrides).
    async fn set_overrides(&self, overrides: ChatOverrides) -> anyhow::Result<()>;
    /// The current server-side conversation id, if any (chat.current_conversation_id).
    async fn current_conversation_id(&self) -> anyhow::Result<Option<String>>;
    /// Create a memory-off ChatGPT Project and return its gizmo id (chat.create_project).
    async fn create_project(&self, name: &str) -> anyhow::Result<Option<String>>;
    /// Rename a conversation by id (PATCH /backend-api/conversation/{id}). Used to
    /// name plain conversations `[proj: <repo>]` for grouping when we deliberately
    /// don't scope them into a real Project. Default no-op so mocks/tests are
    /// unaffected; best-effort at the call site.
    async fn set_conversation_title(&self, _conv_id: &str, _title: &str) -> anyhow::Result<()> {
        Ok(())
    }
    /// Wait until the composer is ready after a (re)load (chat._wait_composer).
    async fn wait_composer(&self) -> anyhow::Result<()>;
    /// Whether the page session is logged in (the `/api/auth/session` probe the
    /// cyrus-setup chrome step uses). A logged-out chatgpt.com tab still evals
    /// fine, so the plain `state()` liveness probe cannot see the login wall —
    /// this can. Default `true` so offline mocks/tests are unaffected.
    async fn is_logged_in(&self) -> anyhow::Result<bool> {
        Ok(true)
    }
}

/// The override axes carried in the `/backend-api/f/conversation` turn body and
/// forced via the fetch-wrapper (chat.set_overrides args). `None` fields keep the
/// account/page default for that axis.
#[derive(Debug, Clone, Default)]
pub struct ChatOverrides {
    pub model: Option<String>,
    pub thinking_effort: Option<String>,
    pub gizmo_id: Option<String>,
    /// `history_and_training_disabled` per-turn (project memory-off doesn't stop
    /// account "Reference chat history"; this does).
    pub no_history: Option<bool>,
}

/// The router callbacks the conductor invokes (`ShadowResponsesShim` methods used
/// by `ThreadConductor`). The real router (responses.rs / a richer shim) provides
/// these; expressed as a trait so the conductor compiles against the current
/// scaffold without importing a not-yet-built concrete shim.
#[async_trait]
pub trait ConductorShim: Send + Sync {
    /// `shim._main_thread_id` — the id of the first non-subagent (root) thread, if
    /// one has been registered yet.
    async fn main_thread_id(&self) -> Option<String>;
    /// `shim._ensure_tabs()` — bring up the shared TabFactory (browser control
    /// socket) once.
    async fn ensure_tabs(&self) -> anyhow::Result<()>;
    /// `shim.tabs.open_tab(url, agent_id, human)` — open a fresh tab and return its
    /// target id.
    async fn open_tab(
        &self,
        url: &str,
        agent_id: Option<&str>,
        human: bool,
    ) -> anyhow::Result<String>;
    /// `shim.tabs.close_tab(target_id)`.
    async fn close_tab(&self, target_id: &str);
 /// Build a page surface bound to this tab's OWN page
    /// socket. Equivalent to `CDPClient.for_target(...)` + `ChatSurface(cdp, cfg)` +
    /// `WSTap(cdp, on_ws).start()`. The `on_ws` callback forwards `(kind, value)`
    /// onto the conductor's per-turn WS queue.
    async fn open_page(
        &self,
        target_id: &str,
        on_ws: WsCallback,
    ) -> anyhow::Result<Arc<dyn ChatSurface>>;
    /// `shim._seed_binds_once()` — claim pre-existing unbound chimera sessions for
    /// "main" before the first subagent tab opens.
    async fn seed_binds_once(&self) -> anyhow::Result<()>;
    /// `shim.enqueue_bind(thread_id)` — queue this non-main thread for elimination
    /// binding.
    async fn enqueue_bind(&self, thread_id: &str);
    /// `shim.resolve_project(cwd, chat)` — map a repo cwd to a memory-off ChatGPT
    /// Project gizmo, creating + caching one the first time a folder is seen.
    async fn resolve_project(
        &self,
        cwd: &str,
        chat: &Arc<dyn ChatSurface>,
    ) -> anyhow::Result<Option<String>>;

    /// Tail chimera's `/events` SSE feed for connector-tool liveness: ChatGPT
    /// emits NO tokens while one of its native MCP connector tools runs
    /// server-side, so chimera's per-tool event feed is the only signal that the
    /// turn is healthy rather than rate-limit dead. Implementations spawn a
    /// background task that invokes `on_event` once per received event — ANY
    /// event kind counts (completions today; started/heartbeat kinds as the
    /// server grows them) — and return its handle so the conductor can abort it
    /// at turn end. A down/unreachable server must fail silently (task simply
    /// ends; at most one debug log).
    ///
    /// `agent` scopes the tail to THIS conversation's chimera agent id
    /// (`/events?agent=<id>`): chimera stamps every event with the agent its
    /// session bound to via `repo_register` — "main" for the root conductor,
    /// "codex:<thread>" for bound sub-threads. Without the filter, ANY
    /// session's tool activity would mask a genuinely dead turn on this one.
    ///
    /// The default returns `None` — no feed, no keepalives — so offline shims
    /// (tests) degrade to the WS-only stall watchdog exactly as before.
    fn spawn_server_events_tail(
        &self,
        agent: &str,
        on_event: ServerEventCallback,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let _ = (agent, on_event);
        None
    }
}

/// The WS tap callback the page layer invokes per forwarded parser event:
/// `(kind, value)` where `kind` is "token" | "thinking" | "turn_complete".
pub type WsCallback = Arc<dyn Fn(&str, &str) + Send + Sync + 'static>;

/// Liveness callback for [`ConductorShim::spawn_server_events_tail`]: invoked
/// once per chimera `/events` event, payload-agnostic on purpose (any activity
/// on the feed proves the turn is alive).
pub type ServerEventCallback = Arc<dyn Fn() + Send + Sync + 'static>;

// ===== the merged per-turn event stream ======================================

/// One item on the conductor's MERGED per-turn stream. Carries ChatGPT's
/// final-answer tokens AND chimera's out-of-band tool calls so their mutual order
/// is preserved (a tool call that interrupts streamed text lands after exactly the
/// tokens that preceded it). Mirrors the Python `("token", str) | ("call", dict)`.
#[derive(Debug)]
pub enum Item {
    /// A visible answer token.
    Token(String),
    /// A chimera tool call to dispatch to codex.
    Call(ToolCallEvent),
    /// Liveness ping: carries no visible text, but proves the turn is still
    /// healthy so the per-turn stall watchdog doesn't abort it. Fed by the WS
    /// tap (a reasoning/"thinking" event during a long reasoning pass) AND by
    /// the chimera `/events` tail (a connector-tool event while ChatGPT is
    /// token-silently blocked on a native MCP tool).
    Keepalive,
    /// A ChatGPT stream error the tap detected (rate-limit / moderation /
    /// server error). Fails the turn immediately with the carried codex-facing
    /// code instead of waiting out the stall watchdog.
    Error(TurnFailed),
}

/// A turn failure with an explicit codex-facing error code, surfaced on the
/// `response.failed` frame. codex's contract (codex-api/src/sse/responses.rs):
/// `"invalid_prompt"` is FATAL (no retry, message shown to the user);
/// `"shim_error"` — like any unknown code — is RETRYABLE (up to 5x, honoring a
/// retry-after parsed from the message). Choose codes accordingly.
#[derive(Debug, Clone)]
pub struct TurnFailed {
    pub code: String,
    pub message: String,
}

impl std::fmt::Display for TurnFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for TurnFailed {}

/// A chimera tool call routed onto the merged stream (the `call` dict). `future`
/// resolves with the tool output once codex has executed it, unblocking the held
/// `/control/toolcall` request.
#[derive(Debug)]
pub struct ToolCallEvent {
    pub name: String,
    /// "function" | "custom" (apply_patch and other freeform tools).
    pub kind: String,
    pub arguments: Value,
    /// raw freeform input text (custom tools), else "".
    pub input: String,
    pub call_id: String,
    /// Resolved by `run_turn_conductor` with the tool output codex returned.
    pub future: oneshot::Sender<String>,
}

/// A cached call record for moderation-loop recovery (`_recent_calls` value).
#[derive(Debug, Clone, Default)]
struct RecentCall {
    count: u32,
    /// The real output; once set, a repeat is re-delivered from cache (never
    /// re-executed).
    result: Option<String>,
}

// ===== the conductor =========================================================

/// Per-thread-id conductor: owns ONE chatgpt.com tab + all per-turn state.
///
/// codex tags every Responses request with a `thread-id` header: the MAIN session
/// and each native subagent thread carry distinct ids, so each gets its own free-
/// ChatGPT brain and renders natively in codex's own TUI. All conductors share one
/// browser control socket (the shim's TabFactory) but each opens its OWN page
/// socket — so two threads' token streams cannot cross-talk.
pub struct ThreadConductor {
    shim: Arc<dyn ConductorShim>,
    cfg: ShadowConfig,
    /// codex's thread-id. Behind a SYNC mutex (never held across an await)
    /// because the router REBINDS the eager pre-boot conductor's placeholder id
    /// (`"__eager_main__"`) to the first real main thread-id, mirroring the
    /// Python `get_conductor`'s `tc = self._eager_main; tc.thread_id = key`.
    /// Read via [`Self::thread_id`].
    thread_id: std::sync::Mutex<String>,
    /// The launch-default model — the FALLBACK when a request omits `body.model`
    /// (and the seed for the eager boot tab). codex's per-request `body.model`
    /// (the live picker choice) overrides it every turn via [`Self::apply_turn_overrides`].
    model: String,
    effort: Option<String>,
    /// The model currently forced on the tab (raw codex spec, e.g. `gpt-5-5-pro`).
    /// Tracks codex's per-request `body.model` so we rewrite the localStorage
    /// override the moment the picker choice changes. `None` until the first turn
    /// applies one — which forces an explicit write on turn 1 (boot pinned only
    /// the launch default, which may not be what codex actually requested).
    applied_model: Mutex<Option<String>>,
    /// The reasoning effort currently forced on the tab (ChatGPT vocabulary:
    /// min/standard/extended/max). Tracks codex's per-request `reasoning.effort`
    /// so we only rewrite the localStorage override — and reassert model/gizmo
    /// scoping — when the effort actually changes between turns.
    applied_effort: Mutex<Option<String>>,

    /// `x-openai-subagent` header value when codex tags this thread as a subagent;
    /// None for the root/main thread. Routing is still by thread-id — this is the
    /// explicit "is a sub-thread" signal (codex sends NO parent-thread-id header).
    subagent_kind: Mutex<Option<String>>,

    // --- this thread's tab ---
    chat: Mutex<Option<Arc<dyn ChatSurface>>>,
    target: Mutex<Option<String>>,
    /// Per-turn WS event queue (the `_q`): receives `(kind, value)` from the tap.
    /// Behind an `Arc` so the WS-watcher task can restore the receiver into this
    /// same slot when the turn ends (without aliasing `&self` across the spawn).
    ws_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<(String, String)>>>>,
    ws_tx: Mutex<Option<mpsc::UnboundedSender<(String, String)>>>,
    /// Serializes turns (the `_turn_lock`).
    turn_lock: Mutex<()>,
    booted: Mutex<bool>,
    boot_lock: Mutex<()>,
    /// connector preamble injected once per thread (`_protocol_sent`).
    protocol_sent: Mutex<bool>,
    /// this thread's ChatGPT Project, memory off (`_gizmo`).
    gizmo: Mutex<Option<String>>,
    /// per-folder resolution runs once per thread (`_project_resolved`).
    project_resolved: Mutex<bool>,
    /// Pending `[proj: <repo>]` title for this thread's conversation. Replaces
    /// gizmo scoping on the critical path: the conversation is created PLAIN
    /// (never gated on a Project that might 404), then named once it exists.
    proj_title: Mutex<Option<String>>,
    title_applied: Mutex<bool>,

    // --- conductor state ---
    /// The MERGED per-turn stream (`_events`): Token | Call, order preserved.
    events_tx: Mutex<Option<mpsc::UnboundedSender<Item>>>,
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<Item>>>,
    /// The call codex is currently executing (`_inflight_call`); its future is
    /// resolved when codex returns the tool output.
    inflight_call: Mutex<Option<oneshot::Sender<String>>>,
    /// Resolves with the final text at turn end (`_turn_done`).
    turn_done_tx: Mutex<Option<oneshot::Sender<String>>>,
    turn_done_rx: Mutex<Option<oneshot::Receiver<String>>>,
    /// Accumulated visible text this turn (`_turn_text`).
    turn_text: Arc<Mutex<String>>,
    /// per-turn sig -> record (moderation-loop recovery) (`_recent_calls`).
    recent_calls: Mutex<HashMap<String, RecentCall>>,
    /// This turn's chimera `/events` liveness tail (connector-tool keepalives).
    /// Spawned per fresh turn alongside the WS watcher; unlike the watcher (whose
    /// loop ends itself on `turn_complete`) the SSE tail is unbounded, so the
    /// conductor aborts it explicitly when the turn ends.
    server_tail: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl ThreadConductor {
    /// `ThreadConductor.__init__`.
    pub fn new(
        shim: Arc<dyn ConductorShim>,
        cfg: ShadowConfig,
        thread_id: impl Into<String>,
        model: impl Into<String>,
        effort: Option<String>,
    ) -> Self {
        Self {
            shim,
            cfg,
            thread_id: std::sync::Mutex::new(thread_id.into()),
            model: model.into(),
            // None on purpose: the first turn always re-pins the model from
            // `body.model`, since boot only forced the launch default (which may
            // differ from what codex's picker actually requests on turn 1).
            applied_model: Mutex::new(None),
            // boot() forces the launch-default effort on the tab; track its mapped
            // form so per-turn codex efforts compare against what's actually set.
            applied_effort: Mutex::new(crate::provider::resolve_effort(effort.as_deref())),
            effort,
            subagent_kind: Mutex::new(None),
            chat: Mutex::new(None),
            target: Mutex::new(None),
            ws_rx: Arc::new(Mutex::new(None)),
            ws_tx: Mutex::new(None),
            turn_lock: Mutex::new(()),
            booted: Mutex::new(false),
            boot_lock: Mutex::new(()),
            protocol_sent: Mutex::new(false),
            gizmo: Mutex::new(None),
            project_resolved: Mutex::new(false),
            proj_title: Mutex::new(None),
            title_applied: Mutex::new(false),
            events_tx: Mutex::new(None),
            events_rx: Mutex::new(None),
            inflight_call: Mutex::new(None),
            turn_done_tx: Mutex::new(None),
            turn_done_rx: Mutex::new(None),
            turn_text: Arc::new(Mutex::new(String::new())),
            recent_calls: Mutex::new(HashMap::new()),
            server_tail: Mutex::new(None),
        }
    }

    /// The current thread id.
    /// Poison-proof: a panic elsewhere under this lock must not take every
    /// thread's routing down with it — the plain String inside stays valid.
    pub fn thread_id(&self) -> String {
        self.thread_id
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Rebind this conductor to a new thread id. ONLY used when the router hands
    /// the eager pre-booted tab to the first main thread (Python `get_conductor`:
    /// `tc = self._eager_main; self._eager_main = None; tc.thread_id = key`).
    pub fn rebind_thread_id(&self, id: &str) {
        *self.thread_id.lock().unwrap_or_else(|e| e.into_inner()) = id.to_string();
    }

    /// Whether this conductor's tab has booted (the Python `_booted` attribute;
    /// read by the router's `/health` `booted` aggregate).
    pub async fn is_booted(&self) -> bool {
        *self.booted.lock().await
    }

    /// The seeded subagent kind, if any (the Python `subagent_kind` attribute;
    /// read by the router for `get_conductor` seeding assertions/diagnostics).
    pub async fn subagent_kind(&self) -> Option<String> {
        self.subagent_kind.lock().await.clone()
    }

    /// Set/seed the subagent kind (the router updates it if it learns the kind
    /// after creation, matching `get_conductor`'s `if subagent_kind and not
    /// tc.subagent_kind`).
    pub async fn set_subagent_kind(&self, kind: Option<String>) {
        let mut g = self.subagent_kind.lock().await;
        match kind {
            Some(k) if g.is_none() => *g = Some(k),
            Some(_) => {} // don't clobber an existing kind
            None => {}
        }
    }

    /// `_is_main` — the root thread (no binding needed; its chimera calls default
    /// to MAIN). Identified by the router as the first non-subagent thread.
    async fn is_main(&self) -> bool {
        let tid = self.thread_id();
        self.shim.main_thread_id().await.as_deref() == Some(tid.as_str())
    }

    /// `_needs_bind` — true for any thread that is NOT the root: codex marked it a
    /// subagent, or a different main thread already exists. The eager pre-boot tab
    /// (no main registered yet, not a subagent) is excluded — it becomes the root.
    async fn needs_bind(&self) -> bool {
        if self.subagent_kind.lock().await.is_some() {
            return true;
        }
        match self.shim.main_thread_id().await {
            None => false,
            Some(m) => m != self.thread_id(),
        }
    }

    /// `_first_turn_message` — the text injected once, ahead of this thread's first
    /// task. A non-main thread is prefixed with the binding directive so chimera can
    /// attribute and route its tool calls to THIS conductor.
    ///
    /// Context replay (fix wave 2): codex is stateless — every request carries
    /// the COMPLETE conversation in `input` — but this fresh ChatGPT chat has
    /// seen none of it. On the first injection into a never-used conversation,
    /// when the request carries prior turns beyond the trailing user message
    /// (restart / `codex resume` onto a rebooted bridge), a bounded transcript
    /// of those turns is sandwiched between the preamble and the current
    /// message. Subsequent turns keep the last-message-only behavior (the
    /// ChatGPT side carries the memory from there). Headless background tasks
    /// (memory/compaction) already receive full context in their prompt.
    async fn first_turn_message(&self, body: &Value, prompt: &str) -> String {
        if is_headless_kind(self.subagent_kind.lock().await.as_deref()) {
            // Background task (memory consolidation / compaction): the connector
            // bridge and bind directive would only confuse a tool-less turn.
            return build_headless_message(body, prompt);
        }
        let pre = build_conductor_preamble(body);
        let pre = if self.is_main().await {
            pre
        } else {
            format!("{}{}", fill_thread_bind(&self.thread_id()), pre)
        };
        let replay = crate::responses::build_context_replay(body).unwrap_or_default();
        format!("{pre}{replay}{prompt}")
    }

    /// `_on_ws` — push a tap event onto the per-turn WS queue (best-effort, like
    /// the Python `q.put_nowait` under try/except). Returns a callback bound to a
    /// freshly created queue sender.
    fn make_ws_callback(tx: mpsc::UnboundedSender<(String, String)>) -> WsCallback {
        Arc::new(move |kind: &str, val: &str| {
            // put_nowait equivalent; a closed/full channel is swallowed.
            let _ = tx.send((kind.to_string(), val.to_string()));
        })
    }

    // ----- project scoping -----

    /// Carry codex's per-request MODEL and `reasoning.effort` through to the
    /// ChatGPT backend for THIS turn. codex's `body.model` (the live picker
    /// choice) and `/effort` (minimal/low/medium/high → min/standard/extended/max)
    /// are forced on the tab via the fetch-wrapper override. We rewrite the
    /// override only when one of them actually changes, and rebuild the FULL
    /// override (model + effort + gizmo + no_history) because the adapter's
    /// `set_overrides` REPLACES the whole `__shadow_overrides` object — a
    /// partial write would drop per-folder project scoping or the other axis.
    ///
    /// The model is tracked as the raw codex spec (`gpt-5-5-pro`); the adapter
    /// resolves it to the live account slug. A request with NO `body.model`
    /// keeps the launch model; one with NO `reasoning.effort` reasserts the
    /// LAUNCH default (a previously raised per-turn effort must not stick).
    /// `applied_*` is recorded only AFTER the write succeeds, so a failed write
    /// retries on the next turn instead of silently sticking. The first turn
    /// always writes (applied_model starts `None`), pinning codex's real choice
    /// over whatever boot defaulted to.
    async fn apply_turn_overrides(&self, body: &Value, chat: &Arc<dyn ChatSurface>) {
        // codex's per-request model, else the launch default.
        let desired_model = body_model(body).unwrap_or_else(|| self.model.clone());
        // codex's per-request effort, else the launch default (None = account
        // default: the override write then clears the forced effort axis).
        let desired_effort = body_reasoning_effort(body)
            .or_else(|| crate::provider::resolve_effort(self.effort.as_deref()));
        let model_same = self.applied_model.lock().await.as_deref() == Some(desired_model.as_str());
        let effort_same = *self.applied_effort.lock().await == desired_effort;
        if model_same && effort_same {
            return; // nothing changed -> no useless override write
        }
        let gizmo = self.gizmo.lock().await.clone();
        let no_hist = is_headless_kind(self.subagent_kind.lock().await.as_deref())
            || matches!(
                std::env::var("SHIM_NO_HISTORY").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE")
            );
        if let Err(e) = chat
            .set_overrides(ChatOverrides {
                model: Some(desired_model.clone()),
                thinking_effort: desired_effort.clone(),
                gizmo_id: gizmo,
                no_history: if no_hist { Some(true) } else { None },
            })
            .await
        {
            // applied_* untouched: still differs from desired, so the next turn
            // retries rather than silently sticking.
            tracing::warn!(
                "[shim] failed to apply turn overrides (model={} effort={}): {e}",
                desired_model,
                desired_effort.as_deref().unwrap_or("default")
            );
            return;
        }
        *self.applied_model.lock().await = Some(desired_model.clone());
        *self.applied_effort.lock().await = desired_effort.clone();
        tracing::info!(
            "[shim] thread={} turn model={} effort={} temp_chat={} (codex body.model/reasoning.effort)",
            self.thread_id(),
            desired_model,
            desired_effort.as_deref().unwrap_or("default"),
            // temp_chat=true means this turn ran with history_and_training_disabled
            // (SHIM_NO_HISTORY / headless). Logged so a temp-chat trial can be
            // correlated with whether the repo connector tools still resolve.
            no_hist,
        );
    }

    /// `_ensure_project` — resolve this thread's per-folder ChatGPT Project from
    /// codex's cwd ONCE, and set the gizmo override so the fetch-wrapper scopes
    /// every turn into it (memory off). No-op if already resolved or per-folder
    /// scoping is disabled (`SHIM_PROJECT_PER_FOLDER=0`).
    async fn ensure_project(&self, body: &Value) -> anyhow::Result<()> {
        {
            let mut resolved = self.project_resolved.lock().await;
            if *resolved {
                return Ok(());
            }
            *resolved = true;
        }
        if is_headless_kind(self.subagent_kind.lock().await.as_deref()) {
            // Background tasks (memory consolidation / compaction) run as
            // ephemeral temp-chats OUTSIDE the per-folder project: they need no
            // connector and shouldn't clutter the project's conversation list.
            // Honor codex's per-request model here too (apply_turn_overrides
            // re-asserts it right after, but keep this write consistent).
            if let Some(chat) = self.chat.lock().await.clone() {
                chat.set_overrides(ChatOverrides {
                    model: Some(body_model(body).unwrap_or_else(|| self.model.clone())),
                    thinking_effort: self.effort.clone(),
                    gizmo_id: None,
                    no_history: Some(true),
                })
                .await?;
            }
            return Ok(());
        }
        // We deliberately do NOT auto-create/scope a real ChatGPT Project here.
        // A per-folder gizmo on a FRESH account can make the conversation-create
        // POST 404 ("project not found") — the chat is never born and not a
        // single event streams (the failure that strands new users). Project
        // membership is an ORGANIZATION feature; it must never sit on the
        // critical path of getting a model response. So the conversation is
        // created PLAIN and named `[proj: <repo>]` for grouping once it exists.
        //
        // Escape hatches: an explicit `SHIM_PROJECT_GIZMO` pin (applied at boot)
        // still scopes for opt-in power users; `SHIM_PROJECT_PER_FOLDER=0` turns
        // the naming off entirely.
        let per_folder_off = std::env::var("SHIM_PROJECT_PER_FOLDER")
            .map(|v| v == "0")
            .unwrap_or(false);
        if self.gizmo.lock().await.is_some() || per_folder_off {
            return Ok(());
        }
        let cwd = match extract_cwd(body) {
            Some(c) => c,
            None => return Ok(()),
        };
        *self.proj_title.lock().await =
            Some(format!("[proj: {}]", project_name_for_cwd(&cwd)));
        Ok(())
    }

    /// Best-effort, once: name this thread's conversation `[proj: <repo>]` after
    /// it exists. The conversation is born on the first turn, so this no-ops
    /// until `current_conversation_id` resolves, then renames and latches.
    async fn apply_pending_title(&self) {
        if *self.title_applied.lock().await {
            return;
        }
        let Some(title) = self.proj_title.lock().await.clone() else {
            return;
        };
        let Some(chat) = self.chat.lock().await.clone() else {
            return;
        };
        if let Ok(Some(conv_id)) = chat.current_conversation_id().await {
            match chat.set_conversation_title(&conv_id, &title).await {
                Ok(()) => *self.title_applied.lock().await = true,
                Err(e) => tracing::warn!("[shim] set conversation title failed: {e}"),
            }
        }
    }

    // ----- boot -----

    /// `boot` — open a FRESH chatgpt.com tab for THIS thread and arm the taps, on
    /// the shim's shared browser control socket. A fresh tab has no service worker,
    /// so the turn streams inline through the `/f/conversation` SSE body (which the
    /// FETCH_WRAPPER tees to WSTap) rather than handing off to the CDP-invisible
    /// shared-worker WebSocket.
    pub async fn boot(&self) -> anyhow::Result<()> {
        let _g = self.boot_lock.lock().await;
        if *self.booted.lock().await {
            return Ok(());
        }
        tracing::debug!("[shim] boot thread={}: ensure_tabs", self.thread_id());
        self.shim.ensure_tabs().await?;
        let needs_bind = self.needs_bind().await;
        if needs_bind {
            // Claim pre-existing unbound chimera sessions for "main" BEFORE this tab
            // exists, so elimination binding can never hand the long-lived main
            // session to this subagent.
            self.shim.seed_binds_once().await?;
        }

        let tid = self.thread_id();
        tracing::debug!("[shim] boot thread={tid}: opening tab");
        let target = self
            .shim
            .open_tab("https://chatgpt.com/", Some(&tid), false)
            .await?;
        tracing::debug!("[shim] boot thread={tid}: tab={target}, opening page");
        *self.target.lock().await = Some(target.clone());

        // This thread's OWN page socket + a per-turn WS queue feeding the tap.
        let (ws_tx, ws_rx) = mpsc::unbounded_channel::<(String, String)>();
        let on_ws = Self::make_ws_callback(ws_tx.clone());
        let chat = self.shim.open_page(&target, on_ws).await?;
        *self.ws_tx.lock().await = Some(ws_tx);
        *self.ws_rx.lock().await = Some(ws_rx);
        *self.chat.lock().await = Some(chat.clone());

        // Resolve the model slug; the page layer's open_page already armed the taps
        // and navigated with the FETCH_WRAPPER (arm_and_navigate). We still resolve
        // the slug for the overrides.
        let slug = chat.resolve_slug(&self.model).await.unwrap_or(None);

        // Project scoping (memory off) is applied by the fetch-wrapper. The gizmo is
        // resolved per-FOLDER on the first request (cwd known then) in ensure_project;
        // SHIM_PROJECT_GIZMO pins an explicit one at boot.
        let pinned = std::env::var("SHIM_PROJECT_GIZMO")
            .ok()
            .filter(|s| !s.is_empty());
        *self.gizmo.lock().await = pinned.clone();
        chat.set_overrides(ChatOverrides {
            model: slug.clone(),
            thinking_effort: self.effort.clone(),
            gizmo_id: pinned,
            no_history: None,
        })
        .await?;

        self.force_fresh_if_resumed(slug.as_deref(), &chat).await;

        *self.booted.lock().await = true;
        if needs_bind {
            // Model-free binding: this tab's chimera session binds to codex:<T>.
            self.shim.enqueue_bind(&self.thread_id()).await;
        }
        Ok(())
    }

    /// `_force_fresh_if_resumed` — bare `chatgpt.com/?model=` RESUMES the last
    /// server-side conversation, priming cross-session apply_patch loops. Detect a
    /// resumed thread (a conversation id present right after boot, before any
    /// message) and force a genuinely fresh chat via an in-SPA new-chat click.
    /// No-op when the tab opened fresh. Toggle with `SHIM_FRESH_CHAT=0`.
    async fn force_fresh_if_resumed(&self, slug: Option<&str>, chat: &Arc<dyn ChatSurface>) {
        if std::env::var("SHIM_FRESH_CHAT")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            return;
        }
        let cid = chat.current_conversation_id().await.unwrap_or(None);
        let cid = match cid {
            Some(c) if !c.is_empty() => c,
            _ => {
                tracing::info!(
                    "[shim] boot thread={} conversation_id=NEW (fresh chat)",
                    self.thread_id()
                );
                return;
            }
        };
        tracing::info!(
            "[shim] boot thread={} RESUMED conversation_id={cid} -> forcing new chat",
            self.thread_id()
        );
        // The SPA new-chat click + composer wait + override re-apply are owned by the
 // page layer. The conductor expresses
        // it as: stop -> wait_composer -> re-apply overrides; the page surface performs
        // the actual click inside `stop`/`wait_composer` plumbing. We re-assert the
        // overrides after, matching the Python's post-click set_overrides.
        let _ = chat.stop().await; // best-effort: interrupt any resumed generation
        if chat.wait_composer().await.is_ok() {
            let _ = chat
                .set_overrides(ChatOverrides {
                    model: slug.map(str::to_string),
                    thinking_effort: self.effort.clone(),
                    gizmo_id: None,
                    no_history: None,
                })
                .await;
        }
        let cid2 = chat.current_conversation_id().await.unwrap_or(None);
        tracing::info!(
            "[shim] fresh-chat thread={} conversation_id={}",
            self.thread_id(),
            cid2.as_deref().unwrap_or("NEW")
        );
    }

    /// `ensure_booted` — boot the tab if needed, OR re-boot if the page socket died
    /// (Chrome restarted / tab closed). A cheap liveness probe avoids a dead-socket
    /// "CDP socket not open" error.
    ///
    /// The probe also catches the silent failure the socket check CANNOT: a
    /// LOGGED-OUT page still evals fine, so without an auth check every turn on
    /// it would 90s-stall forever. A detected logout fails fast with a FATAL
    /// codex error telling the user exactly what to do (no auto-login attempt).
    pub async fn ensure_booted(&self) -> anyhow::Result<()> {
        if *self.booted.lock().await {
            // probe liveness via the page surface state() call (cheap eval).
            let chat = self.chat.lock().await.clone();
            if let Some(chat) = chat {
                if let Ok(st) = chat.state().await {
                    self.fail_if_logged_out(&chat, &st).await?;
                    return Ok(()); // socket alive + not on the login wall
                }
            }
        }
        // never booted, or socket dead: tear down stale state + fresh boot.
        *self.booted.lock().await = false;
        *self.protocol_sent.lock().await = false; // fresh tab = fresh conversation
        if let Some(chat) = self.chat.lock().await.take() {
            // best-effort teardown of the dead surface (page layer closes its socket).
            let _ = chat.stop().await;
        }
        *self.chat.lock().await = None;
        *self.ws_tx.lock().await = None;
        *self.ws_rx.lock().await = None;
        self.boot().await?;
        // A fresh boot can land ON the login wall (logged-out Chrome profile):
        // catch it now instead of stalling the first turn.
        let chat = self.chat.lock().await.clone();
        if let Some(chat) = chat {
            if let Ok(st) = chat.state().await {
                self.fail_if_logged_out(&chat, &st).await?;
            }
        }
        Ok(())
    }

    /// Logout detection: only when the page shows NO ready composer and is not
    /// generating (the healthy states) do we spend the `/api/auth/session`
    /// probe. `Ok(false)` — definitively logged out — fails the turn with a
    /// FATAL codex code ("invalid_prompt" stops codex's retry loop) and an
    /// actionable message; a probe error stays conservative (proceed; the
    /// stall watchdog remains the backstop).
    async fn fail_if_logged_out(
        &self,
        chat: &Arc<dyn ChatSurface>,
        st: &Value,
    ) -> anyhow::Result<()> {
        let composer = st
            .get("composerReady")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let generating = st
            .get("generating")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if composer || generating {
            return Ok(());
        }
        if let Ok(false) = chat.is_logged_in().await {
            tracing::warn!(
                "[shim] thread={} ChatGPT tab is LOGGED OUT — failing turn (no auto-login)",
                self.thread_id()
            );
            return Err(anyhow::Error::new(TurnFailed {
                code: "invalid_prompt".to_string(),
                message: "ChatGPT tab is logged out — log in to chatgpt.com in the cyrus \
Chrome window, then retry."
                    .to_string(),
            }));
        }
        Ok(())
    }

    // ----- the simple (non-conductor) path -----

    /// `_build_injection` — decide the text to send to the chat tab for this codex
    /// request. (a) codex just executed a tool -> forward the tool result; (b) a
    /// normal user turn -> the last user message (with the tool-protocol preamble
    /// prepended once per thread, if tools are offered).
    async fn build_injection_simple(&self, body: &Value, tools: &[Value]) -> String {
        if let Some(tool_out) = last_tool_output(body) {
            return format!(
                "TOOL RESULT (the backend executed your tool call and returned this \
real output):\n```\n{}\n```\nContinue: call another tool if needed, or give your \
final answer with no tool block if the task is complete.",
                tool_out.trim()
            );
        }
        let user = extract_prompt(body);
        let mut sent = self.protocol_sent.lock().await;
        if !tools.is_empty() && !*sent {
            *sent = true;
            return format!("{}{}", build_tool_preamble(tools), user);
        }
        user
    }

    /// `_collect_turn` — inject one prompt and buffer the full visible answer (no
    /// streaming to codex yet — we must see the whole turn to know if it's a tool
    /// call). Used by the simple `run_turn` path / `TurnDriver::collect_turn`.
    async fn collect_turn_text(&self, inject_text: &str) -> anyhow::Result<String> {
        // Fresh per-turn WS queue (`self._q = asyncio.Queue(); self.tap.reset()`).
        // The page layer's tap pushes onto ws_tx; we drain ws_rx here.
        let chat = self
            .chat
            .lock()
            .await
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no chat surface booted"))?;

        // Take the receiver for the duration of this turn (single consumer).
        let mut rx = self
            .ws_rx
            .lock()
            .await
            .take()
            .ok_or_else(|| anyhow::anyhow!("no WS queue (tab not booted)"))?;

        let mut acc = String::new();
        // Definitely assigned exactly once, on the loop exit path (each `break`
        // is preceded by its assignment), so no dead initializer and no `mut`.
        let full;
        let mut idle_ticks = 0u32;

        chat.inject(inject_text).await?;

        // No per-turn wall clock: the turn ends on `turn_complete`, a closed WS
        // queue, or the idle_ticks check below (generation stopped) — progress
        // signals, not a timer.
        loop {
            match tokio::time::timeout(Duration::from_secs_f64(1.5), rx.recv()).await {
                Ok(Some((kind, val))) => {
                    idle_ticks = 0;
                    match kind.as_str() {
                        "token" => acc.push_str(&val),
                        "turn_complete" => {
                            full = if val.is_empty() { acc.clone() } else { val };
                            break;
                        }
                        // "thinking" (chain-of-thought) is dropped.
                        _ => {}
                    }
                }
                Ok(None) => {
                    // queue closed (socket gone): end the turn with what we have.
                    full = acc.clone();
                    break;
                }
                Err(_) => {
                    // timeout tick: poll page state for approve / generating.
                    let st = chat.state().await.unwrap_or_else(|_| json!({}));
                    if st
                        .get("hasApprove")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                        && self.cfg.auto_approve
                    {
                        let _ = chat.approve().await;
                        idle_ticks = 0;
                        continue;
                    }
                    let generating = st
                        .get("generating")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    if !generating && !acc.is_empty() {
                        idle_ticks += 1;
                        if idle_ticks >= 2 {
                            full = acc.clone();
                            break;
                        }
                    }
                }
            }
        }

        // Restore the receiver for the next turn.
        *self.ws_rx.lock().await = Some(rx);
        Ok(full)
    }

    /// `run_turn` (simple path) — one codex request -> one chatgpt turn -> either a
    /// function_call (tool) or a message (final answer), as Responses SSE events.
    /// When `SHIM_CONDUCTOR` is set, defers to [`Self::run_turn_conductor`].
    ///
    /// Frames are sent as `data: {json}\n\n` strings on `tx` (the streaming body),
    /// the Rust analogue of writing to aiohttp's `web.StreamResponse`.
    pub async fn run_turn(&self, tx: &mpsc::Sender<String>, body: &Value) -> anyhow::Result<()> {
        if std::env::var("SHIM_CONDUCTOR")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            return self.run_turn_conductor(tx, body).await;
        }

        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let tools: Vec<Value> = body
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let inject_text = self.build_injection_simple(body, &tools).await;

        send_frame(tx, json!({"type": "response.created", "response": {}})).await;

        if inject_text.is_empty() {
            let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            send_frame(
                tx,
                json!({"type": "response.output_item.added", "item": message_item("", Some(&item_id))}),
            )
            .await;
            send_frame(
                tx,
                json!({"type": "response.output_item.done", "item": message_item("", Some(&item_id))}),
            )
            .await;
            send_frame(tx, completed(&response_id)).await;
            return Ok(());
        }

        let full = {
            let _g = self.turn_lock.lock().await;
            self.collect_turn_text(&inject_text).await?
        };

        let call = if !tools.is_empty() {
            parse_tool_call(&full)
        } else {
            None
        };
        if let Some(call) = call {
            // Tool call: a single function_call output item triggers codex to run it.
            send_frame(
                tx,
                json!({
                    "type": "response.output_item.done",
                    "item": function_call_item(&call.name, &call.arguments(), None),
                }),
            )
            .await;
            send_frame(tx, completed(&response_id)).await;
        } else {
            // Final answer: open the message item, stream the buffered text, done.
            let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            send_frame(
                tx,
                json!({"type": "response.output_item.added", "item": message_item("", Some(&item_id))}),
            )
            .await;
            if !full.is_empty() {
                send_frame(
                    tx,
                    json!({"type": "response.output_text.delta", "delta": full}),
                )
                .await;
            }
            send_frame(
                tx,
                json!({"type": "response.output_item.done", "item": message_item(&full, Some(&item_id))}),
            )
            .await;
            send_frame(tx, completed(&response_id)).await;
        }
        Ok(())
    }

    // ----- the conductor: one ChatGPT turn onto N codex request/response cycles -----

    /// Ensure the merged event stream exists (`if self._events is None: ... Queue()`).
    /// Returns a clone of the sender for producers (the WS watcher, chimera relay).
    async fn ensure_events(&self) -> mpsc::UnboundedSender<Item> {
        let mut tx_guard = self.events_tx.lock().await;
        if tx_guard.is_none() {
            let (tx, rx) = mpsc::unbounded_channel::<Item>();
            *tx_guard = Some(tx);
            *self.events_rx.lock().await = Some(rx);
        }
        tx_guard.as_ref().expect("events tx just set").clone()
    }

    /// `_watch_ws_turn` — drain ChatGPT's token stream for THIS turn. Each visible
    /// token is forwarded onto the merged event queue the moment it arrives (so
    /// codex streams it live) AND accumulated into `turn_text`; on `turn_complete`
    /// we resolve `turn_done` with the final text. A mid-turn MCP tool call is
    /// invisible here (it goes through OpenAI's connector infra) — only chimera
    /// signals those, via the events queue.
    ///
    /// Spawned as a background task; takes ownership of the WS receiver, the events
    /// sender, the turn-done sender, and shared turn_text state.
    /// On turn end it restores the WS receiver into the shared `ws_rx` slot so the
    /// next turn can re-wire it.
    fn spawn_ws_watcher(
        &self,
        mut ws_rx: mpsc::UnboundedReceiver<(String, String)>,
        events_tx: mpsc::UnboundedSender<Item>,
        turn_done: oneshot::Sender<String>,
    ) {
        let turn_text = Arc::clone(&self.turn_text);
        let ws_rx_slot = Arc::clone(&self.ws_rx);
        tokio::spawn(async move {
            let mut turn_done = Some(turn_done);
            while let Some((kind, val)) = ws_rx.recv().await {
                match kind.as_str() {
                    "token" => {
                        {
                            let mut t = turn_text.lock().await;
                            t.push_str(&val);
                        }
                        let _ = events_tx.send(Item::Token(val));
                    }
                    "turn_complete" => {
                        let full = if val.is_empty() {
                            turn_text.lock().await.clone()
                        } else {
                            val
                        };
                        if let Some(td) = turn_done.take() {
                            let _ = td.send(full);
                        }
                        break;
                    }
                    // The tap detected a typed error event (rate-limit / moderation /
                    // server error) — classify and fail the turn NOW instead of
                    // waiting out the stall watchdog. The turn is dead: break so the
                    // WS receiver is restored for the retry.
                    "error" => {
                        let v: Value = serde_json::from_str(&val).unwrap_or_else(|_| json!({}));
                        let pick =
                            |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                        let tf =
                            classify_stream_error(&pick("etype"), &pick("code"), &pick("message"));
                        let _ = events_tx.send(Item::Error(tf));
                        break;
                    }
                    // "thinking" (and any other non-terminal tap activity): forward a
                    // keepalive so the turn's stall watchdog sees ChatGPT is alive and
                    // won't abort during a long reasoning pass. No visible text.
                    _ => {
                        let _ = events_tx.send(Item::Keepalive);
                    }
                }
            }
            // Restore the WS receiver for the next turn.
            *ws_rx_slot.lock().await = Some(ws_rx);
        });
    }

    /// Spawn this turn's chimera `/events` liveness tail, replacing (and
    /// aborting) any stale tail from a previous turn. Each event the tail sees
    /// — whatever its kind — sends one [`Item::Keepalive`] onto the merged
    /// queue, resetting the stall watchdog: a turn that streams no tokens
    /// because ChatGPT is blocked on a long connector (MCP) tool is healthy,
    /// not rate-limited, and must not be aborted. When the shim has no feed
    /// (tests) or chimera is down, there is no tail and behavior degrades to
    /// the WS-only watchdog.
    ///
    /// Scoped to THIS conversation's chimera agent id (chimera stamps every
    /// event with the agent the session bound via `repo_register`): "main" for
    /// the root conductor, "codex:<thread>" for bound sub-threads — the same
    /// identity [`fill_thread_bind`] / the elimination-bind rail register.
    /// Unfiltered, ANY session's tool traffic would keep a dead turn alive.
    async fn spawn_server_tail(&self, events_tx: &mpsc::UnboundedSender<Item>) {
        let agent = if self.is_main().await {
            "main".to_string()
        } else {
            format!("codex:{}", self.thread_id())
        };
        let keep = events_tx.clone();
        let tail = self.shim.spawn_server_events_tail(
            &agent,
            Arc::new(move || {
                let _ = keep.send(Item::Keepalive);
            }),
        );
        let mut slot = self.server_tail.lock().await;
        if let Some(old) = std::mem::replace(&mut *slot, tail) {
            old.abort();
        }
    }

    /// Abort the `/events` liveness tail at turn end. The WS watcher needs no
    /// such call (its loop breaks on `turn_complete`); the SSE tail is unbounded
    /// so every turn-end path must abort it explicitly.
    async fn abort_server_tail(&self) {
        if let Some(tail) = self.server_tail.lock().await.take() {
            tail.abort();
        }
    }

    /// `_stream_turn` — consume one codex request's slice of the ChatGPT turn,
    /// emitting Responses SSE up to (but not including) `response.completed` — the
    /// caller emits that.
    ///
    /// The message item is opened LAZILY on the first token, so a slice that is a
    /// pure tool call emits a clean function_call with no empty message item. Final
    /// answer tokens stream as `response.output_text.delta` as they arrive. The
    /// slice ends when a chimera tool call is dispatched (close any open message,
    /// emit the call) or the turn completes (flush the remainder, close the message).
    async fn stream_turn(&self, tx: &mpsc::Sender<String>) -> anyhow::Result<()> {
        let _events_tx = self.ensure_events().await; // ensure the queue exists
        let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let mut acc = String::new();
        let mut open = false;

        // Take the events receiver + the turn-done receiver for this slice.
        let mut events_rx = self
            .events_rx
            .lock()
            .await
            .take()
            .ok_or_else(|| anyhow::anyhow!("no events queue"))?;
        let mut turn_done_rx = self.turn_done_rx.lock().await.take();

        // No wall clock: runs while the page reports it's generating.
        // Opt-in SHIM_TURN_STALL_SECS caps silence.
        let stall =
            explicit_stall_secs(std::env::var("SHIM_TURN_STALL_SECS").ok().as_deref())
                .map(Duration::from_secs);
        // Watchdog cadence: poll every 15s, but never coarser than the stall
        // budget — otherwise a short opt-in `SHIM_TURN_STALL_SECS` could never
        // fire before the next poll tick.
        let poll = match stall {
            Some(cap) => Duration::from_secs(15).min(cap),
            None => Duration::from_secs(15),
        };
        let mut silent = Duration::ZERO;
        let result: anyhow::Result<()> = async {
            loop {
                tokio::select! {
                    biased;
                    ev = events_rx.recv() => {
                        silent = Duration::ZERO; // any tap activity resets the watchdog
                        match ev {
                            Some(Item::Token(payload)) => {
                                if !open {
                                    send_frame(tx, json!({
                                        "type": "response.output_item.added",
                                        "item": message_item("", Some(&item_id)),
                                    })).await;
                                    open = true;
                                }
                                acc.push_str(&payload);
                                send_frame(tx, json!({
                                    "type": "response.output_text.delta", "delta": payload,
                                })).await;
                            }
                            // Liveness ping only — resets the watchdog, emits nothing.
                            Some(Item::Keepalive) => {}
                            // ChatGPT stream error detected by the tap: fail the turn
                            // IMMEDIATELY with the classified codex-facing code (the
                            // caller maps it onto response.failed) instead of letting
                            // the stall watchdog burn its full budget.
                            Some(Item::Error(err)) => {
                                tracing::warn!(
                                    "[shim] thread={} ChatGPT stream error (code={}): {} — failing turn",
                                    self.thread_id(),
                                    err.code,
                                    err.message,
                                );
                                self.abort_server_tail().await;
                                if let Some(chat) = self.chat.lock().await.clone() {
                                    let _ = chat.stop().await;
                                }
                                return Err(anyhow::Error::new(err));
                            }
                            Some(Item::Call(call)) => {
                                // Close the commentary message first (if ChatGPT streamed
                                // text before reaching for the tool) so codex renders the
                                // message, then the native tool card.
                                if open {
                                    send_frame(tx, json!({
                                        "type": "response.output_item.done",
                                        "item": message_item(&acc, Some(&item_id)),
                                    })).await;
                                }
                                // Park the call's future so codex's tool-result turn can
                                // resolve it.
                                *self.inflight_call.lock().await = Some(call.future);
                                let item = if call.kind == "custom" {
                                    custom_tool_call_item(&call.name, &call.input, Some(&call.call_id))
                                } else {
                                    function_call_item(&call.name, &call.arguments, Some(&call.call_id))
                                };
                                send_frame(tx, json!({
                                    "type": "response.output_item.done", "item": item,
                                })).await;
                                return Ok(());
                            }
                            None => {
                                // events queue closed: treat as turn end with no more text.
                                self.abort_server_tail().await;
                                self.finish_message(tx, &item_id, &mut acc, open, "").await;
                                return Ok(());
                            }
                        }
                    }
                    done = recv_opt(&mut turn_done_rx) => {
                        // turn completed: stop the /events liveness tail, then drain any
                        // tokens buffered just before completion, in order, before closing.
                        self.abort_server_tail().await;
                        let full = done.unwrap_or_default();
                        while let Ok(item) = events_rx.try_recv() {
                            if let Item::Token(v) = item {
                                if !open {
                                    send_frame(tx, json!({
                                        "type": "response.output_item.added",
                                        "item": message_item("", Some(&item_id)),
                                    })).await;
                                    open = true;
                                }
                                acc.push_str(&v);
                                send_frame(tx, json!({
                                    "type": "response.output_text.delta", "delta": v,
                                })).await;
                            }
                            // a stray "call" after turn-done is anomalous; the turn is ending.
                        }
                        self.finish_message(tx, &item_id, &mut acc, open, &full).await;
                        return Ok(());
                    }
                    _ = tokio::time::sleep(poll) => {
                        // No tap activity for `poll`. Before treating it as a stall,
                        // ask the page whether ChatGPT is even still generating.
                        let stopped = match self.chat.lock().await.clone() {
                            Some(chat) => match chat.state().await {
                                Ok(st) => {
                                    let generating = st.get("generating")
                                        .and_then(Value::as_bool).unwrap_or(true);
                                    let ready = st.get("composerReady")
                                        .and_then(Value::as_bool).unwrap_or(false);
                                    // Turn ended iff generation halted AND the composer
                                    // re-enabled (it's disabled for the whole generation).
                                    !generating && ready
                                }
                                // Can't read the page -> assume it's still working.
                                Err(_) => false,
                            },
                            None => false,
                        };
                        if stopped {
                            // The turn is DONE but the tap never forwarded a
                            // completion (shared-worker handoff): close cleanly with
                            // what we streamed instead of burning the budget and
                            // forcing codex into a stall-retry loop.
                            tracing::info!(
                                "[shim] thread={} ChatGPT stopped generating but the tap missed the completion — closing turn cleanly ({} streamed chars)",
                                self.thread_id(),
                                acc.len(),
                            );
                            self.abort_server_tail().await;
                            self.finish_message(tx, &item_id, &mut acc, open, "").await;
                            return Ok(());
                        }
                        silent += poll;
                        match stall {
                            Some(cap) if silent >= cap => {
                                self.abort_server_tail().await;
                                if let Some(chat) = self.chat.lock().await.clone() {
                                    let _ = chat.stop().await;
                                }
                                return Err(anyhow::anyhow!(
                                    "ChatGPT produced no output for {}s (no token stream and no \
connector-tool activity, SHIM_TURN_STALL_SECS) — likely rate-limited or stalled; turn aborted, retry shortly",
                                    silent.as_secs()
                                ));
                            }
                            _ => continue,
                        }
                    }
                }
            }
        }
        .await;

        // Restore the receivers for the next slice.
        *self.events_rx.lock().await = Some(events_rx);
        *self.turn_done_rx.lock().await = turn_done_rx;
        result
    }

    /// `_finish_message` — close out a final-answer slice's message item. If nothing
    /// streamed this slice (an empty/moderation turn), emit the final text as one
    /// delta. If tokens did stream, emit only a trailing remainder when the final
    /// text cleanly extends what we streamed (a backstop; normally full == acc).
    async fn finish_message(
        &self,
        tx: &mpsc::Sender<String>,
        item_id: &str,
        acc: &mut String,
        open: bool,
        full: &str,
    ) {
        if !open {
            send_frame(
                tx,
                json!({"type": "response.output_item.added", "item": message_item("", Some(item_id))}),
            )
            .await;
            if !full.is_empty() {
                send_frame(
                    tx,
                    json!({"type": "response.output_text.delta", "delta": full}),
                )
                .await;
            }
            send_frame(
                tx,
                json!({"type": "response.output_item.done", "item": message_item(full, Some(item_id))}),
            )
            .await;
            return;
        }
        if !full.is_empty() && full.starts_with(acc.as_str()) && full.len() > acc.len() {
            let tail = full[acc.len()..].to_string();
            *acc = full.to_string();
            send_frame(
                tx,
                json!({"type": "response.output_text.delta", "delta": tail}),
            )
            .await;
        }
        send_frame(
            tx,
            json!({"type": "response.output_item.done", "item": message_item(acc, Some(item_id))}),
        )
        .await;
    }

    /// `run_turn_conductor` — the conductor path: maps one ChatGPT turn onto N codex
    /// request/response cycles.
    pub async fn run_turn_conductor(
        &self,
        tx: &mpsc::Sender<String>,
        body: &Value,
    ) -> anyhow::Result<()> {
        // Serialize whole requests on this thread (the same `turn_lock` the
        // non-conductor path holds): two concurrent codex requests for one
        // thread-id would otherwise interleave on turn_done_tx / ws_rx /
        // events_rx / inflight_call / turn_text and corrupt the turn. Safe
        // against /control/toolcall: that path never takes this lock — it only
        // pushes onto the events queue, and its parked future is resolved by
        // the NEXT request, which acquires the lock only after the dispatching
        // slice returned (releasing it).
        let _turn = self.turn_lock.lock().await;
        self.ensure_events().await;
        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        send_frame(tx, json!({"type": "response.created", "response": {}})).await;

        let tool_out = last_tool_output(body);
        let have_inflight = self.inflight_call.lock().await.is_some();

        if tool_out.is_some() && have_inflight {
            // codex finished the tool we dispatched — hand the result to the parked
            // chimera call so ChatGPT resumes its turn.
            if let Some(fut) = self.inflight_call.lock().await.take() {
                let _ = fut.send(tool_out.unwrap_or_default());
            }
        } else if tool_out.is_none() {
            // fresh ChatGPT turn: open a new turn-done future and drive ChatGPT.
            let (td_tx, td_rx) = oneshot::channel::<String>();
            *self.turn_done_tx.lock().await = Some(td_tx);
            *self.turn_done_rx.lock().await = Some(td_rx);
            *self.turn_text.lock().await = String::new();
            self.recent_calls.lock().await.clear();

            let prompt = extract_prompt(body);
            let chat = self.chat.lock().await.clone();
            if let Some(chat) = chat {
                // Fresh per-turn WS queue is already wired (boot created ws_tx/ws_rx);
                // the page layer's tap.reset() is invoked by the page surface on inject.
                let msg = {
                    let mut sent = self.protocol_sent.lock().await;
                    if !*sent {
                        let m = self.first_turn_message(body, &prompt).await;
                        *sent = true;
                        m
                    } else {
                        prompt.clone()
                    }
                };
                // Resolve this folder's memory-off Project BEFORE the first inject.
                self.ensure_project(body).await?;
                // Carry codex's picker model + reasoning effort through per turn.
                self.apply_turn_overrides(body, &chat).await;
                chat.inject(&msg).await?;

                // Spawn the WS watcher: it streams tokens onto the merged queue,
                // resolves turn_done on completion, and restores the WS receiver into
                // the shared slot for the next turn.
                let ws_rx = self.ws_rx.lock().await.take();
                let events_tx = self.ensure_events().await;
                let td_tx = self.turn_done_tx.lock().await.take();
                if let (Some(ws_rx), Some(td_tx)) = (ws_rx, td_tx) {
                    self.spawn_ws_watcher(ws_rx, events_tx, td_tx);
                }
            }
            // (no browser) test mode: a simulated chimera drives via /control/*.

            // Alongside the WS watcher: tail chimera's /events feed for the
            // duration of THIS turn. A ChatGPT-native connector (MCP) tool call
            // is invisible to the WS tap — the model streams nothing while the
            // tool runs — so without this tail the stall watchdog cannot tell
            // "healthy turn blocked on a slow tool" from "rate-limited dead".
            // The tail lives across codex request slices (a chimera-relayed
            // Call ends the slice, not the turn) and is aborted at turn end.
            let events_tx = self.ensure_events().await;
            self.spawn_server_tail(&events_tx).await;
        }

        // If somehow no turn-done future is set (e.g. an inflight-result turn with no
        // prior open future), create a resolved-on-complete one so stream_turn can
        // wait deterministically.
        if self.turn_done_rx.lock().await.is_none() {
            let (td_tx, td_rx) = oneshot::channel::<String>();
            *self.turn_done_tx.lock().await = Some(td_tx);
            *self.turn_done_rx.lock().await = Some(td_rx);
        }

        if let Err(e) = self.stream_turn(tx).await {
            // Aborted turn (stall watchdog / stream error): reset ALL per-turn
            // state so the retry starts clean — a surviving inflight_call would
            // fire a stale future on the next tool-result request, and stale
            // recent_calls/turn buffers would corrupt loop-recovery dedupe.
            self.reset_turn_state().await;
            return Err(e);
        }
        // The conversation exists now (created on the first inject); name it
        // `[proj: <repo>]` if pending. Best-effort, latches after the first hit.
        self.apply_pending_title().await;
        send_frame(tx, completed(&response_id)).await;
        Ok(())
    }

    /// Clear ALL per-turn state after an aborted turn so the next request
    /// starts clean (stall abort, stream error, or any stream_turn failure).
    async fn reset_turn_state(&self) {
        *self.inflight_call.lock().await = None;
        self.recent_calls.lock().await.clear();
        self.turn_text.lock().await.clear();
        *self.turn_done_tx.lock().await = None;
        *self.turn_done_rx.lock().await = None;
        // Drain any stale items (late tokens/keepalives from the dead turn) so
        // they can't leak into the retry's output stream.
        if let Some(rx) = self.events_rx.lock().await.as_mut() {
            while rx.try_recv().is_ok() {}
        }
        self.abort_server_tail().await;
    }

    /// Drop the parked tool-call future. Called by the `/control/toolcall`
    /// timeout path: once chimera's held request 504s, resolving the stale
    /// future later (or leaving it armed for an unrelated tool-result turn)
    /// can only desync — the next turn must start clean.
    pub async fn clear_inflight_call(&self) {
        *self.inflight_call.lock().await = None;
    }

    /// Test-only probe: whether a dispatched tool call is parked.
    #[cfg(test)]
    pub(crate) async fn has_inflight_call(&self) -> bool {
        self.inflight_call.lock().await.is_some()
    }

    // ----- moderation-loop recovery entry point (chimera /control/toolcall) -----

    /// Dispatch a chimera tool call onto the merged stream and (the caller) block
    /// until codex executes it. Implements the `/control/toolcall` body of the
    /// Python `build_app`, but scoped to THIS conductor: signature dedupe, soft
    /// re-delivery from cache (NEVER re-executing), and the hard out-of-band nudge.
    ///
    /// Returns the `/control/toolcall` JSON response body. The held request that
    /// awaits codex's execution is modelled by the returned `oneshot::Receiver`
    /// inside [`ControlToolcall::Dispatched`]; the HTTP layer awaits it (with the
    /// blocking-tool timeout policy) and then calls [`Self::record_tool_result`].
    pub async fn control_toolcall(
        &self,
        name: &str,
        kind: &str,
        arguments: &Value,
        input: &str,
        call_id: &str,
    ) -> ControlToolcall {
        // Moderation-loop recovery: signature = name :: sorted-json(args) :: input.
        let sig = format!("{name}::{}::{input}", canonical_json(arguments));

        // A repo mutation invalidates every cached EXECUTION result: after an edit,
        // the same `cargo build`/test/shell command must RE-RUN against the new
        // code, not be deduped or hard-stopped as a "repeat" against a stale
        // pre-edit result. (This severed the build/verify loop on a real task —
        // edit -> rebuild was hard-stopped at count==2 and the model concluded it
        // had lost tool access and gave up.) Clearing also resets the loop counters,
        // which is correct: progress was made, so the anti-loop window restarts.
        if matches!(name, "apply_patch" | "repo_write" | "repo_edit") {
            self.recent_calls.lock().await.clear();
        }

        // `update_plan` is a planning tool: the model legitimately revises its TODO
        // many times per task and an identical re-post is harmless — it is NOT a
        // stuck-loop signal, so it is never deduped or hard-stopped.
        let dedup = name != "update_plan";

        if dedup {
            let mut recent = self.recent_calls.lock().await;
            if let Some(prev) = recent.get_mut(&sig) {
                if prev.result.is_some() {
                    prev.count += 1;
                    let count = prev.count;
                    let threshold = self.cfg.loop_repeat_threshold;
                    let cached = prev.result.clone().unwrap_or_default();
                    drop(recent);
                    // Re-deliver the cached result through the TOOL RESPONSE rather
                    // than interrupting the model's generation. The old path escalated
                    // to chat.stop() at the threshold, which yanked the model out of
                    // its turn mid-thought and re-prompted it with "[loop broken]"; the
                    // model then emitted a status summary and the codex-exec run ended
                    // — that is what stranded a task one fix away from green. Returning
                    // the output keeps the model IN its turn: a genuine moderation loop
                    // finally sees the result it missed, and a legitimate re-run or
                    // liveness poll just gets nudged to use what it already has. The
                    // wording firms up with the repeat count, but generation is never
                    // interrupted. An edit clears this cache (above), so a real
                    // edit -> rebuild always runs fresh instead of getting a stale hit.
                    let output = if count >= threshold {
                        format!(
                            "[You have called this exact command {count} times and the result \
                             is not changing. STOP re-running it — its output is below; use that \
                             and move on to the next step.]\n{cached}"
                        )
                    } else {
                        format!(
                            "[This tool call already ran — do NOT call it again. Its real output \
                             is below; use it and continue.]\n{cached}"
                        )
                    };
                    return ControlToolcall::Recovered(json!({
                        "output": output,
                        "call_id": call_id,
                        "recovered": true,
                    }));
                }
            }
        }

        // Fresh call: enqueue it on the merged stream and hand the caller a future
        // to await codex's execution.
        let events_tx = self.ensure_events().await;
        let (fut_tx, fut_rx) = oneshot::channel::<String>();
        let call = ToolCallEvent {
            name: name.to_string(),
            kind: if kind.is_empty() {
                "function".to_string()
            } else {
                kind.to_string()
            },
            arguments: arguments.clone(),
            input: input.to_string(),
            call_id: call_id.to_string(),
            future: fut_tx,
        };
        let _ = events_tx.send(Item::Call(call));

        ControlToolcall::Dispatched {
            sig,
            call_id: call_id.to_string(),
            result_rx: fut_rx,
            // Most tools execute in seconds; the blockers (request_user_input,
            // wait_agent, followup_task, send_message) legitimately park up to an hour.
            timeout_secs: if is_blocking_tool(name) { 3600 } else { 300 },
        }
    }

    /// Cache a fresh tool result against its signature (after codex returned it).
    /// Mirrors `tc._recent_calls[sig] = {"count": 1, "result": out}`.
    pub async fn record_tool_result(&self, sig: &str, out: &str) {
        self.recent_calls.lock().await.insert(
            sig.to_string(),
            RecentCall {
                count: 1,
                result: Some(out.to_string()),
            },
        );
    }

    /// `/control/turn_complete`: ChatGPT finished its turn with final text — end the
    /// codex turn by resolving the turn-done future. Returns true if a turn was
    /// active.
    pub async fn control_turn_complete(&self, text: &str) -> bool {
        if let Some(td) = self.turn_done_tx.lock().await.take() {
            let _ = td.send(text.to_string());
            true
        } else {
            false
        }
    }

    /// `close` — tear down this thread's tab. The shared TabFactory is closed by the
    /// router, not here.
    pub async fn close(&self) {
        // Don't leak a live /events SSE connection past the conductor's life.
        self.abort_server_tail().await;
        if let Some(chat) = self.chat.lock().await.take() {
            let _ = chat.stop().await; // page layer closes its own socket on drop
        }
        if let Some(target) = self.target.lock().await.take() {
            self.shim.close_tab(&target).await;
        }
    }
}

/// The outcome of [`ThreadConductor::control_toolcall`].
pub enum ControlToolcall {
    /// A soft re-delivery / hard-recovery response — return as the HTTP body
    /// directly (the call was NOT re-executed).
    Recovered(Value),
    /// A fresh call was enqueued; the HTTP layer awaits `result_rx` (bounded by
    /// `timeout_secs`), then calls [`ThreadConductor::record_tool_result`] with
    /// `sig` and the output and returns `{"output", "call_id"}`.
    Dispatched {
        sig: String,
        call_id: String,
        result_rx: oneshot::Receiver<String>,
        timeout_secs: u64,
    },
}

// ===== TurnDriver impl (router integration) ==================================

/// The conductor satisfies [`TurnDriver`] (the simple `collect_turn` contract the
/// responses.rs router calls). The full conductor path runs through
/// [`ThreadConductor::run_turn_conductor`]; `collect_turn` covers the
/// non-conductor `run_turn` buffered path.
#[async_trait]
impl TurnDriver for ThreadConductor {
    async fn collect_turn(
        &self,
        _thread_id: Option<&str>,
        subagent_kind: Option<&str>,
        _body: &Value,
        inject_text: &str,
    ) -> anyhow::Result<String> {
        if let Some(kind) = subagent_kind {
            self.set_subagent_kind(Some(kind.to_string())).await;
        }
        self.ensure_booted().await?;
        let _g = self.turn_lock.lock().await;
        self.collect_turn_text(inject_text).await
    }

    fn build_injection(&self, _thread_id: Option<&str>, body: &Value, tools: &[Value]) -> String {
        // The trait method is sync; the conductor's per-thread protocol_sent flag is
        // behind an async Mutex. Use a blocking lock acquisition via try_lock, falling
        // back to the stateless default (which is correct on the common first-turn
        // path). The async run_turn paths use the async `build_injection` instead.
        let mut sent = self.protocol_sent.try_lock().map(|g| *g).unwrap_or(false);
        let out = crate::responses::default_build_injection(body, tools, &mut sent);
        if let Ok(mut g) = self.protocol_sent.try_lock() {
            *g = sent;
        }
        out
    }
}

// ===== free helpers ==========================================================

/// An EXPLICIT `SHIM_TURN_STALL_SECS` override, if a valid positive integer.
/// Missing / empty / non-numeric / `0` -> `None` (the caller then uses the
/// model+effort-aware default budget; `0` does NOT disable the watchdog, which
/// would reintroduce the infinite-hang-on-rate-limit). A valid override is
/// honored EXACTLY — the long-think floor never raises it — so tests and power
/// users can pin a precise budget.
fn explicit_stall_secs(raw: Option<&str>) -> Option<u64> {
    raw.and_then(|v| v.trim().parse::<u64>().ok()).filter(|n| *n > 0)
}

/// Send one SSE frame on the streaming-body channel (`_sse` analogue). A closed
/// receiver is swallowed, matching aiohttp best-effort writes after the client
/// disconnects.
async fn send_frame(tx: &mpsc::Sender<String>, frame: Value) {
    let _ = tx.send(sse_frame(&frame)).await;
}

/// Resolve `Option<oneshot::Receiver<String>>` to the received value (or `None`
/// if the sender dropped). When the slot is `None` — no active turn-done future —
/// this NEVER resolves, so the `events_rx` branch of the select drives the loop
/// (the test-mode path where `/control/turn_complete` resolves the future).
///
/// `oneshot::Receiver` is `Unpin`, so it can be polled through a `&mut` without
/// pinning to the stack; `poll_fn` makes the borrow explicit and re-pollable
/// across select iterations WHILE PENDING. Once it resolves, the slot is set to
/// `None` (a oneshot panics if polled after completion, and stream_turn restores
/// this slot for the next turn — so a consumed receiver must not survive).
async fn recv_opt(rx: &mut Option<oneshot::Receiver<String>>) -> Option<String> {
    use std::future::poll_fn;
    use std::future::Future;
    use std::task::Poll;
    match rx {
        Some(r) => {
            let out = poll_fn(|cx| match std::pin::Pin::new(&mut *r).poll(cx) {
                Poll::Ready(res) => Poll::Ready(res.ok()),
                Poll::Pending => Poll::Pending,
            })
            .await;
            // The receiver has now yielded its single value. A oneshot::Receiver
            // panics ("called after complete") if polled again, and stream_turn
            // RESTORES this Option back into self.turn_done_rx after the loop — so a
            // turn that ends via the done-branch would otherwise stash a *consumed*
            // receiver for the next turn to re-poll. Drop it: the restore then puts
            // `None` back, and run_turn_conductor opens a fresh channel next turn.
            *rx = None;
            out
        }
        None => std::future::pending().await,
    }
}

/// Canonical JSON for a value with object keys SORTED (the Python
/// `json.dumps(args, sort_keys=True, ensure_ascii=False)`), so the moderation
/// signature is stable regardless of key order.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = String::from("{");
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                // Python json.dumps default separators are (", ", ": ").
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push_str(": ");
                out.push_str(&canonical_json(&map[*k]));
            }
            out.push('}');
            out
        }
        Value::Array(items) => {
            let mut out = String::from("[");
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&canonical_json(it));
            }
            out.push(']');
            out
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Classify a tap-detected stream error onto codex's error-code contract:
///   - moderation / hard-refusal shapes -> `"invalid_prompt"` (FATAL: codex
///     stops retrying and shows the message — retrying a refusal only loops);
///   - everything else (rate limits, server errors, unknowns) ->
///     `"shim_error"` (RETRYABLE: codex retries up to 5x and parses any
///     retry-after hint out of the message, which is passed through verbatim).
/// Conservative: when unsure, retryable — a false FATAL is worse than slow.
fn classify_stream_error(etype: &str, code: &str, message: &str) -> TurnFailed {
    let hay = format!("{etype} {code} {message}").to_ascii_lowercase();
    let moderation = ["moderation", "content_policy", "content policy", "policy violation", "unsafe content"]
        .iter()
        .any(|k| hay.contains(k));
    let detail = if message.is_empty() {
        if code.is_empty() { etype } else { code }
    } else {
        message
    };
    if moderation {
        TurnFailed {
            code: "invalid_prompt".to_string(),
            message: format!("ChatGPT refused the turn (moderation): {detail}"),
        }
    } else {
        TurnFailed {
            code: "shim_error".to_string(),
            message: format!(
                "ChatGPT stream reported an error (type={etype}, code={code}): {detail}"
            ),
        }
    }
}

/// Tools that legitimately BLOCK (request_user_input waits on a human; wait_agent
/// parks up to its own timeout) — give them the hour-long timeout. Mirrors the
/// `_BLOCKING` set in the Python `control_toolcall`.
fn is_blocking_tool(name: &str) -> bool {
    matches!(
        name,
        "request_user_input" | "wait_agent" | "followup_task" | "send_message"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_name_uses_leaf_folder() {
        assert_eq!(
            project_name_for_cwd("C:\\Users\\alice\\repo"),
            "codex: repo"
        );
        assert_eq!(project_name_for_cwd("/home/x/proj/"), "codex: proj");
        assert_eq!(project_name_for_cwd("/home/x/proj//"), "codex: proj");
        // a bare path with no separators is its own leaf
        assert_eq!(project_name_for_cwd("solo"), "codex: solo");
    }

    #[test]
    fn extract_cwd_from_environment_context() {
        let body = json!({
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "content": [
                    {"type": "input_text", "text": "<environment_context>\n  <cwd>C:\\proj\\app</cwd>\n</environment_context>"}
                ]}
            ]
        });
        assert_eq!(extract_cwd(&body).as_deref(), Some("C:\\proj\\app"));
    }

    #[test]
    fn extract_cwd_handles_text_field_and_case() {
        // The bare `text` field is read directly, and the inner value is trimmed
        // of surrounding whitespace. The opening tag MUST be lowercase `<cwd>`:
        // the Python original gates on `"<cwd>" in txt` (case-SENSITIVE) before
        // ever running its case-insensitive regex, so a lowercase open tag is
        // required even though the close tag may differ in case.
        let body = json!({
            "input": [
                {"text": "junk <cwd>  /repo/here  </CWD> tail"}
            ]
        });
        assert_eq!(extract_cwd(&body).as_deref(), Some("/repo/here"));

        // ORIGINAL-SIDE BUG (matched deliberately for byte/behavioral fidelity):
        // a fully UPPERCASE `<CWD>...</CWD>` yields None, because the Python gate
        // `"<cwd>" in txt` is case-sensitive and never lets the `re.I` regex run.
        // The port reproduces this exactly rather than "fixing" it.
        let upper = json!({
            "input": [
                {"text": "junk <CWD>  /repo/here  </CWD> tail"}
            ]
        });
        assert_eq!(extract_cwd(&upper).as_deref(), None);
    }

    #[test]
    fn extract_cwd_none_when_absent_or_empty() {
        assert!(extract_cwd(&json!({"input": [{"text": "no tag"}]})).is_none());
        assert!(extract_cwd(&json!({"input": [{"text": "<cwd>   </cwd>"}]})).is_none());
        assert!(extract_cwd(&json!({"input": "not a list"})).is_none());
    }

    #[test]
    fn conductor_preamble_default_forwards_instructions() {
        // SHIM_FORWARD_CODEX_PROMPT unset (default on) + non-empty instructions ->
        // bridge + instructions + task sep.
        let body = json!({"instructions": "  Be a good codex.  "});
        // Only assert structure that does not depend on process env toggles being
        // unset in the test runner: when forwarding, the bridge prefix appears.
        let out = build_conductor_preamble(&body);
        if std::env::var("SHIM_FORWARD_CODEX_PROMPT")
            .map(|v| v != "0")
            .unwrap_or(true)
        {
            assert!(out.starts_with(CONDUCTOR_BRIDGE));
            assert!(out.contains("Be a good codex."));
            assert!(out.contains(shell_rules()));
            assert!(out.ends_with("# Task\n"));
        } else {
            assert!(out.starts_with(CONDUCTOR_PREAMBLE_HEAD));
            assert!(out.contains(shell_rules()));
            assert!(out.ends_with(CONDUCTOR_PREAMBLE_TAIL));
        }
    }

    #[test]
    fn conductor_preamble_falls_back_without_instructions() {
        let body = json!({});
        let out = build_conductor_preamble(&body);
        assert!(out.starts_with(CONDUCTOR_PREAMBLE_HEAD));
        assert!(out.contains(shell_rules()));
        assert!(out.ends_with("TASK:\n"));
    }

    #[test]
    fn body_model_reads_the_picker_choice() {
        assert_eq!(
            body_model(&json!({"model": "gpt-5-5-pro"})).as_deref(),
            Some("gpt-5-5-pro")
        );
        // surrounding whitespace is trimmed
        assert_eq!(
            body_model(&json!({"model": "  gpt-5-5-thinking  "})).as_deref(),
            Some("gpt-5-5-thinking")
        );
        // absent / empty / non-string -> None (caller keeps the launch model)
        assert_eq!(body_model(&json!({})), None);
        assert_eq!(body_model(&json!({"model": ""})), None);
        assert_eq!(body_model(&json!({"model": "   "})), None);
        assert_eq!(body_model(&json!({"model": 5})), None);
    }

    #[test]
    fn headless_kinds_are_memory_and_compact_only() {
        assert!(is_headless_kind(Some("memory"))); // codex 0.137 request_kind
        assert!(is_headless_kind(Some("memory_consolidation"))); // legacy
        assert!(is_headless_kind(Some("compact")));
        assert!(is_headless_kind(Some("compaction"))); // current request_kind
        assert!(!is_headless_kind(Some("turn"))); // the interactive session
        assert!(!is_headless_kind(Some("prewarm"))); // background, but not headless-special
        assert!(!is_headless_kind(Some("review")));
        assert!(!is_headless_kind(Some("collab_spawn")));
        assert!(!is_headless_kind(None));
    }

    #[test]
    fn stream_error_classification_is_conservative() {
        // moderation-looking -> FATAL invalid_prompt (codex stops retrying)
        let tf = classify_stream_error("moderation_blocked", "", "removed for content policy");
        assert_eq!(tf.code, "invalid_prompt");
        assert!(tf.message.contains("content policy"));
        // rate-limit-looking -> retryable shim_error, message passed through so
        // codex can parse a retry-after hint out of it.
        let tf = classify_stream_error("rate_limit_error", "", "try again in 30 seconds");
        assert_eq!(tf.code, "shim_error");
        assert!(tf.message.contains("try again in 30 seconds"));
        // unknown error shape -> retryable (false fatal is worse than slow)
        let tf = classify_stream_error("server_error", "internal", "");
        assert_eq!(tf.code, "shim_error");
        assert!(tf.message.contains("internal"));
    }

    #[test]
    fn stall_watchdog_timeout_parsing() {
        // None = "no explicit override" -> caller uses the default budget.
        assert_eq!(explicit_stall_secs(None), None); // unset
        assert_eq!(explicit_stall_secs(Some("")), None); // empty
        assert_eq!(explicit_stall_secs(Some("0")), None); // 0 must NOT disable
        assert_eq!(explicit_stall_secs(Some("bogus")), None); // junk
        assert_eq!(explicit_stall_secs(Some(" 45 ")), Some(45)); // trimmed, honoured
        assert_eq!(explicit_stall_secs(Some("120")), Some(120));
    }

    #[test]
    fn body_effort_maps_codex_reasoning_to_chatgpt_vocab() {
        // codex sends reasoning.effort in its own vocabulary; we map to ChatGPT's.
        let eff = |v: &str| body_reasoning_effort(&json!({"reasoning": {"effort": v}}));
        assert_eq!(eff("none").as_deref(), Some("min"));
        assert_eq!(eff("minimal").as_deref(), Some("min"));
        assert_eq!(eff("low").as_deref(), Some("min"));
        assert_eq!(eff("medium").as_deref(), Some("standard"));
        assert_eq!(eff("high").as_deref(), Some("extended"));
        assert_eq!(eff("xhigh").as_deref(), Some("max"));
        // No reasoning block, or an unknown value -> None (keep launch default).
        assert_eq!(body_reasoning_effort(&json!({})), None);
        assert_eq!(body_reasoning_effort(&json!({"reasoning": {}})), None);
        assert_eq!(eff("bogus"), None);
    }

    #[test]
    fn headless_message_forwards_instructions_without_bridge_or_bind() {
        let body = json!({"instructions": "  Distill memories from this rollout.  "});
        let out = build_headless_message(&body, "rollout_context: ...");
        assert_eq!(
            out,
            "Distill memories from this rollout.\n\nrollout_context: ..."
        );
        assert!(!out.contains("[connector bridge"));
        assert!(!out.contains("[thread binding"));
        // No instructions -> just the prompt.
        assert_eq!(build_headless_message(&json!({}), "p"), "p");
    }

    #[test]
    fn thread_bind_directive_substitutes_thread_id() {
        let filled = fill_thread_bind("T123");
        assert!(filled.contains("codex:T123"));
        assert!(!filled.contains("{thread_id}"));
        // verbatim register-first rail
        assert!(filled.contains("repo_register"));
        assert!(filled.starts_with("[thread binding"));
    }

    #[test]
    fn preambles_are_load_bearing_verbatim() {
        assert!(CONDUCTOR_BRIDGE.contains("BEGIN OPERATING INSTRUCTIONS"));
        assert!(CONDUCTOR_TASK_SEP_HEAD.contains("END OPERATING INSTRUCTIONS"));
        assert!(CONDUCTOR_TASK_SEP_TAIL.contains("# Task"));
        assert!(CONDUCTOR_PREAMBLE_HEAD.contains("repo-agent"));
        assert_eq!(CONDUCTOR_PREAMBLE_TAIL, "TASK:\n");
    }

    #[test]
    fn shell_rules_match_the_runtime_platform() {
        // The repo tools run powershell on Windows and `sh -c` elsewhere; the
        // model-facing guidance must follow the platform it's actually built for.
        if cfg!(windows) {
            assert!(shell_rules().contains("Windows PowerShell 5.1"));
            assert_eq!(shell_word(), "PowerShell");
        } else {
            assert!(shell_rules().contains("POSIX sh"));
            assert_eq!(shell_word(), "sh");
        }
        // The assembled preamble always carries whatever the platform rules are.
        let out = build_conductor_preamble(&json!({"instructions": "x"}));
        assert!(out.contains(shell_rules()));
    }

    #[test]
    fn canonical_json_sorts_keys() {
        let v = json!({"b": 1, "a": 2, "c": {"z": 1, "y": 2}});
        assert_eq!(
            canonical_json(&v),
            "{\"a\": 2, \"b\": 1, \"c\": {\"y\": 2, \"z\": 1}}"
        );
    }

    #[test]
    fn blocking_tools_get_long_timeout() {
        assert!(is_blocking_tool("request_user_input"));
        assert!(is_blocking_tool("wait_agent"));
        assert!(is_blocking_tool("followup_task"));
        assert!(is_blocking_tool("send_message"));
        assert!(!is_blocking_tool("shell_command"));
    }
}
