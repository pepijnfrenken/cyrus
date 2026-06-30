//! Durable repo state: event log, notes, capsules, approvals, path leases, and
//! the session->agent attribution map. Plus the auto-compaction capsule builder.
//!
//! (+ types.ts: ToolEvent, ApprovalRequest, Capsule, PathLease, MemoryNote)
//!
//! Behavioral fidelity notes:
//!   - The TS serializes ALL state mutations + file I/O through an in-process
//!     promise queue (the verified save() race fix). Rust has no event loop here,
//!     so we make every mutator a synchronous `&mut self` method that performs its
//!     in-memory mutation and then its file I/O inline. Callers hold the whole
//!     `RepoState` behind a single async lock (`tokio::sync::Mutex<RepoState>`),
//!     which gives the exact same guarantee: at most one append+rename sequence is
//!     ever in flight, so `state.json` can never be torn-written. The TS deferred
//!     the I/O off the request's await point; we run it inline under the lock —
//!     equivalent ordering, no torn writes either way.
//!   - `event()` captures the request session SYNCHRONOUSLY in the TS (the
//!     AsyncLocalStorage store is only valid on the call stack). Here the session
//!     is an explicit argument that the tool handler reads from its request
//!     context and passes in before anything is deferred — same capture point.
//!   - State-dir writes (events.jsonl, state.json, blobs/, capsule .md) bypass the
//!     `blockedPathGlobs` tool guard: that guard lives in the tool layer; this
//!     module writes the state dir directly with std::fs and is never routed
//!     through it.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use chrono::SecondsFormat;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Config view
//
// `config.rs` is still a skeleton stub (a unit `Config`), and the brief forbids
// editing it, so — exactly as `http.rs` does with its own `HttpConfig` view —
// this module declares the config shape it needs locally. When `config.rs` is
// ported, the real `RepoAgentConfig` either replaces this or `From`-converts into
// it; the field set here is precisely what `RepoState` touches.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCompactConfig {
    pub enabled: bool,
    #[serde(rename = "eventSoftLimit")]
    pub event_soft_limit: usize,
    #[serde(rename = "eventHardLimit")]
    pub event_hard_limit: usize,
    #[serde(rename = "bytesSoftLimit")]
    pub bytes_soft_limit: usize,
    #[serde(rename = "hotEventCount")]
    pub hot_event_count: usize,
    #[serde(rename = "hotFileCount")]
    pub hot_file_count: usize,
    #[serde(rename = "capsuleBudgetChars")]
    pub capsule_budget_chars: usize,
    #[serde(rename = "returnCapsuleEveryNEvents")]
    pub return_capsule_every_n_events: usize,
}

/// Subset of `RepoAgentConfig` that `RepoState` reads. The field names
/// mirror the TS camelCase via serde renames so a real config can round-trip in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoAgentConfig {
    pub root: String,
    #[serde(rename = "homeRoot")]
    pub home_root: String,
    #[serde(rename = "currentProject", default, skip_serializing_if = "Option::is_none")]
    pub current_project: Option<String>,
    #[serde(rename = "sandboxMode")]
    pub sandbox_mode: SandboxMode,
    #[serde(rename = "approvalPolicy")]
    pub approval_policy: ApprovalPolicy,
    #[serde(rename = "approvalsReviewer")]
    pub approvals_reviewer: ApprovalReviewer,
    #[serde(rename = "writableRoots")]
    pub writable_roots: Vec<String>,
    #[serde(rename = "autoCompact")]
    pub auto_compact: AutoCompactConfig,
    // Subagent budgets read by `subagent.rs` (spawn_subagent). Kept as `f64` to
    // mirror the TS `number` config (config.rs::RepoAgentConfig) and the JS-style
    // numeric comparisons in spawn_subagent. Defaults match config.ts:
    // maxSubagents = 2, maxSubagentSpawns = 12.
    #[serde(rename = "maxSubagents", default = "default_max_subagents")]
    pub max_subagents: f64,
    #[serde(rename = "maxSubagentSpawns", default = "default_max_subagent_spawns")]
    pub max_subagent_spawns: f64,
}

fn default_max_subagents() -> f64 {
    2.0
}

fn default_max_subagent_spawns() -> f64 {
    12.0
}

// ---------------------------------------------------------------------------
// Types ported from src/types.ts.
// ---------------------------------------------------------------------------

/// `SandboxMode` — "read-only" | "workspace-write" | "danger-full-access".
pub type SandboxMode = String;
/// `ApprovalPolicy` — "untrusted" | "on-request" | "never".
pub type ApprovalPolicy = String;
/// `ApprovalReviewer` — "user" | "auto_review".
pub type ApprovalReviewer = String;
/// `ApprovalDecision` — "pending" | "approved" | "denied".
pub type ApprovalDecision = String;

/// `SubagentStatus` — "pending" | "spawning" | "running" | "done" | "blocked" |
/// "timeout" | "crashed".
pub type SubagentStatus = String;

const TERMINAL_STATUSES: [&str; 4] = ["done", "blocked", "timeout", "crashed"];

fn is_terminal(status: &str) -> bool {
    TERMINAL_STATUSES.contains(&status)
}

/// `ParsedCommandSummary` from types.ts. Stored opaquely on events; kept as a
/// struct so round-tripping the JSON preserves shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedCommandSummary {
    pub kind: String,
    pub cmd: String,
    pub safe: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNote {
    pub id: String,
    pub ts: String,
    pub kind: String,
    pub text: String,
    pub files: Vec<String>,
}

/// `ToolEvent` from types.ts. Optional fields use `skip_serializing_if` so the
/// serialized JSON omits absent keys exactly like the TS object spread does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolEvent {
    pub id: String,
    pub ts: String,
    /// Event kind. Absent on ordinary completion events; `"tool_started"` marks
    /// the dispatch-time started ping. Started events carry an EMPTY `tool` field
    /// (real name in `args.tool`) because the lipsync /events consumers treat any
    /// event with a non-empty `tool` as a completed call and skip empty ones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    pub tool: String,
    pub ok: bool,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blobs: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(rename = "parsedCommand", default, skip_serializing_if = "Option::is_none")]
    pub parsed_command: Option<Vec<ParsedCommandSummary>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(rename = "sandboxMode", default, skip_serializing_if = "Option::is_none")]
    pub sandbox_mode: Option<SandboxMode>,
    #[serde(rename = "approvalPolicy", default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<ApprovalPolicy>,
    #[serde(rename = "approvalId", default, skip_serializing_if = "Option::is_none")]
    pub approval_id: Option<String>,
    #[serde(rename = "needsApproval", default, skip_serializing_if = "Option::is_none")]
    pub needs_approval: Option<bool>,
    // `exitCode?: number | null` — distinguish absent (None) from explicit null
    // (Some(None)) so an explicit-null exit code still round-trips as `null`.
    #[serde(
        rename = "exitCode",
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with_double_option"
    )]
    pub exit_code: Option<Option<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(rename = "durationMs", default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// Inline `serde_with`-style double-option (de)serializer so `exitCode` can be
/// absent, `null`, or a number — matching `number | null | undefined` in TS.
mod serde_with_double_option {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<Option<i64>>, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            // Outer None is handled by skip_serializing_if and never reaches here.
            Some(Some(n)) => ser.serialize_i64(*n),
            Some(None) => ser.serialize_none(),
            None => ser.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(de: D) -> Result<Option<Option<i64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Present key: parse inner Option<i64> (null -> None, number -> Some).
        Ok(Some(Option::<i64>::deserialize(de)?))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: String,
    pub ts: String,
    pub tool: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<serde_json::Value>,
    #[serde(rename = "sandboxMode")]
    pub sandbox_mode: SandboxMode,
    #[serde(rename = "approvalPolicy")]
    pub approval_policy: ApprovalPolicy,
    pub reviewer: ApprovalReviewer,
    pub status: ApprovalDecision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobRef {
    pub id: String,
    pub sha256: String,
    pub bytes: u64,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionNotice {
    #[serde(rename = "capsuleId")]
    pub capsule_id: String,
    pub reason: String,
    pub short: String,
    #[serde(rename = "nextAction")]
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capsule {
    pub id: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "projectRoot", default, skip_serializing_if = "Option::is_none")]
    pub project_root: Option<String>,
    #[serde(rename = "sandboxMode", default, skip_serializing_if = "Option::is_none")]
    pub sandbox_mode: Option<SandboxMode>,
    #[serde(rename = "approvalPolicy", default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<ApprovalPolicy>,
    #[serde(rename = "eventCount")]
    pub event_count: u64,
    #[serde(rename = "bytesApprox")]
    pub bytes_approx: u64,
    pub markdown: String,
    #[serde(rename = "hotFiles")]
    pub hot_files: Vec<String>,
    #[serde(rename = "hotBlobs")]
    pub hot_blobs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathLease {
    #[serde(rename = "leaseId")]
    pub lease_id: String,
    #[serde(rename = "agentId")]
    pub agent_id: String,
    pub paths: Vec<String>,
    pub mode: String, // "read" | "write"
    pub ts: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandbackCapsule {
    #[serde(rename = "agentId")]
    pub agent_id: String,
    pub status: SubagentStatus,
    pub summary: String,
    #[serde(rename = "filesTouched")]
    pub files_touched: Vec<String>,
    #[serde(rename = "bgIds")]
    pub bg_ids: Vec<String>,
    #[serde(rename = "durationMs")]
    pub duration_ms: u64,
}

/// `{ turns; lastSummary }` progress sub-object on `SubagentJob`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentProgress {
    pub turns: u64,
    #[serde(rename = "lastSummary")]
    pub last_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentJob {
    #[serde(rename = "agentId")]
    pub agent_id: String,
    #[serde(rename = "parentAgentId")]
    pub parent_agent_id: String,
    pub label: String,
    pub task: String,
    #[serde(rename = "scopePaths")]
    pub scope_paths: Vec<String>,
    pub status: SubagentStatus,
    #[serde(rename = "createdTs")]
    pub created_ts: String,
    #[serde(rename = "lastHeartbeatTs", default, skip_serializing_if = "Option::is_none")]
    pub last_heartbeat_ts: Option<String>,
    #[serde(rename = "targetId", default, skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<SubagentProgress>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<HandbackCapsule>,
    pub collected: bool,
    #[serde(rename = "leaseIds")]
    pub lease_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

// ---------------------------------------------------------------------------
// Small helpers reproducing JS primitives byte-for-byte.
// ---------------------------------------------------------------------------

/// `new Date().toISOString()` — ISO-8601 in UTC with millisecond precision and a
/// trailing `Z`, e.g. `2026-06-10T12:34:56.789Z`.
fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// `Date.now().toString(36)` — milliseconds since the Unix epoch in base-36
/// (lowercase, JS radix-36 alphabet 0-9a-z).
fn now_ms_base36() -> String {
    let ms = chrono::Utc::now().timestamp_millis();
    // `Date.now()` is always non-negative in practice; mirror JS's unsigned-ish
    // base36 of a positive integer.
    to_base36(ms.max(0) as u128)
}

fn to_base36(mut n: u128) -> String {
    if n == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = Vec::new();
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    buf.reverse();
    String::from_utf8(buf).unwrap()
}

/// `randomUUID()` lowercase, with dashes (matches Node's `crypto.randomUUID`).
fn random_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// `createHash("sha256").update(text).digest("hex")` — lowercase hex digest.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Port of `trimMiddle` from src/result.ts. Head ratio 0.58, tail ratio 0.32 are
/// load-bearing; the `[trimmed N chars]` count uses JS `toLocaleString()` grouping
/// (en-US thousands separators, e.g. `1,234`).
///
/// IMPORTANT: JS `String.length`/`slice` operate on UTF-16 code units. We mirror
/// that by slicing over the UTF-16 view so byte budgets line up exactly with the
/// original for any input (ASCII inputs are identical either way).
pub fn trim_middle(s: &str, max_chars: usize) -> (String, bool) {
    let units: Vec<u16> = s.encode_utf16().collect();
    let len = units.len();
    if len <= max_chars {
        return (s.to_string(), false);
    }
    let head = (max_chars as f64 * 0.58).floor() as usize;
    let tail = (max_chars as f64 * 0.32).floor() as usize;
    let omitted = len - head - tail;
    let head_str = String::from_utf16_lossy(&units[..head]);
    let tail_str = String::from_utf16_lossy(&units[len - tail..]);
    let text = format!(
        "{head_str}\n\n…[trimmed {} chars]…\n\n{tail_str}",
        group_thousands(omitted)
    );
    (text, true)
}

/// `Number.prototype.toLocaleString()` for a non-negative integer in en-US:
/// thousands grouped with commas (e.g. `1234567` -> `1,234,567`).
fn group_thousands(n: usize) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

/// `rank(items, n)`: count occurrences (truthy items only), sort by count
/// descending, take `n`. JS `Array.prototype.sort` is stable; first-seen order
/// breaks ties, so we preserve insertion order for equal counts.
fn rank(items: &[String], n: usize) -> Vec<String> {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<String, u64> = HashMap::new();
    for item in items.iter().filter(|s| !s.is_empty()) {
        let entry = counts.entry(item.clone()).or_insert(0);
        if *entry == 0 {
            order.push(item.clone());
        }
        *entry += 1;
    }
    // Stable sort by descending count; ties keep first-seen order.
    order.sort_by(|a, b| counts[b].cmp(&counts[a]));
    order.truncate(n);
    order
}

// ---------------------------------------------------------------------------
// PersistedState
// ---------------------------------------------------------------------------

/// `PersistedState` from store.ts — the JSON shape of state.json.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedState {
    #[serde(default)]
    events: Vec<ToolEvent>,
    #[serde(default)]
    notes: Vec<MemoryNote>,
    #[serde(default)]
    capsules: Vec<Capsule>,
    #[serde(default)]
    approvals: Vec<ApprovalRequest>,
}

impl PersistedState {
    fn empty() -> Self {
        PersistedState {
            events: Vec::new(),
            notes: Vec::new(),
            capsules: Vec::new(),
            approvals: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs for `event()` / `requestApproval()` (the `Omit<…, "id" | "ts">` shapes).
// ---------------------------------------------------------------------------

/// Input for `RepoState::event` — `Omit<ToolEvent, "id" | "ts">`. Every field is
/// optional except the four the TS requires positionally (`tool`, `ok`,
/// `summary`); the rest default to `None`/absent and are filled by `event()`.
#[derive(Debug, Clone, Default)]
pub struct EventInput {
    pub tool: String,
    pub ok: bool,
    pub summary: String,
    /// See [`ToolEvent::kind`]; `None` for ordinary completion events.
    pub kind: Option<String>,
    pub seq: Option<u64>,
    pub agent: Option<String>,
    pub args: Option<serde_json::Value>,
    pub files: Option<Vec<String>>,
    pub blobs: Option<Vec<String>>,
    pub command: Option<String>,
    pub parsed_command: Option<Vec<ParsedCommandSummary>>,
    pub project: Option<String>,
    pub sandbox_mode: Option<SandboxMode>,
    pub approval_policy: Option<ApprovalPolicy>,
    pub approval_id: Option<String>,
    pub needs_approval: Option<bool>,
    pub exit_code: Option<Option<i64>>,
    pub bytes: Option<u64>,
    pub duration_ms: Option<u64>,
}

impl EventInput {
    /// Convenience constructor for the common `{ tool, ok, summary }` case.
    pub fn new(tool: impl Into<String>, ok: bool, summary: impl Into<String>) -> Self {
        EventInput {
            tool: tool.into(),
            ok,
            summary: summary.into(),
            ..Default::default()
        }
    }
}

/// Input for `RepoState::request_approval` — `Omit<ApprovalRequest, "id" | "ts">`.
#[derive(Debug, Clone)]
pub struct ApprovalInput {
    pub tool: String,
    pub reason: String,
    pub command: Option<String>,
    pub files: Option<Vec<String>>,
    pub retry: Option<serde_json::Value>,
    pub sandbox_mode: SandboxMode,
    pub approval_policy: ApprovalPolicy,
    pub reviewer: ApprovalReviewer,
    pub status: ApprovalDecision,
}

/// Result of `acquireLease`.
#[derive(Debug, Clone)]
pub struct LeaseResult {
    pub ok: bool,
    pub conflict: Option<String>,
    pub lease_id: Option<String>,
}

/// One entry of `unboundSessions()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnboundSession {
    pub session: String,
    #[serde(rename = "lastSeq")]
    pub last_seq: u64,
    pub tools: Vec<String>,
}

// ---------------------------------------------------------------------------
// RepoState
// ---------------------------------------------------------------------------

/// In-memory + on-disk repo state handle. Port of `RepoState` in store.ts.
///
/// Concurrency contract: hold the whole value behind one async lock
/// (`tokio::sync::Mutex<RepoState>`). Every mutator takes `&mut self` and runs
/// its file I/O inline, so the lock serializes append+rename
/// promise queue did.
pub struct RepoState {
    pub config: RepoAgentConfig,

    pub dir: PathBuf,
    pub events_path: PathBuf,
    pub state_path: PathBuf,
    pub blob_dir: PathBuf,
    pub subagents_path: PathBuf,

    state: PersistedState,
    seq: u64,

    /// In-memory-only events (the dispatcher's "tool started" pings): delivered
    /// to the live /events SSE tail via [`recent_events_since`] but NEVER
    /// appended to events.jsonl or saved into state.json — one started ping per
    /// tool call would otherwise double the durable write volume (each `event()`
    /// is an append + a full state.json rewrite). Capped ring (newest kept).
    ephemeral_events: Vec<ToolEvent>,

    /// ChatGPT x-openai-session token -> agentId. "main" is implicit.
    session_to_agent: HashMap<String, String>,
    /// event seq -> session that produced it (for retroactive re-stamp on bind).
    seq_to_session: HashMap<u64, String>,

    /// Path-level write/read leases for file coordination across subagents.
    leases: Vec<PathLease>,

    /// Durable subagent registry, bound after `bind_root`.
    pub subagents: SubagentRegistry,
}

impl RepoState {
    /// `new RepoState(config)` — binds the state root and loads persisted state.
    pub fn new(config: RepoAgentConfig) -> std::io::Result<Self> {
        let root = config.root.clone();
        // Construct with placeholder paths, then bind for real.
        let mut state = RepoState {
            config,
            dir: PathBuf::new(),
            events_path: PathBuf::new(),
            state_path: PathBuf::new(),
            blob_dir: PathBuf::new(),
            subagents_path: PathBuf::new(),
            state: PersistedState::empty(),
            seq: 0,
            ephemeral_events: Vec::new(),
            session_to_agent: HashMap::new(),
            seq_to_session: HashMap::new(),
            leases: Vec::new(),
            subagents: SubagentRegistry::detached(),
        };
        state.bind_root(&root)?;
        Ok(state)
    }

    /// `bindRoot(root)` — point all paths at `<root>/.repo-agent-mcp`, ensure the
    /// blob dir exists, load state, and (re)bind the subagent registry.
    fn bind_root(&mut self, root: &str) -> std::io::Result<()> {
        self.dir = Path::new(root).join(".repo-agent-mcp");
        self.events_path = self.dir.join("events.jsonl");
        self.state_path = self.dir.join("state.json");
        self.blob_dir = self.dir.join("blobs");
        self.subagents_path = self.dir.join("subagents.jsonl");
        fs::create_dir_all(&self.blob_dir)?; // mkdirSync(blobDir, { recursive: true })
        // Rotate an oversized append-only log down to the tail window a rebuild
        // would keep anyway — events.jsonl otherwise grows without bound.
        self.rotate_events_log();
        self.state = self.load();
        // Resume the seq counter past anything already persisted.
        self.seq = 0;
        for e in &self.state.events {
            self.seq = self.seq.max(e.seq.unwrap_or(0));
        }
        // Ephemeral (started) pings belong to the previous root's seq space.
        self.ephemeral_events.clear();
        self.subagents = SubagentRegistry::new(self.subagents_path.clone());
        Ok(())
    }

    /// When events.jsonl exceeds [`EVENTS_LOG_ROTATE_BYTES`], rewrite it in place
    /// to the last `event_hard_limit*2` parseable lines — exactly the window
    /// [`rebuild_events_from_log`] would keep. Atomic (tmp + rename), best-effort:
    /// any failure leaves the original log untouched.
    fn rotate_events_log(&self) {
        const EVENTS_LOG_ROTATE_BYTES: u64 = 5 * 1024 * 1024;
        let len = match fs::metadata(&self.events_path) {
            Ok(m) => m.len(),
            Err(_) => return,
        };
        if len <= EVENTS_LOG_ROTATE_BYTES {
            return;
        }
        let kept = self.rebuild_events_from_log();
        let mut body = String::new();
        for evt in &kept {
            if let Ok(line) = serde_json::to_string(evt) {
                body.push_str(&line);
                body.push('\n');
            }
        }
        let tmp = self.events_path.with_extension("jsonl.tmp");
        if fs::write(&tmp, &body).is_ok() && fs::rename(&tmp, &self.events_path).is_ok() {
            tracing::warn!(
                path = %self.events_path.display(),
                old_bytes = len,
                new_bytes = body.len(),
                kept_events = kept.len(),
                "rotated oversized events.jsonl to its rebuild tail window"
            );
        }
    }

    /// `switchRoot(root)`.
    pub fn switch_root(&mut self, root: &str) -> std::io::Result<()> {
        self.bind_root(root)
    }

    /// `load()` — read state.json, else rebuild events from the append-only log so
    /// a torn full-file write can never silently discard history.
    ///
    /// A state.json that EXISTS but fails to parse is preserved as
    /// `state.json.corrupt-<ts>` (and logged loudly) before the rebuild: the
    /// event log only restores events — notes/capsules/approvals live solely in
    /// state.json and would otherwise vanish silently.
    fn load(&self) -> PersistedState {
        let mut loaded: Option<PersistedState> = None;
        if self.state_path.exists() {
            let raw = fs::read_to_string(&self.state_path).ok();
            match raw.as_deref().map(serde_json::from_str::<PersistedState>) {
                Some(Ok(parsed)) => loaded = Some(parsed),
                Some(Err(err)) => {
                    let backup = self
                        .state_path
                        .with_extension(format!("json.corrupt-{}", now_ms_base36()));
                    let moved = fs::rename(&self.state_path, &backup).is_ok();
                    tracing::error!(
                        path = %self.state_path.display(),
                        backup = %backup.display(),
                        backup_saved = moved,
                        error = %err,
                        "state.json is corrupt — notes/capsules/approvals in it are NOT \
                         recoverable from events.jsonl; rebuilding events from the log"
                    );
                }
                None => {} // unreadable (I/O) — fall through to the rebuild
            }
        }
        match loaded {
            Some(l) => l,
            None => {
                let mut l = PersistedState::empty();
                l.events = self.rebuild_events_from_log();
                l
            }
        }
    }

    /// `rebuildEventsFromLog()` — read the tail of events.jsonl (cap = hardLimit*2),
    /// skipping torn lines.
    fn rebuild_events_from_log(&self) -> Vec<ToolEvent> {
        if !self.events_path.exists() {
            return Vec::new();
        }
        let raw = match fs::read_to_string(&self.events_path) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let cap = self.config.auto_compact.event_hard_limit.saturating_mul(2);
        let lines: Vec<&str> = raw.split('\n').filter(|l| !l.trim().is_empty()).collect();
        let start = lines.len().saturating_sub(cap);
        let mut events = Vec::new();
        for line in &lines[start..] {
            if let Ok(evt) = serde_json::from_str::<ToolEvent>(line) {
                events.push(evt);
            }
        }
        events
    }

    /// `save()` — atomic full-file save: write `state.json.tmp`, fsync it, then
    /// rename over `state.json`. Without the `sync_all` a power loss after the
    /// rename can leave an empty/stale file (the rename can be durable before
    /// the data blocks are). Mirrors `JSON.stringify(state, null, 2)`.
    pub fn save(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let tmp = self.state_path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(&self.state).unwrap_or_else(|_| "{}".to_string());
        {
            use std::io::Write;
            let mut f = fs::File::create(&tmp)?;
            f.write_all(body.as_bytes())?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &self.state_path)?;
        Ok(())
    }

    // ----- subagent attribution (by ChatGPT per-conversation session token) -----

    /// `bindSession(session, agentId)`.
    pub fn bind_session(&mut self, session: &str, agent_id: &str) {
        self.session_to_agent
            .insert(session.to_string(), agent_id.to_string());
    }

    /// `agentForSession(session)` — unbound/empty resolves to "main".
    pub fn agent_for_session(&self, session: Option<&str>) -> String {
        match session {
            None => "main".to_string(),
            Some(s) => self
                .session_to_agent
                .get(s)
                .cloned()
                .unwrap_or_else(|| "main".to_string()),
        }
    }

    /// `restampAgent(session, agentId)` — bind and retroactively re-label every
    /// in-memory event the session already produced, then persist.
    pub fn restamp_agent(&mut self, session: &str, agent_id: &str) {
        self.session_to_agent
            .insert(session.to_string(), agent_id.to_string());
        for e in &mut self.state.events {
            if let Some(seq) = e.seq {
                if self.seq_to_session.get(&seq).map(String::as_str) == Some(session) {
                    e.agent = Some(agent_id.to_string());
                }
            }
        }
        let _ = self.save();
    }

    /// `unboundSessions()` — distinct sessions in the log not yet bound to an
    /// agentId (and not implicit "main"), with latest seq + last 5 tool names.
    pub fn unbound_sessions(&self) -> Vec<UnboundSession> {
        // Preserve first-seen order of sessions, like JS Map iteration order.
        let mut order: Vec<String> = Vec::new();
        let mut by_session: HashMap<String, (u64, Vec<String>)> = HashMap::new();
        for e in &self.state.events {
            let seq = match e.seq {
                Some(s) => s,
                None => continue,
            };
            let session = match self.seq_to_session.get(&seq) {
                Some(s) => s,
                None => continue,
            };
            if self.session_to_agent.contains_key(session) {
                continue;
            }
            if !by_session.contains_key(session) {
                order.push(session.clone());
                by_session.insert(session.clone(), (0, Vec::new()));
            }
            let entry = by_session.get_mut(session).unwrap();
            entry.0 = entry.0.max(seq);
            entry.1.push(e.tool.clone());
        }
        order
            .into_iter()
            .map(|session| {
                let (last_seq, tools) = by_session.remove(&session).unwrap();
                let tools = tools
                    .iter()
                    .rev()
                    .take(5)
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>();
                UnboundSession {
                    session,
                    last_seq,
                    tools,
                }
            })
            .collect()
    }

    /// `boundAgents()` — distinct agentIds bound to at least one session.
    pub fn bound_agents(&self) -> Vec<String> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut out = Vec::new();
        for v in self.session_to_agent.values() {
            if seen.insert(v.as_str()) {
                out.push(v.clone());
            }
        }
        out
    }

    // ----- path leases (file coordination across subagents) -----

    /// `pathsOverlap(a, b)` — equal, or one is a directory prefix of the other.
    fn paths_overlap(a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let a_dir = if a.ends_with('/') {
            a.to_string()
        } else {
            format!("{a}/")
        };
        let b_dir = if b.ends_with('/') {
            b.to_string()
        } else {
            format!("{b}/")
        };
        b.starts_with(&a_dir) || a.starts_with(&b_dir)
    }

    /// `acquireLease(agentId, paths, mode)`. Write leases conflict on overlap with
    /// a write lease held by a DIFFERENT agent; read leases never conflict.
    pub fn acquire_lease(&mut self, agent_id: &str, paths: Vec<String>, mode: &str) -> LeaseResult {
        if mode == "write" {
            for lease in &self.leases {
                if lease.mode != "write" || lease.agent_id == agent_id {
                    continue;
                }
                for held in &lease.paths {
                    for want in &paths {
                        if Self::paths_overlap(held, want) {
                            return LeaseResult {
                                ok: false,
                                conflict: Some(format!("{want} leased by {}", lease.agent_id)),
                                lease_id: None,
                            };
                        }
                    }
                }
            }
        }
        let lease = PathLease {
            lease_id: format!("lease_{}_{}", now_ms_base36(), &random_uuid()[..6]),
            agent_id: agent_id.to_string(),
            paths,
            mode: mode.to_string(),
            ts: now_iso(),
        };
        let lease_id = lease.lease_id.clone();
        self.leases.push(lease);
        LeaseResult {
            ok: true,
            conflict: None,
            lease_id: Some(lease_id),
        }
    }

    /// `releaseLease(leaseId)`.
    pub fn release_lease(&mut self, lease_id: &str) {
        self.leases.retain(|l| l.lease_id != lease_id);
    }

    /// `releaseLeasesForAgent(agentId)`.
    pub fn release_leases_for_agent(&mut self, agent_id: &str) {
        self.leases.retain(|l| l.agent_id != agent_id);
    }

    /// `activeLeases()`.
    pub fn active_leases(&self) -> Vec<PathLease> {
        self.leases.clone()
    }

    /// `recentEventsSince(sinceSeq, agent?)` — events newer than `sinceSeq`,
    /// optionally filtered to one agent (default agent "main"). Merges the
    /// persisted events with the in-memory-only (started) pings, ordered by seq,
    /// so the /events SSE tail sees both through one monotonic cursor.
    pub fn recent_events_since(&self, since_seq: u64, agent: Option<&str>) -> Vec<ToolEvent> {
        let keep = |e: &&ToolEvent| {
            e.seq.unwrap_or(0) > since_seq
                && match agent {
                    None => true,
                    Some(a) => e.agent.as_deref().unwrap_or("main") == a,
                }
        };
        let mut out: Vec<ToolEvent> = self
            .state
            .events
            .iter()
            .filter(keep)
            .chain(self.ephemeral_events.iter().filter(keep))
            .cloned()
            .collect();
        out.sort_by_key(|e| e.seq.unwrap_or(0));
        out
    }

    /// Build one stamped ToolEvent (id/ts/seq/agent/project/sandbox defaults) and
    /// record the seq->session mapping. Shared by the durable [`event`] and the
    /// in-memory-only [`ephemeral_event`].
    fn build_event(&mut self, input: EventInput, session: Option<&str>) -> ToolEvent {
        self.seq += 1;
        let seq = self.seq;
        let agent = input
            .agent
            .clone()
            .unwrap_or_else(|| self.agent_for_session(session));
        if let Some(s) = session {
            self.seq_to_session.insert(seq, s.to_string());
        }
        ToolEvent {
            id: format!("evt_{}_{}", now_ms_base36(), &random_uuid()[..8]),
            ts: now_iso(),
            kind: input.kind,
            seq: Some(seq),
            agent: Some(agent),
            tool: input.tool,
            ok: input.ok,
            summary: input.summary,
            args: input.args,
            files: input.files,
            blobs: input.blobs,
            command: input.command,
            parsed_command: input.parsed_command,
            project: Some(input.project.unwrap_or_else(|| self.config.root.clone())),
            sandbox_mode: Some(
                input
                    .sandbox_mode
                    .unwrap_or_else(|| self.config.sandbox_mode.clone()),
            ),
            approval_policy: Some(
                input
                    .approval_policy
                    .unwrap_or_else(|| self.config.approval_policy.clone()),
            ),
            approval_id: input.approval_id,
            needs_approval: input.needs_approval,
            exit_code: input.exit_code,
            bytes: input.bytes,
            duration_ms: input.duration_ms,
        }
    }

    /// `event(input)` — assign id/ts/seq, stamp agent/project/sandbox/approval
    /// defaults, push in memory, record seq->session, trim, then append+save.
    ///
    /// `session` is the request session captured synchronously by the caller (the
    /// TS reads it from AsyncLocalStorage at the top of `event()`).
    pub fn event(&mut self, input: EventInput, session: Option<&str>) -> ToolEvent {
        // JS truthiness: an empty session string is falsy. Both `agentForSession`
        // and the `seqToSession` set treat it like "no session".
        let session = session.filter(|s| !s.is_empty());
        let evt = self.build_event(input, session);

        self.state.events.push(evt.clone());
        let hard2 = self.config.auto_compact.event_hard_limit.saturating_mul(2);
        if self.state.events.len() > hard2 {
            let start = self.state.events.len() - hard2;
            self.state.events.drain(..start);
        }

        // Append-only log + atomic save (the deferred I/O, run inline under lock).
        let line = format!("{}\n", serde_json::to_string(&evt).unwrap());
        let _ = append_file(&self.events_path, &line);
        let _ = self.save();

        evt
    }

    /// In-memory-only event: visible to the live /events tail (same seq space as
    /// durable events) but never appended to events.jsonl, never save()d, and
    /// invisible to recent_events / snapshots / capsules. Used for the
    /// dispatcher's per-call "tool started" ping, which would otherwise double
    /// the durable write volume.
    pub fn ephemeral_event(&mut self, input: EventInput, session: Option<&str>) -> ToolEvent {
        const EPHEMERAL_CAP: usize = 256;
        let session = session.filter(|s| !s.is_empty());
        let evt = self.build_event(input, session);
        self.ephemeral_events.push(evt.clone());
        if self.ephemeral_events.len() > EPHEMERAL_CAP {
            let start = self.ephemeral_events.len() - EPHEMERAL_CAP;
            self.ephemeral_events.drain(..start);
        }
        evt
    }

    /// `recentEvents(n = 40)` — the last `n` events.
    pub fn recent_events(&self, n: usize) -> Vec<ToolEvent> {
        let start = self.state.events.len().saturating_sub(n);
        self.state.events[start..].to_vec()
    }

    /// `notes()`.
    pub fn notes(&self) -> Vec<MemoryNote> {
        self.state.notes.clone()
    }
    /// `capsules()`.
    pub fn capsules(&self) -> Vec<Capsule> {
        self.state.capsules.clone()
    }
    /// `approvals()`.
    pub fn approvals(&self) -> Vec<ApprovalRequest> {
        self.state.approvals.clone()
    }

    /// `requestApproval(input)` — unshift a pending approval (cap 40), persist,
    /// then emit a `needs approval` event. Returns the created request.
    pub fn request_approval(&mut self, input: ApprovalInput, session: Option<&str>) -> ApprovalRequest {
        let req = ApprovalRequest {
            id: format!("approval_{}_{}", now_ms_base36(), &random_uuid()[..6]),
            ts: now_iso(),
            tool: input.tool.clone(),
            reason: input.reason.clone(),
            command: input.command.clone(),
            files: input.files.clone(),
            retry: input.retry,
            sandbox_mode: input.sandbox_mode,
            approval_policy: input.approval_policy,
            reviewer: input.reviewer,
            status: input.status,
        };
        self.state.approvals.insert(0, req.clone());
        self.state.approvals.truncate(40);
        let _ = self.save();
        self.event(
            EventInput {
                tool: input.tool,
                ok: false,
                summary: format!("needs approval: {}", input.reason),
                approval_id: Some(req.id.clone()),
                needs_approval: Some(true),
                command: input.command,
                files: input.files,
                ..Default::default()
            },
            session,
        );
        req
    }

    /// `decideApproval(id, status)` — set status, persist, emit an event. Returns
    /// the (mutated) request, or None if unknown.
    pub fn decide_approval(
        &mut self,
        id: &str,
        status: &str,
        session: Option<&str>,
    ) -> Option<ApprovalRequest> {
        let idx = self.state.approvals.iter().position(|a| a.id == id)?;
        self.state.approvals[idx].status = status.to_string();
        let req = self.state.approvals[idx].clone();
        let _ = self.save();
        self.event(
            EventInput {
                tool: "repo_permissions".to_string(),
                ok: true,
                summary: format!("{status} {id}"),
                approval_id: Some(id.to_string()),
                ..Default::default()
            },
            session,
        );
        Some(req)
    }

    /// `remember(kind, text, files?)` — unshift a note (cap 200), persist, emit.
    pub fn remember(
        &mut self,
        kind: &str,
        text: &str,
        files: Vec<String>,
        session: Option<&str>,
    ) -> MemoryNote {
        let note = MemoryNote {
            id: format!("mem_{}_{}", now_ms_base36(), &random_uuid()[..6]),
            ts: now_iso(),
            kind: kind.to_string(),
            text: text.to_string(),
            files: files.clone(),
        };
        self.state.notes.insert(0, note.clone());
        self.state.notes.truncate(200);
        let _ = self.save();
        let (trimmed, _) = trim_middle(text, 140);
        self.event(
            EventInput {
                tool: "repo_remember".to_string(),
                ok: true,
                summary: format!("{kind}: {trimmed}"),
                files: Some(files),
                ..Default::default()
            },
            session,
        );
        note
    }

    /// `putBlob(text, label = "blob")` — write `<sha12>.txt` under blobs/, emit a
    /// `repo_blob` event, return the ref. `bytes` is the UTF-8 byte length.
    pub fn put_blob(&mut self, text: &str, label: &str, session: Option<&str>) -> std::io::Result<BlobRef> {
        let buf = text.as_bytes();
        let sha256 = sha256_hex(buf);
        let id = format!("blob_{}", &sha256[..12]);
        let path = self.blob_dir.join(format!("{id}.txt"));
        fs::write(&path, text)?;
        let bytes = buf.len() as u64;
        self.event(
            EventInput {
                tool: "repo_blob".to_string(),
                ok: true,
                summary: format!("{label}: {id}"),
                blobs: Some(vec![id.clone()]),
                bytes: Some(bytes),
                ..Default::default()
            },
            session,
        );
        Ok(BlobRef {
            id,
            sha256,
            bytes,
            path: path.to_string_lossy().into_owned(),
        })
    }

    /// `readBlob(id)` — read `<id>.txt` from blobs/, or None if missing.
    pub fn read_blob(&self, id: &str) -> Option<String> {
        let path = self.blob_dir.join(format!("{id}.txt"));
        if !path.exists() {
            return None;
        }
        fs::read_to_string(&path).ok()
    }

    /// `maybeCompact(reason = "soft-limit")`. Returns a notice iff a new capsule
    /// was created. No-ops when auto-compact is disabled, below limits, or when the
    /// latest capsule already covers the most recent event id.
    pub fn maybe_compact(&mut self, reason: &str) -> Option<CompactionNotice> {
        if !self.config.auto_compact.enabled {
            return None;
        }
        let recent = self.recent_events(self.config.auto_compact.event_hard_limit);
        // `e.bytes ?? JSON.stringify(e).length`. The fallback uses the serialized
        // length in UTF-16 code units (JS `String.length`). Key ordering in the
        // fallback serialization may differ slightly from V8's, but this only feeds
        // a soft-limit threshold (not a wire format), and `e.bytes` is set on most
        // events anyway, so the heuristic is unaffected in practice.
        let bytes: usize = recent
            .iter()
            .map(|e| {
                e.bytes.map(|b| b as usize).unwrap_or_else(|| {
                    serde_json::to_string(e)
                        .map(|s| s.encode_utf16().count())
                        .unwrap_or(0)
                })
            })
            .sum();
        let should = recent.len() >= self.config.auto_compact.event_soft_limit
            || bytes >= self.config.auto_compact.bytes_soft_limit;
        if !should {
            return None;
        }
        let last_event_id = recent.last().map(|e| e.id.clone());
        if let Some(latest) = self.state.capsules.first() {
            let needle = last_event_id.clone().unwrap_or_else(|| "never".to_string());
            if latest.id.contains(&needle) {
                return None;
            }
        }
        let cap_reason = format!("{reason}:{}", last_event_id.unwrap_or_else(|| "latest".to_string()));
        let capsule = self.create_capsule(&cap_reason);
        Some(CompactionNotice {
            capsule_id: capsule.id.clone(),
            reason: reason.to_string(),
            short: format!(
                "Auto-compacted {} events into {}. Ask repo_resume for the compressed state instead of replaying history.",
                capsule.event_count, capsule.id
            ),
            next_action: "Call repo_resume({ mode: 'groove' }) before continuing the agent loop."
                .to_string(),
        })
    }

    /// `createCapsule(reason = "manual")` — build the no-lostness markdown capsule,
    /// id-seed from its sha256, write `<id>.md`, unshift (cap 20), persist.
    pub fn create_capsule(&mut self, reason: &str) -> Capsule {
        let events = self.recent_events(self.config.auto_compact.event_hard_limit);

        let files_flat: Vec<String> = events
            .iter()
            .flat_map(|e| e.files.clone().unwrap_or_default())
            .collect();
        let hot_files = rank(&files_flat, self.config.auto_compact.hot_file_count);

        let blobs_flat: Vec<String> = events
            .iter()
            .flat_map(|e| e.blobs.clone().unwrap_or_default())
            .collect();
        let hot_blobs = rank(&blobs_flat, 8);

        let notes: Vec<MemoryNote> = self.state.notes.iter().take(16).cloned().collect();
        let pending_approvals: Vec<ApprovalRequest> = self
            .state
            .approvals
            .iter()
            .filter(|a| a.status == "pending")
            .take(8)
            .cloned()
            .collect();
        // `.slice(-8)` of the filtered failures — last 8.
        let failures_all: Vec<&ToolEvent> = events
            .iter()
            // TS: `!e.ok || (e.exitCode !== undefined && e.exitCode !== 0)`. An
            // exitCode present-and-null (Some(None)) counts as a failure too, since
            // JS `null !== 0` is true; only an absent exitCode (None) or an explicit
            // 0 is "not a failure" on its own.
            .filter(|e| {
                !e.ok
                    || match e.exit_code {
                        None => false,
                        Some(None) => true,
                        Some(Some(c)) => c != 0,
                    }
            })
            .collect();
        let fstart = failures_all.len().saturating_sub(8);
        let failures = &failures_all[fstart..];

        let hot_n = self.config.auto_compact.hot_event_count;
        let tstart = events.len().saturating_sub(hot_n);
        let thread = events[tstart..]
            .iter()
            .map(|e| {
                format!(
                    "- {} {} {}: {}",
                    e.ts,
                    if e.ok { "✓" } else { "✗" },
                    e.tool,
                    e.summary
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let notes_section = if notes.is_empty() {
            "none".to_string()
        } else {
            notes
                .iter()
                .map(|n| {
                    let files = if n.files.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", n.files.join(", "))
                    };
                    format!("- [{}] {}{}", n.kind, n.text, files)
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let failures_section = if failures.is_empty() {
            "none".to_string()
        } else {
            failures
                .iter()
                .map(|e| {
                    let exit = match e.exit_code {
                        Some(Some(c)) => format!(" exit={c}"),
                        Some(None) => " exit=null".to_string(),
                        None => String::new(),
                    };
                    format!("- {}: {}{}", e.tool, e.summary, exit)
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let hot_files_str = if hot_files.is_empty() {
            "none yet".to_string()
        } else {
            hot_files.join(", ")
        };
        let pending_str = if pending_approvals.is_empty() {
            "none".to_string()
        } else {
            pending_approvals
                .iter()
                .map(|a| format!("{}:{}", a.id, a.tool))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let hot_blobs_str = if hot_blobs.is_empty() {
            "none".to_string()
        } else {
            hot_blobs.join(", ")
        };
        let thread_str = if thread.is_empty() {
            "none".to_string()
        } else {
            thread.clone()
        };

        let markdown_raw = format!(
            "# Repo Agent Capsule\n\n\
id: pending\nreason: {reason}\ncreated: {created}\nproject: {project}\nsandbox: {sandbox}\napproval: {approval}\n\n\
## Groove State\n\
- Hot files: {hot_files_str}\n\
- Pending approvals: {pending_str}\n\
- Hot blobs: {hot_blobs_str}\n\n\
## Durable Notes\n{notes_section}\n\n\
## Failure Needles\n{failures_section}\n\n\
## Recent Event Thread\n{thread_str}\n\n\
## No-Lostness Contract\n\
Use this capsule as the canonical compressed state. Do not ask the user to restate context; call repo_read/repo_diff for exact bytes when precision matters.\n",
            reason = reason,
            created = now_iso(),
            project = self.config.root,
            sandbox = self.config.sandbox_mode,
            approval = self.config.approval_policy,
        );

        let (shortened, _) = trim_middle(&markdown_raw, self.config.auto_compact.capsule_budget_chars);
        let id_seed = &sha256_hex(shortened.as_bytes())[..12];
        let id = format!("cap_{}_{}", now_ms_base36(), id_seed);
        // `replace("id: pending", ...)` — JS String.replace replaces the FIRST
        // occurrence only.
        let markdown = replace_first(&shortened, "id: pending", &format!("id: {id}"));

        let capsule = Capsule {
            id: id.clone(),
            created_at: now_iso(),
            project_root: Some(self.config.root.clone()),
            sandbox_mode: Some(self.config.sandbox_mode.clone()),
            approval_policy: Some(self.config.approval_policy.clone()),
            event_count: events.len() as u64,
            // JS `markdown.length` is UTF-16 code units.
            bytes_approx: markdown.encode_utf16().count() as u64,
            markdown: markdown.clone(),
            hot_files,
            hot_blobs,
        };
        self.state.capsules.insert(0, capsule.clone());
        self.state.capsules.truncate(20);
        let _ = fs::write(self.dir.join(format!("{id}.md")), &markdown);
        let _ = self.save();
        capsule
    }

    /// `snapshot()` — the JSON object returned by `repo_status`/`repo_ui`.
    pub fn snapshot(&self) -> serde_json::Value {
        let writable_roots: Vec<String> = if self.config.sandbox_mode == "read-only" {
            Vec::new()
        } else {
            let mut v = vec![self.config.root.clone()];
            v.extend(self.config.writable_roots.iter().cloned());
            v
        };
        serde_json::json!({
            "project": {
                "name": self.config.current_project,
                "root": self.config.root,
                "homeRoot": self.config.home_root,
            },
            "permissions": {
                "sandboxMode": self.config.sandbox_mode,
                "approvalPolicy": self.config.approval_policy,
                "reviewer": self.config.approvals_reviewer,
                "writableRoots": writable_roots,
            },
            "approvals": self.state.approvals.iter().take(20).cloned().collect::<Vec<_>>(),
            "recentEvents": self.recent_events(24),
            "notes": self.state.notes.iter().take(20).cloned().collect::<Vec<_>>(),
            "capsules": self.state.capsules.iter().take(6).cloned().collect::<Vec<_>>(),
        })
    }
}

/// JS `String.prototype.replace(str, repl)` semantics: replace only the FIRST
/// occurrence of `needle`.
fn replace_first(haystack: &str, needle: &str, repl: &str) -> String {
    match haystack.find(needle) {
        Some(idx) => {
            let mut s = String::with_capacity(haystack.len());
            s.push_str(&haystack[..idx]);
            s.push_str(repl);
            s.push_str(&haystack[idx + needle.len()..]);
            s
        }
        None => haystack.to_string(),
    }
}

/// `appendFileSync(path, data)` — open-for-append (create if missing), write.
fn append_file(path: &Path, data: &str) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(data.as_bytes())
}

// ---------------------------------------------------------------------------
// SubagentRegistry
// ---------------------------------------------------------------------------

/// Result of `collectResult`.
#[derive(Debug, Clone)]
pub struct CollectResult {
    pub status: SubagentStatus,
    pub result: Option<HandbackCapsule>,
    /// true if THIS call performed the collection.
    pub collected: bool,
}

/// Input for `createJob`.
#[derive(Debug, Clone)]
pub struct CreateJobInput {
    pub task: String,
    pub label: Option<String>,
    pub scope_paths: Option<Vec<String>>,
    pub parent_agent_id: String,
    pub model: Option<String>,
    pub effort: Option<String>,
}

/// Patch for `updateJob` — every field optional; absent = unchanged. `agentId`
/// is always forced back to the job's own id (matching `{ ...job, ...patch,
/// agentId }`).
#[derive(Debug, Clone, Default)]
pub struct SubagentJobPatch {
    pub parent_agent_id: Option<String>,
    pub label: Option<String>,
    pub task: Option<String>,
    pub scope_paths: Option<Vec<String>>,
    pub status: Option<SubagentStatus>,
    pub created_ts: Option<String>,
    pub last_heartbeat_ts: Option<Option<String>>,
    pub target_id: Option<Option<String>>,
    pub progress: Option<Option<SubagentProgress>>,
    pub result: Option<Option<HandbackCapsule>>,
    pub collected: Option<bool>,
    pub lease_ids: Option<Vec<String>>,
    pub model: Option<Option<String>>,
    pub effort: Option<Option<String>>,
}

/// Durable subagent registry. Persisted append-only to subagents.jsonl, one JSON
/// line per write; load = read all lines, last-write-wins by agentId into a map.
///
/// In the TS, all writes serialize through the RepoState mutation queue. Here,
/// the registry is owned by `RepoState` and only mutated while the `RepoState`
/// async lock is held, so its appends are serialized by the same lock — no extra
/// queue needed.
pub struct SubagentRegistry {
    path: PathBuf,
    /// Insertion-ordered map (last-write-wins by agentId), to mirror JS Map order.
    jobs: indexmap::IndexMap<String, SubagentJob>,
    counter: u64,
}

impl SubagentRegistry {
    /// A registry with no backing path yet (used before `bind_root`); never
    /// written to.
    fn detached() -> Self {
        SubagentRegistry {
            path: PathBuf::new(),
            jobs: indexmap::IndexMap::new(),
            counter: 0,
        }
    }

    /// `new SubagentRegistry(path, enqueue)` — load durable jobs from the log.
    pub fn new(path: PathBuf) -> Self {
        let mut reg = SubagentRegistry {
            path,
            jobs: indexmap::IndexMap::new(),
            counter: 0,
        };
        reg.load();
        reg
    }

    /// `load()` — replay the append-only log, last-write-wins by agentId, then
    /// derive the id counter from the max existing `aN` id.
    fn load(&mut self) {
        if !self.path.exists() {
            return;
        }
        let raw = match fs::read_to_string(&self.path) {
            Ok(r) => r,
            Err(_) => return, // unreadable log: start empty
        };
        for line in raw.split('\n').filter(|l| !l.trim().is_empty()) {
            if let Ok(job) = serde_json::from_str::<SubagentJob>(line) {
                if !job.agent_id.is_empty() {
                    // IndexMap::insert preserves the existing slot's position on
                    // overwrite — matching JS Map.set last-write-wins-in-place.
                    self.jobs.insert(job.agent_id.clone(), job);
                }
            }
        }
        for id in self.jobs.keys() {
            if let Some(n) = parse_an_id(id) {
                self.counter = self.counter.max(n);
            }
        }
    }

    /// `persist(job)` — append one JSON line. (Serialized by the RepoState lock.)
    fn persist(&self, job: &SubagentJob) {
        let line = format!("{}\n", serde_json::to_string(job).unwrap());
        let _ = append_file(&self.path, &line);
    }

    /// `createJob(input)`.
    pub fn create_job(&mut self, input: CreateJobInput) -> SubagentJob {
        self.counter += 1;
        let agent_id = format!("a{}", self.counter);
        let label = input
            .label
            .unwrap_or_else(|| js_slice_chars(&input.task, 60));
        let job = SubagentJob {
            agent_id: agent_id.clone(),
            parent_agent_id: input.parent_agent_id,
            label,
            task: input.task,
            scope_paths: input.scope_paths.unwrap_or_default(),
            status: "pending".to_string(),
            created_ts: now_iso(),
            last_heartbeat_ts: None,
            target_id: None,
            progress: None,
            result: None,
            collected: false,
            lease_ids: Vec::new(),
            model: input.model,
            effort: input.effort,
        };
        self.jobs.insert(agent_id, job.clone());
        self.persist(&job);
        job
    }

    /// `getJob(agentId)`.
    pub fn get_job(&self, agent_id: &str) -> Option<SubagentJob> {
        self.jobs.get(agent_id).cloned()
    }

    /// `listJobs()` — all jobs in insertion order.
    pub fn list_jobs(&self) -> Vec<SubagentJob> {
        self.jobs.values().cloned().collect()
    }

    /// `liveJobs()` — non-terminal jobs.
    pub fn live_jobs(&self) -> Vec<SubagentJob> {
        self.jobs
            .values()
            .filter(|j| !is_terminal(&j.status))
            .cloned()
            .collect()
    }

    /// `updateJob(agentId, patch)` — merge patch, force agentId, persist.
    pub fn update_job(&mut self, agent_id: &str, patch: SubagentJobPatch) -> Option<SubagentJob> {
        let job = self.jobs.get(agent_id)?.clone();
        let mut merged = job;
        if let Some(v) = patch.parent_agent_id {
            merged.parent_agent_id = v;
        }
        if let Some(v) = patch.label {
            merged.label = v;
        }
        if let Some(v) = patch.task {
            merged.task = v;
        }
        if let Some(v) = patch.scope_paths {
            merged.scope_paths = v;
        }
        if let Some(v) = patch.status {
            merged.status = v;
        }
        if let Some(v) = patch.created_ts {
            merged.created_ts = v;
        }
        if let Some(v) = patch.last_heartbeat_ts {
            merged.last_heartbeat_ts = v;
        }
        if let Some(v) = patch.target_id {
            merged.target_id = v;
        }
        if let Some(v) = patch.progress {
            merged.progress = v;
        }
        if let Some(v) = patch.result {
            merged.result = v;
        }
        if let Some(v) = patch.collected {
            merged.collected = v;
        }
        if let Some(v) = patch.lease_ids {
            merged.lease_ids = v;
        }
        if let Some(v) = patch.model {
            merged.model = v;
        }
        if let Some(v) = patch.effort {
            merged.effort = v;
        }
        merged.agent_id = agent_id.to_string();
        self.jobs.insert(agent_id.to_string(), merged.clone());
        self.persist(&merged);
        Some(merged)
    }

    /// `setResult(agentId, capsule)` — store result + adopt its status, persist.
    pub fn set_result(&mut self, agent_id: &str, capsule: HandbackCapsule) {
        let job = match self.jobs.get(agent_id) {
            Some(j) => j.clone(),
            None => return,
        };
        let mut merged = job;
        merged.status = capsule.status.clone();
        merged.result = Some(capsule);
        self.jobs.insert(agent_id.to_string(), merged.clone());
        self.persist(&merged);
    }

    /// `collectResult(agentId)` — pull a terminal result exactly once (CAS).
    pub fn collect_result(&mut self, agent_id: &str) -> CollectResult {
        let job = match self.jobs.get(agent_id) {
            Some(j) => j.clone(),
            None => {
                return CollectResult {
                    status: "crashed".to_string(),
                    result: None,
                    collected: false,
                }
            }
        };
        if !is_terminal(&job.status) {
            return CollectResult {
                status: job.status,
                result: None,
                collected: false,
            };
        }
        if job.collected {
            return CollectResult {
                status: job.status,
                result: None,
                collected: true,
            };
        }
        let mut merged = job.clone();
        merged.collected = true;
        self.jobs.insert(agent_id.to_string(), merged.clone());
        self.persist(&merged);
        CollectResult {
            status: job.status,
            result: job.result,
            collected: true,
        }
    }

    /// `firstUncollected()` — first terminal-uncollected job, if any.
    pub fn first_uncollected(&self) -> Option<SubagentJob> {
        self.jobs
            .values()
            .find(|j| is_terminal(&j.status) && !j.collected)
            .cloned()
    }

    /// `findByTask(task)` — running/pending/spawning/done job with an identical
    /// task (dedup at spawn).
    pub fn find_by_task(&self, task: &str) -> Option<SubagentJob> {
        self.jobs
            .values()
            .find(|j| {
                j.task == task
                    && matches!(j.status.as_str(), "running" | "pending" | "spawning" | "done")
            })
            .cloned()
    }

    /// `spawnCount()` — total jobs ever minted this session.
    pub fn spawn_count(&self) -> u64 {
        self.counter
    }
}

/// Parse `^a(\d+)$` -> the numeric suffix.
fn parse_an_id(id: &str) -> Option<u64> {
    let rest = id.strip_prefix('a')?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse::<u64>().ok()
}

/// JS `str.slice(0, n)` over UTF-16 code units (used for the 60-char label cut).
fn js_slice_chars(s: &str, n: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    if units.len() <= n {
        return s.to_string();
    }
    String::from_utf16_lossy(&units[..n])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(root: &Path) -> RepoAgentConfig {
        RepoAgentConfig {
            root: root.to_string_lossy().to_string(),
            home_root: root.to_string_lossy().to_string(),
            current_project: None,
            sandbox_mode: "danger-full-access".to_string(),
            approval_policy: "never".to_string(),
            approvals_reviewer: "user".to_string(),
            writable_roots: Vec::new(),
            auto_compact: AutoCompactConfig {
                enabled: true,
                event_soft_limit: 28,
                event_hard_limit: 64,
                bytes_soft_limit: 220_000,
                hot_event_count: 12,
                hot_file_count: 12,
                capsule_budget_chars: 12_000,
                return_capsule_every_n_events: 10,
            },
            max_subagents: 2.0,
            max_subagent_spawns: 12.0,
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "chimera-state-{tag}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    /// Fix 5: started pings are ephemeral — visible on the /events cursor view,
    /// NEVER in events.jsonl / state.json / the snapshot-feeding recent_events.
    #[test]
    fn ephemeral_events_are_not_persisted() {
        let root = temp_root("ephemeral");
        let mut state = RepoState::new(test_config(&root)).unwrap();

        state.ephemeral_event(
            EventInput {
                kind: Some("tool_started".to_string()),
                ..EventInput::new("", true, "started repo_glob")
            },
            None,
        );
        state.event(EventInput::new("repo_glob", true, "3 files"), None);

        // The SSE view sees both, ordered by seq (started ping first).
        let feed = state.recent_events_since(0, None);
        assert_eq!(feed.len(), 2);
        assert_eq!(feed[0].kind.as_deref(), Some("tool_started"));
        assert_eq!(feed[1].tool, "repo_glob");

        // The durable views see only the completion event.
        assert_eq!(state.recent_events(10).len(), 1);
        let log = fs::read_to_string(&state.events_path).unwrap();
        assert_eq!(log.lines().count(), 1, "events.jsonl must hold only the completion");
        assert!(!log.contains("tool_started"), "started ping leaked into events.jsonl");
        let saved = fs::read_to_string(&state.state_path).unwrap();
        assert!(!saved.contains("tool_started"), "started ping leaked into state.json");

        let _ = fs::remove_dir_all(&root);
    }

    /// Fix 6: a corrupt state.json is preserved as state.json.corrupt-<ts> (and
    /// the event log rebuild still proceeds) instead of being silently replaced.
    #[test]
    fn corrupt_state_json_is_backed_up_before_rebuild() {
        let root = temp_root("corrupt");
        // Seed a real event so both files exist and the log has one line.
        {
            let mut state = RepoState::new(test_config(&root)).unwrap();
            state.event(EventInput::new("repo_status", true, "clean"), None);
        }
        let state_path = root.join(".repo-agent-mcp").join("state.json");
        fs::write(&state_path, "{ this is not json").unwrap();

        let state = RepoState::new(test_config(&root)).unwrap();
        // Events came back from events.jsonl …
        let events = state.recent_events(10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].tool, "repo_status");
        // … and the corrupt original was preserved, not destroyed.
        let dir = root.join(".repo-agent-mcp");
        let backups: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("state.json.corrupt-"))
            .collect();
        assert_eq!(backups.len(), 1, "expected one corrupt backup, got {backups:?}");
        let backup_body = fs::read_to_string(dir.join(&backups[0])).unwrap();
        assert_eq!(backup_body, "{ this is not json");

        let _ = fs::remove_dir_all(&root);
    }

    /// The bind_root rotation rewrites an oversized events.jsonl down to the
    /// rebuild tail window (hard_limit*2 parseable lines).
    #[test]
    fn oversized_events_log_is_rotated_on_bind() {
        let root = temp_root("rotate");
        // Build a >5MB log of valid event lines via one seeded event repeated.
        {
            let mut state = RepoState::new(test_config(&root)).unwrap();
            let evt = state.event(EventInput::new("repo_glob", true, "x".repeat(2_000)), None);
            let line = format!("{}\n", serde_json::to_string(&evt).unwrap());
            let mut body = String::new();
            while body.len() <= 5 * 1024 * 1024 {
                body.push_str(&line);
            }
            fs::write(&state.events_path, body).unwrap();
            // Remove state.json so the rebind rebuilds from the (rotated) log.
            let _ = fs::remove_file(&state.state_path);
        }
        let state = RepoState::new(test_config(&root)).unwrap();
        let log = fs::read_to_string(&state.events_path).unwrap();
        let hard2 = state.config.auto_compact.event_hard_limit * 2;
        assert_eq!(log.lines().count(), hard2, "rotated log keeps the rebuild window");
        assert!(
            fs::metadata(&state.events_path).unwrap().len() < 5 * 1024 * 1024,
            "rotated log must be small again"
        );
        assert_eq!(state.recent_events(1000).len(), hard2);

        let _ = fs::remove_dir_all(&root);
    }
}
