//! cyrus-setup — the engine behind "pick #4 and have it just work".
//!
//! One call, [`ensure_all`], converges the machine onto a working stack:
//!
//!   1. secrets    — per-install bearer token (+ optional JWT signing key)
//!   2. chrome     — CDP endpoint up + a logged-in chatgpt.com tab
//!   3. tunnel     — a public HTTPS URL for chimera (ngrok-static preferred for
//!                   a permanent URL; cloudflared named/quick otherwise)
//!   4. stack      — cyrus-chimera (the MCP connector server) + cyrus-lipsync
//!                   (the /v1/responses server codex talks to, historically
//!                   "the shim"), spawned with the load-bearing env
//!                   (`SHIM_CONDUCTOR=1`, `CHIMERA_RELAY_URL`, ...)
//!   5. connector  — the ChatGPT MCP connector, created or reused via the
//!                   page-context `cyrus_connect.js` flow (no UI automation,
//!                   no password page: loopback-armed nonce auto-issue)
//!   6. codex      — `[model_providers.shadow]` written to ${CODEX_HOME}/config.toml
//!
//! Every step is idempotent: re-running verifies and repairs instead of
//! duplicating. Progress streams as [`SetupEvent`]s over an mpsc channel so a
//! TUI can render it live; the headless `cyrus` binary prints the same events as
//! guided lines. [`diagnose`] gives the same per-component view for `cyrus check`.

pub mod chrome;
pub mod codex_config;
pub mod connector;
pub mod embedded;
pub mod secrets;
pub mod stack;
pub mod state;
pub mod tunnel;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

/// The page-context connector driver, embedded verbatim (the artifact proven
/// live against chatgpt.com on 2026-06-10).
pub const CYRUS_CONNECT_JS: &str = include_str!("../assets/cyrus_connect.js");

/// Default ports — the production wiring validated live.
pub const DEFAULT_CHIMERA_PORT: u16 = 8787;
pub const DEFAULT_SHIM_PORT: u16 = 8765;
pub const DEFAULT_CDP_PORT: u16 = 9222;

/// Read a `u16` port from `var`, falling back to `default` when unset, empty, or
/// unparseable (a bogus value shouldn't silently land on port 0).
fn env_port(var: &str, default: u16) -> u16 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<u16>().ok())
        .filter(|&p| p != 0)
        .unwrap_or(default)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Secrets,
    Chrome,
    Tunnel,
    Stack,
    Connector,
    CodexConfig,
}

impl Step {
    /// All steps in run order — used by the guided intro and the doctor.
    pub const ALL: [Step; 6] = [
        Step::Secrets,
        Step::Chrome,
        Step::Tunnel,
        Step::Stack,
        Step::Connector,
        Step::CodexConfig,
    ];

    /// The short headline shown while the step runs.
    pub fn label(&self) -> &'static str {
        match self {
            Step::Secrets => "Preparing credentials",
            Step::Chrome => "Connecting to Chrome (your ChatGPT login)",
            Step::Tunnel => "Opening the public tunnel (this can take a minute or two)",
            Step::Stack => "Starting the local servers",
            Step::Connector => "Wiring the ChatGPT connector (this can take a minute or two)",
            Step::CodexConfig => "Configuring codex",
        }
    }

    /// One plain-language line on what the step is for — the guided intro.
    pub fn blurb(&self) -> &'static str {
        match self {
            Step::Secrets => "a per-install token that locks the local servers to you",
            Step::Chrome => "the one manual step: a logged-in chatgpt.com tab over CDP",
            Step::Tunnel => "a public HTTPS URL so ChatGPT's connector can reach you",
            Step::Stack => "chimera (the connector server) + lipsync (the model bridge)",
            Step::Connector => "registers the MCP connector on your ChatGPT account",
            Step::CodexConfig => "points codex at cyrus — injected at launch, nothing written to your config",
        }
    }

    /// What to try when this step fails — the actionable hint in diagnostics.
    pub fn remedy(&self) -> &'static str {
        match self {
            Step::Secrets => {
                "Check that ~/.cyrus is writable (it holds secrets.json)."
            }
            Step::Chrome => {
                "Make sure Google Chrome is installed. cyrus opens it with a dedicated \
                 profile and waits for you to log in to ChatGPT; finish that login in the \
                 window it opened. If Chrome is already running on the debugging port, it \
                 must be reachable at the CDP host/port."
            }
            Step::Tunnel => {
                "Install a tunnel: ngrok (recommended — set CYRUS_NGROK_DOMAIN to a free \
                 static domain for a permanent URL) or cloudflared. See docs/TUNNELING.md."
            }
            Step::Stack => {
                "A local server (chimera or lipsync) didn't come up, or its port is held \
                 by another process. Free the port and re-run setup."
            }
            Step::Connector => {
                "The ChatGPT tab must stay logged in, and the tunnel must reach chimera. \
                 If the connector got into a bad state, re-running setup recreates it."
            }
            Step::CodexConfig => {
                "Check that ${CODEX_HOME:-~/.codex}/config.toml is writable and valid TOML."
            }
        }
    }

    /// Log file basenames (under ~/.cyrus/logs/) worth tailing if this step
    /// fails. Empty when the step writes no logs of its own.
    pub fn log_files(&self) -> &'static [&'static str] {
        match self {
            Step::Tunnel => &["cloudflared.log", "cloudflared-quick.log", "ngrok.log"],
            Step::Stack => &["chimera.log", "lipsync.log"],
            _ => &[],
        }
    }
}

/// One probed component in a [`Diagnosis`].
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Display name, e.g. "chimera" or "Tunnel".
    pub name: String,
    pub ok: bool,
    /// A short status line: the URL/model when up, the reason when down.
    pub detail: String,
}

/// The result of [`diagnose`] — a per-component health snapshot for `cyrus check`.
#[derive(Debug, Clone)]
pub struct Diagnosis {
    pub components: Vec<ComponentHealth>,
}

impl Diagnosis {
    /// True iff every probed component is up.
    pub fn healthy(&self) -> bool {
        self.components.iter().all(|c| c.ok)
    }
}

/// Progress events for a front-end. The engine emits StepStarted/StepDone in
/// strict order; NeedsUserAction can interleave (the engine keeps polling on
/// its own — the front-end only has to display the instruction).
#[derive(Debug, Clone)]
pub enum SetupEvent {
    StepStarted { step: Step },
    /// `detail` is a short human line ("reused link link_ab12, 34 tools").
    StepDone { step: Step, detail: String },
    /// The engine is blocked on the human (e.g. "log in to ChatGPT in the
    /// Chrome window"). It keeps polling; no front-end ack needed.
    NeedsUserAction { step: Step, instruction: String },
    /// Cleared a previously-emitted NeedsUserAction.
    UserActionResolved { step: Step },
}

/// Which tunnel lane to bring up. The TUI picks this (was an env var); the
/// engine routes on it instead of auto-detecting when it's not `Auto`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TunnelChoice {
    /// Precedence ladder (env var / config presence). Backward-compatible default.
    #[default]
    Auto,
    /// Zero-config cloudflared quick tunnel. Instant, but the URL churns on
    /// restart (the connector then needs re-registering).
    Quick,
    /// ngrok with a reserved static domain — a permanent URL from one free
    /// signup. `domain` comes from the flag or `CYRUS_NGROK_DOMAIN`.
    Ngrok { domain: Option<String> },
    /// cloudflared named tunnel — you own a domain (`~/.cloudflared/config.yml`).
    Named,
}

#[derive(Debug, Clone)]
pub struct SetupOptions {
    /// Repo root chimera serves (`--repo`). The cwd codex runs in.
    pub repo_root: PathBuf,
    /// Directory holding the `cyrus` binary the stack respawns itself from
    /// (busybox: chimera + lipsync are `cyrus chimera` / `cyrus lipsync`
    /// subcommands). Defaults to the running executable, overridable via
    /// `CYRUS_BIN_DIR` for unusual layouts.
    pub bin_dir: Option<PathBuf>,
    pub chimera_port: u16,
    pub shim_port: u16,
    pub cdp_host: String,
    pub cdp_port: u16,
    /// Model lane + effort for the shim (production: gpt-5-5-thinking / max).
    pub model: String,
    pub effort: String,
    /// Connector display name on the ChatGPT side.
    pub connector_name: String,
    /// Which tunnel lane to bring up (`--tunnel`). `Auto` = the precedence ladder.
    pub tunnel: TunnelChoice,
}

impl SetupOptions {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        SetupOptions {
            repo_root: repo_root.into(),
            bin_dir: None,
            // Ports are read from the environment so EVERY entry point agrees:
            // `cyrus setup`, `cyrus check`, AND the bare-`cyrus` codex launch all
            // build options through here, and the launch path uses these ports to
            // point codex at lipsync + to repair/tear down chimera. A flag on
            // `setup` alone would desync the launch path, so the env var is the
            // load-bearing override (the `--chimera-port`/`--shim-port` flags just
            // set it for the current process).
            chimera_port: env_port("CYRUS_CHIMERA_PORT", DEFAULT_CHIMERA_PORT),
            shim_port: env_port("CYRUS_SHIM_PORT", DEFAULT_SHIM_PORT),
            cdp_host: "127.0.0.1".to_string(),
            cdp_port: env_port("CYRUS_CDP_PORT", DEFAULT_CDP_PORT),
            model: "gpt-5-5-thinking".to_string(),
            effort: "max".to_string(),
            connector_name: "repo".to_string(),
            tunnel: TunnelChoice::Auto,
        }
    }

    /// `~/.cyrus` — secrets, logs, and the dedicated Chrome profile live here.
    /// `CYRUS_HOME` is trimmed: a trailing space (trivially introduced by a
    /// `set CYRUS_HOME=… &` cmd one-liner) otherwise yields a bogus path that
    /// silently breaks embedded extraction and connector lookup.
    pub fn cyrus_home(&self) -> PathBuf {
        if let Ok(h) = std::env::var("CYRUS_HOME") {
            let h = h.trim();
            if !h.is_empty() {
                return PathBuf::from(h);
            }
        }
        home_dir().join(".cyrus")
    }

    pub fn resolve_bin_dir(&self) -> PathBuf {
        if let Some(d) = &self.bin_dir {
            return d.clone();
        }
        if let Ok(d) = std::env::var("CYRUS_BIN_DIR") {
            if !d.is_empty() {
                return PathBuf::from(d);
            }
        }
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Path to the `cyrus` binary the stack respawns itself from. With no
    /// override this is the running executable (busybox: the chimera/lipsync
    /// servers are `cyrus chimera` / `cyrus lipsync` subcommands of this same
    /// exe). An explicit `bin_dir` / `CYRUS_BIN_DIR` instead points at the
    /// directory holding `cyrus(.exe)`.
    pub fn cyrus_exe(&self) -> PathBuf {
        if self.bin_dir.is_none() && std::env::var_os("CYRUS_BIN_DIR").is_none() {
            if let Ok(p) = std::env::current_exe() {
                return p;
            }
        }
        let name = if cfg!(windows) { "cyrus.exe" } else { "cyrus" };
        self.resolve_bin_dir().join(name)
    }
}

/// What a successful run converged onto.
#[derive(Debug, Clone)]
pub struct SetupOutcome {
    pub public_url: String,
    pub shim_base_url: String,
    pub connector_id: String,
    pub link_id: String,
    pub tool_count: usize,
    /// True when every step verified an existing resource instead of creating.
    pub fully_reused: bool,
}

/// Fast pass/fail for "is the stack already serving?" — the every-launch path.
/// Checks the two local servers AND that the recorded public tunnel still
/// reaches our chimera (a dead tunnel means ChatGPT can't call the connector
/// even though the local servers are up). Single-shot, no retry loop; does NOT
/// touch Chrome or re-drive the connector — that's `ensure_all`/`repair_stack`.
pub async fn health_check(opts: &SetupOptions) -> bool {
    if stack::chimera_alive(opts).await.is_none() || !stack::lipsync_alive(opts).await {
        return false;
    }
    // Enrolled implies a recorded connector; its public URL must still be live.
    match connector::recorded_connector(opts) {
        Some(rec) => tunnel::tunnel_alive(rec.mcp_url.trim_end_matches("/mcp")).await,
        None => false,
    }
}

/// Surgical self-heal for the launch path: bring the local stack (chimera +
/// lipsync) back up against the recorded tunnel URL, restarting the tunnel
/// first if it's down. Deliberately does NOT touch Chrome or re-drive the
/// connector — those need a logged-in tab and belong to full `cyrus setup`. If
/// the tunnel came back on a DIFFERENT URL (quick-tunnel churn), the connector
/// is now stale and only setup can re-register it, so we say so instead of
/// limping on.
pub async fn repair_stack(opts: &SetupOptions) -> anyhow::Result<()> {
    use anyhow::Context;
    let secrets = secrets::load_or_create(opts)?;
    let rec = connector::recorded_connector(opts)
        .context("cyrus is not set up (no recorded connector) — run `cyrus setup`")?;
    let public_url = rec.mcp_url.trim_end_matches("/mcp").to_string();

    if !tunnel::tunnel_alive(&public_url).await {
        let out = tunnel::ensure_tunnel(opts).await?;
        anyhow::ensure!(
            out.public_url.trim_end_matches('/') == public_url,
            "tunnel came back on a new URL ({} -> {}) — run `cyrus setup` to re-register the connector",
            public_url,
            out.public_url
        );
    }

    stack::ensure_stack(opts, &secrets, &public_url).await?;
    Ok(())
}

/// Probe every component and return a per-component snapshot for `cyrus check`.
/// Read-only: contacts the local servers, the tunnel edge, and reads local
/// config/state — never launches Chrome or touches the ChatGPT connector.
pub async fn diagnose(opts: &SetupOptions) -> Diagnosis {
    let mut components = Vec::new();

    // Chrome / CDP endpoint.
    let cdp_up = chrome::cdp_alive(opts).await;
    components.push(ComponentHealth {
        name: "Chrome (CDP)".to_string(),
        ok: cdp_up,
        detail: if cdp_up {
            format!("up on {}:{}", opts.cdp_host, opts.cdp_port)
        } else {
            format!(
                "no DevTools endpoint on {}:{} (run `cyrus setup` to launch it)",
                opts.cdp_host, opts.cdp_port
            )
        },
    });

    // chimera (local).
    let chimera = stack::chimera_alive(opts).await;
    components.push(ComponentHealth {
        name: "chimera".to_string(),
        ok: chimera.is_some(),
        detail: match &chimera {
            Some(info) => format!("up on :{} · serving {}", opts.chimera_port, info.root),
            None => format!("down on :{}", opts.chimera_port),
        },
    });

    // lipsync (local).
    let lipsync = stack::lipsync_health(opts).await;
    components.push(ComponentHealth {
        name: "lipsync".to_string(),
        ok: lipsync.is_some(),
        detail: match &lipsync {
            Some(model) if !model.is_empty() => {
                format!("up on :{} · model {model}", opts.shim_port)
            }
            Some(_) => format!("up on :{}", opts.shim_port),
            None => format!("down on :{}", opts.shim_port),
        },
    });

    // Tunnel: read the recorded connector URL and probe the public edge.
    match connector::recorded_connector(opts) {
        Some(rec) => {
            let public = rec.mcp_url.trim_end_matches("/mcp");
            // `cyrus check` is a READ-ONLY doctor: use the single-shot probe
            // (≤6s), never `verify_through_tunnel`'s 180s setup retry loop — a
            // doctor must answer fast precisely when things are down, which is
            // exactly when that loop would otherwise hang for 3 minutes.
            let reaches = tunnel::tunnel_alive(public).await;
            components.push(ComponentHealth {
                name: "Tunnel".to_string(),
                ok: reaches,
                detail: if reaches {
                    format!("{public} reaches chimera")
                } else {
                    format!("{public} did not reach chimera (tunnel down or repointed)")
                },
            });
            components.push(ComponentHealth {
                name: "ChatGPT connector".to_string(),
                // We can't verify the remote connector without driving the tab;
                // having a recorded link + a reachable tunnel is the local proxy.
                ok: !rec.link_id.is_empty(),
                detail: if rec.link_id.is_empty() {
                    "recorded, but no link id (re-run setup to finish wiring)".to_string()
                } else {
                    format!("recorded · link {}", rec.link_id)
                },
            });
        }
        None => {
            components.push(ComponentHealth {
                name: "Tunnel".to_string(),
                ok: false,
                detail: "no connector recorded yet (run `cyrus setup`)".to_string(),
            });
            components.push(ComponentHealth {
                name: "ChatGPT connector".to_string(),
                ok: false,
                detail: "not set up yet (run `cyrus setup`)".to_string(),
            });
        }
    }

    // codex provider. The `shadow` provider is injected at launch (per-invocation
    // `-c`), never persisted — so "enrolled" is the recorded connector, not a
    // block in config.toml. A stale persisted block (from an older cyrus) is
    // worth flagging since it now just clutters the user's config.
    let want = format!("http://127.0.0.1:{}/v1", opts.shim_port);
    let enrolled = connector::recorded_connector(opts).is_some();
    let stale = codex_config::current_provider_base_url().is_some();
    components.push(ComponentHealth {
        name: "codex provider".to_string(),
        ok: enrolled,
        detail: if !enrolled {
            "not set up — run `cyrus setup`".to_string()
        } else if stale {
            format!("injected at launch ({want}) · stale block in config.toml — re-run setup to clean")
        } else {
            format!("injected at launch ({want})")
        },
    });

    Diagnosis { components }
}

/// Converge everything. Emits progress on `tx`; returns the outcome.
pub async fn ensure_all(
    opts: &SetupOptions,
    tx: &UnboundedSender<SetupEvent>,
) -> anyhow::Result<SetupOutcome> {
    let emit = |e: SetupEvent| {
        let _ = tx.send(e);
    };
    let mut fully_reused = true;

    emit(SetupEvent::StepStarted { step: Step::Secrets });
    let secrets = secrets::load_or_create(opts)?;
    emit(SetupEvent::StepDone {
        step: Step::Secrets,
        detail: if secrets.created {
            "generated new install secrets".to_string()
        } else {
            "reusing install secrets".to_string()
        },
    });
    fully_reused &= !secrets.created;

    emit(SetupEvent::StepStarted { step: Step::Chrome });
    let chrome = chrome::ensure_chrome(opts, tx).await?;
    emit(SetupEvent::StepDone {
        step: Step::Chrome,
        detail: format!(
            "{} (tab {})",
            if chrome.launched { "launched Chrome" } else { "reusing running Chrome" },
            &chrome.login_target_id[..8.min(chrome.login_target_id.len())]
        ),
    });
    fully_reused &= !chrome.launched;

    emit(SetupEvent::StepStarted { step: Step::Tunnel });
    let tunnel = tunnel::ensure_tunnel(opts).await?;
    emit(SetupEvent::StepDone {
        step: Step::Tunnel,
        detail: format!(
            "{} -> {}",
            if tunnel.started { "started cloudflared" } else { "reusing tunnel" },
            tunnel.public_url
        ),
    });
    fully_reused &= !tunnel.started;

    emit(SetupEvent::StepStarted { step: Step::Stack });
    let stack = stack::ensure_stack(opts, &secrets, &tunnel.public_url).await?;
    emit(SetupEvent::StepDone {
        step: Step::Stack,
        detail: stack.detail.clone(),
    });
    fully_reused &= stack.reused;

    // Start the connector step BEFORE the tunnel verify: a fresh quick tunnel
    // can take a couple of minutes to become edge-routable, and we'd rather show
    // "Wiring the ChatGPT connector" during that wait than freeze on the last
    // step looking hung. Verify reaches OUR chimera before we touch ChatGPT.
    emit(SetupEvent::StepStarted { step: Step::Connector });
    tunnel::verify_through_tunnel(&tunnel.public_url).await?;
    let connector = connector::ensure_connector(opts, &secrets, &tunnel.public_url, &chrome).await?;
    emit(SetupEvent::StepDone {
        step: Step::Connector,
        detail: format!(
            "{} ({} tools)",
            if connector.reused { "reused link" } else { "created connector" },
            connector.tool_count
        ),
    });
    fully_reused &= connector.reused;

    emit(SetupEvent::StepStarted { step: Step::CodexConfig });
    let shim_base_url = format!("http://127.0.0.1:{}/v1", opts.shim_port);
    // Nothing is persisted to the user's codex config: the wrapper injects the
    // whole `shadow` provider as per-invocation `-c` overrides at launch, so a
    // plain `codex` stays pristine. We DO strip any block an older cyrus wrote,
    // and record the repair hints (tunnel lane + repo) cyrus needs to self-heal
    // the stack on a later launch without asking again.
    let removed = codex_config::remove_shadow_provider()?;
    state::save(
        opts,
        &state::CyrusState {
            repo_root: Some(opts.repo_root.clone()),
            tunnel: opts.tunnel.clone(),
        },
    );
    emit(SetupEvent::StepDone {
        step: Step::CodexConfig,
        detail: if removed {
            "provider injected at launch (removed the old persisted block)".to_string()
        } else {
            "provider injected at launch (nothing written to your codex config)".to_string()
        },
    });

    Ok(SetupOutcome {
        public_url: tunnel.public_url,
        shim_base_url,
        connector_id: connector.connector_id,
        link_id: connector.link_id,
        tool_count: connector.tool_count,
        fully_reused,
    })
}

/// Home directory without an extra dependency: USERPROFILE on Windows, HOME
/// elsewhere.
pub(crate) fn home_dir() -> PathBuf {
    #[cfg(windows)]
    {
        if let Ok(p) = std::env::var("USERPROFILE") {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
    }
    if let Ok(p) = std::env::var("HOME") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    PathBuf::from(".")
}
