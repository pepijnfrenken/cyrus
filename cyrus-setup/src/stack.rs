//! The two local servers: cyrus-chimera (the MCP connector server ChatGPT
//! calls) and cyrus-lipsync (the /v1/responses server codex calls — historically
//! "the shim"). Spawned as detached children with the load-bearing env:
//!
//!   chimera:  REPO_AGENT_TOKEN, REPO_AGENT_PUBLIC_URL, JWT_SIGNING_KEY,
//!             CHIMERA_RELAY_URL=http://127.0.0.1:<lipsync>/control/toolcall
//!   lipsync:  SHIM_CONDUCTOR=1 (the env var keeps its legacy name; without it
//!             tool calls hang for the full 300s hold — responses_shim.py:744
//!             semantics), REPO_AGENT_URL, CDP_HOST/CDP_PORT
//!
//! Reuse-first: a healthy instance with matching identity (repo root + public
//! URL for chimera, model for lipsync) is kept; OUR stale instance on the port
//! is killed and respawned; a foreign service on the port is an error.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use serde_json::Value;

use crate::secrets::Secrets;
use crate::SetupOptions;

pub struct StackOutcome {
    pub reused: bool,
    pub detail: String,
}

#[derive(Debug)]
pub struct ChimeraInfo {
    pub root: String,
}

fn norm_path(p: &str) -> String {
    p.trim_start_matches(r"\\?\")
        .replace('\\', "/")
        .to_ascii_lowercase()
        .trim_end_matches('/')
        .to_string()
}

/// GET / on the chimera port; Some(info) iff it identifies as repo-agent-mcp.
pub async fn chimera_alive(opts: &SetupOptions) -> Option<ChimeraInfo> {
    let url = format!("http://127.0.0.1:{}/", opts.chimera_port);
    let v: Value = reqwest::Client::new()
        .get(url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    if v.get("name").and_then(Value::as_str) == Some("repo-agent-mcp") {
        Some(ChimeraInfo {
            root: v.get("root").and_then(Value::as_str).unwrap_or("").to_string(),
        })
    } else {
        None
    }
}

/// The public URL the running chimera was configured with, read off its
/// protected-resource metadata (loopback).
async fn chimera_public_url(opts: &SetupOptions) -> Option<String> {
    let url = format!(
        "http://127.0.0.1:{}/.well-known/oauth-protected-resource",
        opts.chimera_port
    );
    let v: Value = reqwest::Client::new()
        .get(url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    v.get("resource")
        .and_then(Value::as_str)
        .map(|s| s.trim_end_matches('/').to_string())
}

/// GET /health on the lipsync port; `Some(model)` iff it answers the lipsync
/// shape, else `None`. (lipsync = the `/v1/responses` server codex talks to;
/// historically "the shim".)
pub async fn lipsync_health(opts: &SetupOptions) -> Option<String> {
    let url = format!("http://127.0.0.1:{}/health", opts.shim_port);
    let v: Value = reqwest::Client::new()
        .get(url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    if v.get("ok").and_then(Value::as_bool) == Some(true) {
        Some(
            v.get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        )
    } else {
        None
    }
}

/// `true` iff the lipsync server is up — the health gate used everywhere we just
/// need a yes/no.
pub async fn lipsync_alive(opts: &SetupOptions) -> bool {
    lipsync_health(opts).await.is_some()
}

/// The `cyrus` binary's file name (lowercased) — the image the stack children
/// run under. Busybox: chimera/lipsync are `cyrus chimera` / `cyrus lipsync`, so
/// a child's process image is `cyrus(.exe)`, not a separate per-server exe. Used
/// to gate `kill_port_owner` so we only ever kill our own image.
fn cyrus_image(opts: &SetupOptions) -> String {
    opts.cyrus_exe()
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_else(|| "cyrus".to_string())
}

fn spawn_detached(mut cmd: std::process::Command, log: PathBuf) -> anyhow::Result<()> {
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let out = std::fs::File::create(&log)?;
    let err = out.try_clone()?;
    cmd.stdout(out).stderr(err).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0200); // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    }
    cmd.spawn().context("spawn stack binary")?;
    Ok(())
}

/// Kill whatever owns `port` — ONLY if its image name matches `expect_image`
/// (we never kill a foreign service; that's an error instead).
#[cfg(windows)]
fn kill_port_owner(port: u16, expect_image: &str) -> anyhow::Result<()> {
    let out = std::process::Command::new("netstat").args(["-ano"]).output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let needle = format!(":{port}");
    let mut pid: Option<u32> = None;
    for line in text.lines() {
        if line.contains("LISTENING") && line.contains(&needle) {
            // local address column must end with :port (avoid :8787x matches)
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() >= 5 && cols[1].ends_with(&needle) {
                pid = cols[4].parse().ok();
                break;
            }
        }
    }
    let Some(pid) = pid else { return Ok(()) };
    let tl = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()?;
    let tl_text = String::from_utf8_lossy(&tl.stdout).to_ascii_lowercase();
    anyhow::ensure!(
        tl_text.contains(&expect_image.to_ascii_lowercase()),
        "port {port} is held by a foreign process (pid {pid}) — free it and re-run setup"
    );
    std::process::Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .output()?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    Ok(())
}

// macOS/Linux counterpart: lsof finds the listener, ps confirms it's OUR image
// before SIGKILL. Empty port -> no-op (Ok), foreign image -> error. No lsof on
// PATH -> no-op and let the spawn/poll below surface a real failure.
#[cfg(not(windows))]
fn kill_port_owner(port: u16, expect_image: &str) -> anyhow::Result<()> {
    let out = match std::process::Command::new("lsof")
        .args(["-ti", &format!("tcp:{port}"), "-sTCP:LISTEN"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Ok(()),
    };
    let pids: Vec<u32> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    for pid in pids {
        // ps comm is the full path on macOS, the (possibly truncated) name on
        // Linux — match on the basename either way.
        let comm = std::process::Command::new("ps")
            .args(["-o", "comm=", "-p", &pid.to_string()])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_ascii_lowercase())
            .unwrap_or_default();
        let image = comm.rsplit('/').next().unwrap_or(comm.as_str());
        anyhow::ensure!(
            image.contains(&expect_image.to_ascii_lowercase()),
            "port {port} is held by a foreign process (pid {pid}) — free it and re-run setup"
        );
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .output();
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    Ok(())
}

/// Best-effort teardown of OUR chimera + lipsync (kill the cyrus-image owner of
/// each local port). Called when the codex session that wanted them exits: the
/// local servers are cheap to respawn and shouldn't outlive the session holding
/// the cyrus binary open. The TUNNEL is deliberately NOT touched — its public
/// URL (and the connector registered against it) must survive across relaunches,
/// or every restart churns a fresh quick-tunnel URL and strands the connector.
pub fn stop_servers(opts: &SetupOptions) {
    let image = cyrus_image(opts);
    for port in [opts.chimera_port, opts.shim_port] {
        // kill_port_owner refuses foreign images and no-ops an empty port, so a
        // bare `let _ =` is the right best-effort: only ever kills our own.
        let _ = kill_port_owner(port, &image);
    }
    // Defense-in-depth: also close the EPHEMERAL quick tunnel cyrus spawned (a
    // disposable public socket). With chimera already dead there's nothing behind
    // it, but a throwaway socket shouldn't linger. A NAMED/user tunnel is left
    // running — it's the user's own auth'd infra with a stable URL.
    kill_quick_tunnel(opts);
}

/// Kill the cyrus-spawned QUICK (ephemeral trycloudflare) tunnel via the pid it
/// recorded at spawn, guarded against pid-reuse by confirming the image is still
/// cloudflared. No pid file (or a named tunnel) -> no-op.
#[cfg(windows)]
pub(crate) fn kill_quick_tunnel(opts: &SetupOptions) {
    let pid_file = opts.cyrus_home().join("quick-tunnel.pid");
    let Ok(s) = std::fs::read_to_string(&pid_file) else {
        return;
    };
    let Ok(pid) = s.trim().parse::<u32>() else {
        return;
    };
    if let Ok(out) = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
    {
        if String::from_utf8_lossy(&out.stdout)
            .to_ascii_lowercase()
            .contains("cloudflared")
        {
            let _ = std::process::Command::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .output();
        }
    }
    let _ = std::fs::remove_file(&pid_file);
}

#[cfg(not(windows))]
pub(crate) fn kill_quick_tunnel(opts: &SetupOptions) {
    let pid_file = opts.cyrus_home().join("quick-tunnel.pid");
    let Ok(s) = std::fs::read_to_string(&pid_file) else {
        return;
    };
    let Ok(pid) = s.trim().parse::<u32>() else {
        return;
    };
    let is_cf = std::process::Command::new("ps")
        .args(["-o", "comm=", "-p", &pid.to_string()])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .to_ascii_lowercase()
                .contains("cloudflared")
        })
        .unwrap_or(false);
    if is_cf {
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .output();
    }
    let _ = std::fs::remove_file(&pid_file);
}

async fn poll_until<F, Fut>(what: &str, secs: u64, mut f: F) -> anyhow::Result<()>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    loop {
        if f().await {
            return Ok(());
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "{what} did not come up within {secs}s (logs in ~/.cyrus/logs/)"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

pub async fn ensure_stack(
    opts: &SetupOptions,
    secrets: &Secrets,
    public_url: &str,
) -> anyhow::Result<StackOutcome> {
    let want_root = norm_path(
        &std::fs::canonicalize(&opts.repo_root)
            .unwrap_or_else(|_| opts.repo_root.clone())
            .to_string_lossy(),
    );
    let mut reused_chimera = false;
    let mut reused_lipsync = false;

    // ---- chimera ----
    let healthy = match chimera_alive(opts).await {
        Some(info) => {
            let root_ok = norm_path(&info.root) == want_root;
            let url_ok = chimera_public_url(opts).await.as_deref()
                == Some(public_url.trim_end_matches('/'));
            root_ok && url_ok
        }
        None => false,
    };
    if healthy {
        reused_chimera = true;
    } else {
        // Whatever is there is stale (ours) or foreign — kill_port_owner
        // refuses foreign images, so this is safe.
        kill_port_owner(opts.chimera_port, &cyrus_image(opts))?;
        let exe = opts.cyrus_exe();
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("chimera")
            .arg("--http")
            .arg("--port")
            .arg(opts.chimera_port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--repo")
            .arg(&opts.repo_root)
            .env("REPO_AGENT_TOKEN", &secrets.bearer_token)
            .env("REPO_AGENT_PUBLIC_URL", public_url)
            .env(
                "CHIMERA_RELAY_URL",
                format!("http://127.0.0.1:{}/control/toolcall", opts.shim_port),
            )
            .current_dir(&opts.repo_root);
        if let Some(k) = &secrets.jwt_signing_key {
            cmd.env("JWT_SIGNING_KEY", k);
        }
        spawn_detached(cmd, opts.cyrus_home().join("logs/chimera.log"))?;
        poll_until("chimera", 20, || async {
            chimera_alive(opts).await.is_some()
        })
        .await?;
    }

    // ---- lipsync ----
    if lipsync_alive(opts).await {
        reused_lipsync = true;
    } else {
        kill_port_owner(opts.shim_port, &cyrus_image(opts))?;
        let exe = opts.cyrus_exe();
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("lipsync")
            .arg("--port")
            .arg(opts.shim_port.to_string())
            .arg("--model")
            .arg(&opts.model)
            .arg("--effort")
            .arg(&opts.effort)
            // lazy: setup itself doesn't need a MAIN tab yet, and a boot
            // failure here would block setup on a browser hiccup. The first
            // codex turn boots it (and logs loudly if that fails).
            .arg("--lazy")
            .env("SHIM_CONDUCTOR", "1")
            .env(
                "REPO_AGENT_URL",
                format!("http://127.0.0.1:{}", opts.chimera_port),
            )
            .env("CDP_HOST", &opts.cdp_host)
            .env("CDP_PORT", opts.cdp_port.to_string());
        spawn_detached(cmd, opts.cyrus_home().join("logs/lipsync.log"))?;
        poll_until("lipsync", 20, || async { lipsync_alive(opts).await }).await?;
    }

    let detail = match (reused_chimera, reused_lipsync) {
        (true, true) => "reusing running chimera + lipsync".to_string(),
        (true, false) => "reusing chimera; started lipsync".to_string(),
        (false, true) => "started chimera; reusing lipsync".to_string(),
        (false, false) => "started chimera + lipsync".to_string(),
    };
    Ok(StackOutcome {
        reused: reused_chimera && reused_lipsync,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_path_unifies_separators_and_case() {
        assert_eq!(
            norm_path(r"\\?\C:\Users\X\Repo\"),
            norm_path("c:/users/x/repo")
        );
    }
}
