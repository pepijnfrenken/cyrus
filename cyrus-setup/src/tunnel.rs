//! cloudflared bring-up. Two lanes:
//!
//! * **Named tunnel (preferred, stable URL):** `~/.cloudflared/config.yml`
//!   exists with an ingress hostname — reuse the hostname, repoint the local
//!   service at our chimera port if needed (backup first), make sure the
//!   tunnel process is registered.
//! * **Quick tunnel (zero-config fallback):** `cloudflared tunnel --url ...`
//!   and parse the ephemeral trycloudflare.com URL. URL churn is fine: the
//!   connector step dedupes by mcp_url and recreates stale connectors.
//!
//! Probe trick: Cloudflare's edge answers **530** for a hostname whose tunnel
//! is down, and anything else (200/4xx/502) once a tunnel connection is
//! registered — so "GET https://host/" distinguishes down vs up without
//! process inspection.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;

use crate::{home_dir, SetupOptions, TunnelChoice};

pub struct TunnelOutcome {
    /// `https://<hostname>` — no trailing slash.
    pub public_url: String,
    pub started: bool,
}

struct NamedTunnel {
    config_path: PathBuf,
    tunnel_id: String,
    hostname: String,
    /// Raw service value of the chosen rule, e.g. `http://localhost:9999` —
    /// kept verbatim so repoint can match the exact host the user wrote.
    service: String,
    service_port: Option<u16>,
}

fn cloudflared_dir() -> PathBuf {
    home_dir().join(".cloudflared")
}

fn cloudflared_exe_name() -> &'static str {
    if cfg!(windows) { "cloudflared.exe" } else { "cloudflared" }
}

fn find_cloudflared_exe() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CYRUS_CLOUDFLARED_EXE") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // Embedded in the single-binary build — extracted to ~/.cyrus/bin on first
    // use. Preferred over a system install so a release `cyrus.exe` needs nothing
    // else on the machine.
    if let Some(extracted) = crate::embedded::embedded_cloudflared_path() {
        return Some(extracted);
    }
    // Bundled next to the cyrus binary (the ship layout: cyrus.exe + codex.exe +
    // cloudflared.exe in one folder). Prefer it over a system install so the
    // bundle is self-contained and needs no separate cloudflared install.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(cloudflared_exe_name());
            if sibling.exists() {
                return Some(sibling);
            }
        }
    }
    #[cfg(windows)]
    {
        for base in [
            std::env::var("ProgramFiles(x86)").ok(),
            std::env::var("ProgramFiles").ok(),
        ]
        .into_iter()
        .flatten()
        {
            let p = PathBuf::from(base).join("cloudflared/cloudflared.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }
    // PATH lookup: spawnable by bare name?
    let probe = std::process::Command::new("cloudflared")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Some(PathBuf::from("cloudflared"));
    }
    None
}

/// Light-touch parse of cloudflared's config.yml — no YAML dependency. (The
/// file is machine-written and flat.) Selects the ingress rule cyrus should
/// drive against `chimera_port` (see `select_ingress`).
fn parse_named_tunnel(chimera_port: u16) -> Option<NamedTunnel> {
    let path = cloudflared_dir().join("config.yml");
    let text = std::fs::read_to_string(&path).ok()?;
    let (tunnel_id, hostname, service, service_port) = select_ingress(&text, chimera_port)?;
    Some(NamedTunnel {
        config_path: path,
        tunnel_id,
        hostname,
        service,
        service_port,
    })
}

/// Pick the ingress rule to drive: the one already pointing at `chimera_port`
/// if any, else the first hostname rule. Returns (tunnel_id, hostname, raw
/// service, port). Hostnames are paired with the service line that follows
/// them, so a multi-ingress config doesn't get rule 0's hostname stapled to
/// rule 1's service.
fn select_ingress(text: &str, chimera_port: u16) -> Option<(String, String, String, Option<u16>)> {
    let mut tunnel_id = None;
    let mut rules: Vec<(String, String, Option<u16>)> = Vec::new();
    let mut pending_host: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("tunnel:") {
            tunnel_id = Some(v.trim().to_string());
        } else if let Some(v) = t.strip_prefix("- hostname:") {
            pending_host = Some(v.trim().to_string());
        } else if let Some(v) = t.strip_prefix("service:") {
            // The catch-all `- service: http_status:404` keeps its `- ` prefix,
            // so it never matches here and has no pending host anyway.
            if let Some(host) = pending_host.take() {
                let service = v.trim().to_string();
                let port = service.rsplit(':').next().and_then(|p| p.parse().ok());
                rules.push((host, service, port));
            }
        }
    }
    let (hostname, service, port) = rules
        .iter()
        .find(|(_, _, p)| *p == Some(chimera_port))
        .or_else(|| rules.first())?;
    Some((tunnel_id?, hostname.clone(), service.clone(), *port))
}

/// Repoint the named tunnel's local service at our chimera port (backup first).
fn repoint_service(nt: &NamedTunnel, port: u16) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(&nt.config_path)?;
    let backup = nt.config_path.with_extension("yml.bak-cyrus");
    std::fs::write(&backup, &text)?;
    let from = format!("service: {}", nt.service);
    let to = format!("service: http://127.0.0.1:{port}");
    let updated = text.replace(&from, &to);
    anyhow::ensure!(
        updated != text,
        "could not find '{from}' in {} to repoint",
        nt.config_path.display()
    );
    std::fs::write(&nt.config_path, updated)?;
    Ok(())
}

/// 530 == tunnel down at the edge; anything else == a connection is registered.
async fn tunnel_registered(public_url: &str) -> bool {
    let res = reqwest::Client::new()
        .get(format!("{public_url}/"))
        .timeout(Duration::from_secs(10))
        .send()
        .await;
    match res {
        Ok(r) => r.status().as_u16() != 530,
        Err(_) => false,
    }
}

fn spawn_detached(mut cmd: std::process::Command, log: PathBuf) -> anyhow::Result<u32> {
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let out = std::fs::File::create(&log)?;
    let err = out.try_clone()?;
    cmd.stdout(out).stderr(err).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP — child outlives us.
        cmd.creation_flags(0x0800_0200);
    }
    let child = cmd.spawn().context("spawn tunnel agent")?;
    Ok(child.id())
}

/// Provider precedence (stability first; ngrok-static is the recommended path):
///
///   1. ngrok static   — `CYRUS_NGROK_DOMAIN` set + ngrok available. A permanent
///                        `*.ngrok-free.app` from one free signup: the connector
///                        is created ONCE and survives reboots. ⭐ recommended.
///   2. cloudflared named — `~/.cloudflared/config.yml` present (you own a domain).
///                          Also stable.
///   3. cloudflared quick — zero-config `*.trycloudflare.com`. URL changes on
///                          every cloudflared restart, so the connector is
///                          re-created on cold starts (cheap + automatic now that
///                          cleanup is surgical). The out-of-box fallback.
///   4. ngrok ephemeral — only if cloudflared is absent. Random URL, same churn.
pub async fn ensure_tunnel(opts: &SetupOptions) -> anyhow::Result<TunnelOutcome> {
    // An explicit choice (from the TUI's tunnel picker, via `--tunnel`) routes
    // directly to that lane with an actionable error if its prereqs are missing.
    // `Auto` falls through to the precedence ladder below (backward-compatible).
    match &opts.tunnel {
        TunnelChoice::Quick => {
            anyhow::ensure!(
                find_cloudflared_exe().is_some(),
                "cloudflared not found — install it (winget install Cloudflare.cloudflared) \
                 and re-run, or pick the ngrok tunnel instead"
            );
            return cloudflared_quick(opts).await;
        }
        TunnelChoice::Named => {
            anyhow::ensure!(
                parse_named_tunnel(opts.chimera_port).is_some(),
                "no named cloudflared tunnel found at ~/.cloudflared/config.yml — set one up \
                 (you need your own domain), or pick the quick or ngrok tunnel"
            );
            return cloudflared_named(opts).await;
        }
        TunnelChoice::Ngrok { domain } => {
            anyhow::ensure!(
                ngrok_available(),
                "ngrok not found — install it (https://ngrok.com/download), run \
                 `ngrok config add-authtoken <token>`, then re-run"
            );
            let domain = domain.clone().or_else(|| {
                std::env::var("CYRUS_NGROK_DOMAIN").ok().filter(|s| !s.is_empty())
            });
            return ensure_ngrok(opts, domain).await;
        }
        TunnelChoice::Auto => {}
    }

    let ngrok_domain = std::env::var("CYRUS_NGROK_DOMAIN")
        .ok()
        .filter(|s| !s.is_empty());

    // 1. ngrok static (the recommended stable path).
    if let Some(domain) = &ngrok_domain {
        anyhow::ensure!(
            ngrok_available(),
            "CYRUS_NGROK_DOMAIN is set but the ngrok binary was not found — install ngrok \
             (https://ngrok.com/download) or unset CYRUS_NGROK_DOMAIN to use cloudflared"
        );
        return ensure_ngrok(opts, Some(domain.clone())).await;
    }

    // 2. cloudflared named.
    if parse_named_tunnel(opts.chimera_port).is_some() {
        return cloudflared_named(opts).await;
    }

    // 3. cloudflared quick (zero-config default).
    if find_cloudflared_exe().is_some() {
        return cloudflared_quick(opts).await;
    }

    // 4. ngrok ephemeral (only if cloudflared is unavailable).
    if ngrok_available() {
        return ensure_ngrok(opts, None).await;
    }

    anyhow::bail!(
        "no tunnel provider found. Recommended: sign up at ngrok.com (free), run \
         `ngrok config add-authtoken <token>`, reserve a static domain, and set \
         CYRUS_NGROK_DOMAIN=<your-domain>. Alternative: install cloudflared."
    )
}

async fn cloudflared_named(opts: &SetupOptions) -> anyhow::Result<TunnelOutcome> {
    if let Some(nt) = parse_named_tunnel(opts.chimera_port) {
        let public_url = format!("https://{}", nt.hostname);

        if nt.service_port != Some(opts.chimera_port) {
            repoint_service(&nt, opts.chimera_port)?;
            // A running cloudflared keeps the OLD ingress; it must be restarted
            // to pick up the repoint. We can't safely kill arbitrary processes,
            // so surface this loudly instead.
            anyhow::bail!(
                "tunnel config repointed from port {:?} to {} — restart cloudflared and re-run setup",
                nt.service_port,
                opts.chimera_port
            );
        }

        if tunnel_registered(&public_url).await {
            return Ok(TunnelOutcome { public_url, started: false });
        }

        let exe = find_cloudflared_exe().context(
            "cloudflared not found — install it (winget install Cloudflare.cloudflared) and re-run",
        )?;
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("tunnel")
            .arg("--config")
            .arg(&nt.config_path)
            .arg("run")
            .arg(&nt.tunnel_id);
        spawn_detached(cmd, opts.cyrus_home().join("logs/cloudflared.log"))?;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
        loop {
            if tunnel_registered(&public_url).await {
                return Ok(TunnelOutcome { public_url, started: true });
            }
            anyhow::ensure!(
                tokio::time::Instant::now() < deadline,
                "cloudflared did not register the tunnel within 45s (see ~/.cyrus/logs/cloudflared.log)"
            );
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    // No named config — fall back to a quick tunnel.
    cloudflared_quick(opts).await
}

async fn cloudflared_quick(opts: &SetupOptions) -> anyhow::Result<TunnelOutcome> {
    let exe = find_cloudflared_exe().context(
        "cloudflared not found — install it (winget install Cloudflare.cloudflared) and re-run",
    )?;
    let log = opts.cyrus_home().join("logs/cloudflared-quick.log");

    // Isolate from any existing `~/.cloudflared/config.yml`. If the user already
    // runs a NAMED cloudflared tunnel, cloudflared would otherwise inherit its
    // config + credentials here and register this ephemeral quick tunnel under
    // the named tunnel's account — so the fresh trycloudflare URL 404s at the
    // Cloudflare edge (it never routes to our origin). An empty `--config`
    // forces a clean quick tunnel. (A true fresh user has no config to inherit,
    // so this is a no-op for them.)
    let empty_cfg = opts.cyrus_home().join("cloudflared-quick.empty.yml");
    if let Some(parent) = empty_cfg.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&empty_cfg, "").ok();

    let mut cmd = std::process::Command::new(exe);
    cmd.arg("tunnel")
        .arg("--no-autoupdate")
        // Force HTTP/2 (TCP) transport instead of cloudflared's QUIC default.
        // Many networks block or mangle QUIC (UDP/443), which leaves the tunnel
        // looping on `CRYPTO_ERROR ... tls: no application protocol` and never
        // routing to the origin — the "tunnel does not reach chimera" failure.
        // HTTP/2 rides ordinary TCP/443 and sails through. Override with
        // CYRUS_CF_PROTOCOL (e.g. `quic`, `auto`) if a network prefers QUIC.
        .arg("--protocol")
        .arg(std::env::var("CYRUS_CF_PROTOCOL").unwrap_or_else(|_| "http2".to_string()))
        .arg("--config")
        .arg(&empty_cfg)
        .arg("--url")
        .arg(format!("http://127.0.0.1:{}", opts.chimera_port));
    let pid = spawn_detached(cmd, log.clone())?;
    // Record the ephemeral tunnel's pid so the session-end teardown can close
    // this throwaway public socket (a NAMED tunnel is the user's own; we don't).
    let _ = std::fs::write(opts.cyrus_home().join("quick-tunnel.pid"), pid.to_string());

    // cloudflared prints the assigned URL into its log; poll-parse it.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    loop {
        if let Ok(text) = std::fs::read_to_string(&log) {
            if let Some(url) = extract_trycloudflare_url(&text) {
                return Ok(TunnelOutcome { public_url: url, started: true });
            }
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "quick tunnel did not come up within 45s (see {})",
            log.display()
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

// ---- ngrok ------------------------------------------------------------------
//
// The ngrok agent exposes a local API at http://127.0.0.1:4040. We use it to
// REUSE a running agent (ngrok's free plan allows one agent session, so we must
// never start a second), and otherwise spawn the CLI. A static `--domain` gives
// a permanent URL; without one the agent assigns a random `*.ngrok-free.app`.

const NGROK_API: &str = "http://127.0.0.1:4040/api/tunnels";

fn find_ngrok_exe() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CYRUS_NGROK_EXE") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let probe = std::process::Command::new("ngrok")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Some(PathBuf::from("ngrok"));
    }
    None
}

fn ngrok_available() -> bool {
    find_ngrok_exe().is_some()
}

/// An https tunnel from a running agent whose upstream addr targets `port`.
async fn ngrok_existing_for_port(port: u16) -> Option<String> {
    let v: serde_json::Value = reqwest::Client::new()
        .get(NGROK_API)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    let needle = format!(":{port}");
    for t in v.get("tunnels")?.as_array()? {
        let addr = t
            .get("config")
            .and_then(|c| c.get("addr"))
            .and_then(|a| a.as_str())
            .unwrap_or("");
        let url = t.get("public_url").and_then(|u| u.as_str()).unwrap_or("");
        if addr.ends_with(&needle) && url.starts_with("https://") {
            return Some(url.trim_end_matches('/').to_string());
        }
    }
    None
}

fn ngrok_agent_up() -> bool {
    // A blocking probe is fine here (called rarely); reqwest blocking would pull
    // an extra feature, so reuse the async client via a oneshot block.
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:4040".parse().expect("valid addr"),
        Duration::from_millis(400),
    )
    .is_ok()
}

async fn ensure_ngrok(opts: &SetupOptions, domain: Option<String>) -> anyhow::Result<TunnelOutcome> {
    let port = opts.chimera_port;

    // Reuse a tunnel the running agent already has for our port.
    if let Some(url) = ngrok_existing_for_port(port).await {
        return Ok(TunnelOutcome { public_url: url, started: false });
    }

    if ngrok_agent_up() {
        // An agent is running (maybe for another app) — add our tunnel via its
        // API rather than starting a second agent (which the free plan rejects).
        let mut body = json_addr(port);
        if let Some(d) = &domain {
            body["domain"] = serde_json::Value::String(d.clone());
        }
        let res = reqwest::Client::new()
            .post(NGROK_API)
            .json(&body)
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .context("ngrok API: add tunnel")?;
        anyhow::ensure!(
            res.status().is_success(),
            "ngrok API rejected the tunnel ({}). If you set a static domain, make sure it is \
             reserved on your ngrok dashboard and the authtoken is configured.",
            res.status()
        );
        let v: serde_json::Value = res.json().await.context("ngrok API response")?;
        let url = v
            .get("public_url")
            .and_then(|u| u.as_str())
            .context("ngrok API returned no public_url")?;
        return Ok(TunnelOutcome {
            public_url: url.trim_end_matches('/').to_string(),
            started: true,
        });
    }

    // No agent running: spawn the CLI.
    let exe = find_ngrok_exe().context("ngrok binary not found")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("http").arg(port.to_string()).arg("--log=stdout");
    if let Some(d) = &domain {
        cmd.arg(format!("--domain={d}"));
    }
    spawn_detached(cmd, opts.cyrus_home().join("logs/ngrok.log"))?;

    // Poll the agent API for our tunnel.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    loop {
        if let Some(url) = ngrok_existing_for_port(port).await {
            return Ok(TunnelOutcome { public_url: url, started: true });
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "ngrok did not come up within 45s. If this is the first run, set your authtoken: \
             `ngrok config add-authtoken <token>` (see ~/.cyrus/logs/ngrok.log)"
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

fn json_addr(port: u16) -> serde_json::Value {
    serde_json::json!({
        "name": "cyrus-chimera",
        "proto": "http",
        "addr": port.to_string(),
        "schemes": ["https"],
    })
}

fn extract_trycloudflare_url(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some(start) = line.find("https://") {
            let rest = &line[start..];
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '|')
                .unwrap_or(rest.len());
            let url = &rest[..end];
            if url.contains(".trycloudflare.com") {
                return Some(url.trim_end_matches('/').to_string());
            }
        }
    }
    None
}

/// Single-shot "does the public URL still reach OUR chimera right now?" — the
/// launch-path probe (no retry loop, tight timeout). A dead tunnel, a churned
/// URL, or a foreign service answering the host all return false. Confirms
/// chimera's identity in the body so a 530/placeholder edge can't pass.
pub async fn tunnel_alive(public_url: &str) -> bool {
    let res = reqwest::Client::new()
        .get(format!("{}/", public_url.trim_end_matches('/')))
        .timeout(Duration::from_secs(6))
        .send()
        .await;
    match res {
        Ok(r) => r.text().await.map(|b| b.contains("repo-agent-mcp")).unwrap_or(false),
        Err(_) => false,
    }
}

/// The end-to-end proof: the public hostname reaches OUR chimera. A fresh
/// cloudflared QUICK tunnel can take a couple of minutes to become routable at
/// the edge (named/ngrok-static come up fast and return early on success), so we
/// wait up to 3 minutes rather than failing a slow-but-fine tunnel.
pub async fn verify_through_tunnel(public_url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    loop {
        let last = match client
            .get(format!("{public_url}/"))
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                if body.contains("repo-agent-mcp") {
                    return Ok(());
                }
                format!("HTTP {status}: {}", &body[..body.len().min(120)])
            }
            Err(e) => e.to_string(),
        };
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "tunnel does not reach chimera ({last})"
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_url_parse() {
        let log = "2026-06-10T00:00:00Z INF |  https://random-words-here.trycloudflare.com  |";
        assert_eq!(
            extract_trycloudflare_url(log).as_deref(),
            Some("https://random-words-here.trycloudflare.com")
        );
        assert_eq!(extract_trycloudflare_url("no url here"), None);
    }

    #[test]
    fn select_ingress_prefers_rule_already_on_chimera_port() {
        // Two hostname rules; the chimera-port one is second, and the first uses
        // `localhost` (not 127.0.0.1). cyrus must pick the matching rule, not #0.
        let cfg = "tunnel: abc-123\n\
                   ingress:\n\
                   \x20 - hostname: sb.solvero.app\n\
                   \x20   service: http://localhost:9999\n\
                   \x20 - hostname: chimera.solvero.app\n\
                   \x20   service: http://127.0.0.1:8787\n\
                   \x20 - service: http_status:404\n";
        let (id, host, service, port) = select_ingress(cfg, 8787).expect("a rule");
        assert_eq!(id, "abc-123");
        assert_eq!(host, "chimera.solvero.app");
        assert_eq!(service, "http://127.0.0.1:8787");
        assert_eq!(port, Some(8787));
    }

    #[test]
    fn select_ingress_falls_back_to_first_rule_and_keeps_host_literal() {
        // No rule on the chimera port → fall back to the first hostname rule,
        // preserving its `localhost` literal so repoint can match it.
        let cfg = "tunnel: abc-123\n\
                   ingress:\n\
                   \x20 - hostname: sb.solvero.app\n\
                   \x20   service: http://localhost:9999\n\
                   \x20 - service: http_status:404\n";
        let (_, host, service, port) = select_ingress(cfg, 8787).expect("a rule");
        assert_eq!(host, "sb.solvero.app");
        assert_eq!(service, "http://localhost:9999");
        assert_eq!(port, Some(9999));
    }

    #[test]
    fn ngrok_add_tunnel_body_targets_our_port_over_https() {
        let body = json_addr(8787);
        assert_eq!(body["addr"], "8787");
        assert_eq!(body["proto"], "http");
        assert_eq!(body["schemes"][0], "https");
    }
}
