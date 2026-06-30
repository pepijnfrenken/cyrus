//! Chrome bring-up: a CDP endpoint on the configured port with at least one
//! chatgpt.com tab whose session is logged in.
//!
//! Reuse-first: if CDP already answers (the user's automation Chrome is
//! running), use it. Otherwise launch Chrome with a DEDICATED profile dir
//! (`~/.cyrus/chrome-profile`) — never the user's daily profile (the Chrome
//! singleton would swallow our --remote-debugging-port flag, and we don't want
//! cyrus cookies entangled with their main browser).
//!
//! Login is the ONE human step in all of setup: we open chatgpt.com, check
//! `/api/auth/session` from page context, and if logged out, emit
//! NeedsUserAction and poll until the human signs in.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use cyrus_lipsync::cdp::CdpClient;
use cyrus_lipsync::tab_factory::BrowserControl;
use tokio::sync::mpsc::UnboundedSender;

use crate::{SetupEvent, SetupOptions, Step};

pub struct ChromeOutcome {
    /// A chatgpt.com page target with a live login — the connector step
    /// drives this tab.
    pub login_target_id: String,
    pub launched: bool,
}

/// `true` iff a Chrome DevTools (CDP) endpoint answers on the configured
/// host/port. Read-only probe reused by `cyrus check`.
pub async fn cdp_alive(opts: &SetupOptions) -> bool {
    let url = format!("http://{}:{}/json/version", opts.cdp_host, opts.cdp_port);
    reqwest::Client::new()
        .get(url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn find_chrome_exe() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CYRUS_CHROME_EXE") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    #[cfg(windows)]
    {
        let candidates = [
            std::env::var("ProgramFiles").ok().map(|p| {
                PathBuf::from(p).join("Google/Chrome/Application/chrome.exe")
            }),
            std::env::var("ProgramFiles(x86)").ok().map(|p| {
                PathBuf::from(p).join("Google/Chrome/Application/chrome.exe")
            }),
            std::env::var("LocalAppData").ok().map(|p| {
                PathBuf::from(p).join("Google/Chrome/Application/chrome.exe")
            }),
        ];
        for c in candidates.into_iter().flatten() {
            if c.exists() {
                return Some(c);
            }
        }
    }
    #[cfg(not(windows))]
    {
        for c in [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ] {
            let p = PathBuf::from(c);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

/// Carbonyl (a terminal-rendering Chromium, github.com/jmagly/carbonyl) speaks
/// standard CDP on `--remote-debugging-port` just like Chrome, so it's a drop-in
/// alternate browser — CDP stays the one interface. `CYRUS_CARBONYL_EXE` wins,
/// else bare `carbonyl` on PATH.
fn find_carbonyl_exe() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CYRUS_CARBONYL_EXE") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // PATH lookup: spawnable bare name?
    let probe = std::process::Command::new("carbonyl")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if probe.map(|s| s.success()).unwrap_or(false) {
        return Some(PathBuf::from("carbonyl"));
    }
    None
}

fn launch_chrome(opts: &SetupOptions) -> anyhow::Result<()> {
    // CYRUS_BROWSER picks the engine; both drive over CDP. Default is Chrome.
    if std::env::var("CYRUS_BROWSER").as_deref() == Ok("carbonyl") {
        return launch_carbonyl(opts);
    }
    let exe = find_chrome_exe()
        .context("Chrome not found — install Google Chrome or set CYRUS_CHROME_EXE")?;
    let profile = opts.cyrus_home().join("chrome-profile");
    std::fs::create_dir_all(&profile).ok();

    let mut cmd = std::process::Command::new(exe);
    cmd.arg(format!("--remote-debugging-port={}", opts.cdp_port))
        .arg(format!("--user-data-dir={}", profile.display()))
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("https://chatgpt.com/");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NEW_PROCESS_GROUP: Chrome outlives us and ignores our Ctrl-C.
        cmd.creation_flags(0x0000_0200);
    }
    cmd.spawn().context("spawn chrome")?;
    Ok(())
}

/// Launch Carbonyl with the SAME CDP flag set as Chrome. Two differences that
/// matter: (1) Carbonyl renders to the TTY, so its stdout/stderr MUST be
/// redirected to a log or it garbles the setup UI; (2) a SEPARATE profile dir,
/// since Carbonyl's bundled Chromium revision differs from system Chrome and the
/// two shouldn't share a `--user-data-dir`.
fn launch_carbonyl(opts: &SetupOptions) -> anyhow::Result<()> {
    let exe = find_carbonyl_exe().context(
        "carbonyl not found — install it (github.com/jmagly/carbonyl) or set CYRUS_CARBONYL_EXE",
    )?;
    let profile = opts.cyrus_home().join("carbonyl-profile");
    std::fs::create_dir_all(&profile).ok();
    let log_dir = opts.cyrus_home().join("logs");
    std::fs::create_dir_all(&log_dir).ok();
    let log = std::fs::File::create(log_dir.join("carbonyl.log"))
        .context("create carbonyl.log")?;
    let log_err = log.try_clone().context("clone carbonyl log handle")?;

    let mut cmd = std::process::Command::new(exe);
    cmd.arg(format!("--remote-debugging-port={}", opts.cdp_port))
        .arg(format!("--user-data-dir={}", profile.display()))
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("https://chatgpt.com/")
        // Critical: keep Carbonyl's terminal rendering off our stdio.
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0000_0200);
    }
    cmd.spawn().context("spawn carbonyl")?;
    Ok(())
}

/// Find an existing chatgpt.com page target, or open one.
async fn ensure_chatgpt_tab(browser: &BrowserControl) -> anyhow::Result<String> {
    let targets = browser.get_targets().await?;
    for t in &targets {
        let is_page = t.r#type.as_deref() == Some("page");
        let url = t.url.as_deref().unwrap_or("");
        if is_page && url.contains("chatgpt.com") {
            if let Some(id) = t.resolve_id() {
                return Ok(id.to_string());
            }
        }
    }
    browser.create_target("https://chatgpt.com/").await
}

/// `true` iff the tab's session carries an access token. Evaluated from page
/// context (same-origin; the page's own cookies do the authentication).
async fn is_logged_in(cdp: &CdpClient) -> bool {
    let expr = r#"fetch('/api/auth/session',{credentials:'include'})
        .then(r=>r.json()).then(s=>!!(s&&s.accessToken)).catch(()=>false)"#;
    matches!(cdp.eval(expr, 15.0).await, Ok(v) if v.as_bool() == Some(true))
}

pub async fn ensure_chrome(
    opts: &SetupOptions,
    tx: &UnboundedSender<SetupEvent>,
) -> anyhow::Result<ChromeOutcome> {
    let mut launched = false;
    if !cdp_alive(opts).await {
        launch_chrome(opts)?;
        launched = true;
        // Chrome takes a beat to open the DevTools endpoint.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            if cdp_alive(opts).await {
                break;
            }
            anyhow::ensure!(
                tokio::time::Instant::now() < deadline,
                "Chrome did not open the CDP endpoint on port {} within 30s",
                opts.cdp_port
            );
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    let browser = BrowserControl::new(opts.cdp_host.clone(), opts.cdp_port);
    browser.connect().await.context("browser control socket")?;
    let target_id = ensure_chatgpt_tab(&browser).await?;

    // Page may still be loading right after creation; give the first probe a
    // moment before declaring the user logged out.
    let cdp = attach_with_retry(opts, &target_id).await?;
    tokio::time::sleep(Duration::from_secs(1)).await;

    if !is_logged_in(&cdp).await {
        browser.activate_target(&target_id).await;
        let _ = tx.send(SetupEvent::NeedsUserAction {
            step: Step::Chrome,
            instruction: "Log in to ChatGPT in the Chrome window that just opened, \
                          then come back here — setup continues automatically."
                .to_string(),
        });
        let deadline = tokio::time::Instant::now() + Duration::from_secs(600);
        loop {
            tokio::time::sleep(Duration::from_secs(3)).await;
            if is_logged_in(&cdp).await {
                let _ = tx.send(SetupEvent::UserActionResolved { step: Step::Chrome });
                break;
            }
            anyhow::ensure!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for the ChatGPT login (10 minutes)"
            );
        }
    }

    cdp.close().await;
    browser.close().await;
    Ok(ChromeOutcome {
        login_target_id: target_id,
        launched,
    })
}

/// Attach a page socket; retry briefly (a just-created tab can lag in /json).
pub(crate) async fn attach_with_retry(
    opts: &SetupOptions,
    target_id: &str,
) -> anyhow::Result<CdpClient> {
    let mut last_err = None;
    for _ in 0..10 {
        match CdpClient::for_target(
            opts.cdp_host.clone(),
            opts.cdp_port,
            target_id,
            "chatgpt.com",
        )
        .await
        {
            Ok(c) => return Ok(c),
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    Err(anyhow::anyhow!(
        "could not attach page socket to {target_id}: {:?}",
        last_err
    ))
}
