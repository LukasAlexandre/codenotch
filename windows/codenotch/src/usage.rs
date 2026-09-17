//! Claude usage adapter (official), implemented from the upstream Codenotch's documented behaviour.
//! Endpoint: GET https://api.anthropic.com/api/oauth/usage
//! Headers: Authorization: Bearer <token>; anthropic-beta: oauth-2025-04-20; 15 s timeout
//! Rules (upstream's discipline):
//!   - the credential comes from Claude Code's own store (Windows: ~/.claude/.credentials.json), read only
//!   - 401/403 → re-read the credential once and retry (Claude Code may have just refreshed the token) → still failing means needsAuth
//!   - 429 → back off 60 s × 2^n capped at 15 min, Retry-After only raises it, even past the cap; the deadline is persisted
//!   - an expired token is never sent: the endpoint answers it with 429 + Retry-After ≈ 3600, not 401, so sending it
//!     reads as "rate limited" for as long as the token stays stale (upstream's credentialExpired, no network)
//!   - the token is renewed by running the standalone `claude -p` with an empty stdin shortly before it expires
//!     (upstream's ClaudeTokenRefresher). Only that CLI writes ~/.claude/.credentials.json — Claude Code inside the
//!     desktop app renews its own copy elsewhere — so without this the file rots eight hours after the last CLI run
//!   - never invent a percentage on failure: keep the last reading marked stale, and the UI shows how old it is
//!
//! Reply (snake_case): { limits:[{kind,percent,resets_at}], five_hour:{utilization,resets_at}, seven_day:{...} }
//! limits is the forward-compatible main shape; five_hour/seven_day are merged in as a fallback (a window that just rolled over disappears from limits).

use crate::claude_account;
use crate::AppState;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

const ENDPOINT: &str = "https://api.anthropic.com/api/oauth/usage";
const POLL_ACTIVE_SECS: u64 = 60;
const POLL_IDLE_SECS: u64 = 300;
const BACKOFF_BASE_SECS: u64 = 60;
const BACKOFF_CAP_SECS: u64 = 900;
/// Renew when this close to expiry. Must stay under Claude Code's own five minutes: its start-up renews the token
/// only when now + 300 s >= expiresAt, so launching any earlier is a no-op that would be judged a failure
const RENEW_MARGIN_MS: u64 = 4 * 60 * 1000;
const RENEW_COOLDOWN_MS: u64 = 10 * 60 * 1000;
const RENEW_TIMEOUT_SECS: u64 = 30;
const EXPIRED_NOTE: &str = "Credential expired — run claude once in a terminal to renew it";

/// Upstream's `ClaudeUsageCLI.timeout` (ClaudeUsageCLI.swift:32): long enough for a cold Node
/// start on a busy machine, short enough that a wedged process cannot hold a refresh open.
const CLI_USAGE_TIMEOUT_SECS: u64 = 20;
/// The CLI flags, matching upstream's `ClaudeUsageCLI.arguments` (ClaudeUsageCLI.swift:54).
/// `--print` skips the interactive/workspace-trust prompt, `--no-session-persistence` (print-mode
/// only) skips writing a transcript, and `--strict-mcp-config` with no `--mcp-config` starts no
/// MCP server at all (upstream measured this keeps the process talking only to
/// api.anthropic.com and Claude Code's own feature-gate host).
///
/// Deliberate divergence from upstream: Swift passes `/usage` as a fourth, trailing argv element
/// with stdin nulled. On the Claude Code CLI actually installed here (2.1.267), that made `/usage`
/// land as a literal chat prompt ("I notice this message is just a path with no actual request"),
/// not the slash command — verified by hand before writing this parser, not assumed. Piping
/// `/usage` over stdin instead (see `run_cli_usage`) reliably invokes the command and produces the
/// exact `Current session: …` report. Argv-only is kept here as a documented fact, in case a
/// future CLI version's behaviour changes back.
const CLI_USAGE_ARGS: &[&str] = &["--print", "--no-session-persistence", "--strict-mcp-config"];
/// Windows-side observability for T-CLAUDE-USAGE-01 (doctor/logs only, never the UI): which
/// source fed the last Claude usage snapshot.
static LAST_CLAUDE_SOURCE: std::sync::Mutex<Option<&'static str>> = std::sync::Mutex::new(None);

static REFRESH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Immediate refresh from the tray or a command
pub fn request_refresh() {
    REFRESH.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Sleep in slices so request_refresh can interrupt it
fn sleep_interruptible(total_secs: u64) {
    for _ in 0..total_secs {
        if REFRESH.swap(false, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LimitWindow {
    pub id: String,
    pub label: String,
    /// 0.0–1.0 (fraction used)
    pub used: f64,
    /// Reset time, ms epoch (None = unknown)
    pub resets_at: Option<u64>,
    /// Pure count window (no published denominator, e.g. Antigravity's requests today) — the cell shows ~N and the ring draws only its track
    #[serde(default)]
    pub count: Option<i64>,
    /// The number is ours, not the vendor's (upstream fidelity=.derived) — the card adds a ~ prefix
    #[serde(default)]
    pub derived: bool,
    /// The heading the window sits under on the card, for a provider that reports the same windows
    /// for several things (Antigravity: a 5-hour and a weekly lane per model family). None = ungrouped
    #[serde(default)]
    pub group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageSnapshot {
    /// ok | stale | needsAuth | backoff | error
    pub status: String,
    pub windows: Vec<LimitWindow>,
    pub fetched_at: u64,
    pub note: String,
    #[serde(default)]
    pub backoff_until: u64,
    /// The Claude account this snapshot's numbers belong to (`claude_account::ClaudeAccountStatus
    /// ::fingerprint`), so a restart can tell "still the same account, safe to show as stale"
    /// from "a different account, discard" — Ajuste 2. `#[serde(default)]` means an old
    /// pre-T-CLAUDE-ACCOUNT-SWITCHER `usage.json` deserializes to `None` here, which
    /// `usage::start()` treats as untrusted/legacy, never as "confirmed same account".
    #[serde(default)]
    pub account_fingerprint: Option<String>,
}

fn store_path() -> std::path::PathBuf {
    crate::config::config_path().with_file_name("usage.json")
}

pub fn load_persisted() -> UsageSnapshot {
    std::fs::read_to_string(store_path())
        .ok()
        .and_then(|t| serde_json::from_str::<UsageSnapshot>(&t).ok())
        .map(|mut s| {
            if !s.windows.is_empty() {
                s.status = "stale".into(); // an old reading after a restart is labelled as such
            }
            s
        })
        .unwrap_or_default()
}

fn persist(s: &UsageSnapshot) {
    if let Ok(t) = serde_json::to_string_pretty(s) {
        let _ = std::fs::write(store_path(), t);
    }
}

struct Credential {
    token: String,
    /// ms epoch (None = the file names no expiry)
    expires_at: Option<u64>,
}

impl Credential {
    fn expired(&self, now: u64) -> bool {
        self.expires_at.map(|e| e <= now).unwrap_or(false)
    }
}

/// Reads Claude Code's OAuth credential.
fn read_credentials() -> Option<Credential> {
    let home = dirs::home_dir()?;
    for name in [".credentials.json", "credentials.json"] {
        let p = home.join(".claude").join(name);
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let oauth = v.get("claudeAiOauth").unwrap_or(&v);
        if let Some(tok) = oauth.get("accessToken").and_then(|x| x.as_str()) {
            let expires_at = oauth.get("expiresAt").and_then(|x| x.as_f64()).map(|ms| ms as u64);
            return Some(Credential { token: tok.to_string(), expires_at });
        }
    }
    None
}

/// For doctor: credential probe report (prints no secret values)
pub fn probe_credentials() -> String {
    let cli = match find_cli() {
        Some(p) => format!("renews via {}", p.display()),
        None => "no standalone claude CLI found to renew it".into(),
    };
    let source = LAST_CLAUDE_SOURCE.lock().unwrap().unwrap_or("none yet");
    let base = match read_credentials() {
        Some(c) => format!(
            "credential: found (token {} chars, {}; {cli})",
            c.token.len(),
            if c.expired(now_ms()) { "expired" } else { "valid" }
        ),
        None => "credential: ~/.claude/.credentials.json not found (needsAuth; the desktop app may use another store — signing in once with the Claude Code CLI creates it)".into(),
    };
    format!("{base}\n  Claude usage source: {source}")
}

// ---------------- token renewal (upstream's ClaudeTokenRefresher) ----------------

/// Anything under these belongs to the desktop app: its bundled Claude Code keeps its token in the desktop app's
/// own store and never writes ~/.claude/.credentials.json, so renewing with it would change nothing here
fn is_desktop_owned(p: &std::path::Path) -> bool {
    let s = p.to_string_lossy().to_ascii_lowercase().replace('/', "\\");
    s.contains("\\anthropicclaude\\") || s.contains("\\claude\\claude-code\\") || s.contains("\\windowsapps\\")
}

/// The standalone Claude Code command: its own installer's location first, then global npm/pnpm/Volta, then PATH
pub(crate) fn find_cli() -> Option<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Some(h) = dirs::home_dir() {
        v.push(h.join(".local").join("bin").join("claude.exe"));
    }
    if let Some(d) = dirs::config_dir() {
        v.push(d.join("npm").join("claude.cmd"));
    }
    if let Some(d) = dirs::data_local_dir() {
        v.push(d.join("pnpm").join("claude.cmd"));
    }
    if let Some(h) = dirs::home_dir() {
        v.push(h.join(".volta").join("bin").join("claude.exe"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            v.push(dir.join("claude.exe"));
            v.push(dir.join("claude.cmd"));
        }
    }
    v.into_iter().find(|p| p.is_file() && !is_desktop_owned(p))
}

// ---------------- Claude Code CLI usage (T-CLAUDE-USAGE-01, upstream's ClaudeUsageCLI) ----------------
//
// Windows equivalent of macOS's `ClaudeOAuthProvider.fetchSnapshot()` priority order
// (ClaudeOAuthProvider.swift:150-198): try `claude "/usage"` first — off the same credential the
// CLI already holds, no keychain/token involved — and only fall back to the raw
// `/api/oauth/usage` token endpoint if the CLI is absent or declines to answer. `desktopWindows()`
// (the Claude Desktop cache reader) is explicitly out of scope for this task.
//
// Root cause this addresses: the token endpoint answers for whichever OAuth grant is in
// `~/.claude/.credentials.json`, which is not necessarily the credential doing the account's real
// work (e.g. a desktop-app-hosted session uses its own separate credential store — see
// `is_desktop_owned`). `claude "/usage"` instead asks the CLI itself, which reports on its own
// actual usage the same way the terminal command does.

/// Reasons the CLI source did not produce a usage reading. Every variant means "fall back to the
/// OAuth token path" — none of them is treated as "usage is 0%".
#[derive(Debug)]
enum CliUsageErr {
    /// No standalone `claude` binary found.
    NotFound,
    /// The process exited non-zero: upstream's own reading of this is "declining to answer",
    /// in practice meaning it has no login of its own.
    NeedsAuth,
    /// The process did not finish within `CLI_USAGE_TIMEOUT_SECS` and was killed.
    Timeout,
    /// Empty output, or output that did not contain a recognizable `Current session: NN% used`
    /// line — never treated as a real reading, real or zero.
    BadResponse,
}

/// `%APPDATA%\codenotch\usage-scratch` — one fixed directory, created once and reused across
/// calls. Mirrors upstream's `ClaudeUsageCLI.scratchDirectory` (ClaudeUsageCLI.swift:65-75): a
/// fresh temporary directory per call left a new, never-revisited project folder under
/// `~/.claude/projects` on every poll (twelve an hour, indefinitely); one fixed directory means at
/// most one, and `--no-session-persistence` above means none at all.
fn cli_usage_scratch_dir() -> Option<std::path::PathBuf> {
    let dir = dirs::config_dir()?.join("codenotch").join("usage-scratch");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Successful process exit (any status code — callers decide what a non-zero exit means for
/// their own command) plus its captured stdout text.
pub(crate) struct ClaudeProcOutput {
    pub success: bool,
    pub stdout: String,
}

/// Reasons `run_claude_subprocess` could not produce an output at all. Distinct from "the process
/// ran and exited non-zero" (that is `ClaudeProcOutput.success = false`, a normal outcome).
pub(crate) enum ClaudeProcErr {
    NotFound,
    Timeout,
    /// The OS-level wait on the process itself failed (distinct from a timeout) — vanishingly
    /// rare in practice, kept as its own variant only so callers can tell it apart from a normal
    /// timeout if they ever need to.
    WaitFailed,
}

/// Runs any `claude <args>` invocation hidden, with no window, no shell, and a hard timeout,
/// returning its stdout text and exit status. Shared by every Claude CLI caller in this crate
/// (`/usage` here, `auth status`/`auth login`/`auth logout` in `claude_account.rs`) — extracted
/// from what was originally `run_cli_usage`'s own body (T-CLAUDE-USAGE-01) so the one proven,
/// safety-reviewed mechanism is reused rather than re-implemented per caller.
///
/// Safety properties: no window (`CREATE_NO_WINDOW`), argv passed as a plain array (no shell
/// string, no injection surface), stdin is either closed immediately or fed exactly the caller's
/// bytes (never a real prompt/response, never anything from an actual session), and the whole
/// process tree is contained in a Windows Job Object so a `claude.cmd` → `node.exe` wrapper
/// cannot leave an orphaned `node.exe` behind on timeout — the same concern `agy_cli.rs`'s
/// bounded runner exists for. No ConPTY: `claude --print`/`claude auth ...` are designed for
/// non-interactive capture (this is exactly what upstream's own `Process`+`Pipe` does on macOS,
/// with no pseudo-terminal either), so a plain redirected pipe is sufficient and simpler.
#[cfg(windows)]
pub(crate) fn run_claude_subprocess(
    cli: &std::path::Path,
    args: &[&str],
    stdin_data: Option<&[u8]>,
    cwd: &std::path::Path,
    timeout: Duration,
) -> Result<ClaudeProcOutput, ClaudeProcErr> {
    use std::io::{Read, Write};
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::sync::mpsc::channel;
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    if !cli.is_file() {
        return Err(ClaudeProcErr::NotFound);
    }

    let mut child = Command::new(cli)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
        .spawn()
        .map_err(|_| ClaudeProcErr::NotFound)?;
    // Observability for every Claude CLI subprocess (Fase 14): args are just flags (e.g.
    // "auth login"), never a token/URL/code — safe to log in full.
    crate::applog(&format!("claude subprocess: spawned pid={:?} args={}", child.id(), args.join(" ")));

    // Fixed, caller-supplied bytes only — never a real prompt/response, never anything from an
    // actual session. stdin is then closed by dropping the handle, so the process proceeds
    // instead of waiting for more input.
    if let Some(mut stdin) = child.stdin.take() {
        if let Some(data) = stdin_data {
            let _ = stdin.write_all(data);
        }
        // dropping `stdin` here closes the handle even when `stdin_data` is None
    }

    // Contained as a whole in a job object killed on timeout — a bare child.kill() only ends the
    // immediate process and can orphan a Node process a .cmd shim spawned underneath it.
    let job = unsafe { CreateJobObjectW(None, PCWSTR::null()) }.ok();
    if let Some(job) = job {
        unsafe {
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let _ = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _,
                std::mem::size_of_val(&limits) as u32,
            );
            let _ = AssignProcessToJobObject(job, HANDLE(child.as_raw_handle() as *mut core::ffi::c_void));
        }
    }

    // Draining stdout happens on its own thread: the read blocks until EOF, which arrives either
    // because the process exited on its own or because the timeout below killed it — the two are
    // never waited on in sequence, which is what would risk a deadlock on a full pipe buffer.
    let mut stdout = child.stdout.take();
    let (tx, rx) = channel();
    let reader = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(s) = stdout.as_mut() {
            let _ = s.read_to_string(&mut buf);
        }
        let _ = tx.send(buf);
    });

    let deadline = std::time::Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    timed_out = true;
                    if let Some(job) = job {
                        unsafe {
                            let _ = TerminateJobObject(job, 1);
                        }
                    }
                    let _ = child.kill();
                    break child.wait().ok();
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => break None,
        }
    };
    let out = rx.recv_timeout(Duration::from_secs(2)).unwrap_or_default();
    let _ = reader.join();
    if let Some(job) = job {
        unsafe {
            let _ = CloseHandle(job);
        }
    }

    // Length only, never the stdout content itself (an OAuth authorize URL or a paste-code
    // prompt can be in there) — this is what caught the Switch-account deadlock live.
    crate::applog(&format!(
        "claude subprocess: finished timed_out={timed_out} stdout_len={} exit_success={:?}",
        out.len(),
        status.map(|s| s.success())
    ));
    if timed_out {
        return Err(ClaudeProcErr::Timeout);
    }
    let Some(status) = status else {
        return Err(ClaudeProcErr::WaitFailed);
    };
    Ok(ClaudeProcOutput { success: status.success(), stdout: out })
}

#[cfg(not(windows))]
pub(crate) fn run_claude_subprocess(
    _cli: &std::path::Path,
    _args: &[&str],
    _stdin_data: Option<&[u8]>,
    _cwd: &std::path::Path,
    _timeout: Duration,
) -> Result<ClaudeProcOutput, ClaudeProcErr> {
    Err(ClaudeProcErr::NotFound)
}

/// `claude --print --no-session-persistence --strict-mcp-config` with `/usage` on stdin, via
/// `run_claude_subprocess`. Behaviour unchanged from before the T002 refactor: the same three
/// outcomes (not found / timeout / bad-or-declined response) map to the same `CliUsageErr`
/// variants — this function only re-expresses `run_cli_usage`'s old body in terms of the shared
/// runner, it does not change what `/usage` does.
fn run_cli_usage(cli: &std::path::Path, timeout: Duration) -> Result<String, CliUsageErr> {
    let scratch = cli_usage_scratch_dir().ok_or(CliUsageErr::BadResponse)?;
    let out = run_claude_subprocess(cli, CLI_USAGE_ARGS, Some(b"/usage\n"), &scratch, timeout)
        .map_err(|e| match e {
            ClaudeProcErr::NotFound => CliUsageErr::NotFound,
            ClaudeProcErr::Timeout => CliUsageErr::Timeout,
            ClaudeProcErr::WaitFailed => CliUsageErr::BadResponse,
        })?;
    if !out.success {
        // Upstream's own reading of a non-zero exit (ClaudeUsageCLI.swift:223-230): Claude Code
        // declining to answer, in practice meaning it has no login of its own — not an error
        // worth surfacing, the caller falls back to the token path.
        return Err(CliUsageErr::NeedsAuth);
    }
    if out.stdout.trim().is_empty() {
        return Err(CliUsageErr::BadResponse);
    }
    Ok(out.stdout)
}

/// `all models` -> `weekly_all`, `Opus` -> `weekly_opus`: the endpoint's own vocabulary, copied
/// from upstream's `ClaudeUsageCLI.kind(forWeek:)` (ClaudeUsageCLI.swift:294-299), so
/// `label_for` below can name both the same way regardless of which source produced the window.
fn cli_week_kind(text: &str) -> String {
    let lower = text.to_lowercase();
    let name = if lower == "all models" { "all".to_string() } else { lower.replace(' ', "_") };
    format!("weekly_{name}")
}

/// One `Current session: 38% used · resets Sep 7 at 2:59pm (Asia/Jakarta)` style line, upstream's
/// exact wording (`ClaudeUsageCLI.line`, ClaudeUsageCLI.swift:247-250) parsed by hand instead of
/// with a regex engine — the grammar is small and fixed, and a hand parser fails closed on
/// anything unrecognized instead of silently matching too much.
fn parse_cli_line(line: &str) -> Option<(String, f64, Option<String>)> {
    let line = line.trim();
    let rest = line.strip_prefix("Current ")?;
    let (kind, after_kind) = if let Some(r) = rest.strip_prefix("session") {
        ("session".to_string(), r)
    } else if let Some(r) = rest.strip_prefix("week (") {
        let close = r.find(')')?;
        let label = &r[..close];
        (cli_week_kind(label), &r[close + 1..])
    } else {
        return None;
    };
    let after_colon = after_kind.strip_prefix(':')?.trim_start();
    let digit_end = after_colon.find(|c: char| !c.is_ascii_digit()).unwrap_or(after_colon.len());
    if digit_end == 0 {
        return None;
    }
    let percent: f64 = after_colon[..digit_end].parse().ok()?;
    let after_pct = after_colon[digit_end..].trim_start().strip_prefix('%')?.trim_start();
    let after_used = after_pct.strip_prefix("used")?.trim_start();
    if after_used.is_empty() {
        return Some((kind, percent, None));
    }
    let after_dot = after_used.strip_prefix('·')?.trim_start();
    let reset_text = after_dot.strip_prefix("resets")?.trim();
    if reset_text.is_empty() {
        return Some((kind, percent, None));
    }
    Some((kind, percent, Some(reset_text.to_string())))
}

/// `Sep 17, 7:49am (America/Sao_Paulo)` -> a ms-epoch timestamp, adapted from upstream's
/// `ClaudeUsageCLI.resetDate(from:now:)` (ClaudeUsageCLI.swift:307-348).
///
/// Known, deliberate simplification vs. upstream: the parenthesized IANA zone name is read only
/// to be stripped off, then the wall-clock time is interpreted in the machine's own local
/// timezone rather than resolved against a real timezone database (this crate has no `chrono-tz`
/// dependency, and adding one for this alone was judged not worth it). In practice the zone
/// Claude Code prints here already matches the machine's own — this only diverges from upstream
/// if that ever stops being true.
///
/// Date/time separator: upstream's own doc comment (and its regex) says `"Sep 7 at 2:59pm"`, but
/// the Claude Code CLI actually installed here (2.1.267) prints `"Sep 17, 7:49am"` — a comma, no
/// `"at"` — verified against real `/usage` output, not assumed. Both separators are accepted so a
/// future CLI version matching upstream's documented wording still parses.
///
/// No year is printed, so — exactly like upstream — the year is chosen as whichever of last year,
/// this year or next year lands nearest to `now`.
fn parse_cli_reset(text: &str, now: chrono::DateTime<chrono::Local>) -> Option<u64> {
    use chrono::{Datelike, TimeZone};

    let text = text.trim();
    let stamp = match text.rfind('(') {
        Some(open) if text.ends_with(')') => text[..open].trim(),
        _ => text,
    };
    let (date_part, time_part) = stamp.split_once(" at ").or_else(|| stamp.split_once(", "))?;

    let mut dp = date_part.split_whitespace();
    let month_str = dp.next()?;
    let day: u32 = dp.next()?.parse().ok()?;
    const MONTHS: [&str; 12] =
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
    let month = MONTHS.iter().position(|m| m.eq_ignore_ascii_case(&month_str[..3.min(month_str.len())]))? as u32 + 1;

    let tp = time_part.trim();
    let (ampm, digits) = if let Some(d) = tp.strip_suffix("am").or_else(|| tp.strip_suffix("AM")) {
        (false, d)
    } else if let Some(d) = tp.strip_suffix("pm").or_else(|| tp.strip_suffix("PM")) {
        (true, d)
    } else {
        return None;
    };
    let (hour12, minute) = match digits.split_once(':') {
        Some((h, m)) => (h.parse::<u32>().ok()?, m.parse::<u32>().ok()?),
        None => (digits.parse::<u32>().ok()?, 0),
    };
    if !(1..=12).contains(&hour12) || minute > 59 {
        return None;
    }
    let hour24 = match (hour12, ampm) {
        (12, false) => 0,  // 12am == midnight
        (12, true) => 12,  // 12pm == noon
        (h, false) => h,
        (h, true) => h + 12,
    };

    let this_year = now.year();
    [this_year - 1, this_year, this_year + 1]
        .into_iter()
        .filter_map(|year| {
            let date = chrono::NaiveDate::from_ymd_opt(year, month, day)?;
            let time = chrono::NaiveTime::from_hms_opt(hour24, minute, 0)?;
            let naive = chrono::NaiveDateTime::new(date, time);
            match chrono::Local.from_local_datetime(&naive) {
                chrono::LocalResult::Single(dt) => Some(dt),
                chrono::LocalResult::Ambiguous(dt, _) => Some(dt),
                chrono::LocalResult::None => None,
            }
        })
        .min_by_key(|dt| (dt.signed_duration_since(now)).num_seconds().abs())
        .map(|dt| dt.timestamp_millis().max(0) as u64)
}

/// Turns `claude "/usage"`'s stdout into `LimitWindow`s. Fails closed: any line that is not the
/// exact `Current session: NN% used [· resets ...]` shape (upstream's `ClaudeUsageCLI.line`) is
/// silently skipped as prose, never coerced into a reading — and a report with no recognizable
/// session line at all is rejected outright (upstream's own rule, ClaudeUsageCLI.swift:283-288:
/// "without the session window there is no headline... better to fall back to the token path than
/// to draw a ring with a hole in it").
fn parse_cli_usage(text: &str) -> Result<Vec<LimitWindow>, CliUsageErr> {
    let now = chrono::Local::now();
    let mut out: Vec<LimitWindow> = Vec::new();
    for line in text.lines() {
        let Some((kind, percent, reset_text)) = parse_cli_line(line) else {
            continue;
        };
        if out.iter().any(|w| w.id == kind) {
            continue; // first occurrence wins, same as upstream
        }
        let resets_at = reset_text.and_then(|t| parse_cli_reset(&t, now));
        out.push(LimitWindow {
            id: kind.clone(),
            label: label_for(&kind),
            used: (percent / 100.0).clamp(0.0, 1.0),
            resets_at,
            ..Default::default()
        });
    }
    if !out.iter().any(|w| w.id == "session") {
        return Err(CliUsageErr::BadResponse);
    }
    out.sort_by_key(|w| if w.id == "session" { 0 } else { 1 });
    Ok(out)
}

/// Whether a launch is worth making. Pure, so every branch is testable without a clock or a subprocess
fn should_renew(expires_at: Option<u64>, now: u64, attempted_for: Option<u64>, last_attempt: Option<u64>) -> bool {
    // Nothing read yet: never launch on a guess
    let Some(exp) = expires_at else { return false };
    // Plenty of time left — also where launching would do nothing, because the CLI's own gate has not opened
    if exp > now + RENEW_MARGIN_MS {
        return false;
    }
    // One attempt per token: a launch that failed to move the expiry leaves the same value here, and never runs again
    if attempted_for == Some(exp) {
        return false;
    }
    if let Some(t) = last_attempt {
        if now.saturating_sub(t) < RENEW_COOLDOWN_MS {
            return false;
        }
    }
    true
}

/// `claude -p` with a null stdin starts up (which is where it renews an aged token), then exits non-zero for want
/// of a prompt: no conversation, no transcript. Output goes nowhere — a token could in principle be echoed into it.
fn run_renewal(cli: &std::path::Path) -> std::io::Result<()> {
    use std::process::{Command, Stdio};
    let mut cmd = Command::new(cli);
    cmd.arg("-p").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    // Launched from inside a Claude Code session, the child would take the host's auth and leave the file alone
    for (k, _) in std::env::vars_os() {
        let k = k.to_string_lossy();
        if k == "CLAUDECODE" || k.starts_with("CLAUDE_CODE_") {
            cmd.env_remove(k.as_ref());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn()?;
    let deadline = std::time::Instant::now() + Duration::from_secs(RENEW_TIMEOUT_SECS);
    while child.try_wait()?.is_none() {
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

#[derive(Default)]
struct Renewer {
    attempted_for: Option<u64>,
    last_attempt: Option<u64>,
}

impl Renewer {
    /// Renews if the token is about to expire. Some(true) = the expiry moved; judged on the outcome, never on the
    /// exit status, because refusing the empty prompt is a non-zero exit and a successful renewal at the same time
    fn maybe_renew(&mut self, cred: &Credential) -> Option<bool> {
        let now = now_ms();
        if !should_renew(cred.expires_at, now, self.attempted_for, self.last_attempt) {
            return None;
        }
        self.last_attempt = Some(now);
        self.attempted_for = cred.expires_at;
        let Some(cli) = find_cli() else {
            crate::applog("claude: token about to expire and no standalone claude CLI found to renew it");
            return Some(false);
        };
        if let Err(e) = run_renewal(&cli) {
            crate::applog(&format!("claude: token renewal could not start ({}): {e}", cli.display()));
            return Some(false);
        }
        let after = read_credentials().and_then(|c| c.expires_at);
        let renewed = matches!((after, cred.expires_at), (Some(a), Some(b)) if a > b);
        crate::applog(&if renewed {
            format!("claude: token renewed via {}", cli.display())
        } else {
            format!("claude: ran {} but the token expiry did not move", cli.display())
        });
        Some(renewed)
    }
}

fn parse_reset(v: &serde_json::Value) -> Option<u64> {
    v.as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.timestamp_millis().max(0) as u64)
}

fn label_for(kind: &str) -> String {
    match kind {
        "session" => "Current session".into(),
        "seven_day" | "weekly_all" => "Weekly (all models)".into(),
        "seven_day_opus" | "weekly_opus" => "Weekly (Opus)".into(),
        "weekly_scoped" => "Weekly (model-scoped)".into(),
        other => {
            // Forward compatibility: an unknown kind gets a readable label
            let mut s = other.replace('_', " ");
            if let Some(c) = s.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            s
        }
    }
}

fn parse_response(v: &serde_json::Value) -> Vec<LimitWindow> {
    let mut out: Vec<LimitWindow> = Vec::new();
    if let Some(arr) = v.get("limits").and_then(|x| x.as_array()) {
        for l in arr {
            let Some(kind) = l.get("kind").and_then(|x| x.as_str()) else {
                continue;
            };
            let Some(pct) = l.get("percent").and_then(|x| x.as_f64()) else {
                continue;
            };
            let resets = l.get("resets_at").and_then(parse_reset);
            if resets.is_none() {
                continue; // upstream rule: a window without a reset time is not shown
            }
            out.push(LimitWindow {
                id: kind.to_string(),
                label: label_for(kind),
                used: (pct / 100.0).clamp(0.0, 1.0),
                resets_at: resets, ..Default::default()
            });
        }
    }
    // Fallback merge: a window that just rolled over disappears from limits while the named field remains.
    // In practice the kinds in limits are weekly_all/weekly_scoped, not seven_day — deduplicating by id
    // alone would add the seven_day fallback a second time (the card showed "Weekly all" and
    // "Weekly (all models)" as twins). Three dedupe rules: id alias / same resets_at and percentage / same label.
    let aliases: [(&str, &str, &[&str]); 2] = [
        ("five_hour", "session", &["session", "five_hour"]),
        ("seven_day", "seven_day", &["seven_day", "weekly_all", "weekly"]),
    ];
    for (field, id, alias) in aliases {
        let Some(w) = v.get(field) else { continue };
        let Some(u) = w.get("utilization").and_then(|x| x.as_f64()) else { continue };
        let used = (u / 100.0).clamp(0.0, 1.0);
        let resets_at = w.get("resets_at").and_then(parse_reset);
        let label = label_for(id);
        let dup = out.iter().any(|x| {
            alias.contains(&x.id.as_str())
                || x.label == label
                || (resets_at.is_some()
                    && x.resets_at.map(|r| r / 1000) == resets_at.map(|r| r / 1000)
                    && (x.used - used).abs() < 0.005)
        });
        if dup {
            continue;
        }
        out.push(LimitWindow { id: id.into(), label, used, resets_at, ..Default::default() });
    }
    // session always comes first (upstream display order)
    out.sort_by_key(|w| if w.id == "session" { 0 } else { 1 });
    out
}

enum FetchErr {
    NeedsAuth,
    RateLimited(u64), // suggested wait in seconds (the Retry-After before the floor is applied)
    Other(String),
}

fn fetch_once(token: &str) -> Result<Vec<LimitWindow>, FetchErr> {
    let resp = ureq::get(ENDPOINT)
        .set("Authorization", &format!("Bearer {token}"))
        .set("anthropic-beta", "oauth-2025-04-20")
        .timeout(Duration::from_secs(15))
        .call();
    match resp {
        Ok(r) => {
            let v: serde_json::Value = r
                .into_json()
                .map_err(|e| FetchErr::Other(format!("parse: {e}")))?;
            Ok(parse_response(&v))
        }
        Err(ureq::Error::Status(401, _)) | Err(ureq::Error::Status(403, _)) => {
            Err(FetchErr::NeedsAuth)
        }
        Err(ureq::Error::Status(429, r)) => {
            let ra = r
                .header("retry-after")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            Err(FetchErr::RateLimited(ra))
        }
        Err(ureq::Error::Status(code, _)) => Err(FetchErr::Other(format!("HTTP {code}"))),
        Err(e) => Err(FetchErr::Other(format!("{e}"))),
    }
}

fn backoff_secs(consecutive: u32, retry_after_floor: u64) -> u64 {
    let exp = BACKOFF_BASE_SECS.saturating_mul(1u64 << consecutive.min(4));
    // The server's Retry-After is honoured in full: with expired tokens no longer
    // sent, a long one is a real rate limit, and retrying early only earns another.
    exp.clamp(BACKOFF_BASE_SECS, BACKOFF_CAP_SECS).max(retry_after_floor)
}

fn set_and_broadcast(app: &AppHandle, mutate: impl FnOnce(&mut UsageSnapshot)) {
    let st = app.state::<AppState>();
    let snap = {
        let mut u = st.usage.lock().unwrap();
        mutate(&mut u);
        u.clone()
    };
    persist(&snap);
    let _ = app.emit("usage", &snap);
}

pub fn start(app: AppHandle) {
    std::thread::spawn(move || {
        // Ajuste 2: validate (or discard) the persisted reading's account binding BEFORE ever
        // showing it — "stale beats blank" must never mean "the wrong account's numbers beat
        // blank". A persisted snapshot with no fingerprint at all (pre-T-CLAUDE-ACCOUNT-SWITCHER,
        // legacy) is always untrusted; one with a fingerprint is kept only if it equals the
        // account this first check actually finds. If the account cannot be resolved at all right
        // now (no CLI, or the call fails), that also counts as "not confirmed" — startup is the
        // one place this stays strict, unlike the steady-state check further down.
        {
            let persisted_fp = app.state::<AppState>().usage.lock().unwrap().account_fingerprint.clone();
            let current = find_cli()
                .and_then(|cli| claude_account::run_status(&cli, Duration::from_secs(claude_account::AUTH_STATUS_TIMEOUT_SECS)).ok());
            if let Some(status) = current.clone() {
                crate::applog(&format!("claude account: {:?}", status.state));
                crate::set_claude_account(&app, status);
            }
            let current_fp = current.and_then(|s| s.fingerprint);
            let trusted = matches!((&persisted_fp, &current_fp), (Some(p), Some(c)) if p == c);
            let st = app.state::<AppState>();
            let mut u = st.usage.lock().unwrap();
            if trusted {
                u.account_fingerprint = current_fp;
            } else {
                if !u.windows.is_empty() {
                    crate::applog("claude usage: discarding persisted snapshot (no confirmed account match at startup)");
                }
                u.windows.clear();
                u.status = "stale".into();
                u.account_fingerprint = current_fp;
            }
            let snap = u.clone();
            drop(u);
            let _ = app.emit("usage", &snap);
        }
        let mut consecutive_429: u32 = 0;
        let mut renewer = Renewer::default();
        loop {
            // Account identity check, every cycle, ahead of everything else: detects
            // ACTIVE ACCOUNT CHANGED before either usage source below is asked to interpret new
            // numbers under the old account's fingerprint (Fase 3/4). Resolved once and reused
            // for the `/usage` attempt right after, so this never doubles the CLI lookup cost.
            let cli = find_cli();
            if let Some(cli) = cli.as_deref() {
                if let Ok(status) = claude_account::run_status(cli, Duration::from_secs(claude_account::AUTH_STATUS_TIMEOUT_SECS)) {
                    let (changed, state_changed) = {
                        let st = app.state::<AppState>();
                        let prev = st.claude_account.lock().unwrap();
                        (prev.fingerprint != status.fingerprint, prev.state != status.state)
                    };
                    if changed {
                        crate::applog("claude account changed");
                        set_and_broadcast(&app, |u| {
                            u.windows.clear();
                            u.status = "stale".into();
                            u.account_fingerprint = status.fingerprint.clone();
                        });
                        crate::applog("claude usage refresh after account change");
                    } else if state_changed {
                        crate::applog(&format!("claude account: {:?}", status.state));
                    }
                    crate::set_claude_account(&app, status);
                }
            }
            // Ahead of the OAuth back-off on purpose, exactly like upstream's fetchSnapshot()
            // (ClaudeOAuthProvider.swift:150-163): the CLI does not share the token endpoint's
            // rate limit, so there is no reason for a 429 on one to darken a ring the other can
            // still fill. When the CLI answers, this whole cycle is done — the OAuth path below
            // (credential read, renewal, backoff, the token fetch itself) is skipped entirely, so
            // a working CLI can never spend an OAuth attempt or touch its persisted backoff.
            if let Some(cli) = cli.as_deref() {
                match run_cli_usage(cli, Duration::from_secs(CLI_USAGE_TIMEOUT_SECS)).and_then(|text| parse_cli_usage(&text)) {
                    Ok(windows) => {
                        *LAST_CLAUDE_SOURCE.lock().unwrap() = Some("claude_cli");
                        crate::applog(&format!("claude usage source: claude_cli ({} windows)", windows.len()));
                        let fp = app.state::<AppState>().claude_account.lock().unwrap().fingerprint.clone();
                        set_and_broadcast(&app, |u| {
                            u.status = "ok".into();
                            u.windows = windows;
                            u.fetched_at = now_ms();
                            u.note.clear();
                            u.account_fingerprint = fp;
                            // backoff_until is deliberately left untouched: it is the OAuth
                            // path's own state, and the CLI succeeding says nothing about
                            // whether the token endpoint is still rate limited.
                        });
                        let active = {
                            let st = app.state::<AppState>();
                            let store = st.store.lock().unwrap();
                            !store.snapshot("en", "en", false, false).sessions.is_empty()
                        };
                        sleep_interruptible(if active { POLL_ACTIVE_SECS } else { POLL_IDLE_SECS });
                        continue;
                    }
                    Err(e) => {
                        // NotFound/NeedsAuth/Timeout/BadResponse all mean the same thing here:
                        // fall through to the OAuth path below, unchanged. Logged (no secret in
                        // any of these variants) so a doctor/run.log read can tell why.
                        crate::applog(&format!("claude usage: CLI source unavailable ({e:?}), falling back to OAuth"));
                    }
                }
            }
            *LAST_CLAUDE_SOURCE.lock().unwrap() = Some("oauth_endpoint");
            // Ahead of the back-off: renewing never touches the usage endpoint, and a fresh token deserves a fresh try
            if let Some(cred) = read_credentials() {
                if renewer.maybe_renew(&cred) == Some(true) {
                    consecutive_429 = 0;
                    set_and_broadcast(&app, |u| u.backoff_until = 0);
                }
            }
            // No requests inside the backoff window
            let bu = {
                let st = app.state::<AppState>();
                let u = st.usage.lock().unwrap();
                u.backoff_until
            };
            let now = now_ms();
            if bu > now {
                sleep_interruptible(((bu - now) / 1000).clamp(1, 30));
                continue;
            }
            match read_credentials() {
                None => set_and_broadcast(&app, |u| {
                    u.status = "needsAuth".into();
                    u.note = "No Claude Code credential found".into();
                }),
                // Expired is not signed out: keep the last reading, dimmed and dated, and send nothing
                Some(cred) if cred.expired(now_ms()) => set_and_broadcast(&app, |u| {
                    u.status = if u.windows.is_empty() { "needsAuth" } else { "stale" }.into();
                    u.note = EXPIRED_NOTE.into();
                }),
                Some(cred) => {
                    let token = cred.token;
                    // On 401/403 re-read the credential and retry once (Claude Code may have just refreshed it)
                    let result = match fetch_once(&token) {
                        Err(FetchErr::NeedsAuth) => match read_credentials() {
                            Some(c2) if c2.token != token => fetch_once(&c2.token),
                            _ => Err(FetchErr::NeedsAuth),
                        },
                        other => other,
                    };
                    let auth_note = "Credential rejected (switched accounts?)";
                    match result {
                        Ok(windows) => {
                            consecutive_429 = 0;
                            set_and_broadcast(&app, |u| {
                                u.status = "ok".into();
                                u.windows = windows;
                                u.fetched_at = now_ms();
                                u.note.clear();
                                u.backoff_until = 0;
                            });
                        }
                        Err(FetchErr::NeedsAuth) => set_and_broadcast(&app, |u| {
                            u.status = "needsAuth".into();
                            u.note = auth_note.into();
                        }),
                        Err(FetchErr::RateLimited(ra)) => {
                            consecutive_429 += 1;
                            let wait = backoff_secs(consecutive_429 - 1, ra);
                            set_and_broadcast(&app, |u| {
                                if !u.windows.is_empty() {
                                    u.status = "stale".into();
                                }
                                u.note = format!("Rate limited, retrying in {wait}s");
                                u.backoff_until = now_ms() + wait * 1000;
                            });
                        }
                        Err(FetchErr::Other(msg)) => set_and_broadcast(&app, |u| {
                            if u.windows.is_empty() {
                                u.status = "error".into();
                            } else {
                                u.status = "stale".into();
                            }
                            u.note = msg;
                        }),
                    }
                }
            }
            // 60 s while a session is active, 300 s otherwise (upstream throttling discipline)
            let active = {
                let st = app.state::<AppState>();
                let store = st.store.lock().unwrap();
                let s = store.snapshot("en", "en", false, false);
                !s.sessions.is_empty()
            };
            sleep_interruptible(if active {
                POLL_ACTIVE_SECS
            } else {
                POLL_IDLE_SECS
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXP: u64 = 1_000_000_000;

    #[test]
    fn renews_only_inside_the_margin() {
        assert!(!should_renew(None, EXP, None, None), "never launch on a guess");
        assert!(!should_renew(Some(EXP), EXP - RENEW_MARGIN_MS - 1, None, None), "plenty of time left");
        assert!(should_renew(Some(EXP), EXP - RENEW_MARGIN_MS, None, None));
        assert!(should_renew(Some(EXP), EXP + 3_600_000, None, None), "already expired still renews");
    }

    #[test]
    fn one_attempt_per_token_and_a_cooldown() {
        let now = EXP + 1;
        assert!(!should_renew(Some(EXP), now, Some(EXP), None), "same token is never retried");
        assert!(!should_renew(Some(EXP + 5), now, Some(EXP), Some(now - 1000)), "cooldown holds a new token back");
        assert!(should_renew(Some(EXP + 5), now, Some(EXP), Some(now - RENEW_COOLDOWN_MS)));
    }

    #[test]
    fn retry_after_never_exceeds_the_cap() {
        assert_eq!(backoff_secs(0, 3600), 3600);
        assert_eq!(backoff_secs(0, 0), BACKOFF_BASE_SECS);
        assert_eq!(backoff_secs(1, 300), 300);
        assert_eq!(backoff_secs(9, 0), BACKOFF_CAP_SECS);
    }

    // ---------------- T-CLAUDE-USAGE-01: claude "/usage" text parsing ----------------

    #[test]
    fn cli_line_session_with_reset_parses() {
        let (kind, pct, reset) =
            parse_cli_line("Current session: 38% used · resets Sep 7 at 2:59pm (Asia/Jakarta)").unwrap();
        assert_eq!(kind, "session");
        assert_eq!(pct, 38.0);
        assert_eq!(reset.as_deref(), Some("Sep 7 at 2:59pm (Asia/Jakarta)"));
    }

    #[test]
    fn cli_line_weekly_all_models_parses() {
        let (kind, pct, _) =
            parse_cli_line("Current week (all models): 4% used · resets Sep 14 at 5:59am (Asia/Jakarta)").unwrap();
        assert_eq!(kind, "weekly_all");
        assert_eq!(pct, 4.0);
    }

    #[test]
    fn cli_line_weekly_named_model_parses() {
        let (kind, pct, reset) = parse_cli_line("Current week (Opus): 12% used").unwrap();
        assert_eq!(kind, "weekly_opus");
        assert_eq!(pct, 12.0);
        assert!(reset.is_none(), "no resets clause present must not invent one");
    }

    #[test]
    fn cli_line_percent_scale_is_preserved_not_rescaled() {
        // 37 must become the fraction 0.37 exactly once — never 3700, never re-divided.
        let (_, pct, _) = parse_cli_line("Current session: 37% used").unwrap();
        assert_eq!(pct / 100.0, 0.37);
    }

    #[test]
    fn cli_line_explicit_zero_percent_is_a_real_reading_not_a_rejection() {
        let (kind, pct, _) = parse_cli_line("Current session: 0% used").unwrap();
        assert_eq!(kind, "session");
        assert_eq!(pct, 0.0);
    }

    #[test]
    fn cli_line_unrecognized_text_is_none_not_zero() {
        assert!(parse_cli_line("Estimated based on your recent usage.").is_none());
        assert!(parse_cli_line("").is_none());
        assert!(parse_cli_line("Current session used 38%").is_none(), "missing the colon shape must not fuzzy-match");
    }

    #[test]
    fn cli_usage_full_valid_report_parses_both_windows_session_first() {
        let text = "Some plan header\n\
                     Current session: 38% used · resets Sep 7 at 2:59pm (Asia/Jakarta)\n\
                     Current week (all models): 4% used · resets Sep 14 at 5:59am (Asia/Jakarta)\n\
                     \n\
                     This estimate is based on recent activity and may not be exact.\n";
        let windows = parse_cli_usage(text).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].id, "session");
        assert_eq!(windows[0].used, 0.38);
        assert_eq!(windows[1].id, "weekly_all");
        assert_eq!(windows[1].used, 0.04);
        assert!(windows[0].resets_at.is_some());
    }

    #[test]
    fn cli_usage_session_only_report_has_no_weekly_window_not_a_zero_one() {
        let text = "Current session: 12% used\n";
        let windows = parse_cli_usage(text).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].id, "session");
        assert!(!windows.iter().any(|w| w.id.starts_with("weekly")), "absent weekly must not appear as 0%");
    }

    #[test]
    fn cli_usage_without_a_session_line_is_rejected() {
        let text = "Current week (all models): 4% used\n";
        assert!(matches!(parse_cli_usage(text), Err(CliUsageErr::BadResponse)));
    }

    #[test]
    fn cli_usage_unknown_format_is_rejected_not_read_as_zero() {
        let text = "Claude Code v2.1.259\nSomething changed in the output format.\n";
        assert!(matches!(parse_cli_usage(text), Err(CliUsageErr::BadResponse)));
    }

    #[test]
    fn cli_usage_partial_garbled_output_does_not_crash() {
        let text = "Current session: %% used\nCurrent week (\nrandom\x00binary\x01noise";
        // Must not panic; garbled lines are simply skipped, and with no valid session line the
        // whole report is rejected rather than partially trusted.
        assert!(matches!(parse_cli_usage(text), Err(CliUsageErr::BadResponse)));
    }

    #[test]
    fn cli_usage_duplicate_session_line_keeps_the_first() {
        let text = "Current session: 10% used\nCurrent session: 90% used\n";
        let windows = parse_cli_usage(text).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].used, 0.10);
    }

    #[test]
    fn cli_reset_with_minutes_parses_exact_time() {
        use chrono::{Datelike, TimeZone, Timelike};
        let now = chrono::Local.with_ymd_and_hms(2026, 9, 1, 12, 0, 0).unwrap();
        let ms = parse_cli_reset("Sep 7 at 2:59pm (Asia/Jakarta)", now).unwrap();
        let dt = chrono::Local.timestamp_millis_opt(ms as i64).unwrap();
        assert_eq!(dt.month(), 9);
        assert_eq!(dt.day(), 7);
        assert_eq!(dt.hour(), 14);
        assert_eq!(dt.minute(), 59);
        assert_eq!(dt.year(), 2026, "must pick the year nearest `now`, not always the current one");
    }

    #[test]
    fn cli_reset_real_cli_comma_format_parses() {
        // The exact wording the installed Claude Code CLI (2.1.267) actually prints — verified by
        // hand, not the " at " form upstream's own doc comment describes.
        use chrono::{Datelike, TimeZone, Timelike};
        let now = chrono::Local.with_ymd_and_hms(2026, 9, 17, 3, 0, 0).unwrap();
        let ms = parse_cli_reset("Sep 17, 7:49am (America/Sao_Paulo)", now).unwrap();
        let dt = chrono::Local.timestamp_millis_opt(ms as i64).unwrap();
        assert_eq!(dt.month(), 9);
        assert_eq!(dt.day(), 17);
        assert_eq!(dt.hour(), 7);
        assert_eq!(dt.minute(), 49);
    }

    #[test]
    fn cli_usage_real_cli_output_shape_parses_correctly() {
        // A trimmed-down real capture (percentages/format only) from this machine's own
        // `claude --print --no-session-persistence --strict-mcp-config` with `/usage` on stdin.
        let text = "You are currently using your subscription to power your Claude Code usage\n\n\
                     Current session: 2% used · resets Sep 17, 7:49am (America/Sao_Paulo)\n\
                     Current week (all models): 0% used · resets Sep 23, 10:59pm (America/Sao_Paulo)\n\n\
                     What's contributing to your limits usage?\n";
        let windows = parse_cli_usage(text).unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].id, "session");
        assert_eq!(windows[0].used, 0.02);
        assert_eq!(windows[1].id, "weekly_all");
        assert_eq!(windows[1].used, 0.0, "an explicit 0% from the real CLI is still a valid reading");
        assert!(windows[0].resets_at.is_some());
        assert!(windows[1].resets_at.is_some());
    }

    #[test]
    fn cli_reset_without_minutes_parses_on_the_hour() {
        use chrono::{TimeZone, Timelike};
        let now = chrono::Local.with_ymd_and_hms(2026, 9, 1, 12, 0, 0).unwrap();
        let ms = parse_cli_reset("Sep 7 at 3pm (Asia/Jakarta)", now).unwrap();
        let dt = chrono::Local.timestamp_millis_opt(ms as i64).unwrap();
        assert_eq!(dt.hour(), 15);
        assert_eq!(dt.minute(), 0);
    }

    #[test]
    fn cli_reset_unparseable_text_is_none_not_a_guess() {
        let now = chrono::Local::now();
        assert!(parse_cli_reset("sometime soon", now).is_none());
        assert!(parse_cli_reset("", now).is_none());
    }

    // ---------------- subprocess plumbing (mirrors agy_cli.rs's own test style) ----------------

    #[cfg(windows)]
    fn write_test_cmd(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("codenotch-usagecli-{}-{name}.cmd", std::process::id()));
        std::fs::write(&p, body).unwrap();
        p
    }

    #[cfg(windows)]
    #[test]
    fn run_cli_usage_reports_not_found_for_a_missing_binary() {
        let missing = std::path::PathBuf::from(r"C:\does\not\exist\claude.exe");
        assert!(matches!(run_cli_usage(&missing, Duration::from_secs(5)), Err(CliUsageErr::NotFound)));
    }

    #[cfg(windows)]
    #[test]
    fn run_cli_usage_treats_nonzero_exit_as_needs_auth() {
        let script = write_test_cmd("nonzero", "@echo off\r\nexit /b 3\r\n");
        let result = run_cli_usage(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(CliUsageErr::NeedsAuth)));
    }

    #[cfg(windows)]
    #[test]
    fn run_cli_usage_captures_stdout_on_success() {
        let script = write_test_cmd("valid", "@echo off\r\necho Current session: 37%% used\r\n");
        let result = run_cli_usage(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        let text = result.expect("script exits 0 with output");
        assert!(text.contains("Current session: 37% used"));
    }

    #[cfg(windows)]
    #[test]
    fn run_cli_usage_kills_a_wedged_process_on_timeout() {
        // ping to an address that will not answer, well past the timeout below.
        let script = write_test_cmd("hang", "@echo off\r\nping -n 30 127.0.0.1 >nul\r\n");
        let started = std::time::Instant::now();
        let result = run_cli_usage(&script, Duration::from_millis(300));
        let elapsed = started.elapsed();
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(CliUsageErr::Timeout)));
        assert!(elapsed < Duration::from_secs(10), "must not wait anywhere near the script's own 30s");
    }

    #[cfg(windows)]
    #[test]
    fn run_cli_usage_empty_output_is_bad_response() {
        let script = write_test_cmd("empty", "@echo off\r\n");
        let result = run_cli_usage(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(CliUsageErr::BadResponse)));
    }

    #[test]
    fn desktop_bundled_cli_is_refused() {
        use std::path::Path;
        assert!(is_desktop_owned(Path::new(r"C:\Users\u\AppData\Local\AnthropicClaude\app-1.2.3\claude.exe")));
        assert!(is_desktop_owned(Path::new(r"C:\Users\u\AppData\Roaming\Claude\claude-code\2.1.0\claude.exe")));
        assert!(!is_desktop_owned(Path::new(r"C:\Users\u\.local\bin\claude.exe")));
        assert!(!is_desktop_owned(Path::new(r"C:\Users\u\AppData\Roaming\npm\claude.cmd")));
    }

    #[test]
    #[ignore = "Runs the installed standalone claude CLI; opt in for integration verification"]
    fn live_renewal_runs_the_standalone_cli() {
        let cli = find_cli().expect("a standalone claude CLI");
        assert!(!is_desktop_owned(&cli));
        let before = read_credentials().and_then(|c| c.expires_at);
        let t = std::time::Instant::now();
        run_renewal(&cli).expect("spawned");
        assert!(t.elapsed() < Duration::from_secs(RENEW_TIMEOUT_SECS), "returned before the timeout");
        let after = read_credentials().and_then(|c| c.expires_at);
        assert!(after >= before, "the expiry never moves backwards");
        eprintln!("cli: {}", cli.display());
    }

    #[test]
    fn expired_is_judged_against_now() {
        let c = Credential { token: "t".into(), expires_at: Some(EXP) };
        assert!(c.expired(EXP));
        assert!(!c.expired(EXP - 1));
        assert!(!Credential { token: "t".into(), expires_at: None }.expired(EXP));
    }
}
