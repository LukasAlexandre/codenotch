//! Claude account identity: detect which Claude account is active, without ever owning
//! authentication ourselves.
//!
//! Claude Code remains the sole owner of login/logout/token management. This module only:
//!   1. asks the standalone CLI who is signed in (`claude auth status --json`, an official,
//!      documented, JSON-emitting subcommand — never `~/.claude/.credentials.json` parsing, see
//!      the architecture report's Fase 2);
//!   2. starts the CLI's own official browser-based login/logout flow on request
//!      (`claude auth login` / `claude auth logout`) and waits for it to finish;
//!   3. derives a local-only fingerprint so the rest of the app can tell "same account" from
//!      "different account" without ever holding the email in the clear anywhere serializable.
//!
//! Every subprocess call here goes through `crate::usage::run_claude_subprocess` — the same
//! hidden/Job-Object/timeout-bounded runner `/usage` already uses (T-CLAUDE-USAGE-01).

use crate::usage::ClaudeProcErr;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const AUTH_STATUS_ARGS: &[&str] = &["auth", "status"]; // --json is documented as the default
const AUTH_LOGIN_ARGS: &[&str] = &["auth", "login"];
const AUTH_LOGOUT_ARGS: &[&str] = &["auth", "logout"];

/// `claude auth status` measured at ~0.5 s on this machine (a local read, not a slow network
/// round trip like `/usage`'s "cold Node start") — generous headroom, still short enough to run
/// every poll cycle without being felt.
pub const AUTH_STATUS_TIMEOUT_SECS: u64 = 10;
/// `claude auth login` opens the system browser and waits for a human to complete an OAuth PKCE
/// flow there (verified by hand — see the architecture report's Fase "Official login mechanism").
/// Long enough for a real person to actually do that, short enough that an abandoned flow doesn't
/// hold a process open forever.
pub const AUTH_LOGIN_TIMEOUT_SECS: u64 = 300;
/// `claude auth logout` is expected to be a fast local operation, like `auth status` — kept
/// separate as its own constant since it is a distinct command or upstream could change it.
pub const AUTH_LOGOUT_TIMEOUT_SECS: u64 = 15;

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn scratch_dir() -> Option<std::path::PathBuf> {
    // Shares the same fixed scratch directory as `/usage` (see usage.rs::cli_usage_scratch_dir):
    // one directory, created once, so `auth status`/`login`/`logout` never scatter new folders
    // under whatever directory the app happens to run from.
    let dir = dirs::config_dir()?.join("codenotch").join("usage-scratch");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Where a `ClaudeAccountStatus` reading came from. An enum rather than a bare marker so a future
/// second source (e.g. a desktop-app cache, symmetrical with `/usage`'s own two sources) does not
/// need a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountSource {
    ClaudeCli,
}

/// UI/lifecycle state for the account panel. Distinct from `usage::UsageSnapshot::status`, which
/// is about the *usage numbers'* freshness — this is about the *identity/session* itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaudeAccountState {
    Connected,
    NotConnected,
    SwitchingAccount,
    SigningIn,
    SigningOut,
    /// Not a login/switch — the periodic identity+usage recheck already in flight.
    Refreshing,
    /// `auth login` did not finish inside its timeout: not an error, and not "still working" —
    /// upstream's own browser-redirect step needs a human, and our subprocess has no interactive
    /// stdin to fall back to the CLI's own "paste code here" prompt. See Ajuste 4.
    ManualActionRequired,
    Error,
}

/// How much can actually be trusted about `fingerprint`. `claude auth status` is allowed to come
/// back missing `email` (its own schema marks nothing as required beyond `loggedIn`) — a weaker
/// fingerprint is still usable for change *detection*, but must never be presented or reasoned
/// about as if it were the strong one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityStrength {
    /// sha256(normalized email + "|" + org id).
    Strong,
    /// Org id only — no email in the reply.
    OrgOnly,
    /// Connected, but neither field was present to fingerprint.
    None,
}

/// A single, point-in-time reading of which Claude account (if any) `claude auth status` reports
/// as active. Never carries a full email address, a token, or anything from
/// `~/.claude/.credentials.json` — see `mask_email`/`fingerprint_of`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeAccountStatus {
    pub state: ClaudeAccountState,
    /// e.g. `"someo********@example.test"` — see `mask_email`. `None` when not connected or the
    /// reply carried no email.
    pub email_masked: Option<String>,
    pub plan: Option<String>,
    pub organization: Option<String>,
    /// Local-only identity marker — see `IdentityStrength`. Never the email itself in any form,
    /// masked or not; safe to persist and to log.
    pub fingerprint: Option<String>,
    pub identity_strength: IdentityStrength,
    pub last_sync_at: u64,
    pub source: AccountSource,
    /// A short, fixed, internal string for `ManualActionRequired`/`Error` states — never a raw
    /// subprocess error message, which could embed a path or other machine detail.
    pub note: Option<String>,
}

impl Default for ClaudeAccountStatus {
    fn default() -> Self {
        Self {
            state: ClaudeAccountState::Refreshing,
            email_masked: None,
            plan: None,
            organization: None,
            fingerprint: None,
            identity_strength: IdentityStrength::None,
            last_sync_at: 0,
            source: AccountSource::ClaudeCli,
            note: None,
        }
    }
}

impl ClaudeAccountStatus {
    pub fn not_connected(now: u64) -> Self {
        Self { state: ClaudeAccountState::NotConnected, last_sync_at: now, ..Self::default() }
    }
}

/// `"someone123@example.test"` -> `"someo********@example.test"`. Always exactly 8 mask
/// characters regardless of the real local-part length, so the mask itself never leaks how long
/// the address is. Anything that is not `local@domain` shaped (empty, no `@`, empty side) masks
/// to a fixed placeholder rather than being echoed back as-is.
pub fn mask_email(email: &str) -> String {
    let email = email.trim();
    match email.split_once('@') {
        Some((local, domain)) if !local.is_empty() && !domain.is_empty() => {
            let keep: String = local.chars().take(5).collect();
            format!("{keep}********@{domain}")
        }
        _ => "********".to_string(),
    }
}

fn normalize_email(email: &str) -> String {
    email.trim().to_lowercase()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex(&hasher.finalize())
}

/// Ajuste 1: email alone or org id alone cannot tell two accounts apart on their own (two
/// people, or two profiles, can share an org; the same person can hold accounts in different
/// orgs) — the fingerprint is derived from *both* together. Only ever returns a sha256 hex
/// digest or `None`; never the raw email, masked or not.
fn fingerprint_of(email: Option<&str>, org_id: Option<&str>) -> (Option<String>, IdentityStrength) {
    let email = email.map(str::trim).filter(|e| !e.is_empty());
    let org_id = org_id.map(str::trim).filter(|o| !o.is_empty());
    match (email, org_id) {
        (Some(email), Some(org)) => {
            let input = format!("{}|{org}", normalize_email(email));
            (Some(sha256_hex(&input)), IdentityStrength::Strong)
        }
        (None, Some(org)) => (Some(sha256_hex(&format!("org-only|{org}"))), IdentityStrength::OrgOnly),
        _ => (None, IdentityStrength::None),
    }
}

/// The exact shape of `claude auth status`'s JSON reply, as captured live from the CLI actually
/// installed here — see the architecture report. Every field but `loggedIn` is optional: the
/// schema makes no promise beyond that one, and a future field this struct doesn't know about is
/// silently ignored by serde rather than failing the whole parse.
#[derive(Debug, Deserialize)]
struct AuthStatusWire {
    #[serde(rename = "loggedIn")]
    logged_in: bool,
    email: Option<String>,
    #[serde(rename = "orgId")]
    org_id: Option<String>,
    #[serde(rename = "orgName")]
    org_name: Option<String>,
    #[serde(rename = "subscriptionType")]
    subscription_type: Option<String>,
}

/// Reasons a `ClaudeAccountStatus` could not be produced at all.
#[derive(Debug)]
pub enum AccountStatusErr {
    NotFound,
    Timeout,
    /// Empty output, invalid JSON, or missing the one required field (`loggedIn`) — never
    /// silently read as "not connected": that is a distinct, positive `loggedIn: false` reply,
    /// not the absence of one.
    BadResponse,
}

/// `claude auth status` -> JSON -> `ClaudeAccountStatus`. No text parsing: the CLI's own
/// documented `--json` (the default) output is decoded directly with `serde_json`.
pub fn parse_auth_status(text: &str, now: u64) -> Result<ClaudeAccountStatus, AccountStatusErr> {
    let text = text.trim();
    if text.is_empty() {
        return Err(AccountStatusErr::BadResponse);
    }
    let wire: AuthStatusWire = serde_json::from_str(text).map_err(|_| AccountStatusErr::BadResponse)?;
    if !wire.logged_in {
        return Ok(ClaudeAccountStatus::not_connected(now));
    }
    let (fingerprint, identity_strength) = fingerprint_of(wire.email.as_deref(), wire.org_id.as_deref());
    Ok(ClaudeAccountStatus {
        state: ClaudeAccountState::Connected,
        email_masked: wire.email.as_deref().map(mask_email),
        plan: wire.subscription_type,
        organization: wire.org_name,
        fingerprint,
        identity_strength,
        last_sync_at: now,
        source: AccountSource::ClaudeCli,
        note: None,
    })
}

/// Runs `claude auth status` and parses its reply. The one function the polling loop in
/// `usage.rs` calls every cycle.
pub fn run_status(cli: &Path, timeout: Duration) -> Result<ClaudeAccountStatus, AccountStatusErr> {
    let scratch = scratch_dir().ok_or(AccountStatusErr::BadResponse)?;
    let out = crate::usage::run_claude_subprocess(cli, AUTH_STATUS_ARGS, None, &scratch, timeout).map_err(
        |e| match e {
            ClaudeProcErr::NotFound => AccountStatusErr::NotFound,
            ClaudeProcErr::Timeout => AccountStatusErr::Timeout,
            ClaudeProcErr::WaitFailed => AccountStatusErr::BadResponse,
        },
    )?;
    parse_auth_status(&out.stdout, now_ms())
}

/// Reasons `claude auth login`/`claude auth logout` did not complete successfully.
#[derive(Debug)]
pub enum AuthActionErr {
    NotFound,
    /// Login only, in practice: the flow did not finish inside the timeout — most likely the
    /// browser step was never completed. See `ClaudeAccountState::ManualActionRequired`.
    Timeout,
    /// The process ran and exited non-zero, or produced no confirmable result.
    Failed,
}

/// Starts the official `claude auth login` flow (opens the system browser to Anthropic's own
/// OAuth page — verified by hand, see the architecture report) and blocks the calling thread
/// until it finishes or `timeout` elapses. Never reads, writes, or even looks at a token: the
/// subprocess's own stdout/exit code are the only signals used, and stdout is not logged.
///
/// Known limitation (Ajuste 4, deliberate for V1): the CLI's own "paste code here if prompted"
/// fallback cannot work through this call — stdin is never wired to anything interactive, exactly
/// like `/usage`'s call. The expected path (automatic browser redirect) does not need it; if that
/// path fails, this returns `Timeout` and the caller surfaces `ManualActionRequired`, not a
/// destructive retry or an invented success.
pub fn run_login(cli: &Path, timeout: Duration) -> Result<(), AuthActionErr> {
    let scratch = scratch_dir().ok_or(AuthActionErr::Failed)?;
    let out = crate::usage::run_claude_subprocess(cli, AUTH_LOGIN_ARGS, None, &scratch, timeout).map_err(
        |e| match e {
            ClaudeProcErr::NotFound => AuthActionErr::NotFound,
            ClaudeProcErr::Timeout => AuthActionErr::Timeout,
            ClaudeProcErr::WaitFailed => AuthActionErr::Failed,
        },
    )?;
    if out.success {
        Ok(())
    } else {
        Err(AuthActionErr::Failed)
    }
}

/// Starts the official `claude auth logout`. Implemented and tested (Ajuste 3), but **not**
/// wired to a UI action yet — the "Sign out" button stays disabled/hidden until a supervised
/// manual run confirms this has no surprising side effect (see the architecture report's Fase
/// 15). Calling this from anywhere in production code before that confirmation is a policy
/// violation of this task, not a technical one — the function itself is safe to call in tests
/// against a fake `cli` binary.
pub fn run_logout(cli: &Path, timeout: Duration) -> Result<(), AuthActionErr> {
    let scratch = scratch_dir().ok_or(AuthActionErr::Failed)?;
    let out = crate::usage::run_claude_subprocess(cli, AUTH_LOGOUT_ARGS, None, &scratch, timeout).map_err(
        |e| match e {
            ClaudeProcErr::NotFound => AuthActionErr::NotFound,
            ClaudeProcErr::Timeout => AuthActionErr::Timeout,
            ClaudeProcErr::WaitFailed => AuthActionErr::Failed,
        },
    )?;
    if out.success {
        Ok(())
    } else {
        Err(AuthActionErr::Failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAKE_EMAIL_A: &str = "user@example.test";
    const FAKE_EMAIL_A_MIXED_CASE: &str = "User@Example.Test";
    const FAKE_EMAIL_B: &str = "other@example.test";
    const FAKE_ORG_X: &str = "org-x-00000000-0000-0000-0000-000000000000";
    const FAKE_ORG_Y: &str = "org-y-00000000-0000-0000-0000-000000000000";

    // ---------------- T001: mask_email ----------------

    #[test]
    fn mask_email_keeps_first_five_chars_and_domain() {
        assert_eq!(mask_email("someone123456@example.test"), "someo********@example.test");
    }

    #[test]
    fn mask_email_short_local_part_still_masks_fixed_length() {
        assert_eq!(mask_email("ab@example.test"), "ab********@example.test");
    }

    #[test]
    fn mask_email_never_returns_the_original_string() {
        for email in [FAKE_EMAIL_A, FAKE_EMAIL_B, "a@b.co", "x.y.z@sub.example.test"] {
            let masked = mask_email(email);
            assert_ne!(masked, email);
            assert!(!masked.contains("@example.test") || masked.ends_with("@example.test"), "domain must be exact, not embedded elsewhere");
        }
    }

    #[test]
    fn mask_email_malformed_input_is_a_fixed_placeholder() {
        assert_eq!(mask_email(""), "********");
        assert_eq!(mask_email("not-an-email"), "********");
        assert_eq!(mask_email("@nodomain"), "********");
        assert_eq!(mask_email("nolocal@"), "********");
    }

    // ---------------- T001/T004: fingerprint ----------------

    #[test]
    fn fingerprint_is_sha256_hex_and_never_contains_the_email() {
        let (fp, strength) = fingerprint_of(Some(FAKE_EMAIL_A), Some(FAKE_ORG_X));
        let fp = fp.unwrap();
        assert_eq!(strength, IdentityStrength::Strong);
        assert_eq!(fp.len(), 64, "sha256 hex digest is 64 chars");
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!fp.contains(FAKE_EMAIL_A) && !fp.to_lowercase().contains("example.test"));
    }

    #[test]
    fn fingerprint_email_normalization_trim_and_lowercase() {
        let (fp1, _) = fingerprint_of(Some(FAKE_EMAIL_A), Some(FAKE_ORG_X));
        let (fp2, _) = fingerprint_of(Some(FAKE_EMAIL_A_MIXED_CASE), Some(FAKE_ORG_X));
        let (fp3, _) = fingerprint_of(Some("  user@example.test  "), Some(FAKE_ORG_X));
        assert_eq!(fp1, fp2, "case must not affect the fingerprint");
        assert_eq!(fp1, fp3, "surrounding whitespace must not affect the fingerprint");
    }

    /// Ajuste 1's whole point: two different accounts in the *same* org must fingerprint
    /// differently, and the same account read twice must fingerprint identically.
    #[test]
    fn same_org_different_accounts_have_different_fingerprints() {
        let (fp_a, _) = fingerprint_of(Some(FAKE_EMAIL_A), Some(FAKE_ORG_X));
        let (fp_b, _) = fingerprint_of(Some(FAKE_EMAIL_B), Some(FAKE_ORG_X));
        assert_ne!(fp_a, fp_b, "same org must not collapse two different accounts into one fingerprint");
    }

    #[test]
    fn same_account_read_twice_has_identical_fingerprint() {
        let (fp1, _) = fingerprint_of(Some(FAKE_EMAIL_A), Some(FAKE_ORG_X));
        let (fp2, _) = fingerprint_of(Some(FAKE_EMAIL_A), Some(FAKE_ORG_X));
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn same_account_different_org_has_different_fingerprint() {
        let (fp_x, _) = fingerprint_of(Some(FAKE_EMAIL_A), Some(FAKE_ORG_X));
        let (fp_y, _) = fingerprint_of(Some(FAKE_EMAIL_A), Some(FAKE_ORG_Y));
        assert_ne!(fp_x, fp_y);
    }

    #[test]
    fn fingerprint_falls_back_to_org_only_when_email_absent() {
        let (fp, strength) = fingerprint_of(None, Some(FAKE_ORG_X));
        assert!(fp.is_some());
        assert_eq!(strength, IdentityStrength::OrgOnly);
    }

    #[test]
    fn fingerprint_is_none_when_nothing_usable_is_present() {
        let (fp, strength) = fingerprint_of(None, None);
        assert!(fp.is_none());
        assert_eq!(strength, IdentityStrength::None);
    }

    #[test]
    fn fingerprint_empty_strings_are_treated_as_absent() {
        let (fp, strength) = fingerprint_of(Some(""), Some("   "));
        assert!(fp.is_none());
        assert_eq!(strength, IdentityStrength::None);
    }

    // ---------------- T003: parse_auth_status ----------------

    #[test]
    fn parse_auth_status_logged_in_full_reply() {
        let json = format!(
            r#"{{"loggedIn":true,"authMethod":"claude.ai","email":"{FAKE_EMAIL_A}","orgId":"{FAKE_ORG_X}","orgName":"Example Org","subscriptionType":"pro"}}"#
        );
        let status = parse_auth_status(&json, 1000).unwrap();
        assert_eq!(status.state, ClaudeAccountState::Connected);
        assert_eq!(status.email_masked.as_deref(), Some("user********@example.test"));
        assert_eq!(status.plan.as_deref(), Some("pro"));
        assert_eq!(status.organization.as_deref(), Some("Example Org"));
        assert_eq!(status.identity_strength, IdentityStrength::Strong);
        assert!(status.fingerprint.is_some());
        assert_eq!(status.last_sync_at, 1000);
    }

    #[test]
    fn parse_auth_status_logged_out() {
        let status = parse_auth_status(r#"{"loggedIn":false}"#, 2000).unwrap();
        assert_eq!(status.state, ClaudeAccountState::NotConnected);
        assert!(status.email_masked.is_none());
        assert!(status.fingerprint.is_none());
        assert_eq!(status.identity_strength, IdentityStrength::None);
    }

    #[test]
    fn parse_auth_status_logged_in_without_email() {
        let json = format!(r#"{{"loggedIn":true,"orgId":"{FAKE_ORG_X}","subscriptionType":"pro"}}"#);
        let status = parse_auth_status(&json, 3000).unwrap();
        assert_eq!(status.state, ClaudeAccountState::Connected);
        assert!(status.email_masked.is_none());
        assert_eq!(status.identity_strength, IdentityStrength::OrgOnly);
        assert!(status.fingerprint.is_some());
    }

    #[test]
    fn parse_auth_status_logged_in_without_org_id() {
        let json = format!(r#"{{"loggedIn":true,"email":"{FAKE_EMAIL_A}"}}"#);
        let status = parse_auth_status(&json, 4000).unwrap();
        assert_eq!(status.identity_strength, IdentityStrength::None, "email alone is not enough for Ajuste 1's fingerprint");
        assert!(status.fingerprint.is_none());
    }

    #[test]
    fn parse_auth_status_partial_json_missing_required_field_is_bad_response() {
        // No "loggedIn" at all — the one field the schema actually requires.
        assert!(matches!(parse_auth_status(r#"{"email":"user@example.test"}"#, 0), Err(AccountStatusErr::BadResponse)));
    }

    #[test]
    fn parse_auth_status_invalid_json_is_bad_response() {
        assert!(matches!(parse_auth_status("not json at all", 0), Err(AccountStatusErr::BadResponse)));
    }

    #[test]
    fn parse_auth_status_empty_output_is_bad_response() {
        assert!(matches!(parse_auth_status("", 0), Err(AccountStatusErr::BadResponse)));
        assert!(matches!(parse_auth_status("   \n  ", 0), Err(AccountStatusErr::BadResponse)));
    }

    #[test]
    fn parse_auth_status_unknown_extra_fields_are_ignored_not_fatal() {
        let json = format!(
            r#"{{"loggedIn":true,"email":"{FAKE_EMAIL_A}","orgId":"{FAKE_ORG_X}","someBrandNewField":{{"nested":true}}}}"#
        );
        assert!(parse_auth_status(&json, 0).is_ok());
    }

    // ---------------- security: no secret markers anywhere ----------------

    #[test]
    fn no_serialized_output_ever_contains_the_full_email_or_credential_markers() {
        let json = format!(
            r#"{{"loggedIn":true,"email":"{FAKE_EMAIL_A}","orgId":"{FAKE_ORG_X}","orgName":"Example Org","subscriptionType":"pro"}}"#
        );
        let status = parse_auth_status(&json, 5000).unwrap();
        let serialized = serde_json::to_string(&status).unwrap();
        let debug = format!("{status:?}");
        for surface in [&serialized, &debug] {
            assert!(!surface.contains(FAKE_EMAIL_A), "full email leaked: {surface}");
            for marker in ["accessToken", "refreshToken", "authorization", "bearer", "credentials"] {
                assert!(!surface.to_lowercase().contains(&marker.to_lowercase()), "{marker} leaked: {surface}");
            }
        }
    }

    // ---------------- subprocess plumbing (mirrors usage.rs's own test style) ----------------

    #[cfg(windows)]
    fn write_test_cmd(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("codenotch-claudeaccount-{}-{name}.cmd", std::process::id()));
        std::fs::write(&p, body).unwrap();
        p
    }

    #[cfg(windows)]
    #[test]
    fn run_status_reports_not_found_for_a_missing_binary() {
        let missing = std::path::PathBuf::from(r"C:\does\not\exist\claude.exe");
        assert!(matches!(run_status(&missing, Duration::from_secs(5)), Err(AccountStatusErr::NotFound)));
    }

    #[cfg(windows)]
    #[test]
    fn run_status_parses_real_looking_cli_output() {
        let body = format!(
            "@echo off\r\necho {{\"loggedIn\":true,\"email\":\"{FAKE_EMAIL_A}\",\"orgId\":\"{FAKE_ORG_X}\",\"subscriptionType\":\"pro\"}}\r\n"
        );
        let script = write_test_cmd("status-ok", &body);
        let result = run_status(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        let status = result.expect("valid JSON on stdout");
        assert_eq!(status.state, ClaudeAccountState::Connected);
        assert_eq!(status.plan.as_deref(), Some("pro"));
    }

    #[cfg(windows)]
    #[test]
    fn run_status_unknown_output_is_bad_response_not_a_crash() {
        let script = write_test_cmd("status-garbage", "@echo off\r\necho not json\r\n");
        let result = run_status(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(AccountStatusErr::BadResponse)));
    }

    #[cfg(windows)]
    #[test]
    fn run_status_timeout_is_reported_not_hung_forever() {
        let script = write_test_cmd("status-hang", "@echo off\r\nping -n 30 127.0.0.1 >nul\r\n");
        let started = std::time::Instant::now();
        let result = run_status(&script, Duration::from_millis(300));
        let elapsed = started.elapsed();
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(AccountStatusErr::Timeout)));
        assert!(elapsed < Duration::from_secs(10));
    }

    #[cfg(windows)]
    #[test]
    fn run_login_reports_failed_on_nonzero_exit() {
        let script = write_test_cmd("login-fail", "@echo off\r\nexit /b 1\r\n");
        let result = run_login(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(AuthActionErr::Failed)));
    }

    #[cfg(windows)]
    #[test]
    fn run_login_timeout_maps_to_timeout_not_failed() {
        // Simulates the browser-flow-never-completed case: the process just hangs.
        let script = write_test_cmd("login-hang", "@echo off\r\nping -n 30 127.0.0.1 >nul\r\n");
        let result = run_login(&script, Duration::from_millis(300));
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(AuthActionErr::Timeout)));
    }

    #[cfg(windows)]
    #[test]
    fn run_login_success_on_zero_exit() {
        let script = write_test_cmd("login-ok", "@echo off\r\nexit /b 0\r\n");
        let result = run_login(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        assert!(result.is_ok());
    }

    /// Ajuste 3 / Fase 15: `run_logout` itself is implemented and tested — but only against a
    /// fake, harmless `.cmd` fixture. Never against the real installed `claude` binary from an
    /// automated test; that only happens by hand, later, with explicit human supervision.
    #[cfg(windows)]
    #[test]
    fn run_logout_against_fake_binary_success_on_zero_exit() {
        let script = write_test_cmd("logout-ok", "@echo off\r\nexit /b 0\r\n");
        let result = run_logout(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        assert!(result.is_ok());
    }

    #[cfg(windows)]
    #[test]
    fn run_logout_against_fake_binary_failed_on_nonzero_exit() {
        let script = write_test_cmd("logout-fail", "@echo off\r\nexit /b 2\r\n");
        let result = run_logout(&script, Duration::from_secs(5));
        let _ = std::fs::remove_file(&script);
        assert!(matches!(result, Err(AuthActionErr::Failed)));
    }
}
