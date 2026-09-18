//! Four-state machine: attention > running > done > idle (ordered by attention cost).
//! done persists: it is cleared only by a new UserPromptSubmit for that session, the user's ✕, or the > 24 h stale sweep.

use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const ST_RUNNING: &str = "running";
pub const ST_ATTENTION: &str = "attention";
pub const ST_DONE: &str = "done";
pub const ST_IDLE: &str = "idle";

const RUNNING_STALE_MS: u64 = 30 * 60 * 1000; // running with no event for 30 min is treated as an abnormal exit
const DONE_STALE_MS: u64 = 24 * 3600 * 1000; // stale done entries are removed after 24 h
const IDLE_DROP_MS: u64 = 10 * 60 * 1000; // idle entries leave the list after 10 min
/// Per-session token/cost totals are re-derived from the whole transcript file (see
/// session_usage.rs), which is heavier than the tail-only reads elsewhere in this module; this
/// floor keeps a burst of hook events from re-scanning the same multi-MB file dozens of times a second.
const USAGE_SCAN_MIN_GAP_MS: u64 = 2_000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Serialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub state: String,
    /// Start of the current activity (ms epoch)
    pub started: u64,
    /// Total elapsed time frozen at done (ms)
    pub total: u64,
    pub last: String,
    /// What attention is about (permission request / question summary)
    pub attn: String,
    /// The user's latest input (card subtitle: "you: …" — show what you said rather than the agent's action)
    pub prompt: String,
    /// The model the session actually uses (message.model of a transcript assistant entry)
    pub model: String,
    /// Summed across every assistant turn in the transcript so far (session_usage::compute_usage_totals).
    /// No dollar cost is derived from these — see session_usage.rs for why.
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_read_tokens: u64,
    /// Whether any event for this session has ever carried a transcript path — independent of
    /// whether that transcript has parseable usage data yet (see `usage_available`). Lets the UI
    /// tell "no transcript known" apart from "transcript known, just hasn't produced usage lines
    /// yet" instead of collapsing both into a guessed zero.
    pub has_transcript_path: bool,
    /// Whether `session_usage::compute_usage_totals` has ever returned `Some` for this session —
    /// i.e. the token fields above reflect a real scan, not just their zero default.
    pub usage_available: bool,
    #[serde(skip)]
    pub ppid: u32,
    #[serde(skip)]
    pub last_event: u64,
    #[serde(skip)]
    pub cwd: String,
    /// Time of the last real hook event; watcher inference is ignored while hook data is fresh
    #[serde(skip)]
    pub last_hook: u64,
    /// Throttle for the transcript-wide token rescan — see USAGE_SCAN_MIN_GAP_MS
    #[serde(skip)]
    pub last_usage_scan: u64,
}

/// Hook data is considered fresh within this window, and watcher inference yields to it
const HOOK_FRESH_MS: u64 = 5 * 60 * 1000;

#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub sessions: Vec<Session>,
    pub agg: String,
    pub counts: HashMap<String, usize>,
    /// The language the user chose (may be "auto"; used to highlight the menu item)
    pub lang: String,
    /// The actual language resolved on the Rust side (WebView2's navigator.language is unreliable)
    pub lang_resolved: String,
    /// Whether reset times use a 24-hour clock, from the Windows region settings
    pub clock_24h: bool,
    /// Whether dragging / wheel resizing is allowed (the page enables the gestures from it)
    pub drag: bool,
}

#[derive(Default)]
pub struct Store {
    map: HashMap<String, Session>,
}

pub struct HookEvent {
    pub e: String,
    pub session_id: String,
    pub ppid: u32,
    pub cwd: String,
    pub prompt: String,
    pub message: String,
    pub tool_name: String,
    pub tool_cmd: String,
    pub model: String,
    /// "hook" (a real event) or "watch" (transcript inference, the desktop app's fallback)
    pub src: &'static str,
    /// Path to the session's own .jsonl transcript, when known (the hook's stdin JSON carries
    /// this as `transcript_path`; the watcher already has it, being the file it is tailing).
    /// Empty when unknown — usage totals are then simply left as last known, not rescanned.
    pub transcript_path: String,
}

fn truncate(s: &str, n: usize) -> String {
    let mut out: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        out.push('…');
    }
    out
}

/// Re-derives `s`'s token totals from its transcript, throttled by USAGE_SCAN_MIN_GAP_MS.
/// Returns whether any of the totals actually changed (the caller folds this into its own
/// change-detection so a rescan that finds nothing new does not trigger a broadcast).
/// `has_transcript_path` is recorded unconditionally (not throttled) the moment a path is seen —
/// it is a cheap boolean, and gating it on the same throttle as the actual file read would make
/// the "do we even know a transcript for this session" signal lag behind reality for no reason.
fn refresh_usage(s: &mut Session, transcript_path: &str, now: u64) -> bool {
    if transcript_path.is_empty() {
        return false;
    }
    s.has_transcript_path = true;
    if now.saturating_sub(s.last_usage_scan) < USAGE_SCAN_MIN_GAP_MS {
        return false;
    }
    s.last_usage_scan = now;
    let Some(t) = crate::session_usage::compute_usage_totals(Path::new(transcript_path)) else {
        return false;
    };
    s.usage_available = true;
    let changed = s.input_tokens != t.input_tokens
        || s.output_tokens != t.output_tokens
        || s.cache_write_tokens != t.cache_write_tokens
        || s.cache_read_tokens != t.cache_read_tokens;
    s.input_tokens = t.input_tokens;
    s.output_tokens = t.output_tokens;
    s.cache_write_tokens = t.cache_write_tokens;
    s.cache_read_tokens = t.cache_read_tokens;
    changed
}

fn title_of(cwd: &str, id: &str) -> String {
    let base = cwd
        .replace('\\', "/")
        .rsplit('/')
        .find(|p| !p.is_empty())
        .unwrap_or("claude")
        .to_string();
    let short: String = id.chars().take(4).collect();
    format!("{base} · {short}")
}

impl Store {
    pub fn apply(&mut self, ev: HookEvent) -> bool {
        let now = now_ms();
        if ev.e == "session_end" {
            return self.map.remove(&ev.session_id).is_some();
        }
        let s = self
            .map
            .entry(ev.session_id.clone())
            .or_insert_with(|| Session {
                id: ev.session_id.clone(),
                title: title_of(&ev.cwd, &ev.session_id),
                state: ST_IDLE.into(),
                started: now,
                total: 0,
                last: String::new(),
                attn: String::new(),
                prompt: String::new(),
                model: String::new(),
                input_tokens: 0,
                output_tokens: 0,
                cache_write_tokens: 0,
                cache_read_tokens: 0,
                has_transcript_path: false,
                usage_available: false,
                ppid: 0,
                last_event: now,
                cwd: ev.cwd.clone(),
                last_hook: 0,
                last_usage_scan: 0,
            });
        // Source arbitration: a session with fresh hook data does not accept watcher inference
        if ev.src == "watch" && s.last_hook > 0 && now.saturating_sub(s.last_hook) < HOOK_FRESH_MS {
            return false;
        }
        if ev.src == "hook" {
            s.last_hook = now;
        }
        let before = (
            s.state.clone(),
            s.last.clone(),
            s.attn.clone(),
            s.prompt.clone(),
            s.model.clone(),
        );
        s.last_event = now;
        if ev.ppid != 0 {
            s.ppid = ev.ppid;
        }
        if !ev.model.is_empty() {
            s.model = ev.model.clone();
        }
        if !ev.cwd.is_empty() && s.cwd.is_empty() {
            s.cwd = ev.cwd.clone();
            s.title = title_of(&ev.cwd, &s.id);
        }
        match ev.e.as_str() {
            "session_start" => {
                if s.state != ST_RUNNING {
                    s.state = ST_IDLE.into();
                }
            }
            "running" => {
                if s.state != ST_RUNNING {
                    s.started = now;
                }
                s.state = ST_RUNNING.into();
                s.attn.clear();
                if !ev.prompt.is_empty() {
                    s.prompt = truncate(&ev.prompt, 120);
                }
                if !ev.tool_name.is_empty() {
                    s.last = if ev.tool_cmd.is_empty() {
                        format!("🔧 {}", ev.tool_name)
                    } else {
                        format!("🔧 {}: {}", ev.tool_name, truncate(&ev.tool_cmd, 60))
                    };
                }
            }
            "attention" => {
                s.state = ST_ATTENTION.into();
                if !ev.message.is_empty() {
                    s.attn = truncate(&ev.message, 200);
                }
            }
            "done" => {
                if s.state != ST_DONE {
                    s.total = now.saturating_sub(s.started);
                }
                s.state = ST_DONE.into();
                s.attn.clear();
            }
            _ => {}
        }
        let usage_changed = refresh_usage(s, &ev.transcript_path, now);
        // Broadcast only on a visible change, so the watcher's rapid appends cannot cause a storm
        (
            s.state.clone(),
            s.last.clone(),
            s.attn.clone(),
            s.prompt.clone(),
            s.model.clone(),
        ) != before
            || usage_changed
    }

    pub fn dismiss(&mut self, id: &str) -> bool {
        self.map.remove(id).is_some()
    }

    pub fn has_done(&self) -> bool {
        self.map.values().any(|s| s.state == ST_DONE)
    }

    /// Seen-clears-it: done sessions matching the predicate become idle (and the sweep removes them later)
    pub fn ack_done<F: Fn(&Session) -> bool>(&mut self, f: F) -> bool {
        let now = now_ms();
        let mut changed = false;
        for s in self.map.values_mut() {
            if s.state == ST_DONE && f(s) {
                s.state = ST_IDLE.into();
                s.last_event = now;
                changed = true;
            }
        }
        changed
    }

    /// Stale sweep; returns whether anything changed
    pub fn sweep(&mut self) -> bool {
        let now = now_ms();
        let mut changed = false;
        for s in self.map.values_mut() {
            if s.state == ST_RUNNING && now.saturating_sub(s.last_event) > RUNNING_STALE_MS {
                s.state = ST_IDLE.into();
                changed = true;
            }
        }
        let before = self.map.len();
        self.map.retain(|_, s| {
            !(s.state == ST_IDLE && now.saturating_sub(s.last_event) > IDLE_DROP_MS
                || s.state == ST_DONE && now.saturating_sub(s.last_event) > DONE_STALE_MS)
        });
        changed || self.map.len() != before
    }

    pub fn ppid_of(&self, id: &str) -> Option<u32> {
        self.map.get(id).map(|s| s.ppid).filter(|p| *p != 0)
    }

    pub fn snapshot(&self, lang: &str, lang_resolved: &str, clock_24h: bool, drag: bool) -> Snapshot {
        let mut sessions: Vec<Session> = self.map.values().cloned().collect();
        let rank = |st: &str| match st {
            ST_ATTENTION => 0,
            ST_RUNNING => 1,
            ST_DONE => 2,
            _ => 3,
        };
        sessions.sort_by(|a, b| {
            rank(&a.state)
                .cmp(&rank(&b.state))
                .then(b.started.cmp(&a.started))
        });
        let mut counts = HashMap::new();
        for k in [ST_ATTENTION, ST_RUNNING, ST_DONE] {
            counts.insert(
                k.to_string(),
                sessions.iter().filter(|s| s.state == k).count(),
            );
        }
        let agg = [ST_ATTENTION, ST_RUNNING, ST_DONE]
            .iter()
            .find(|k| counts.get(**k).copied().unwrap_or(0) > 0)
            .map(|k| k.to_string())
            .unwrap_or_else(|| ST_IDLE.to_string());
        Snapshot {
            sessions,
            agg,
            counts,
            lang: lang.to_string(),
            lang_resolved: lang_resolved.to_string(),
            clock_24h,
            drag,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn ev(e: &str, session_id: &str, transcript_path: &str) -> HookEvent {
        HookEvent {
            e: e.to_string(),
            session_id: session_id.to_string(),
            ppid: 0,
            cwd: format!("C:\\proj\\{session_id}"),
            prompt: String::new(),
            message: String::new(),
            tool_name: String::new(),
            tool_cmd: String::new(),
            model: "claude-sonnet-5".to_string(),
            src: "hook",
            transcript_path: transcript_path.to_string(),
        }
    }

    fn write_transcript(name: &str, usage_lines: usize) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "codenotch-state-usage-{}-{name}.jsonl",
            std::process::id()
        ));
        let mut f = std::fs::File::create(&p).unwrap();
        for _ in 0..usage_lines {
            writeln!(f, r#"{{"type":"assistant","message":{{"model":"claude-sonnet-5","usage":{{"input_tokens":2,"output_tokens":3,"cache_creation_input_tokens":100,"cache_read_input_tokens":1000}}}}}}"#).unwrap();
        }
        p
    }

    fn write_transcript_no_cache(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "codenotch-state-usage-{}-{name}.jsonl",
            std::process::id()
        ));
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, r#"{{"type":"assistant","message":{{"model":"claude-sonnet-5","usage":{{"input_tokens":2,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}}}}"#).unwrap();
        p
    }

    #[test]
    fn empty_store_has_no_sessions() {
        let store = Store::default();
        let snap = store.snapshot("auto", "en", false, false);
        assert!(snap.sessions.is_empty());
        assert_eq!(snap.agg, ST_IDLE);
    }

    #[test]
    fn one_active_session_with_full_usage_data() {
        let p = write_transcript("one", 1);
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p.to_string_lossy()));
        let snap = store.snapshot("auto", "en", false, false);
        let _ = std::fs::remove_file(&p);
        assert_eq!(snap.sessions.len(), 1);
        let s = &snap.sessions[0];
        assert_eq!(s.state, ST_RUNNING);
        assert_eq!(s.input_tokens, 2);
        assert_eq!(s.output_tokens, 3);
        assert_eq!(s.cache_write_tokens, 100);
        assert_eq!(s.cache_read_tokens, 1000);
        assert!(s.has_transcript_path);
        assert!(s.usage_available);
    }

    #[test]
    fn multiple_active_sessions_stay_independent() {
        let p1 = write_transcript("multi1", 1);
        let p2 = write_transcript("multi2", 2);
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p1.to_string_lossy()));
        store.apply(ev("running", "s2", &p2.to_string_lossy()));
        let snap = store.snapshot("auto", "en", false, false);
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        assert_eq!(snap.sessions.len(), 2);
        let s1 = snap.sessions.iter().find(|s| s.id == "s1").unwrap();
        let s2 = snap.sessions.iter().find(|s| s.id == "s2").unwrap();
        assert_eq!(s1.input_tokens, 2);
        assert_eq!(s2.input_tokens, 4); // two usage lines summed
    }

    #[test]
    fn session_without_cache_activity_has_zero_cache_fields() {
        let p = write_transcript_no_cache("nocache");
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p.to_string_lossy()));
        let snap = store.snapshot("auto", "en", false, false);
        let _ = std::fs::remove_file(&p);
        let s = &snap.sessions[0];
        assert_eq!(s.cache_write_tokens, 0);
        assert_eq!(s.cache_read_tokens, 0);
        // Tokens are still real numbers, not omitted just because cache is idle
        assert_eq!(s.input_tokens, 2);
    }

    #[test]
    fn session_with_no_transcript_path_has_no_usage_data() {
        let mut store = Store::default();
        store.apply(ev("running", "s1", ""));
        let snap = store.snapshot("auto", "en", false, false);
        let s = &snap.sessions[0];
        assert_eq!(s.input_tokens, 0);
        assert!(!s.has_transcript_path, "no path was ever supplied");
        assert!(!s.usage_available, "no scan could have happened without a path");
    }

    #[test]
    fn transcript_path_known_but_not_yet_scanned_reports_no_usage() {
        // A path pointing at a file with no assistant/usage lines yet (mid-session, first turn
        // still streaming): has_transcript_path must flip true immediately, but usage_available
        // must stay false rather than reporting a fabricated zero.
        let p = std::env::temp_dir().join(format!("codenotch-state-usage-{}-nolines.jsonl", std::process::id()));
        std::fs::write(&p, "").unwrap();
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p.to_string_lossy()));
        let snap = store.snapshot("auto", "en", false, false);
        let _ = std::fs::remove_file(&p);
        let s = &snap.sessions[0];
        assert!(s.has_transcript_path);
        assert!(!s.usage_available);
        assert_eq!(s.input_tokens, 0);
    }

    #[test]
    fn idle_session_is_excluded_from_the_active_filter_the_ui_applies() {
        // The UI's "active sessions" list is `sessions.filter(state !== 'idle')` (notch.html);
        // this locks in that a freshly created, never-run session starts idle.
        let mut store = Store::default();
        store.apply(ev("session_start", "s1", ""));
        let snap = store.snapshot("auto", "en", false, false);
        assert_eq!(snap.sessions[0].state, ST_IDLE);
    }

    #[test]
    fn switching_account_does_not_touch_unrelated_sessions() {
        // Sessions are local-process state, not account-scoped (see claude_account.rs) — an
        // account switch must not clear, rename, or merge them. Simulated here by asserting a
        // second session's apply() never mutates a first, already-tracked session.
        let p1 = write_transcript("acct1", 1);
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p1.to_string_lossy()));
        let before = store.snapshot("auto", "en", false, false).sessions[0].clone();
        store.apply(ev("running", "s2", ""));
        let after_s1 = store
            .snapshot("auto", "en", false, false)
            .sessions
            .into_iter()
            .find(|s| s.id == "s1")
            .unwrap();
        let _ = std::fs::remove_file(&p1);
        assert_eq!(before.input_tokens, after_s1.input_tokens);
        assert_eq!(before.usage_available, after_s1.usage_available);
    }

    #[test]
    fn usage_rescan_is_throttled_within_the_min_gap() {
        let p = write_transcript("throttle", 1);
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p.to_string_lossy()));
        // Append more usage to the same file without advancing wall-clock time past the gap —
        // a second apply() right away must not pick it up yet.
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
            writeln!(f, r#"{{"type":"assistant","message":{{"model":"claude-sonnet-5","usage":{{"input_tokens":999,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}}}}"#).unwrap();
        }
        store.apply(ev("running", "s1", &p.to_string_lossy()));
        let snap = store.snapshot("auto", "en", false, false);
        let _ = std::fs::remove_file(&p);
        assert_eq!(snap.sessions[0].input_tokens, 2, "throttle must hold within USAGE_SCAN_MIN_GAP_MS");
    }

    #[test]
    fn session_json_never_serializes_cwd_or_transcript_path() {
        let p = write_transcript("nosecrets", 1);
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p.to_string_lossy()));
        let snap = store.snapshot("auto", "en", false, false);
        let _ = std::fs::remove_file(&p);
        let json = serde_json::to_string(&snap).unwrap();
        assert!(!json.contains("\"cwd\""), "cwd must stay server-side (#[serde(skip)])");
        assert!(!json.contains("\"transcript_path\""), "no local file path may reach the UI");
        assert!(!json.contains(&p.to_string_lossy().replace('\\', "\\\\")), "the raw path itself must never leak into the snapshot");
    }

    #[test]
    fn session_json_never_serializes_a_cost_field() {
        let p = write_transcript("nocost", 1);
        let mut store = Store::default();
        store.apply(ev("running", "s1", &p.to_string_lossy()));
        let snap = store.snapshot("auto", "en", false, false);
        let _ = std::fs::remove_file(&p);
        let json = serde_json::to_string(&snap).unwrap().to_lowercase();
        assert!(!json.contains("cost"), "no cost/dollar field may be serialized to the UI: {json}");
        assert!(!json.contains("usd"), "no cost/dollar field may be serialized to the UI: {json}");
    }
}
