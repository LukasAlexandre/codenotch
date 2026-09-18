//! Per-session token totals, derived from a Claude Code transcript (.claude/projects/**/*.jsonl).
//!
//! Tokens are real, local numbers: each assistant transcript entry carries its own
//! `message.usage` object (input_tokens / output_tokens / cache_creation_input_tokens /
//! cache_read_input_tokens), written by Claude Code itself — verified by hand against a live
//! transcript before writing this parser, the same discipline `watcher.rs`'s tail reader follows.
//!
//! Deliberately NOT here: a dollar cost. An earlier revision of this module priced tokens against
//! a hardcoded per-model rate card, but nothing on this machine confirms those numbers are still
//! correct for the models actually seen in transcripts (no local `costUSD` field exists in current
//! Claude Code transcripts, and Pro/Max subscriptions are not billed per token at all) — showing a
//! dollar figure nobody has verified is exactly the "invented number" this app's own usage.rs
//! already refuses to do for percentages. If cost is wanted later, it needs a verified pricing
//! source, not a guess baked into this file.

use serde::Serialize;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct UsageTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_write_tokens: u64,
    pub cache_read_tokens: u64,
}

/// A runaway multi-hour session's transcript grows without bound; this caps how much of it a
/// single rescan reads, matching the spirit of watcher.rs's own TAIL_BYTES safety valve (that one
/// reads the tail for the *latest* entry, this reads from the start for a running *sum* — a
/// session past this cap under-counts its earliest turns rather than stalling the poll loop).
const MAX_SCAN_BYTES: u64 = 16 * 1024 * 1024;

/// Sums `usage` across every assistant message in the transcript at `path`. Returns `None` when
/// the file cannot be read at all, or no assistant message with usage data was found (a brand-new
/// session, or a non-Claude-Code file) — never a zeroed-out `Some`, so the caller can tell "no
/// data yet" from "confirmed zero".
pub fn compute_usage_totals(path: &Path) -> Option<UsageTotals> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut raw = Vec::new();
    f.by_ref().take(MAX_SCAN_BYTES).read_to_end(&mut raw).ok()?;
    let buf = String::from_utf8_lossy(&raw);

    let mut totals = UsageTotals::default();
    let mut seen_any_usage = false;

    for line in buf.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("type").and_then(|t| t.as_str()) != Some("assistant") {
            continue;
        }
        let Some(usage) = v.pointer("/message/usage") else {
            continue;
        };
        seen_any_usage = true;
        totals.input_tokens += usage.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
        totals.output_tokens += usage.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
        totals.cache_write_tokens += usage
            .get("cache_creation_input_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        totals.cache_read_tokens += usage
            .get("cache_read_input_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
    }

    seen_any_usage.then_some(totals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_transcript(name: &str, lines: &[&str]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "codenotch-session-usage-{}-{name}.jsonl",
            std::process::id()
        ));
        let mut f = std::fs::File::create(&p).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        p
    }

    const ASSISTANT_1: &str = r#"{"type":"assistant","message":{"model":"claude-sonnet-5","usage":{"input_tokens":10,"output_tokens":20,"cache_creation_input_tokens":100,"cache_read_input_tokens":1000}}}"#;
    const ASSISTANT_2: &str = r#"{"type":"assistant","message":{"model":"claude-sonnet-5","usage":{"input_tokens":5,"output_tokens":7,"cache_creation_input_tokens":0,"cache_read_input_tokens":500}}}"#;
    const ASSISTANT_UNKNOWN_MODEL: &str = r#"{"type":"assistant","message":{"model":"claude-totally-unknown-model","usage":{"input_tokens":10,"output_tokens":20,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}"#;
    const USER_LINE: &str = r#"{"type":"user","message":{"content":"hi"}}"#;

    #[test]
    fn missing_file_is_none() {
        let p = std::path::PathBuf::from(r"C:\does\not\exist\nope.jsonl");
        assert!(compute_usage_totals(&p).is_none());
    }

    #[test]
    fn no_assistant_usage_lines_is_none() {
        let p = write_transcript("empty", &[USER_LINE]);
        let r = compute_usage_totals(&p);
        let _ = std::fs::remove_file(&p);
        assert!(r.is_none(), "no usage data yet must stay None, never a guessed zero");
    }

    #[test]
    fn sums_tokens_across_multiple_assistant_turns() {
        let p = write_transcript("sums", &[USER_LINE, ASSISTANT_1, ASSISTANT_2]);
        let r = compute_usage_totals(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(r.input_tokens, 15);
        assert_eq!(r.output_tokens, 27);
        assert_eq!(r.cache_write_tokens, 100);
        assert_eq!(r.cache_read_tokens, 1500);
    }

    #[test]
    fn model_identity_never_affects_token_totals() {
        // Whatever model wrote a turn, its tokens count the same way — no pricing/model gate here.
        let p = write_transcript("anymodel", &[ASSISTANT_UNKNOWN_MODEL]);
        let r = compute_usage_totals(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(r.input_tokens, 10);
        assert_eq!(r.output_tokens, 20);
    }

    #[test]
    fn malformed_lines_are_skipped_not_fatal() {
        let p = write_transcript("malformed", &["not json at all", ASSISTANT_1, "{broken"]);
        let r = compute_usage_totals(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(r.input_tokens, 10);
    }

    #[test]
    fn no_cost_field_exists_on_usage_totals() {
        // Regression guard for the removal itself: UsageTotals must not grow a cost/dollar field
        // back without a deliberate, reviewed decision to do so.
        let p = write_transcript("nocost", &[ASSISTANT_1]);
        let r = compute_usage_totals(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.to_lowercase().contains("cost"), "cost must not reappear in UsageTotals: {json}");
        assert!(!json.to_lowercase().contains("usd"), "cost must not reappear in UsageTotals: {json}");
    }
}
