//! Canonical AI session model (T001). Pure data: no I/O, no provider logic, no Tauri wiring.
//! Every provider adapter (T004/T005, not built yet) will have to *prove* which enum variant
//! applies — this module only defines the vocabulary, it never guesses on a provider's behalf.
//!
//! Not yet wired into `AppState` or any `#[tauri::command]` — that is T004/T005/T007, explicitly
//! out of scope for this pass. `dead_code` is allowed at the module level for exactly that reason
//! and must be removed once an adapter actually constructs an `AiSession`.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Providers this tracker knows how to represent. Deliberately not the same concept as the
/// `provider: String` ids used across `activity.rs`/`config.rs`/the UI (there is no reusable
/// enum for that today — the rest of the codebase passes "claude"/"codex"/"cursor"/"gemini" as
/// raw strings end to end, e.g. `Activity::provider` and `TraySlot::provider`). Introducing an
/// enum here does not replace that: it only gives this new, still-unwired module compile-time
/// safety for the two providers this feature actually targets (see Fase 14 scope decision).
/// Cursor/Antigravity are intentionally absent until a real adapter needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Claude,
    Codex,
}

/// Where a `SessionStatus` reading came from. Named after the actual mechanism that produced it,
/// not a trust rating — "how reliable is this" is a judgement call for the UI/adapter, not
/// something this enum should pre-decide by calling itself "Direct".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatusSource {
    /// Claude Code's own hook events (SessionStart/UserPromptSubmit/PreToolUse/PostToolUse/
    /// Notification/Stop), forwarded by codenotch-hook.exe — see hooks_install.rs.
    ClaudeCodeHook,
    /// Claude desktop fallback: transcript .jsonl silence-based inference — see watcher.rs.
    ClaudeTranscriptWatcher,
    /// Codex desktop app's own `thread_turns` table in `thread_history_1.sqlite` — the app's
    /// own bookkeeping, read read-only — see activity.rs::codex_turns_in_progress.
    CodexThreadTurnsTable,
    /// Codex CLI/extension fallback: last rollout entry type + silence threshold — see
    /// activity.rs::codex_last_step.
    CodexRolloutTail,
    /// A session/thread was found, but no source above produced enough evidence to classify it.
    Unknown,
}

/// Working / waiting / idle / done / unknown, in the sense defined in the architecture report:
/// evidence-based, never guessed. `Unknown` is the conservative default when no source qualifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    /// Recent, direct evidence of active execution.
    Working,
    /// Explicit signal that the provider is waiting on a human (a Claude Notification, a Codex
    /// approval/permission request) — never inferred from silence alone.
    Waiting,
    /// The session is known to exist, but no recent activity was observed.
    Idle,
    /// A turn/session was observably concluded, per its own source (Claude's Stop hook, Codex's
    /// task_complete/turn_aborted).
    Done,
    /// A session/thread exists, but no source gives enough evidence to classify it. The safe
    /// default — never replaced by a guess.
    Unknown,
}

/// What HEAD actually says, as a single source of truth — never represented as a `branch: Option`
/// plus a `detached: bool` that could disagree with each other. `Unknown` is its own state,
/// distinct from `Detached`: a missing or corrupt HEAD is not evidence of a detached checkout, it
/// is simply an absence of evidence, and must not be reported as either a branch or a detachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "branch", rename_all = "snake_case")]
pub enum GitHeadState {
    /// `ref: refs/heads/<name>` resolved to a symbolic branch name.
    Branch(String),
    /// HEAD holds a raw commit id directly, not a symbolic ref: a real, positively observed
    /// detached checkout.
    Detached,
    /// HEAD is missing, unreadable, or its content matches neither a symbolic ref nor a
    /// plausible commit id. No claim is made either way.
    Unknown,
}

impl GitHeadState {
    /// The branch name, only when `self` actually is `Branch(_)`.
    pub fn branch_name(&self) -> Option<&str> {
        match self {
            GitHeadState::Branch(name) => Some(name),
            GitHeadState::Detached | GitHeadState::Unknown => None,
        }
    }
}

/// The working directory turned into something a human recognizes as "a project" — resolved by
/// `session_git` (T002). Represents the actual working tree the session is running in: a linked
/// worktree is represented as itself, never silently collapsed into its main repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectContext {
    /// The working tree's own root directory (for a linked worktree, this is the worktree's
    /// path, e.g. `C:\Worktrees\LK-wallet` — never replaced by the main repo's path).
    pub worktree_root: String,
    /// Basename of `worktree_root`. Deliberately not derived from the remote: a remote can be
    /// absent, point at a mirror, or be named differently from the local checkout.
    pub repository_name: String,
    /// The single source of truth for HEAD. See `GitHeadState` — deliberately not a
    /// `branch: Option<String>` plus a separate `detached: bool`, which could contradict each
    /// other (e.g. `branch: None, detached: false` used to mean at least three different things).
    pub head: GitHeadState,
    /// `origin`'s URL, read from the common git directory's `config`, with any embedded
    /// credentials stripped. `None` when there is no `origin` remote or no readable config.
    pub remote: Option<String>,
    /// True when this working tree is a linked worktree (`.git` was a file pointing elsewhere),
    /// false for the main checkout (`.git` is a directory).
    pub linked_worktree: bool,
}

/// A single AI coding session/thread, as far as it can be honestly known right now. Every field
/// that we cannot currently prove is `Option`/a conservative enum variant — nothing here is
/// invented to fill a gap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AiSession {
    /// The provider's own identifier: Claude Code's `session_id`, or Codex's `thread_id`. Both
    /// are provider-issued UUIDs; this crate never generates or reuses one.
    pub id: String,
    pub provider: Provider,
    /// `Some` only where the source that reported this session also reported a process id it can
    /// vouch for (today: Claude's hook, which reads the real parent pid). A provider's process
    /// merely existing on the machine (e.g. Codex found via ToolHelp32) does NOT establish a
    /// link to this particular thread_id, so Codex sessions carry `None` until a deterministic
    /// thread_id<->pid correlation exists — see the architecture report's Codex PID adjustment.
    pub pid: Option<u32>,
    /// When this session was first observed (ms epoch). Not necessarily "continuously working
    /// since" — see `current_activity_started_at`.
    pub session_started_at: u64,
    /// Last time any source reported activity for this session (ms epoch).
    pub last_activity_at: u64,
    /// When the *current* activity burst started, if a source can actually say so. A Codex
    /// thread's `started_at` is when the thread began, not when its current turn began — so it
    /// must never be reused here. `None` until a source proves this specifically.
    pub current_activity_started_at: Option<u64>,
    pub cwd: Option<String>,
    pub project: Option<ProjectContext>,
    pub status: SessionStatus,
    pub status_source: StatusSource,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_project() -> ProjectContext {
        ProjectContext {
            worktree_root: "C:\\Worktrees\\LK-wallet".into(),
            repository_name: "LK-wallet".into(),
            head: GitHeadState::Branch("feat/wallet-login".into()),
            remote: Some("https://github.com/example/lk-wallet.git".into()),
            linked_worktree: true,
        }
    }

    fn sample_session() -> AiSession {
        AiSession {
            id: "11111111-1111-1111-1111-111111111111".into(),
            provider: Provider::Claude,
            pid: Some(4242),
            session_started_at: 1_000,
            last_activity_at: 2_000,
            current_activity_started_at: Some(1_800),
            cwd: Some("C:\\Worktrees\\LK-wallet".into()),
            project: Some(sample_project()),
            status: SessionStatus::Working,
            status_source: StatusSource::ClaudeCodeHook,
        }
    }

    #[test]
    fn ai_session_round_trips_through_json_with_data_present() {
        let s = sample_session();
        let json = serde_json::to_string(&s).unwrap();
        let back: AiSession = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn ai_session_round_trips_with_every_option_absent() {
        let s = AiSession {
            id: "codex-thread-abc".into(),
            provider: Provider::Codex,
            pid: None,
            session_started_at: 500,
            last_activity_at: 500,
            current_activity_started_at: None,
            cwd: None,
            project: None,
            status: SessionStatus::Unknown,
            status_source: StatusSource::Unknown,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"pid\":null"));
        assert!(json.contains("\"project\":null"));
        assert!(json.contains("\"current_activity_started_at\":null"));
        let back: AiSession = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn every_session_status_serializes_as_lowercase_and_round_trips() {
        for status in [
            SessionStatus::Working,
            SessionStatus::Waiting,
            SessionStatus::Idle,
            SessionStatus::Done,
            SessionStatus::Unknown,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            assert_eq!(json, json.to_lowercase(), "expected lowercase for {status:?}");
            let back: SessionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }

    #[test]
    fn every_status_source_round_trips() {
        for source in [
            StatusSource::ClaudeCodeHook,
            StatusSource::ClaudeTranscriptWatcher,
            StatusSource::CodexThreadTurnsTable,
            StatusSource::CodexRolloutTail,
            StatusSource::Unknown,
        ] {
            let json = serde_json::to_string(&source).unwrap();
            let back: StatusSource = serde_json::from_str(&json).unwrap();
            assert_eq!(source, back);
        }
    }

    #[test]
    fn git_head_state_round_trips_for_every_variant_and_never_confuses_unknown_with_detached() {
        let cases = [
            (GitHeadState::Branch("feat/wallet-login".into()), "\"state\":\"branch\""),
            (GitHeadState::Detached, "\"state\":\"detached\""),
            (GitHeadState::Unknown, "\"state\":\"unknown\""),
        ];
        for (state, expected_tag) in cases {
            let json = serde_json::to_string(&state).unwrap();
            assert!(json.contains(expected_tag), "expected {expected_tag} in {json}");
            let back: GitHeadState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, back);
        }
        // The whole point of this enum: a missing/corrupt HEAD (Unknown) must never compare equal
        // to a positively observed detached checkout, or vice versa.
        assert_ne!(GitHeadState::Unknown, GitHeadState::Detached);
        assert_eq!(GitHeadState::Branch("main".into()).branch_name(), Some("main"));
        assert_eq!(GitHeadState::Detached.branch_name(), None);
        assert_eq!(GitHeadState::Unknown.branch_name(), None);
    }

    #[test]
    fn project_context_without_remote_round_trips() {
        let mut p = sample_project();
        p.remote = None;
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"remote\":null"));
        let back: ProjectContext = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    /// Debug/serde output of a session built only from non-sensitive sample data must never
    /// accidentally spell out anything that looks like a secret marker — a cheap guard against a
    /// future field being added with a misleading name or a copy-pasted fixture value.
    #[test]
    fn debug_and_json_output_carry_no_credential_like_markers() {
        let s = sample_session();
        let debug = format!("{s:?}");
        let json = serde_json::to_string(&s).unwrap();
        for marker in ["token", "password", "bearer", "authorization", "credentials"] {
            assert!(!debug.to_lowercase().contains(marker), "Debug output must not mention {marker}");
            assert!(!json.to_lowercase().contains(marker), "JSON output must not mention {marker}");
        }
    }
}
