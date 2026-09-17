//! Git Project Context resolver (T002): cwd -> ProjectContext, by parsing `.git` on disk.
//!
//! Pure filesystem reads only. No `git.exe`, no network, no writes, no hooks, no checkout/fetch/
//! pull. If a real case turns out to be impossible to resolve this way, this module returns
//! honestly partial data (e.g. a `ProjectContext` with `head: GitHeadState::Unknown`) rather than
//! shelling out — a subprocess fallback, if ever needed, is explicitly out of scope for this pass
//! (see the architecture report).
//!
//! No caching here either: this is a pure, directly testable resolver. Caching is T003.
//!
//! Not yet called from any adapter/command (that is T004/T005/T007) — `dead_code` is allowed at
//! the module level for exactly that reason and must be removed once a caller exists.
#![allow(dead_code)]

use crate::session_model::{GitHeadState, ProjectContext};
use std::path::{Path, PathBuf};

/// The three directories a resolved `.git` pointer can involve. Not part of the public API:
/// callers only need the final `ProjectContext`.
struct GitDirs {
    /// The working tree's own root — for a linked worktree, this is the worktree's directory,
    /// never silently replaced by the main repository's path.
    worktree_root: PathBuf,
    /// Where this working tree's own HEAD/index live. For the main checkout this is `<root>/.git`;
    /// for a linked worktree it is `<main>/.git/worktrees/<name>`.
    git_dir: PathBuf,
    /// Where shared data (refs, config, objects) lives. Equal to `git_dir` unless `git_dir`
    /// itself declares a `commondir`.
    common_git_dir: PathBuf,
    /// True when `.git` at `worktree_root` was a file (linked worktree) rather than a directory.
    linked_worktree: bool,
}

/// Resolves the AI session's Git project context from its working directory. Returns `None` only
/// when no `.git` was found walking up from `cwd`; once a `.git` is found, every other missing
/// piece (HEAD, config, commondir) degrades to an honest `GitHeadState::Unknown`/`None` field
/// instead of failing the whole resolution.
pub fn resolve(cwd: &Path) -> Option<ProjectContext> {
    let dirs = find_git_dirs(cwd)?;
    let head = read_head(&dirs.git_dir);
    let remote = read_origin_remote(&dirs.common_git_dir);
    let worktree_root = strip_verbatim_prefix(&dirs.worktree_root);
    let repository_name = dirs
        .worktree_root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| worktree_root.clone());
    Some(ProjectContext {
        worktree_root,
        repository_name,
        head,
        remote,
        linked_worktree: dirs.linked_worktree,
    })
}

/// Walks up from `start` looking for `.git` (directory or file). Purely lexical (`Path::parent`),
/// so it always terminates and cannot loop even if the path no longer exists on disk.
fn find_git_dirs(start: &Path) -> Option<GitDirs> {
    let mut dir = start.to_path_buf();
    loop {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Some(GitDirs {
                worktree_root: dir,
                git_dir: dot_git.clone(),
                common_git_dir: dot_git,
                linked_worktree: false,
            });
        }
        if dot_git.is_file() {
            // A malformed pointer resolves to the (unusable) `.git` file path itself: the HEAD/
            // config reads that follow simply fail to find anything under it and degrade to
            // `None`, which is the honest outcome for "we found a worktree marker but cannot
            // parse it" — no separate error path needed.
            let git_dir = std::fs::read_to_string(&dot_git)
                .ok()
                .and_then(|content| resolve_gitdir_pointer(&content, &dir))
                .unwrap_or_else(|| dot_git.clone());
            let common_git_dir = resolve_commondir(&git_dir);
            return Some(GitDirs {
                worktree_root: dir,
                git_dir,
                common_git_dir,
                linked_worktree: true,
            });
        }
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => return None,
        }
    }
}

/// Parses a `.git` file's first line: `gitdir: <path>`. Accepts absolute or relative paths (the
/// latter resolved against the directory the `.git` file lives in), and nothing else — any other
/// content in the file is not interpreted.
fn resolve_gitdir_pointer(content: &str, containing_dir: &Path) -> Option<PathBuf> {
    let line = content.lines().next()?.trim();
    let path_str = line.strip_prefix("gitdir:")?.trim();
    if path_str.is_empty() {
        return None;
    }
    let p = Path::new(path_str);
    Some(if p.is_absolute() { p.to_path_buf() } else { containing_dir.join(p) })
}

/// `<git_dir>/commondir`, if present: the path (relative to `git_dir` unless absolute) to the
/// common git directory. Absent file => `common_git_dir = git_dir`, per the Git worktree format.
fn resolve_commondir(git_dir: &Path) -> PathBuf {
    let Ok(content) = std::fs::read_to_string(git_dir.join("commondir")) else {
        return git_dir.to_path_buf();
    };
    let line = content.lines().next().unwrap_or("").trim();
    if line.is_empty() {
        return git_dir.to_path_buf();
    }
    let p = Path::new(line);
    if p.is_absolute() { p.to_path_buf() } else { git_dir.join(p) }
}

/// Strict HEAD parsing: `ref: refs/heads/X` => `Branch(X)`; a bare, plausible commit id =>
/// `Detached`; anything else (missing file, unrecognized content) => `Unknown` — never guessed
/// into either of the other two states, since there is no positive evidence for them.
fn read_head(git_dir: &Path) -> GitHeadState {
    let Ok(content) = std::fs::read_to_string(git_dir.join("HEAD")) else {
        return GitHeadState::Unknown;
    };
    let line = content.lines().next().unwrap_or("").trim();
    if let Some(rest) = line.strip_prefix("ref:") {
        let ref_name = rest.trim();
        let branch = ref_name.strip_prefix("refs/heads/").unwrap_or(ref_name);
        if branch.is_empty() {
            return GitHeadState::Unknown;
        }
        return GitHeadState::Branch(branch.to_string());
    }
    if is_plausible_commit_id(line) {
        return GitHeadState::Detached;
    }
    GitHeadState::Unknown
}

fn is_plausible_commit_id(s: &str) -> bool {
    let len = s.len();
    (4..=64).contains(&len) && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Reads `origin`'s `url` from `<common_git_dir>/config`. Only the exact `[remote "origin"]`
/// section is inspected; nothing else in the file is interpreted or executed.
fn read_origin_remote(common_git_dir: &Path) -> Option<String> {
    let content = std::fs::read_to_string(common_git_dir.join("config")).ok()?;
    let mut in_origin = false;
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_origin = line == "[remote \"origin\"]";
            continue;
        }
        if !in_origin {
            continue;
        }
        let Some(rest) = line.strip_prefix("url") else { continue };
        let rest = rest.trim_start();
        let Some(value) = rest.strip_prefix('=') else { continue };
        let value = value.trim();
        if !value.is_empty() {
            return Some(sanitize_remote_url(value));
        }
    }
    None
}

/// Strips embedded userinfo from a `scheme://` remote URL when — and only when — it looks like a
/// credential rather than a normal SSH identity. Goal: remove sensitive credentials, never mutilate
/// a normal remote.
///
/// - scp-style remotes (`git@github.com:org/repo.git`) have no `://` at all and are returned
///   unchanged outright — that `git@` is the SSH user, not a leaked credential.
/// - A `ssh://` URL's userinfo follows the same convention: a bare username with no `:` (`git@`,
///   `ci@`, …) is the normal SSH login and is kept as-is. A `user:password@` form in `ssh://` is
///   not a real SSH login method, so it is still stripped.
/// - Any other scheme (`http(s)://`, `git://`, …) always has its userinfo stripped when present,
///   with or without a `:` — both `user:token@host` and a bare `token@host` are the two common
///   ways a PAT ends up embedded in an http(s) remote, and neither is a legitimate identity to
///   preserve the way `ssh://git@` is.
/// - In every case, `@` is only ever treated as a userinfo delimiter when it appears before the
///   first `/` — an `@` inside the path (e.g. a repository literally named `org/repo@special`)
///   can never be userinfo, since userinfo cannot contain `/`.
fn sanitize_remote_url(raw: &str) -> String {
    let Some(scheme_end) = raw.find("://") else {
        return raw.to_string();
    };
    let scheme = &raw[..scheme_end];
    let (prefix, rest) = raw.split_at(scheme_end + 3);
    let Some(at) = rest.find('@') else {
        return raw.to_string();
    };
    let userinfo = &rest[..at];
    if userinfo.is_empty() || userinfo.contains('/') {
        return raw.to_string();
    }
    if scheme.eq_ignore_ascii_case("ssh") && !userinfo.contains(':') {
        return raw.to_string();
    }
    format!("{prefix}{}", &rest[at + 1..])
}

/// Strips the Windows extended-length prefix (`\\?\` / `\\?\UNC\`) for display purposes only.
/// Never used before a filesystem call — those work fine with the prefix present.
fn strip_verbatim_prefix(path: &Path) -> String {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return rest.to_string();
    }
    s.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh scratch directory under the OS temp dir, following the project's own test
    /// convention (see diag.rs) instead of adding a `tempfile` dependency. Never touches any real
    /// repository, and never `~/.claude` or `~/.codex`.
    fn scratch_dir(case: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("codenotch-session-git-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir); // a recycled pid must not see a stale fixture
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, content: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    /// A normal (non-worktree) repo: `.git` is a directory with HEAD/config directly inside.
    fn make_normal_repo(root: &Path, branch: &str, origin: Option<&str>) {
        write(&root.join(".git").join("HEAD"), &format!("ref: refs/heads/{branch}\n"));
        let cfg = match origin {
            Some(url) => format!("[core]\n\trepositoryformatversion = 0\n[remote \"origin\"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n"),
            None => "[core]\n\trepositoryformatversion = 0\n".to_string(),
        };
        write(&root.join(".git").join("config"), &cfg);
    }

    #[test]
    fn cwd_without_git_returns_none() {
        let dir = scratch_dir("no-git");
        assert!(resolve(&dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn normal_repo_resolves_branch_and_origin() {
        let dir = scratch_dir("normal-repo");
        make_normal_repo(&dir, "main", Some("https://github.com/example/repo.git"));
        let ctx = resolve(&dir).expect("expected a project context");
        assert_eq!(ctx.head, GitHeadState::Branch("main".into()));
        assert_eq!(ctx.remote.as_deref(), Some("https://github.com/example/repo.git"));
        assert!(!ctx.linked_worktree);
        assert_eq!(ctx.repository_name, dir.file_name().unwrap().to_string_lossy());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cwd_in_subdirectory_of_repo_still_resolves() {
        let dir = scratch_dir("subdir-repo");
        make_normal_repo(&dir, "main", None);
        let sub = dir.join("src").join("nested");
        std::fs::create_dir_all(&sub).unwrap();
        let ctx = resolve(&sub).expect("expected a project context found from a subdirectory");
        assert_eq!(ctx.head, GitHeadState::Branch("main".into()));
        assert_eq!(
            strip_verbatim_prefix(&dir),
            ctx.worktree_root,
            "worktree_root must be the repo root, not the subdirectory"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn branch_name_with_slash_is_kept_whole() {
        let dir = scratch_dir("branch-slash");
        make_normal_repo(&dir, "feat/wallet-login", None);
        let ctx = resolve(&dir).unwrap();
        assert_eq!(ctx.head, GitHeadState::Branch("feat/wallet-login".into()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detached_head_has_no_branch_name() {
        let dir = scratch_dir("detached");
        write(&dir.join(".git").join("HEAD"), "3f786850e387550fdab836ed7e6dc881de23001b\n");
        let ctx = resolve(&dir).unwrap();
        assert_eq!(ctx.head, GitHeadState::Detached);
        assert_eq!(ctx.head.branch_name(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn linked_worktree_resolves_its_own_head_and_the_common_config() {
        // Layout:
        //   <main>\.git\                (real repo)
        //     config                    (origin lives here)
        //     worktrees\wt1\HEAD        (the worktree's own HEAD)
        //   <worktree>\.git             (FILE) -> gitdir: <main>\.git\worktrees\wt1
        let main = scratch_dir("worktree-main");
        let worktree = scratch_dir("worktree-linked");
        write(&main.join(".git").join("HEAD"), "ref: refs/heads/main\n");
        write(
            &main.join(".git").join("config"),
            "[remote \"origin\"]\n\turl = git@github.com:example/repo.git\n",
        );
        let wt_git_dir = main.join(".git").join("worktrees").join("wt1");
        write(&wt_git_dir.join("HEAD"), "ref: refs/heads/feat/from-worktree\n");
        write(&wt_git_dir.join("commondir"), &format!("{}\n", main.join(".git").display()));
        write(&worktree.join(".git"), &format!("gitdir: {}\n", wt_git_dir.display()));

        let ctx = resolve(&worktree).expect("expected a project context for the linked worktree");
        assert!(ctx.linked_worktree);
        assert_eq!(
            ctx.head,
            GitHeadState::Branch("feat/from-worktree".into()),
            "must read the worktree's own HEAD, not the main repo's"
        );
        assert_eq!(ctx.remote.as_deref(), Some("git@github.com:example/repo.git"), "must read origin from the common config");
        assert_eq!(
            ctx.worktree_root,
            strip_verbatim_prefix(&worktree),
            "worktree_root must be the linked worktree's own path, never silently replaced by the main repo"
        );

        let _ = std::fs::remove_dir_all(&main);
        let _ = std::fs::remove_dir_all(&worktree);
    }

    #[test]
    fn two_worktrees_of_the_same_repo_resolve_independently() {
        let main = scratch_dir("two-wt-main");
        let wt_a = scratch_dir("two-wt-a");
        let wt_b = scratch_dir("two-wt-b");
        write(&main.join(".git").join("HEAD"), "ref: refs/heads/main\n");
        write(&main.join(".git").join("config"), "[remote \"origin\"]\n\turl = https://github.com/example/repo.git\n");

        for (wt_dir, name, branch) in [(&wt_a, "wt-a", "feat/a"), (&wt_b, "wt-b", "feat/b")] {
            let wt_git_dir = main.join(".git").join("worktrees").join(name);
            write(&wt_git_dir.join("HEAD"), &format!("ref: refs/heads/{branch}\n"));
            write(&wt_git_dir.join("commondir"), &format!("{}\n", main.join(".git").display()));
            write(wt_dir.join(".git").as_path(), &format!("gitdir: {}\n", wt_git_dir.display()));
        }

        let ctx_a = resolve(&wt_a).unwrap();
        let ctx_b = resolve(&wt_b).unwrap();
        assert_eq!(ctx_a.head, GitHeadState::Branch("feat/a".into()));
        assert_eq!(ctx_b.head, GitHeadState::Branch("feat/b".into()));
        assert_ne!(ctx_a.worktree_root, ctx_b.worktree_root);
        // Both share the same origin, resolved through the same common config.
        assert_eq!(ctx_a.remote, ctx_b.remote);

        let _ = std::fs::remove_dir_all(&main);
        let _ = std::fs::remove_dir_all(&wt_a);
        let _ = std::fs::remove_dir_all(&wt_b);
    }

    #[test]
    fn missing_commondir_file_falls_back_to_git_dir_itself() {
        // A `.git` file pointing straight at a directory that itself holds config/HEAD (no
        // `commondir` file) — a degenerate but not-malformed case the resolver must still handle.
        let root = scratch_dir("no-commondir");
        let git_dir = root.join("actual-git-dir");
        write(&git_dir.join("HEAD"), "ref: refs/heads/main\n");
        write(&git_dir.join("config"), "[remote \"origin\"]\n\turl = https://github.com/example/repo.git\n");
        write(&root.join(".git"), &format!("gitdir: {}\n", git_dir.display()));

        let ctx = resolve(&root).unwrap();
        assert_eq!(ctx.head, GitHeadState::Branch("main".into()));
        assert_eq!(ctx.remote.as_deref(), Some("https://github.com/example/repo.git"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn origin_https_url_is_kept_as_is() {
        let dir = scratch_dir("origin-https");
        make_normal_repo(&dir, "main", Some("https://github.com/org/repo.git"));
        let ctx = resolve(&dir).unwrap();
        assert_eq!(ctx.remote.as_deref(), Some("https://github.com/org/repo.git"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn origin_scp_style_ssh_url_is_kept_as_is() {
        let dir = scratch_dir("origin-scp");
        make_normal_repo(&dir, "main", Some("git@github.com:org/repo.git"));
        let ctx = resolve(&dir).unwrap();
        assert_eq!(ctx.remote.as_deref(), Some("git@github.com:org/repo.git"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn origin_with_embedded_credentials_is_sanitized() {
        const FAKE_TOKEN: &str = "ghp_fakeTestTokenNotReal0000";
        let dir = scratch_dir("origin-credential");
        make_normal_repo(&dir, "main", Some(&format!("https://user:{FAKE_TOKEN}@github.com/org/repo.git")));
        let ctx = resolve(&dir).unwrap();
        let remote = ctx.remote.as_deref().unwrap();
        assert_eq!(remote, "https://github.com/org/repo.git");
        assert!(!remote.contains(FAKE_TOKEN), "sanitized remote must not contain the credential");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repo_without_origin_has_no_remote() {
        let dir = scratch_dir("no-origin");
        make_normal_repo(&dir, "main", None);
        let ctx = resolve(&dir).unwrap();
        assert!(ctx.remote.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn windows_path_with_spaces_resolves() {
        let dir = scratch_dir("with spaces in name");
        make_normal_repo(&dir, "main", None);
        let ctx = resolve(&dir).unwrap();
        assert_eq!(ctx.head, GitHeadState::Branch("main".into()));
        assert!(ctx.repository_name.contains("with spaces in name"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unicode_path_resolves() {
        let dir = scratch_dir("projeto-café-日本語");
        make_normal_repo(&dir, "main", None);
        let ctx = resolve(&dir).unwrap();
        assert_eq!(ctx.head, GitHeadState::Branch("main".into()));
        assert!(ctx.repository_name.contains("café") && ctx.repository_name.contains("日本語"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verbatim_prefix_is_stripped_from_worktree_root() {
        let dir = scratch_dir("verbatim-prefix");
        make_normal_repo(&dir, "main", None);
        let verbatim = Path::new(r"\\?\").join(&dir);
        let ctx = resolve(&verbatim).expect("resolution must work even when cwd carries the \\\\?\\ prefix");
        assert!(!ctx.worktree_root.starts_with(r"\\?\"), "displayed worktree_root must not carry the verbatim prefix: {}", ctx.worktree_root);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_head_is_none_without_crashing() {
        let dir = scratch_dir("missing-head");
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        // HEAD intentionally absent.
        let ctx = resolve(&dir).expect("a bare .git directory is still a resolvable (if empty) project context");
        assert_eq!(ctx.head, GitHeadState::Unknown, "missing HEAD is not positive evidence of a detached state, nor a branch");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_config_has_no_remote_without_crashing() {
        let dir = scratch_dir("missing-config");
        write(&dir.join(".git").join("HEAD"), "ref: refs/heads/main\n");
        // config intentionally absent.
        let ctx = resolve(&dir).unwrap();
        assert!(ctx.remote.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repo_removed_after_discovery_is_handled_gracefully() {
        let dir = scratch_dir("removed-repo");
        make_normal_repo(&dir, "main", None);
        std::fs::remove_dir_all(&dir).unwrap();
        // The path no longer exists at all: no crash, just no project context.
        assert!(resolve(&dir).is_none());
    }

    #[test]
    fn malformed_git_file_degrades_to_partial_context_without_crashing() {
        let dir = scratch_dir("malformed-gitfile");
        write(&dir.join(".git"), "this is not a gitdir pointer at all\n");
        let ctx = resolve(&dir).expect("a .git file, even unparseable, still marks a worktree root");
        assert!(ctx.linked_worktree);
        assert_eq!(ctx.head, GitHeadState::Unknown);
        assert!(ctx.remote.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---------------- remote sanitizer hardening ----------------
    // Direct unit tests of `sanitize_remote_url`, kept separate from the end-to-end fixtures
    // above so each case is a one-liner and the intent (what must change vs. what must not) is
    // impossible to miss. All secrets below are fake, invented for the test only.

    #[test]
    fn sanitizer_strips_https_user_and_password() {
        assert_eq!(
            sanitize_remote_url("https://user:fakePass123@github.com/org/repo.git"),
            "https://github.com/org/repo.git"
        );
    }

    #[test]
    fn sanitizer_strips_https_bare_token_userinfo() {
        assert_eq!(
            sanitize_remote_url("https://ghp_fakeToken000000000000@github.com/org/repo.git"),
            "https://github.com/org/repo.git"
        );
    }

    #[test]
    fn sanitizer_leaves_https_without_userinfo_unchanged() {
        assert_eq!(
            sanitize_remote_url("https://github.com/org/repo.git"),
            "https://github.com/org/repo.git"
        );
    }

    #[test]
    fn sanitizer_leaves_scp_style_ssh_unchanged() {
        assert_eq!(
            sanitize_remote_url("git@github.com:org/repo.git"),
            "git@github.com:org/repo.git"
        );
    }

    /// The important case from this hardening round: `ssh://` with a bare username (no `:`) is
    /// the normal SSH identity convention, not a leaked credential — `git@` must survive.
    #[test]
    fn sanitizer_leaves_ssh_scheme_bare_username_unchanged() {
        assert_eq!(
            sanitize_remote_url("ssh://git@github.com/org/repo.git"),
            "ssh://git@github.com/org/repo.git"
        );
        // Any legitimate-looking SSH username, not just "git", must be preserved the same way.
        assert_eq!(
            sanitize_remote_url("ssh://deploy-ci@example.com/org/repo.git"),
            "ssh://deploy-ci@example.com/org/repo.git"
        );
    }

    /// Unlike a bare SSH username, a `user:password@` form in `ssh://` is not a real SSH login
    /// method and is still treated as a credential to strip.
    #[test]
    fn sanitizer_still_strips_ssh_scheme_userinfo_with_a_password() {
        assert_eq!(
            sanitize_remote_url("ssh://git:fakeSecret@github.com/org/repo.git"),
            "ssh://github.com/org/repo.git"
        );
    }

    /// An `@` that is part of the path (a repository name containing `@`) must never be mistaken
    /// for a userinfo delimiter — userinfo cannot contain `/`, so this must be left untouched.
    #[test]
    fn sanitizer_does_not_touch_an_at_sign_inside_the_path() {
        assert_eq!(
            sanitize_remote_url("https://github.com/org/rep@o.git"),
            "https://github.com/org/rep@o.git"
        );
    }

    #[test]
    fn origin_ssh_scheme_with_bare_username_is_kept_as_is_end_to_end() {
        let dir = scratch_dir("origin-ssh-bare-user");
        make_normal_repo(&dir, "main", Some("ssh://git@github.com/org/repo.git"));
        let ctx = resolve(&dir).unwrap();
        assert_eq!(ctx.remote.as_deref(), Some("ssh://git@github.com/org/repo.git"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
