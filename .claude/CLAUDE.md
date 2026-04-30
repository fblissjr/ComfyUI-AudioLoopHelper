Last updated: 2026-04-30

<!-- path-privacy: skip-file — this file is about the harness's privacy + path conventions, so example placeholders necessarily appear -->

# Claude instructions — working in `.claude/`

You are editing the project's Claude Code harness: agents, skills, hooks,
and settings. This file orients you to the conventions specific to this
directory; the human-oriented overview is `.claude/README.md`. Root
project conventions live in `../CLAUDE.md`.

## Mental model

`.claude/` is **executable configuration**, not application code. The
contents change Claude Code's behavior for every contributor working in
this repo. Treat edits with the same care you'd give to CI configuration
or a pre-commit hook — a typo in a hook command silently no-ops; a stale
agent description silently misroutes invocations.

## File layout (committed vs local)

| Path | Tracked? | Purpose |
|---|---|---|
| `agents/*.md` | yes | Subagent definitions (autonomous reviewers) |
| `skills/*/SKILL.md` | yes | User-invocable workflows |
| `hooks/*.{py,sh}` | yes | PreToolUse/PostToolUse/Stop/SessionStart automations |
| `settings.json` | yes | Hook wiring; uses `${CLAUDE_PROJECT_DIR}` for portability |
| `README.md` | yes | Human contributor overview |
| `CLAUDE.md` | yes | This file — Claude-facing conventions |
| `*.local.json.example` | yes | Templates for the gitignored variants |
| `settings.local.json` | **NO** | Per-user permissions + ComfyUI-loader smoke test |
| `privacy_patterns.local.json` | **NO** | Literal leak patterns for `hooks/privacy_guard.py` |
| `skills/cross-repo-handoff/` | **NO** | Sister-repo coordination (sage-fork symlink dependency) |

`.gitignore` carries selective rules — see root `.gitignore` lines for
`.claude/`. Don't blanket-ignore `.claude/`; do add new gitignores for
any new `*.local.*` file you introduce.

## Conventions

### Hook authoring

- **stdlib only**. Hooks run on a fresh clone before `uv sync`, so they
  cannot import `orjson` or any other installed package. The project's
  global "always orjson" rule has an explicit carve-out for `.claude/hooks/`.
- **Fail open by default.** A hook that wedges on an unexpected payload
  blocks every Write/Edit in the session. Catch broad exceptions, return
  exit 0 unless the rule deliberately blocks (return exit 2 + stderr).
- **Walk to find project root.** Don't hardcode paths. Use the same
  `_find_project_root()` walk-up pattern that `privacy_guard.py` and
  `doc_date_check.py` use — looks for `.git` or `pyproject.toml`. Or
  honor `$CLAUDE_PROJECT_DIR` if set.
- **Hot-path awareness.** Hooks fire on EVERY matched tool call. Keep the
  per-call cost low. Inline `bash -c '...grep...'` is fine for a single
  conditional; anything heavier should defer to a Python script.
- **Hardening for SessionStart hooks**: see `hooks/check_memo_inbox.sh`'s
  preamble for the canonical "no eval / no network / SIGPIPE-safe / audit-able
  in <100 lines" rule set.

### Agent authoring

- The frontmatter `description:` is what Claude reads to decide whether to
  dispatch the agent. Make it specific — "validate workflow JSON for the
  audio-looped music video pipeline; checks AudioLoopController schema, …"
  beats "validate workflows."
- **Read-only by default.** Agents that report findings (privacy-scrubber,
  workflow-validator, ltx-constraints-auditor, conditioning-path-auditor)
  must say so explicitly so the user knows fixes are still their job.
- **Don't enumerate audit IDs in agent prose.** The list rots. Point to
  `docs/reference/debug_tools.md` and the live `record(...)` calls in
  `scripts/audit_workflows.py`. Same rule applies to skill prose.

### Skill authoring

- Skills are user-invokable workflows. Frontmatter `description:` controls
  triggering — be specific about WHEN to use it ("when the user mentions
  'workflow won't run' or 'dependency cycle detected'") not just WHAT.
- **No hardcoded absolute paths.** Use `cd "$(git rev-parse --show-toplevel)"`
  for cwd setup. Use `${COMFYUI_ROOT}` or ask the user when a skill needs
  to know where their ComfyUI install lives.
- **Reference scripts by repo-relative path.** `scripts/audit_workflows.py`,
  not `internal/scripts/...`. Verify the path exists before adding it —
  the broken-hook-path bug from 2026-04-30 came from a moved script
  whose reference wasn't updated.

### Privacy

- The hook `hooks/privacy_guard.py` blocks public-file writes that contain
  patterns from `privacy_patterns.local.json` (gitignored). If absent, it
  falls back to a generic `/home/<user>/` regex.
- **Never inline literal private patterns** (your specific username, your
  ComfyUI path) into committed `.claude/` files. The `privacy-scrubber`
  agent and `scrub-for-public` skill reference the gitignored config by
  pointer, not by inlining.
- The hook's own pattern source contains exempt examples (the regex strings
  themselves) — that's intentional, but the comment at the relevant call
  site explains why so future reviewers don't strip them as leaks.

### Settings split

- **Committed `settings.json`** — portable hooks; uses `${CLAUDE_PROJECT_DIR}`.
- **Per-user `settings.local.json`** — machine-specific permissions
  (model dir reads, scratch dir reads) + the ComfyUI-loader smoke test
  (which needs the absolute path to the user's ComfyUI install).
- **Don't add machine-specific entries to the committed file.** If you're
  about to write `/home/anyone/...` into `settings.json`, stop — it goes
  in `settings.local.json` instead, with a corresponding update to
  `settings.local.json.example` so other clones know about the slot.

## When you're about to ship a change here

1. **Privacy**: run `python3 .claude/hooks/privacy_guard.py` mentally —
   does anything you wrote contain `/home/<user>/`, your username, or a
   personal-storage path?
2. **JSON validity**: `python3 -c "import json; json.load(open('.claude/X.json'))"`
   on every JSON file you touched.
3. **Hook smoke test**: if you changed a hook, run it against a synthetic
   payload via stdin (see the `_scan` smoke test pattern in privacy_guard's
   docstring) before declaring it shipped.
4. **gitignore check**: `git ls-files --others --exclude-standard .claude/`
   to confirm gitignores are doing what you expect.
5. **Update root CLAUDE.md** if a convention changed (where the root file
   talks about hooks/skills/agents/settings).

## Drift protection

The harness rots when CLAUDE.md changes faster than `.claude/` does. Two
mechanisms in place:

1. **Periodic harness drift audit** — schedule a remote-trigger routine
   (via `/schedule`) that re-runs the audit logic against current CLAUDE.md
   and opens a PR if drift is unambiguous. The maintainer of this clone
   handles their own routine — routine IDs are per-account, not portable.
2. **Audit baseline** — when one exists, the prior-pass findings live at
   `internal/analysis/harness_analysis.md` (gitignored, so public-clone
   readers won't see it). Read it before authoring a new audit doc, and
   update it when fixes ship.

Add a third mechanism (`scripts/validate_skills_consistency.py`) when the
maintenance load justifies it. Today the audit routine is enough.

## Pointers

- Root project rules: `../CLAUDE.md`
- Human contributor overview: `./README.md`
- Audit baseline (when present): `../internal/analysis/harness_analysis.md` (gitignored — won't exist on a fresh clone)
- Tooling reference: `../docs/reference/debug_tools.md`
