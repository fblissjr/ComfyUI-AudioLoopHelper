Last updated: 2026-04-30

# Claude Code harness for ComfyUI-AudioLoopHelper

This directory holds the project-specific Claude Code configuration:
agents, skills, hooks, and settings. Most files here are tracked in git so
contributors share the same automation. A small set of `*.local.*` files
stay per-user (see "Local-only files" below).

## Layout

```
.claude/
├── README.md                       (this file)
├── settings.json                   (committed — portable hooks)
├── settings.local.json             (gitignored — per-user permissions + ComfyUI loader test)
├── agents/
│   ├── conditioning-path-auditor.md
│   ├── ltx-constraints-auditor.md
│   └── workflow-validator.md
├── hooks/
│   ├── check_memo_inbox.sh         (SessionStart)
│   └── doc_date_check.py           (PostToolUse Write/Edit)
└── skills/
    ├── apply-script-scaffold/
    ├── audio-analyze/
    ├── comfyui-test/
    ├── compare-workflows/
    ├── cross-repo-handoff/         (gitignored — sister-repo coordination)
    ├── diagnose-workflow/
    ├── prompt-schedule/
    ├── release-notes/
    ├── sage-trace-summary/
    └── workflow-edit/
```

Path-privacy enforcement (Write/Edit + git pre-commit/commit-msg) and the
companion `scrub` workflow now come from the `path-privacy` plugin in the
`fb-claude-skills` marketplace, not from in-repo hooks/agents/skills. See
"Path privacy" below.

## Local-only files

These are not in git. Each new clone needs to either ignore them
(reasonable defaults work) or create their own copy.

| File | Purpose | What to do on first clone |
|---|---|---|
| `settings.local.json` | Per-user permissions + ComfyUI loader smoke test (needs the path to your ComfyUI install) | Copy from `settings.local.json.example` and edit; OR omit entirely — settings.json provides the portable hooks |
| `<repo-root>/.path-privacy.local.json` | Suggestion config consumed by the `path-privacy` plugin (literal-substring → placeholder mappings used in `→ use:` diagnostic and by `scrub-paths.sh`). | Copy the starter from `<plugin-root>/skills/path-privacy/skills/path-privacy/references/path-privacy.local.json.example` and edit for your machine; OR omit — the plugin's scanner falls back to a generic `/home/<user>/` regex |
| `skills/cross-repo-handoff/` | Memo channel between this Claude session and a sister sage-fork session. Depends on `coderef/sage-fork/` symlink. | Skip unless you're working on the sage-fork too |

## How the hooks compose

`settings.json` (committed) provides:

- **PreToolUse Write/Edit**: notice when editing `example_workflows/*.json`
  directly — convention is to use `scripts/apply_*.py` instead.
- **PostToolUse Write/Edit**: workflow integrity validator
  (`scripts/test_workflow_integrity.py`) on JSON edits;
  schema-change warning on `nodes*.py` edits; `doc_date_check.py` for
  missing/stale `Last updated:` stamps in markdown.
- **Stop**: reminds you to write a session log entry to
  `internal/log/log_YYYY-MM-DD.md` if missing.
- **SessionStart**: `check_memo_inbox.sh` surfaces any new memo from the
  sage-fork claude session.

`settings.local.json` (per-user) adds the ComfyUI-loader smoke test which
needs to know where your ComfyUI install lives — this is intentionally
not committed because it differs across machines.

The `path-privacy` plugin (installed via the `fb-claude-skills`
marketplace) layers on:

- **PreToolUse Write/Edit**: blocks writes that would introduce a path
  leak before bytes hit disk.
- **SessionStart**: injects the path-privacy directive so Claude knows
  the rule.
- **Git pre-commit + commit-msg**: hard-block leaks in staged files,
  commit messages, and branch names. Install via
  `bash <plugin-root>/skills/path-privacy/skills/path-privacy/scripts/install-git-hooks.sh`.

> **Note**: the agent + skill tables below are hand-maintained for at-a-glance
> readability. Each agent's `description:` frontmatter is the source of truth
> if these go stale.

## How the agents compose

| Agent | Use case |
|---|---|
| `workflow-validator` | "is this workflow JSON shaped correctly?" — runs `scripts/audit_workflows.py` first, then drills into IMAGE vs LATENT specifics |
| `conditioning-path-auditor` | "are CONDITIONING flows symmetric between initial render and loop body?" — catches the frame_rate-metadata-asymmetry bug class |
| `ltx-constraints-auditor` | "does this workflow honor LTX 2.3 critical constraints?" — sigma chain, decoder tile, noise_mask, etc. (semantic, not structural) |

For privacy auditing, use the `path-privacy` plugin: `bash
<plugin-root>/skills/path-privacy/skills/path-privacy/scripts/find-external-paths.sh
-d .` (audit) or `... scrub-paths.sh -d .` (preview fixes).

## How the skills compose

| Skill | Trigger / use case |
|---|---|
| `apply-script-scaffold` | starting a new `scripts/apply_*.py` — bakes in idempotence, `--revert`, paired audit check |
| `audio-analyze` | analyzing a track for prompt scheduling (or diagnosing generated audio) |
| `comfyui-test` | running the test suite with the right uv groups + ComfyUI loader verification |
| `compare-workflows` | structural diff between two workflow JSONs (filters link IDs / positions / ordering) |
| `diagnose-workflow` | canonical first-pass when a workflow won't run / fails validation |
| `prompt-schedule` | generating prompt-schedule variations from audio + init image |
| `release-notes` | drafting next CHANGELOG.md section from commits since last release |
| `sage-trace-summary` | post-gen sage telemetry report (per-prompt + attribution + gate verdict) |
| `workflow-edit` | editing workflow JSON via WorkflowEditor (avoids hand-rolled link splices) |

## Path privacy

This repo previously shipped a home-grown `privacy_guard.py` PreToolUse
hook + `privacy-scrubber` agent + `scrub-for-public` skill. They were
retired once the `path-privacy` plugin (in the `fb-claude-skills`
marketplace) reached feature parity:

| Old (in-repo) | New (plugin) |
|---|---|
| `.claude/hooks/privacy_guard.py` (PreToolUse Write/Edit blocker) | `path-privacy` plugin's `pre-tool-use.sh` hook |
| `.claude/agents/privacy-scrubber.md` (audit) | `find-external-paths.sh` script (same surface, more robust) |
| `.claude/skills/scrub-for-public/` (apply fixes) | `scrub-paths.sh` script (with `--apply` opt-in + diff preview) |
| `.claude/privacy_patterns.local.json` (regex + replacements) | `<repo-root>/.path-privacy.local.json` (literal-substring suggestions; same purpose) |

The plugin's `pre-commit` and `commit-msg` git hooks add a second
defensive layer that the home-grown version didn't have. Install them
once per clone via the plugin's `install-git-hooks.sh`.

## Maintenance

The harness rots quickly when CLAUDE.md changes. Two protections:

1. **Schedule a periodic re-audit.** Use `/schedule` to create a
   remote-trigger routine that re-runs the audit logic against current
   CLAUDE.md and opens a PR with fixes when drift is unambiguous.
   Routine IDs are per-account; each maintainer schedules their own.
2. **Audit baseline.** When one exists, prior findings live at
   `internal/analysis/harness_analysis.md` (gitignored — present on
   the working clone, absent on fresh public clones). It documents
   what each drift finding meant + how it was remediated.

When you add a new fix that ships an apply script, also update any skill
or agent that references the audit's named-check inventory — same rule
as CLAUDE.md's "Bake new topology constraints into `audit_workflows.py`"
mandate.
