Last updated: 2026-05-04

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
| `settings.local.json.example` | yes | Template for the gitignored `settings.local.json` |
| `settings.local.json` | **NO** | Per-user permissions + ComfyUI-loader smoke test |
| `<repo-root>/.path-privacy.local.json` | **NO** | Literal-substring suggestion config consumed by the `path-privacy` plugin (loaded by its `pre-tool-use.sh` + `find-external-paths.sh` + `scrub-paths.sh`) |
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

- Path-privacy enforcement comes from the `path-privacy` plugin in the
  `fb-claude-skills` marketplace, NOT from in-repo hooks. The plugin
  ships a PreToolUse Write/Edit hook (blocks at edit time), git
  pre-commit + commit-msg hooks (block at commit time), and a
  SessionStart directive (so Claude knows the rule).
- Suggestion config lives at `<repo-root>/.path-privacy.local.json`
  (gitignored). The plugin's scanner uses it to emit `→ use:`
  actionable hints; its `scrub-paths.sh` uses the same entries for
  one-command fixes (with diff preview + opt-in `--apply`).
- **Never inline literal private patterns** (your specific username,
  your ComfyUI path) into committed `.claude/` files. Use placeholder
  forms (`<user>`, `<comfyui>`, `${CLAUDE_PROJECT_DIR}`) or repo-relative
  paths. The plugin's pre-commit hook will hard-block leaks; do not
  bypass with `--no-verify`.
- For files that legitimately need to mention path shapes (regex source,
  doc examples, this file), use the plugin's escape hatches:
  `<!-- path-privacy: skip-file -->` near the top of the file, or
  `# path-privacy: ignore` on individual lines.

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

1. **Privacy**: the path-privacy plugin's PreToolUse hook will block
   leaks at edit time, and its pre-commit/commit-msg hooks block at
   commit time. To audit the working tree on demand:
   `bash <plugin-root>/skills/path-privacy/skills/path-privacy/scripts/find-external-paths.sh -d .`
2. **JSON validity**: `python3 -c "import json; json.load(open('.claude/X.json'))"`
   on every JSON file you touched.
3. **Hook smoke test**: if you changed a hook, run it against a synthetic
   payload via stdin before declaring it shipped.
4. **gitignore check**: `git ls-files --others --exclude-standard .claude/`
   to confirm gitignores are doing what you expect.
5. **Update root CLAUDE.md** if a convention changed (where the root file
   talks about hooks/skills/agents/settings).

## CLAUDE.md governance

This is the policy that keeps root and subtree CLAUDE.md files load-bearing
without growing monotonically. Established 2026-05-04 after a curation pass
brought root CLAUDE.md from 51 KB / 237 lines back under budget.

### The four layers

Knowledge in this repo lives in one of four layers, in order of reliability:

| Layer | Where | What lives here | Reliability |
|---|---|---|---|
| 1. Rules-as-code | `scripts/audit_workflows.py`, `tests/test_*.py`, hooks | Mechanically enforceable invariants (F-checks, schema AST tests, link integrity) | Highest — can't drift |
| 2. Wiki / canonical docs | `docs/reference/`, `docs/guides/`, `docs/analysis/` | The "why" + atomic-note deep dives | Medium — humans must update |
| 3. CLAUDE.md (root + subtree) | `./CLAUDE.md`, subtree CLAUDE.md files | Rules a fresh session must know on turn 1 + pointers down to layers 1–2 | Variable — depends on discipline |
| 4. Findings ledger | `internal/findings_ledger.md`, `internal/log/`, `internal/postmortem_*.md` | Raw findings before they earn promotion; investigation narratives | Transient |

The trap: findings naturally land in layer 3 (easiest place to write) and
never get promoted down to 1–2 or demoted to 4. The lifecycle below is the
counter-pressure.

### Rule lifecycle

```
[experimental finding]                layer 4 (findings_ledger / log / postmortem)
        ↓ stabilizes after N reproductions
[stable rule]                         layer 1 (audit/test) + layer 2 (atomic note)
        ↓ load-bearing on turn 1?
[CLAUDE.md rule]                      layer 3 (one-line rule + pointer down)
```

Promotion criteria — layer 4 → layer 1+2:
- Reproduced at least twice, OR
- Has a one-line audit/test that catches it, OR
- Has a paired apply-script (the F-pair convention).

Promotion to layer 3 requires **all three**:
- A fresh Claude session would silently get it wrong without this rule.
- The cost of getting it wrong exceeds the cost of carrying the line.
- It can't be expressed as a runnable check in layer 1.

### Budget (root CLAUDE.md)

Hard cap: **200 lines / 30 KB**. Enforced by `tests/test_claude_md_budget.py`.
New rule in = old rule earns its way out (down to layer 1/2/4) or compresses
to a one-line pointer. Subtree CLAUDE.md files soft-warn at 500 lines.

### Pointer discipline

1. **Each fact has exactly one canonical home.** No restatement. If
   `docs/reference/sampler_reference.md` is canonical for sigma-chain rules,
   root CLAUDE.md says "Sigma chain: see `docs/reference/sampler_reference.md`."
   That's it — the long prose lives in the canonical doc.
2. **Pointers are repo-relative paths**, not section names. Section names
   drift; paths break loudly when files move (the budget test catches this).
3. **`docs/README.md` is the master index.** Root CLAUDE.md points there for
   anything that's not a turn-1 must-know.

### Subtree CLAUDE.md

Threshold for creating one: **≥ 5 substantive subtree-specific rules** that
don't apply elsewhere. Below that bar, rules stay in root.

Active subtree files:
- `./scripts/CLAUDE.md` — apply-script conventions, audit invariants, WorkflowEditor patterns.
- `./tests/CLAUDE.md` — pytest invocation, AST patterns, fakes hierarchy.
- `./internal/autoresearch/CLAUDE.md` — experiment-runner framework (target-agnostic).

Loading: Claude Code auto-discovers CLAUDE.md files along the directory walk.
A session working in `scripts/` loads `./CLAUDE.md` + `./scripts/CLAUDE.md`
and does NOT load tests or autoresearch conventions. This is the progressive-
disclosure mechanism.

Cross-link rule: when a subtree rule could plausibly apply outside its
subtree (e.g. someone editing from project root who'd benefit from knowing a
scripts/ rule), root CLAUDE.md adds a one-line pointer ("Working in
`scripts/`? See `scripts/CLAUDE.md`.") so the gotcha doesn't bite.

### Capture-then-review

When you (or a `#`-key capture) wants to land a new rule, drop it in the
**"Pending review"** section at the bottom of root CLAUDE.md instead of
inserting inline into the curated body. The pending section gets drained on
the next curation pass — most pending entries demote to layer 4 (archive),
some promote to layer 1 (audit/test), few earn a slot in the curated body.

This separation prevents the curation discipline from being slowly eroded by
in-the-moment additions.

### Wiki direction (Karpathy LLM wiki pattern)

Layer 2 is evolving toward Karpathy's LLM-wiki shape (gist:
`https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`). Three
implications for how new docs are written:

1. **Atomic notes with uniform shape.** Each `docs/reference/X.md` follows
   the template seeded by `docs/reference/frame_planner_reference.md`:
   Last-updated date → Role → Key facts → Mechanism / Wiring → Gotchas →
   Migration → Audit + tests → References (in + out). New reference docs
   match this shape.
2. **Strong cross-linking.** Every atomic note ends with a References
   section listing both outgoing pointers (other docs/scripts cited) and
   the upstream `docs/README.md` index entry. The lint mode in
   `tests/test_claude_md_budget.py` fails on orphan reference docs.
3. **Three operational modes** (informal, no skill yet):
   - *Ingest*: when a new finding stabilizes, write the atomic note → update
     `docs/README.md` index → update related notes that should cross-link →
     append to `internal/log/`.
   - *Query*: read the index → traverse cross-links → synthesize.
   - *Lint*: `audit_workflows.py` (rules-as-code), budget test (size +
     orphans + broken pointers), `validate_docs_consistency.py` (stale
     phrases). Together these are the wiki's lint pass.

A formal ingest skill is a P5 follow-up; for now the convention is in your
hands.

### Curation cadence

- **Per-feature ship**: drain "Pending review" section. ~5 min.
- **Quarterly**: full pass via `claude-md-improver` skill. Scheduled via
  `/schedule` (per-account routine ID; the maintainer of this clone owns it).
- **Whenever the budget test fails**: fix immediately, don't paper over.

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

Add a third mechanism (a skills-consistency validator under `scripts/`) when
the maintenance load justifies it. Today the audit routine is enough.

## Pointers

- Root project rules: `../CLAUDE.md`
- Human contributor overview: `./README.md`
- Audit baseline (when present): `../internal/analysis/harness_analysis.md` (gitignored — won't exist on a fresh clone)
- Tooling reference: `../docs/reference/debug_tools.md`
