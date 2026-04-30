---
name: apply-script-scaffold
description: Scaffold a new scripts/apply_*.py workflow migration script from scripts/templates/, enforcing the project's idempotence + paired-audit-check convention. Use when adding a new fix that mutates example_workflows/*.json or stages an experimental variant.
---

Wraps the convention codified in `scripts/templates/README.md` and CLAUDE.md "Apply scripts" section. Every apply script must be idempotent (`md5sum`-stable on re-run), support `--revert` and `--dry-run`, use `WorkflowEditor`, and ship with a paired audit check in `scripts/audit_workflows.py` so a sibling branch can't silently regress the fix.

## When to invoke

- Adding a new bug fix that mutates ALL `example_workflows/*.json` (e.g. a topology constraint, a node-wiring fix).
- Staging a new experimental variant under `internal/scratch/` or `example_workflows/experimental/`.
- Migrating an existing one-off ad-hoc edit script to the canonical shape.

**Don't use** for one-time data exploration (those go in `internal/scratch/`).

## Steps

1. **Pick the right template** based on scope:

```bash
cd "$(git rev-parse --show-toplevel)"
ls scripts/templates/
```

| Scope | Template | Canonical example |
|---|---|---|
| Mutate all `example_workflows/*.json` in place | `apply_script_all_workflows.py` | `apply_sage_mode.py`, `apply_loop_guide_preprocess_symmetry.py` |
| Stage experimental variant | `apply_script_staged_variant.py` | `apply_iclora_initial_render.py`, `apply_spectrogram_iclora_minimal.py` |

**Rule of thumb:** Bug fix → all-workflows. Experimental → staged.

2. **Copy the template** to a new file:

```bash
cp scripts/templates/apply_script_all_workflows.py scripts/apply_<NAME>.py
# or
cp scripts/templates/apply_script_staged_variant.py scripts/apply_<NAME>.py
```

3. **Substitute placeholders** in the new file's docstring:
   - `<SCRIPT_NAME>`, `<YYYY-MM-DD>` (use today's date)
   - `<SYMPTOM>`, `<ROOT_CAUSE>`, `<FIX>` — explain the WHY
   - `<COMPATIBILITY_NOTES>` — list other apply scripts this one composes with
   - `NODE_ID_*` constants — replace with actual integer node IDs from the workflow

4. **Implement `_apply_one(wf_path, revert, dry_run)`** using `WorkflowEditor`:
   - Use `ed.find_node`, `ed.has_node`, `ed.require_nodes` for guards (NEVER hand-roll link lookups).
   - Use `ed.find_link_to_slot(tgt, slot)` and `ed.rewire_input(tgt, slot, src, src_slot, dtype)` instead of manual `add_link`/`remove_link` splices.
   - Subgraph edits: use `ed.find_subgraph_link_to_slot` and `ed.rewire_subgraph_input`.
   - Idempotence guard at top: `if _is_already_built(wf): return "already applied"`.
   - Return string status: `"applied"`, `"already applied"`, `"reverted"`, `"no change"`, or error message.

5. **Verify idempotence** before committing:

```bash
md5sum example_workflows/audio-loop-music-video_latent.json > /tmp/pre.md5
uv run --group dev python scripts/apply_<NAME>.py
md5sum example_workflows/audio-loop-music-video_latent.json > /tmp/run1.md5
uv run --group dev python scripts/apply_<NAME>.py  # second run should be no-op
md5sum example_workflows/audio-loop-music-video_latent.json > /tmp/run2.md5
diff /tmp/run1.md5 /tmp/run2.md5 && echo "idempotent OK"

# Verify --revert returns to original:
uv run --group dev python scripts/apply_<NAME>.py --revert
md5sum example_workflows/audio-loop-music-video_latent.json > /tmp/revert.md5
diff /tmp/pre.md5 /tmp/revert.md5 && echo "revert OK"
```

6. **Add the paired audit check** in `scripts/audit_workflows.py` (CLAUDE.md-mandatory):
   - Add an inline check function that detects the post-fix state (or pre-fix state for staged variants).
   - ERR status with a remediation pointer to your apply script: `"Run scripts/apply_<NAME>.py to fix."`
   - Canonical pairs to study: F2 (`preprocess_symmetry`), F3 (`loop_cropguides_symmetry`).
   - Don't DRY against `WorkflowEditor` — `audit_workflows.py` is intentionally `WorkflowEditor`-independent (CLAUDE.md: "Debug tool must stay usable when the editor it audits has a bug").

7. **Run audit + tests**:

```bash
uv run --group dev python scripts/audit_workflows.py
uv run --group dev --group analysis python -m pytest tests/test_workflows.py -v --rootdir=.
```

8. **Update CHANGELOG** under `### Added` (new feature) or `### Fixed` (bug fix). Cross-reference the symptom and the audit-check name.

## Common pitfalls (caught by `/simplify` reviews historically)

- **Hand-rolling link splices.** Always `rewire_input` / `rewire_subgraph_input`. The `remove_link` + `add_link` pattern leaves stale local refs because `remove_link` rebinds the target list via filter.
- **Forgetting the `--revert` path.** Idempotent forward + idempotent reverse is the contract. Both must round-trip via `md5sum`.
- **Skipping the audit check.** Sibling branches can silently revert your fix. The audit check is the only durable guard.
- **Using `find_node` with strict assertions.** Use `require_nodes` when missing-is-fatal; `find_node` returns None for soft checks.
- **Bulk `replace_all` with quoted patterns.** If the apply script's prose annotation references the pattern being replaced, the bulk edit sweeps the annotation. Add prose AFTER bulk edits, or scope the pattern.

## Reference

- Templates: `scripts/templates/apply_script_all_workflows.py`, `scripts/templates/apply_script_staged_variant.py`
- Convention: `scripts/templates/README.md` and CLAUDE.md "Apply scripts" + "Subgraph editing"
- Editor API: `scripts/workflow_utils.py` (`WorkflowEditor` class)
- Helpers: `scripts/_apply_helpers.py` (shared utilities — import with aliases to preserve call-site names)
