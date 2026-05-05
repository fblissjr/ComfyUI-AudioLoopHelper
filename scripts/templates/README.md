Last updated: 2026-05-05

# scripts/templates/

Scaffolds for new `scripts/apply_*.py` workflow migrations. 14+ existing apply
scripts share the same shape — these templates codify it so new ones stay
consistent (idempotent, `--revert`, `--dry-run`, WorkflowEditor-only).

## Which template?

| Scope | File | Canonical example |
|---|---|---|
| Mutate all `example_workflows/*.json` in place | `apply_script_all_workflows.py` | `apply_sage_mode.py` (archived), `apply_loop_guide_preprocess_symmetry.py` |
| Stage experimental variant to `internal/scratch/<base>_<feature>_<phase>.json` | `apply_script_staged_variant.py` | `apply_iclora_initial_render.py` |

**Bug fix → all-workflows. Experimental variant → staged.** Per `internal/PLAN.md`,
staged variants promote to `example_workflows/` only when they ship AND stabilize.

## Scaffold a new apply script

```bash
cp scripts/templates/apply_script_all_workflows.py scripts/apply_<name>.py
```

Then substitute:
- `<SCRIPT_NAME>`, `<YYYY-MM-DD>`, `<SYMPTOM>`, `<ROOT_CAUSE>`, `<FIX>`, `<COMPATIBILITY_NOTES>` in the docstring
- `NODE_ID_*` constants with the actual IDs
- `_apply_one` body with the rewire logic (see the template's TODO comments for a rewire skeleton)

## Invariants every apply script must preserve

1. Uses `WorkflowEditor` exclusively — never edit JSON by hand. Top-level
   helpers: `find_node`, `has_node`, `require_nodes`, `find_link_to_slot`,
   `add_link`, `remove_link`, `rewire_input`. Subgraph: `add_subgraph_link`,
   `remove_subgraph_link`, `find_subgraph_link`.
2. `require_nodes([...])` guard up front — skip workflows whose layout doesn't match.
3. Idempotent: re-running reports "no change (already ...)" without writing.
4. `--revert` undoes exactly what apply did (same guards, inverse wiring).
5. `--dry-run` reports what WOULD change without writing. HyDE pattern:
   `scripts/apply_X.py --dry-run | scripts/audit_workflows.py --stdin` lets
   you audit a hypothetical state before committing to it.
6. Exit code 1 on any load error; 0 otherwise.
7. Docstring starts with `Last updated: YYYY-MM-DD`; documents symptom →
   root cause → fix → compatibility notes with other apply scripts.

## After scaffolding

```bash
uv run --group dev python scripts/apply_<name>.py --dry-run   # preview diff
uv run --group dev python scripts/apply_<name>.py             # apply
uv run --group dev python scripts/apply_<name>.py             # idempotent re-run
uv run --group dev python scripts/apply_<name>.py --revert    # undo
uv run --group dev python scripts/apply_<name>.py             # re-apply (verify round-trip)
uv run --group dev python scripts/audit_workflows.py          # 0 ERR required
uv run --group dev --group analysis python -m pytest tests/   # 286+ pass
```

## Compatibility check

Before writing: grep existing scripts for the node IDs you'll touch.

```bash
grep -rnE 'NODE_IDS = .*<your-id>' scripts/apply_*.py
grep -nE '<your-id>' scripts/apply_*.py
```

Flag conflicts in your new docstring's "Compatibility" section. Reference cases:
`apply_iclora_initial_render.py` and `apply_loop_guide_preprocess_symmetry.py`
both touch `#446 LTXVPreprocess` — their docstrings explain why they coexist.
