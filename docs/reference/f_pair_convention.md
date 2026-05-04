# F-pair convention

Last updated: 2026-05-04

## Role

Every fix that changes workflow JSON ships as a pair: an `apply_*.py` migration script AND a matching audit check in `scripts/audit_workflows.py` that fires ERR (with a `Run scripts/apply_<X>.py` remediation pointer) when the invariant is violated. The pair prevents silent regression of fixes a sibling branch might revert. F-numbers (F2 through F13 currently) name *pairs*, not individual checks.

## Disambiguation

- An **F-number names a pair** (one apply script + one or more audit checks), not a single audit check. F12 has three sub-audits (F12a, F12b, F12c) all pairing with `apply_iclora_video_reference.py`.
- **Not all audit checks have F-numbers.** `audit_workflows.py` runs other named pattern checks (`sage`, `sage_mode`, `iteration_stamp`, `prompt_schedule`, etc.) that predate the F-pair convention. They still pair with apply scripts but aren't numbered.
- **Apply script ≠ scaffold script.** `scripts/templates/apply_script_*.py` are scaffolds. `scripts/apply_*.py` are the actual migrations. Don't run a template.
- **Live inventory ≠ this doc.** This doc explains the convention; the current F-pair list lives in `docs/reference/debug_tools.md` "Workflow validation" section AND the live `record(...)` calls in `audit_workflows.py`. Treat those two as canonical when looking up "what F-pairs exist now."

## Key facts

- Apply scripts ALWAYS support `--revert`, `--dry-run`, idempotence, and `require_nodes` pre-flight.
- Audit-check ID is the exact string passed to `record(...)` in `audit_workflows.py`. Cited in docs as `` `<id>` (F<N>) `` or `Audit: F<N> (`<id>`)`.
- F-numbering is sequential by ship date; no semantic meaning to the number itself.
- Pre-flight chaining: when one F-pair depends on another, the dependent apply script's pre-flight detects the prerequisite's signature and refuses with an actionable message. Reference: `apply_iclora_video_reference.py` refuses if Step 0 (`apply_strip_dead_lora_loaders.py`, F11) hasn't run.
- `audit_workflows.py` is intentionally `WorkflowEditor`-independent — raw `orjson.loads` + inline link scans. Debug tools must stay usable when the editor they audit has a bug; don't DRY against `WorkflowEditor`.

## How to add a new F-pair

1. Scaffold apply script from `scripts/templates/apply_script_all_workflows.py` (in-place edits) or `apply_script_staged_variant.py` (experimental staging). Both ship with `--revert`, `--dry-run`, `require_nodes` already wired.
2. Add matching `record(...)` call(s) to `scripts/audit_workflows.py`. Use ERR status for hard violations; WARN for soft (e.g. F10 `vae_decode_no_tile` is WARN since `[2,2,1]` is the safe fallback for ≤16GB).
3. Cite both from `scripts/CLAUDE.md` (apply-script conventions section) AND root CLAUDE.md if turn-1-load-bearing.
4. Add to `docs/reference/debug_tools.md` "Workflow validation" inventory table.
5. Pre-flight chaining: if your migration depends on a prior F-pair's state, detect the prerequisite's signature in your script's pre-flight and refuse with `Run scripts/apply_<prerequisite>.py first`.

## Pre-flight chaining example

```
apply_iclora_video_reference.py (F12 set)
    ├─ requires: apply_strip_dead_lora_loaders.py (F11) has run
    └─ pre-flight: detect bypassed #1625/#1626/#1627; refuse if present

apply_planner_break_stride_cycle.py (F7)
    └─ no prerequisites; standalone
```

## Failure modes

| Symptom | Likely cause |
|---|---|
| Sibling branch reverts a fix; nothing fires | Apply script shipped without paired audit; silent regression |
| Audit ERR fires but users unsure how to fix | Audit shipped without paired apply script; no remediation pointer |
| Dependent migration applies over un-migrated state, corrupted JSON | Pre-flight chaining missed; dependent didn't detect prerequisite signature |
| Two checks claim the same F-number | F-number reused; pick the next free number, update doc inventory |
| Cited audit ID in CLAUDE.md / wiki note doesn't resolve | Renamed in `audit_workflows.py` without propagating to docs (caught by `tests/test_claude_md_budget.py::test_cited_audit_ids_exist`) |

Edge cases:
- A single apply script can pair with multiple audit checks (F12 → F12a/F12b/F12c). The audit IDs are independent strings; the F-number groups them.
- An audit check can fire WARN-level (F10) when the rule has acceptable exceptions; these don't fail CI but surface in `audit_workflows.py --verbose`.
- Generic invariants (`graph_acyclic`, `widget_shape`, `link_integrity`) run unconditionally and aren't numbered — they catch CLASSES of drift without per-bug rules.

## Audit + tests

- Cited audit IDs verified by `tests/test_claude_md_budget.py::test_cited_audit_ids_exist` against truth set extracted from `record(...)` calls in `audit_workflows.py`.
- "Every `apply_*.py` has a matching audit check" — NOT YET enforced as a test. Could be added: parse apply script names, verify each has at least one audit reference. Deferred until the first time it bites.

## References

- `docs/reference/debug_tools.md` — live F-pair inventory + apply-script three-tier staging + WorkflowEditor independence rationale
- `scripts/CLAUDE.md` — apply-script conventions + WorkflowEditor patterns
- `scripts/audit_workflows.py` — truth set for audit-check IDs
- `scripts/templates/apply_script_all_workflows.py`, `apply_script_staged_variant.py` — scaffolds
- `tests/test_claude_md_budget.py::test_cited_audit_ids_exist` — drift detection between docs and live audits
- `docs/reference/_atomic_note_template.md` — concept-note variant template
