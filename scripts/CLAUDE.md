# scripts/ — apply scripts, audit, utilities

Last updated: 2026-05-04

This subtree holds workflow-mutation scripts (`apply_*.py`), workflow validators (`audit_workflows.py`, `validate_docs_consistency.py`, `analyze_workflow_dag.py`, `trace_node_source.py`), the canonical edit utility (`workflow_utils.py`), audio analysis tools, and templates. Loads only when Claude is operating inside `scripts/`. Root project rules: `../CLAUDE.md`. Governance policy: `../.claude/CLAUDE.md`.

## WorkflowEditor — the only sanctioned JSON edit path

**Use `WorkflowEditor` (`scripts/workflow_utils.py`) for every workflow JSON edit.** Hand-rolled link-list traversal or raw `orjson` mutation skips invariants the editor enforces (link-array ↔ node-body sync, dtype propagation, subgraph slot-index drift on insert/remove).

Top-level helpers: `find_node`, `has_node`, `require_nodes`, `find_link_to_slot(tgt, slot)`, `add_link`, `remove_link`, `rewire_input(tgt, slot, new_src, new_src_slot, dtype)`, `find_links_to/from`. Subgraph helpers (mirror naming): `find_subgraph_invoker`, `find_subgraph_node`, `find_subgraph_link`, `find_subgraph_link_to_slot(tgt, slot)`, `add_subgraph_link`, `remove_subgraph_link`, `rewire_subgraph_input`. `find_input_slot` works on both. **Don't hand-roll link lookups or rewires** — `find_link_to_slot` replaces the `next(lk for lk in ed.wf["links"] if lk[0] == link_id)` pattern; `rewire_input` / `rewire_subgraph_input` replace the `remove_link` + `add_link` splice.

Slot-dict shape helpers (static methods): `WorkflowEditor.io_in(name, dtype, link=None)`, `widget_in(name, dtype, link=None)`, `out(name, dtype)`. Use these in `from_scratch` builders (`scripts/build_*_workflow.py`) instead of open-coding the `{"name": ..., "type": ..., "link": ...}` dict literals. The helpers preserve the slot-dict contract that `add_top_level_node` consumes.

`scripts/_apply_helpers.py` is for **RAW-orjson fork-and-strip scripts only** (debug-tool stability when `WorkflowEditor` itself is suspect) — NOT a general utility module. Apply scripts that use `WorkflowEditor` (the canonical path) don't import from it. Confirm by reading its docstring before extracting helpers there.

## Subgraph editing mechanics

- Top-level links are arrays `[id, src, src_slot, tgt, tgt_slot, type]`; subgraph internal links are dicts `{id, origin_id, origin_slot, target_id, target_slot, type}`. Subgraph def at `wf['definitions']['subgraphs'][0]`.
- Distributor `-10` / output collector `-20` are virtual — not in `sg["nodes"]`. Their slot indices map 1-to-1 with `sg["inputs"]` / `sg["outputs"]` order.
- Output slots use `"links"` (plural list); subgraph boundary entries use `"linkIds"`. Don't conflate.
- DynamicCombo widgets: `[num, strength_1..N, index_1..N]` — strengths FIRST, not interleaved.
- **`remove_link` rebinds the target list** via filter — locals holding `ed.wf["links"]` go stale. Use editor methods or re-fetch.
- **Subgraph schema changes force a UI re-add** (slot indices baked at save time). Removing a subgraph input shifts higher slot indices — decrement `origin_slot` refs.

## Apply-script conventions

Every `apply_*.py` script ships with:

- **`--revert`** — undo the migration. Idempotent; safe to run on already-reverted state.
- **`--dry-run`** — show what would change without writing. HyDE pattern: `apply_X.py --dry-run | audit_workflows.py` verifies a hypothetical state before committing.
- **Idempotence** — applying twice is a no-op. Achieved via signature checks (e.g. detecting whether the target node already exists with expected wires).
- **`require_nodes` pre-flight guards** — refuse with an actionable message when expected upstream nodes are missing.
- **Pre-flight chaining** — when one migration depends on another, detect the prerequisite's signature and refuse with "Run scripts/apply_<X>.py first." Reference: `apply_iclora_video_reference.py` refuses if `#1625/#1626/#1627` are still present (Step 0 strip unrun).

**Scaffold new scripts from `scripts/templates/`**:
- `apply_script_all_workflows.py` — in-place edits across `example_workflows/`.
- `apply_script_staged_variant.py` — experimental staging into `internal/scratch/` or sibling files.

Both templates include the canonical `--revert`, `--dry-run`, idempotence, and `require_nodes` patterns.

## Audit + apply F-pair convention

Every fix that ships an apply script ships a matching audit check in `audit_workflows.py`. The check returns ERR with a `Run scripts/apply_<X>.py` remediation pointer when the invariant is violated. This prevents silent regression of fixes a sibling branch might revert.

Inventory (canonical list + remediation pointers): **`docs/reference/debug_tools.md`**. Pairs are referenced by F-number throughout root CLAUDE.md and elsewhere. The audit IS the rule — when in doubt, look at the live `record(...)` call sites in `audit_workflows.py` rather than re-deriving from prose.

In addition to F-pairs, three **generic invariants** run unconditionally: `graph_acyclic`, `widget_shape`, `link_integrity`. Plus one AST-shaped test: `tests/test_node_schemas.py::test_keyframe_idxs_cleared_to_none_not_empty_list`. Together these catch CLASSES of drift without per-bug rules.

## Audit usage

```bash
# Default sweep (example_workflows/ + audited subset of experimental/)
uv run --group dev python scripts/audit_workflows.py

# Audit a specific file (e.g. an apply-script-produced staged variant)
uv run --group dev python scripts/audit_workflows.py internal/scratch/foo.json

# Audit a single experimental fork (warn-level only on F8)
uv run --group dev python scripts/audit_workflows.py example_workflows/experimental/<name>.json
```

Exits 0 on all-green, 1 on any ERR. WARN-level findings don't fail the run (e.g. `vae_decode_no_tile` is WARN-level since `[2,2,1]` is the safe fallback for ≤16GB cards).

## Workflow JSON discipline (recap from root CLAUDE.md)

Three rules whose details belong here even though the rule itself is in root:

- **Workflow JSON references inputs by NAME, not slot index.** Each node's `inputs[]` entry stores `{"name": ..., "type": ..., "widget": {"name": ...}, "link": ...}`; ComfyUI matches the saved name to the schema's input list when reattaching wires. So a bare schema rename (e.g. `"seed"` → `"base_seed"`) without a paired migration script that rewrites `inputs[].name` and `widget.name` in every saved JSON will dangle every existing wire on the renamed input. Canonical migration: `scripts/apply_alc_seed_rename.py`.
- **A schema rename is not enough — strip leftover widget values too.** When `apply_alc_seed_rename.py` renamed `seed`→`base_seed`, it updated `inputs[].name` but did NOT prune the leftover `'randomize'` string at `widgets_values[4]`. ComfyUI's backend pops widgets positionally; 6 saved values into 5 schema slots shifts `'randomize'` into the `fps` slot, INT-parse fails. Companion: `scripts/apply_strip_alc_control_after_generate.py`. Audit: `alc_widget_drift`.
- **Don't ship two schema changes that touch the same iteration-state plane in one session.** When adding an auto-wire that closes a control loop, walk every existing edge between the involved nodes and confirm none of them produces a cycle. ComfyUI's prompt validator rejects with "Dependency cycle detected" before any node runs. Reference: `apply_planner_break_stride_cycle.py`. Audit: `planner_no_stride_input`.

## Bypass + dead-node detection

- `"mode": 0` = active, `"mode": 4` = bypassed. **Bypass passes inputs to outputs of same TYPE only**; inputs with no matching-type output dead-end silently.
- **`workflow_utils.is_active(node)`** is the canonical bypass check (`mode != 4`). Use it instead of inline `node.get("mode", 0) != 4` — 5 call sites across `audit_workflows.py`, `apply_no_tile_vae_decode.py`, `apply_melband_default_off.py`. The bare integer obscures that `4` means bypass.
- **Dead-node detection requires a live-consumer check, not a link-count check.** A node with output links can still be runtime-dead if every consumer is `mode=4`. Pattern: walk consumer ids, return True only if at least one consumer satisfies `is_active`. See `apply_no_tile_vae_decode.py::_has_live_consumer`.

## Audio analysis scripts

- `scripts/analyze_audio.py` — ffmpeg-only energy/structure detection, zero Python deps.
- `scripts/analyze_audio_features.py` — librosa: BPM, key, F0, structure, JSON export for LLM (`--scene-diversity`, `--montage`, `--style`). Full guide: `docs/guides/audio_analysis_guide.md`. **Works on generated audio**: extract via `ffmpeg -i <mp4> -vn -acodec pcm_s16le <wav>` then analyze.
- `scripts/spectrogram_to_reference.py` — Mel spectrogram → PNG frame sequence for IC-LoRA spectrogram-as-reference (Phase 2.0). **Global normalization runs ONCE in `prepare_mel_for_render`** (do NOT switch to per-frame — washes out beat-amplitude). **Dual-use**: primary use is reference rendering; diagnostic use is visualizing generated audio via `--audio <wav>`. Supports `--colormap {gray,viridis,spectrum}`; B&W triggers vintage-broadcast audio priors in LTX 2.3 — use color for V2A experiments.

## Inspection utilities

- `scripts/audit_workflows.py` — see above.
- `scripts/analyze_workflow_dag.py` — DAG structural view of a workflow.
- `scripts/trace_node_source.py <wf> <node_id>` — show node definition + widget shape from upstream (e.g. KJNodes registry). Useful when a saved workflow's widget order disagrees with what apparent.
- `scripts/validate_docs_consistency.py` — STALE_PATTERNS scan against `docs/`. CI runs this; failure = update STALE_PATTERNS or fix the doc.

Full inventory + the canonical first-pass-when-a-workflow-won't-run flow: `docs/reference/debug_tools.md`. Or invoke `/diagnose-workflow`.

## Retired apply scripts

`scripts/archive/` holds apply scripts whose migration is baked into the
canonical workflow permanently and whose source/output files are no
longer in tree (`apply_audio_latent_pre_encode.py`,
`apply_iclora_video_reference.py` as of 2026-05-04). Kept as design
records of the topology each migration introduced; not for re-running.
Audit remediation pointers reference `scripts/archive/...` paths so
they're recoverable if a reader needs to inspect the original migration.
See `scripts/archive/README.md`.

## When `WorkflowEditor` itself is suspect

Rare but possible: a bug in the editor that produces malformed output the audit doesn't catch. In that case fall back to `_apply_helpers.py`'s RAW-orjson fork-and-strip primitives. Don't paper over by writing a hand-rolled traversal in a feature apply-script — fix the editor.

## References

- `../CLAUDE.md` — root project rules
- `../docs/reference/debug_tools.md` — full debug + apply-script inventory
- `../.claude/CLAUDE.md` — CLAUDE.md governance policy
- `templates/README.md` — apply-script template overview
- `../docs/guides/audio_analysis_guide.md` — audio-analysis script usage
