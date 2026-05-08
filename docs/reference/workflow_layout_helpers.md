# workflow_layout_helpers

Last updated: 2026-05-08

## Role

Concept note. The repo's column-grid layout primitive for ComfyUI workflow JSON, exposed by `scripts/_layout_grid.py`. Consumed by apply scripts that need to deterministically (re)position nodes, derive group bounding boxes, and anchor Note nodes to groups. Replaces freelance pixel-coord layout in apply scripts.

## Disambiguation

- **Layout helper ≠ workflow editor.** `_layout_grid.py` only touches `pos`, `size`, `groups[]`, and Note `pos`/`size`. Topology (links, node insertion, widget values) lives in `WorkflowEditor` (`scripts/workflow_utils.py`).
- **Layout spec ≠ workflow JSON.** A `LayoutSpec` is the apply-script's *intent*; the helper materializes it onto the workflow. Same spec applied to two workflows produces two different layouts (different node sets) but with the same column origins and tier structure.
- **Tier sub-groups ≠ litegraph subgraphs.** A "tier sub-group" is just a regular Group at a specific (x, y) origin — sub-tier structure comes from y-stacking groups in the same x-band. Litegraph subgraphs (loop body) are a separate concept.

## Key facts

- **Pattern source.** Extracted from `scripts/apply_intro_workflow.py::_layout_workflow` (the seed). For why a partial-layout pass fails (v0 only positioned new nodes; existing nodes kept their chaotic positions): `internal/design/intro_workflow_design.md` "v1 layout fix" (private clone only).
- **Group bounding format.** `[x, y, w, h]` — verified against `example_workflows/audio-loop-music-video_latent.json` group entries.
- **Render order = array order.** No `z` field. Earlier entries in `wf["groups"]` render below later ones; put background/parent groups first if you nest visually.
- **Pills.** Collapsed Get/Set reroutes (`flags.collapsed=true` + `type in {"GetNode", "SetNode"}`) sort to the top of their group with `COLLAPSED_GAP=12` packing, forming a thin plumbing strip at the column head. Full nodes follow with `INTRA_NODE_GAP=50`.
- **Group binding precedence.** Property tag (`node["properties"][group_tag_key]`) wins over the explicit id-mapping. Lets apply scripts tag runtime-added nodes without threading IDs through every function.
- **Note anchoring.** Notes are positioned by `(group, dx, dy, w, h)` offsets from the anchor group's `[x, y]` bounding origin. Reflows track. Notes that don't carry a `note_key_tag` in `properties` are ignored (the helper looks for tagged notes only).
- **Stored size = rendered size.** ComfyUI saves the post-render pixel size into `node["size"]`; the layout helper trusts it. No widget-height computation needed.

## Surface

```
scripts/_layout_grid.py
├── @dataclass GroupSpec(origin, color, title, font_size=24)
├── @dataclass NoteAnchor(group, dx, dy, w, h)
├── @dataclass LayoutSpec(groups, node_groups={}, note_anchors={}, group_tag_key, note_key_tag)
├── apply_layout(wf, spec) -> None             # mutates wf in place
├── extract_template(wf) -> LayoutSpec         # for --from-template golden round-trip
├── assigned_nodes(spec, wf) -> {gkey: [ids]}  # diagnostics
├── unassigned_node_ids(spec, wf) -> [ids]     # spec-coverage gaps
├── summarize(spec, wf) -> str                 # --dry-run output
└── is_pill(node) -> bool                      # collapsed Get/Set predicate

scripts/_layout_classifications.py
├── SHARED_NODE_FUNCTIONS: dict[node_id, functional_column]
└── compose(function_to_group, *, overrides={}) -> dict[node_id, group_key]
```

## Shared classifications

`SHARED_NODE_FUNCTIONS` is the single source of truth for `node_id → functional_column` bindings ("inputs", "models", "sampler", etc.) across the audio-loop family workflows. Apply scripts call `compose(function_to_group, overrides={...})` to map the shared bindings through their own group-key vocabulary.

Per-script override patterns:
- **Identity** (intro): `compose({"inputs": G_INPUTS, "models": G_MODELS, ...})` — every functional column maps to its corresponding group.
- **Tier shift** (polish): `compose({"inputs": G_COMMON, ...}, overrides={565: G_REQUIRED, 444: G_REQUIRED, ...})` — the inputs column defaults to a COMMON tier, with specific nodes pinned to REQUIRED.
- **Per-script additions**: nodes not in `SHARED_NODE_FUNCTIONS` (e.g. post-intro additions) are added via plain dict union: `compose(...) | {2013: G_COMMON, ...}`.

## Tier sub-groups (the column-grid pattern)

Sub-tiers stack vertically within a single column's x-band. Each sub-tier is a regular `GroupSpec` entry whose origin shares an x with the others. Apply-script author sets per-tier y origins (estimate from node count + size, then refine via visual feedback — see *Round-trip iteration*).

Conventional palette for the inputs column:

| Tier | Meaning | Color |
|---|---|---|
| REQUIRED | Change every render (audio file, init image, seed, prompt schedule) | `#29699c` (bright blue) |
| COMMON | Tune occasionally (audio trims, overlap target, image strength) | `#3f789e` (medium blue) |
| FROZEN-with-widgets | Set once (loaders, sampler triplet) — use color signal on the *containing* functional column rather than a separate tier band | `#322` / `#533` (dark red — "do not touch") |

Per `apply_layout_polish_audio_loop_latent.py`'s decision (deviating from the original 4-tier proposal), ADVANCED + FROZEN don't get their own sub-tier in the inputs column — the column would overflow past 5000px. Instead, those nodes stay in their existing functional columns (Models, Sampler) which carry the FROZEN color signal.

## Round-trip iteration (`--from-template`)

For pixel-perfect tuning without editing Python source: hand-lay-out a workflow in ComfyUI (drag nodes, save), then point an apply script at the saved file via `--from-template <golden.json>`. `extract_template(wf)` reads existing `wf["groups"]` for origins/colors and bins each node into whichever group's bounding box contains its `pos`. The resulting spec, applied back to the target workflow, reproduces the layout.

This closes the "user shows me the pattern once, system replicates" loop. ComfyUI is the visual layout designer; the apply script is the round-trip channel. No separate UI needed.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Notes float in space miles from their group | Anchor `dy` set when target group's bounds were larger; group shrunk after | Re-tune anchor `dy` against current group height; or drop overlapping anchors |
| Tier sub-tier overlaps the next | Hand-tuned tier y origin too tight | Increase y-origin gap between tiers; layout helper doesn't auto-stack tiers (yet) |
| New apply script lands a node at `(0, 0)` | Spec's `node_groups` doesn't cover that node id and no property tag was set | Add to `NODE_GROUPS` table or tag the node via `node["properties"][group_tag_key]` |
| `summarize()` shows `unassigned: [...]` after workflow grew | Spec was authored against an older workflow | Update the apply script's `NODE_GROUPS` table; classify the new ids |
| `--from-template` extraction puts nodes in the wrong group | Bounding boxes overlap (e.g. row 0 and row 1 both include `y=200`) | Make sure golden workflow's group bounds don't overlap before extracting |
| Layout looks fine in Python output but wrong in ComfyUI | Multiline text widget expanded post-save and pushed neighbors out | Re-save the workflow in ComfyUI to refresh `node["size"]`, then re-run apply |

## Migration

There is no migration for the helper itself — it's a new module. Apply scripts that want to use it import from it directly. The seed reference `apply_intro_workflow.py::_layout_workflow` has not been refactored to use the helper yet (deferred follow-up — refactor must produce byte-identical output to avoid regression on the canonical latent workflow).

## Audit + tests

- **`layout_no_orphans`** (generic invariant): any non-Note node at `pos=[0, 0]` is flagged ERR. Catches the silent-failure mode where an apply script inserts a node and never runs a layout pass. Tests: `tests/test_audit_layout_no_orphans.py`.
- **F-pair group_layout_invariants** (deferred): a per-workflow check that the expected groups exist with expected titles + tier coverage. Lands when `apply_layout_polish_audio_loop_latent.py`'s output promotes from `internal/workflows/` to `example_workflows/`. Per `scripts/CLAUDE.md` "Carve-out for staged-variant scripts", staged variants skip F-pair until promotion.

## References

- `scripts/_layout_grid.py` — implementation
- `scripts/apply_intro_workflow.py` — seed reference (`_layout_workflow`); pre-extraction pattern, still in use
- `scripts/apply_layout_polish_audio_loop_latent.py` — first consumer of the extracted helper; tier sub-groups + `--from-template` mode
- `internal/design/intro_workflow_design.md` — v0→v1 layout-fix history (private clone only)
- `scripts/CLAUDE.md` "Apply-script conventions" — layout pointer entry
- `docs/reference/_atomic_note_template.md` — template this note follows
