---
name: compare-workflows
description: Structural diff of two ComfyUI workflow JSON files — node types, widget values, link topology, subgraph schemas — with noise filtered out (link IDs, node positions, ordering). Use when debugging "why does workflow A behave differently than B" or verifying a migration script produced the intended shape change.
---

# Compare Workflows

Structural diff between two ComfyUI workflow JSONs. Answers questions like:
- "What actually changed between my scratch run that worked and the one that didn't?"
- "Did `apply_vae_and_cleanup.py` produce the intended changes to this workflow?"
- "Which widgets differ between `_latent.json` and `_latent_stg.json`?"

Noise is filtered: link IDs (workflow-local, noisy), node positions (visual-only), node ordering. What remains is behavioral: matched-by-ID node type/mode/widget diffs, added/removed nodes, link-topology changes, and subgraph input/output schema changes.

## How to run

```bash
uv run --group dev python \
  .claude/skills/compare-workflows/diff_workflows.py \
  <workflow_a.json> <workflow_b.json>
```

Works on any ComfyUI workflow JSON — shipped `example_workflows/*.json`, user scratch saves under `internal/scratch/`, or archived copies in `internal/archive/`.

## What the output looks like

Each scope (top level + every subgraph) is headed with `[SCOPE]`. Inside a scope:

- `-#NN Type` / `+#NN Type` — node removed / added
- `#NN Type changed:` followed by indented list of what changed (`type`, `mode`, `widget[i]: old → new`)
- `-SRC:slot → TGT:slot (dtype)` / `+...` — link removed / added (link IDs ignored; topology only)
- `subgraph inputs[N]: ...` — subgraph slot-N schema change (name or type)

If there are zero differences, the report ends with `(no structural differences)`.

## Common uses in this repo

### Debug unexpected behavior differences between two runs
```bash
uv run --group dev python \
  .claude/skills/compare-workflows/diff_workflows.py \
  internal/scratch/LTX-2_00211.json internal/scratch/LTX-2_00214.json
```

### Verify an `apply_*.py` migration did what it intended
```bash
# Before: backup of pre-migration shape
cp example_workflows/audio-loop-music-video_latent.json /tmp/pre.json

# Apply the migration
uv run --group dev python scripts/archive/apply_vae_and_cleanup.py

# Compare — should show only the intended changes
uv run --group dev python \
  .claude/skills/compare-workflows/diff_workflows.py \
  /tmp/pre.json example_workflows/audio-loop-music-video_latent.json
```

### A/B two shipped variants
```bash
uv run --group dev python \
  .claude/skills/compare-workflows/diff_workflows.py \
  example_workflows/audio-loop-music-video_latent.json \
  example_workflows/audio-loop-music-video_latent_stg.json
```

## Scope

- Compares the top-level graph + every subgraph definition.
- Ignores link numeric IDs. Link identity is the fingerprint `src_type#id:slot → tgt_type#id:slot (dtype)`.
- Virtual subgraph-boundary nodes render as `<sg_input>` / `<sg_output>` in link fingerprints so slot-level schema changes show up clearly.
- `widgets_values` diffs preserve element order (behaviorally meaningful in ComfyUI — widget positions map to schema slots).

## Limits

- Compares subgraphs by positional index, not by name/UUID. Works for the canonical "one subgraph at index 0" shape this repo uses; would need extending for multi-subgraph workflows.
- Does not compare `properties` (node metadata like `cnr_id`, `Node name for S&R`) — these are noise for behavioral comparison.
- Does not compare `size` or `pos` — purely visual.
