Last updated: 2026-05-16

# working_docs/archive — superseded planning docs

Docs in this directory were planning + research scratch space for the
`fml2v_var_d_audio_loop` variant build. They were superseded once the V1
design crystallized in `../fml2v_audio_loop_v1_design.md` (with key
content merged into that doc's "Why Option B" + "Sampler chain" sections).

Kept here as design records — useful for understanding why V1 took the
shape it did, not for active reference. The live design doc is the
source of truth.

## What's here + why archived

| File | Original purpose | Why archived |
|---|---|---|
| `workflow_quality_delta_analysis.md` | Axis-by-axis A-vs-B structural diff + hypothesis ranking | Hypothesis ranking + "What's NOT a quality lever" list merged into V1 design doc's "Why Option B" section |
| `adapt_benchmark_to_full_audio_plan.md` | Older migration plan (graft B's quality stack into A's subgraph) | Different design choices than V1 (V1 = flat canvas, start from B, all patches inside loop). Steps 1/4/7 + risk #8 merged into V1 design doc |
| `var_d_node_inventory.md` | Per-node port classification for benchmark variant D | Fully superseded by `scripts/_node_templates_fml2v.json` (operational version of the same inventory) |
| `from_b_node_inventory.md` | Per-node port classification for variant B | Same as above; V1 chose var_d not var_b |
| `loop_machinery_package.md` | Top-level loop-spine grafting inventory | Superseded by V1 design doc's HANDOFF section + `docs/reference/audio_loop_controller.md` |
| `subgraph_chain_reference.md` | Canonical subgraph IO reference | V1 is flat-canvas (no subgraph); canonical subgraph reference lives in `docs/reference/audio_loop_controller.md` |
| `audio_loop_mechanism_reference.md` | Loop mechanism explanation | Superseded by `docs/reference/audio_loop_controller.md` + `docs/reference/pipeline_flow_latent.md` |

## What's still in `working_docs/` (not archived)

- `fml2v_audio_loop_v1_design.md` — the live V1 design + HANDOFF section
- `ltx23_max_length_research.md` — durable reference (RoPE max_pos, latent-volume budget, 8k+1 formula, 24GB ceiling); fps-independent body
