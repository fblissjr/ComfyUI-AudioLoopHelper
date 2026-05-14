# scripts/ — apply scripts, audit, utilities

Last updated: 2026-05-10

This subtree holds workflow-mutation scripts (`apply_*.py`), workflow validators (`audit_workflows.py`, `validate_docs_consistency.py`, `analyze_workflow_dag.py`, `trace_node_source.py`), the canonical edit utility (`workflow_utils.py`), audio analysis tools, and templates. Loads only when Claude is operating inside `scripts/`. Root project rules: `../CLAUDE.md`. Governance policy: `../.claude/CLAUDE.md`.

## WorkflowEditor — the only sanctioned JSON edit path

**Use `WorkflowEditor` (`scripts/workflow_utils.py`) for every workflow JSON edit.** Hand-rolled link-list traversal or raw `orjson` mutation skips invariants the editor enforces (link-array ↔ node-body sync, dtype propagation, subgraph slot-index drift on insert/remove).

Top-level helpers: `find_node`, `has_node`, `require_nodes`, `find_link_to_slot(tgt, slot)`, `add_link`, `remove_link`, `rewire_input(tgt, slot, new_src, new_src_slot, dtype)`, `find_links_to/from`. Subgraph helpers (mirror naming): `find_subgraph_invoker`, `find_subgraph_node`, `find_subgraph_link`, `find_subgraph_link_to_slot(tgt, slot)`, `add_subgraph_link`, `remove_subgraph_link`, `rewire_subgraph_input`. `find_input_slot` works on both. **Don't hand-roll link lookups or rewires** — `find_link_to_slot` replaces the `next(lk for lk in ed.wf["links"] if lk[0] == link_id)` pattern; `rewire_input` / `rewire_subgraph_input` replace the `remove_link` + `add_link` splice.

Slot-dict shape helpers (static methods): `WorkflowEditor.io_in(name, dtype, link=None)`, `widget_in(name, dtype, link=None)`, `out(name, dtype)`. Use these in `from_scratch` builders (`scripts/build_*_workflow.py`) instead of open-coding the `{"name": ..., "type": ..., "link": ...}` dict literals. The helpers preserve the slot-dict contract that `add_top_level_node` consumes.

`scripts/_helpers/_apply_helpers.py` is for **RAW-orjson fork-and-strip scripts only** (debug-tool stability when `WorkflowEditor` itself is suspect) — NOT a general utility module. Apply scripts that use `WorkflowEditor` (the canonical path) don't import from it. Confirm by reading its docstring before extracting helpers there.

**Helper modules live under `scripts/_helpers/`** (underscore-prefix = private helper). Import via qualified path: `from _helpers._apply_helpers import ...`, `from _helpers._layout_grid import ...`, `from _helpers._layout_classifications import ...`. PEP 420 namespace package — no `__init__.py` needed. `scripts/workflow_utils.py` stays at `scripts/` because it's the canonical edit API, not a private helper.

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
- **Don't stash revert metadata as keys on workflow JSON nodes/groups** (e.g. `_X_pre`). The keys persist into shipped JSON on apply. For revert: use hardcoded canonical defaults (legacy shapes are stable across shipped variants); for non-trivial dynamic state, a sidecar at `internal/.apply_state/<script>.json`.
- **Widget defaults that DROP user content must be opt-in, not opt-out.** The 2026-05-10 `#567 TrimAudioDuration [start_index=5, duration=300]` default ate the first 5 seconds of every song silently because the user had to notice the widget to disable it. When defining a new node or setting widget defaults via apply scripts, ANY default that removes/clips/transforms user input should default to a no-op (`start_index=0`, `strength=0`, `bypass=True` / `mode=4`). Make destructive behavior require explicit user opt-in. Footgun-by-default eats render time + user trust.

**Scaffold new scripts from `scripts/templates/`**:
- `apply_script_all_workflows.py` — in-place edits across `example_workflows/`.
- `apply_script_staged_variant.py` — experimental staging into `internal/scratch/` or sibling files.

Both templates include the canonical `--revert`, `--dry-run`, idempotence, and `require_nodes` patterns.

**Layout work**: extend `scripts/_helpers/_layout_grid.py` (column-grid + tier sub-groups + note anchors). Don't freelance pixel coords inline in apply scripts — `internal/design/intro_workflow_design.md` "v1 layout fix" (private clone only) records why partial-layout passes drift. Reference users: `apply_intro_workflow.py::_layout_workflow` (seed pattern), `apply_layout_polish_audio_loop_latent.py` (tier sub-groups + `--from-template` template extraction). Full reference: `docs/reference/workflow_layout_helpers.md`.

**Byte-identical refactor validation**: when refactoring an apply script (extracting helpers, moving classifications), capture `md5sum` of the script's output before, re-run after, diff. Catches accidental behavior change before it ships. Used during `_layout_grid` + `_layout_classifications` extractions (2026-05-08) — both byte-identical confirmed via this loop.

**Self-targeting apply scripts** (input path == output path; today: `apply_intro_workflow.py`) overwrite user manual edits on re-application. Before re-running, `git diff` the target — off-grid positions or unclassified node ids may be intentional manual edits worth preserving.

## Before archiving an apply script

Ref-counting (grep across `docs/`, `tests/`, and CLAUDE.md files) is necessary but **not sufficient** — it misses three failure modes:

1. **Unapplied emergency fallback** — script never ran on canonical (e.g. `apply_audio_vae_fix.py`), so canonical state diverges from the script's "after" state. Verify by inspecting node types in `example_workflows/*.json`.
2. **Active CLI tool driven from `internal/`** — script has zero refs in public surface but is the documented invocation path in a private action-item ladder. Grep `internal/` before deciding.
3. **Superseded but re-runnable** — script's output shape conflicts with the current canonical (e.g. produces a removed node type); re-running breaks the workflow. Check what node types the script adds vs what the canonical now has.

Verdict rule: archive only if (a) the migration is baked AND idempotent on current canonical, OR (b) the migration is superseded AND re-running would break shape, OR (c) the script generates a now-shipped variant whose JSON is the source of truth.

## Audit + apply F-pair convention

Every fix that ships an apply script ships a matching audit check in `audit_workflows.py`. The check returns ERR with a `Run scripts/apply_<X>.py` remediation pointer when the invariant is violated. This prevents silent regression of fixes a sibling branch might revert.

**Carve-out for staged-variant scripts:** apply scripts that stage drafts into `internal/workflows/` (don't mutate `example_workflows/`) skip the F-pair audit-invariant requirement. F-pair applies at promotion time — when a draft graduates to `example_workflows/` and a regression-protection invariant earns its keep on the shipped surface. Reference: `apply_lanczos_init_preprocess.py`, `apply_p3_retake_edit_lora.py` ship without paired audits because their outputs are gitignored drafts.

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
- `scripts/analyze_sage_traces.py <sage.jsonl>...` — aggregate per-shape kernel timing (p50/p95 masked + unmasked) across one or more sage trace files. Used for cross-run reproducibility checks + masked-vs-unmasked deltas.
- `scripts/bench_aimdo_vram.py --output <ndjson>` — poll ComfyUI's `/aimdo/vram` endpoint (from `ComfyUI-MemoryVisualization` custom node) at 1Hz; writes per-model VRAM-residency NDJSON. Companion to `analyze_sage_traces.py` for the dynamic-VRAM-offload-pressure question.
- `scripts/startup/start.sh` — canonical deploy template for ComfyUI's launcher (six modes: `default | safe | extreme | minimal | nodynvram | highvram`). The `nodynvram` mode is the load-bearing config for kernel-OOM testing — disables dynamic VRAM, async offload, and node cache. Full methodology at `docs/reference/benchmarking_memory_pressure.md`.

Full inventory + the canonical first-pass-when-a-workflow-won't-run flow: `docs/reference/debug_tools.md`. Or invoke `/diagnose-workflow`.

## Retired apply scripts

`scripts/archive/` holds apply scripts whose migration is baked into the
canonical workflow permanently, was superseded by a different pattern,
or generated a now-shipped variant whose JSON is the source of truth.
Kept as design records; **not for re-running** (would conflict with
current canonical for several entries). Audit remediation pointers
reference `scripts/archive/...` paths so they're recoverable if a
reader needs to inspect the original migration. **Per-script inventory
with original purpose + reason archived: `scripts/archive/CLAUDE.md`.**

## Inventory (current scripts/, post-2026-05-05 cleanup)

74 scripts (+ 3 helpers under `scripts/_helpers/`) grouped by purpose. Each row: **script** — purpose · *primary callers*. Scripts with **no callers** are leaf utilities (CLI-invoked directly or emergency fallbacks). Archive entries: `scripts/archive/CLAUDE.md`.

### Core editing + audit (always-live foundations)

*Add here if you're building infrastructure for workflow validation or JSON editing.*

| Script | Purpose · callers |
|---|---|
| `workflow_utils.py` | Canonical `WorkflowEditor` API for JSON edits · imported by ~all `apply_*.py` |
| `_helpers/_apply_helpers.py` | Raw-orjson primitives for fork-and-strip scripts when `WorkflowEditor` is suspect · `apply_audio_loop_retake.py`, `apply_spectrogram_iclora_minimal.py`, `apply_keyframe_batch_encode.py` |
| `_helpers/_layout_grid.py` | Column-grid + tier sub-group + note-anchor primitives for workflow layout · `apply_intro_workflow.py` (seed reference), `apply_layout_polish_audio_loop_latent.py` |
| `_helpers/_layout_classifications.py` | Shared `node_id → functional column` table for the audio-loop family · `apply_intro_workflow.py`, `apply_layout_polish_audio_loop_latent.py` |
| `audit_workflows.py` | Health audit (F-pair invariants + generic checks) · CI, README, `/diagnose-workflow` |
| `validate_docs_consistency.py` | STALE_PATTERNS scan · CI, root CLAUDE.md |
| `test_workflow_integrity.py` | Structural integrity + widget consistency check · `.claude/settings.json` smoke test |

### Workflow inspection / diagnostics

*Add here if you're diagnosing workflow structure, resolution, or runtime behavior.*

| Script | Purpose · callers |
|---|---|
| `analyze_workflow_dag.py` | Static DAG + execution-order view · README, `apply_keyframe_batch_encode.py` |
| `trace_node_source.py` | Show node Python source from a workflow + node_id · `analyze_workflow_dag.py`, `debug_tools.md` |
| `validate_workflow_decoder.py` | Decoder-config check across example workflows · `debugging_guide.md`, `ltx-constraints-auditor` agent, `tests/test_decoder_validator.py` |
| `validate_workflow_resolution.py` | LTX-2.3 resolution-compliance check · `debugging_guide.md`, `ltx-constraints-auditor` agent |
| `extract_workflow_from_png.py` | Dump embedded workflow JSON from PNG · `debugging_guide.md` |
| `diagnose_overlap_seams.py` | Detect seam-zone artifacts in assembled loop output · `build_seam_refinement_workflow.py` |
| `calc_ltx_resolution.py` | Offline companion to `LTXResolutionFromAspect` — resolve aspect+long-edge to LTX-valid dims · CLI-only (no callers) |
| `promote_latent_for_upscale.py` | Find the most recent `segment_*.latent` saved by a loop's bypassed-SaveLatent toggle and copy it to ComfyUI's input dir under a deterministic name. Reads `COMFYUI_OUTPUT_DIR` / `COMFYUI_INPUT_DIR` env or `--output-dir` / `--input-dir` flags · `docs/guides/upscale_guide.md` |

### From-scratch workflow builders (output: new variant JSON)

*Add here if you're generating new variant JSON (not mutating existing).*

| Script | Purpose · callers |
|---|---|
| `build_keyframe_workflow.py` | Build keyframe-conditioned workflow from latent base · `debug_tools.md` |
| `build_seam_refinement_workflow.py` | Build post-loop seam-zone refinement workflow. Ingress migrated 2026-05-10 from `VHS_LoadVideo + VAEEncode` (OOM at 24 GB) to `LoadLatent + LoadAudio`; sizes empty audio latent via `LatentFrameCount`. Chain `apply_trim_video_latent_to_audio.py` + `apply_run_id_layout.py` after rebuild · `docs/guides/upscale_guide.md` |
| `build_upscale_workflow.py` | Build post-loop spatial-upscale workflow. Same ingress migration 2026-05-10 — reads the loop's assembled `.latent` directly (~855 MB) instead of decoding the mp4 (~16 GB pixel batch). 27 nodes / 32 links. Pre-step: `apply_run_id_layout.py` on the loop workflow + toggle SaveLatent in UI · `docs/README.md`, `docs/guides/upscale_guide.md`, `debug_tools.md` |

### Apply scripts — sigma chain + sampler (4)

*Add here if you're tuning sampler scheduling, scheduler swap, or VAE decode config.*

| Script | Purpose · callers |
|---|---|
| `apply_canonical_sigmas.py` | Replace `BasicScheduler` w/ `ManualSigmas` (canonical 8-step distilled values) · root CLAUDE.md, `audit_workflows.py`, `sampler_reference.md` |
| `apply_strip_sd3_shift_node.py` | Strip dead `ModelSamplingSD3` (orphaned post-sigma-migration) · root CLAUDE.md, `debug_tools.md`, `audit_workflows.py` |
| `apply_no_tile_vae_decode.py` | Set `LTXVTiledVAEDecode` to `[1,1,1]` (24GB+ optimization) · root CLAUDE.md, `audit_workflows.py` |
| `apply_ltx_decoder.py` | Swap generic `VAEDecodeTiled` → `LTXVTiledVAEDecode` · `validate_workflow_decoder.py`, `debugging_guide.md` |

### Apply scripts — audio + planner topology (9)

*Add here if you're wiring planner/controller/audio-slicer autowires or fixing topology bugs.*

| Script | Purpose · callers |
|---|---|
| `apply_audio_latent_slice_iter_wiring_fix.py` | Fix two long-standing AudioLatentSlice wiring bugs · `audit_workflows.py` |
| `apply_audio_latent_slice_source_seconds_autowire.py` | Replace hardcoded `source_seconds=300` widget with autowire · `apply_audio_latent_slice_iter_wiring_fix.py`, `audit_workflows.py` |
| `apply_initial_render_audio_duration_autowire.py` | Wire `LTXFramePlanner.actual_seconds` → `TrimAudioDuration.duration` · `audit_workflows.py` |
| `apply_overlap_seconds_single_source.py` | Eliminate AudioLoopController ↔ AudioLoopPlanner overlap_seconds drift · `audit_workflows.py` |
| `apply_iterations_autowire.py` | Wire `AudioLoopPlanner.total_iterations` → `TensorLoopOpen.iterations_in` · `debug_tools.md`, `audit_workflows.py` |
| `apply_planner_break_stride_cycle.py` | Break planner-stride dependency cycle · `audit_workflows.py`, `f_pair_convention.md` |
| `apply_trim_video_latent_to_audio.py` (F14, latent half) | Splice `TrimVideoLatentToAudio` between the assembled video latent and the VAE decode's LATENT input. Latent trim snaps UP to LTX boundary so video ≥ audio (efficiency: decoder skips overshoot). Pair with the image-trim apply (next row) for exact precision · `audit_workflows.py`, `build_upscale_workflow.py`, `build_seam_refinement_workflow.py` |
| `apply_trim_image_batch_to_audio.py` (F14, image half) | Splice `TrimImageBatchToAudio` between the VAE decoder's IMAGE output and `VHS_VideoCombine.images`. Clips the 0-7 pixel-frame residue from the latent trim's snap-UP to exact `int(audio*fps)` frames. Restored 2026-05-10 after user-reported audio clipping showed the latent-only Option A was insufficient · `audit_workflows.py`, `docs/guides/upscale_guide.md` |
| `apply_fix_source_audio_trim_defaults.py` | Change `TrimAudioDuration #567` widgets from buggy `[start_index=5, duration=300]` to `[0, 600]` (full song by default; user can set start_index > 0 to skip intro explicitly). The historical canonical default ate the first 5 seconds of every song silently. Title also updated to make the new default discoverable. Applied 2026-05-10 across 13 affected workflows · CLI |
| `apply_run_id_layout.py` (F15) | Insert `RunIdPrefix` and wire it into `VHS_VideoCombine.filename_prefix` (+ any existing `SaveLatent`) so every render's artifacts cluster under `<output>/<workflow_name>/<timestamp>/`. For loop workflows, also adds a **bypassed** `SaveLatent` wired to `LatentConcat #1605` — user toggles `mode=0` in the UI to capture the assembled latent for the latent-load upscale path · `audit_workflows.py`, `docs/guides/upscale_guide.md` |
| `apply_audio_vae_fix.py` | **Emergency fallback**: swap `VAELoaderKJ` → core `VAELoader` if KJ breaks · CLI-only, unapplied to canonical |

### Apply scripts — controller + frame planner schema (4)

*Add here if you're renaming/refactoring AudioLoopController or LTXFramePlanner widget schemas.*

| Script | Purpose · callers |
|---|---|
| `apply_alc_seed_rename.py` | Rename `seed` → `base_seed` (avoids ComfyUI's auto-`control_after_generate`) · `audit_workflows.py`, `apply_strip_alc_control_after_generate.py` |
| `apply_strip_alc_control_after_generate.py` | Strip leftover `randomize` widget value post-rename · `audit_workflows.py`, `audio_loop_controller.md` |
| `apply_frame_planner_consolidation.py` | Migrate to `LTXFramePlanner` as single dim source · `apply_skip_under_seq_len.py`, `apply_initial_render_audio_duration_autowire.py` |
| `apply_canonical_resolution_fix.py` | Bring `EmptyLTXVLatentVideo` widgets to LTX-valid resolution · `audit_workflows.py` |

### Apply scripts — conditioning + prompt schedule (3)

*Add here if you're migrating conditioning encode paths or prompt scheduling.*

| Script | Purpose · callers |
|---|---|
| `apply_batch_encode_fix.py` | Migrate per-iter `CachedTextEncode + ConditioningBlend` → `TimestampPromptScheduleBatchEncode` (CLIP outside loop) · `apply_keyframe_batch_encode.py`, `nag_technical_reference.md`, `debugging_guide.md` |
| `apply_keyframe_batch_encode.py` | Migrate keyframe `KeyframeImageSchedule + per-iter VAEEncode` → `KeyframeLatentScheduleBatchEncode` · `_helpers/_apply_helpers.py`, `internal/PLAN.md` (private clone only) |
| `apply_prompt_relay_initial_render.py` | Phase 1: wire `PromptRelayEncode` on initial-render path · `audit_workflows.py`, `tests/test_apply_prompt_relay_initial_render.py` |

### Apply scripts — guide / cropguides / preprocess symmetry (4)

*Add here if you're matching init/loop preprocessing or guide topology (F2/F3).*

| Script | Purpose · callers |
|---|---|
| `apply_loop_cropguides_symmetry.py` | Wire loop-body `CFGGuider` through `LTXVCropGuides` (F3) · `debug_tools.md`, `debugging_guide.md`, `audit_workflows.py` |
| `apply_loop_guide_preprocess_symmetry.py` | Match init + loop `LTXVPreprocess(img_compression=18)` (F2) · `apply_loop_cropguides_symmetry.py`, `templates/README.md`, `debug_tools.md` |
| `apply_split_cropguides.py` | Split `LTXVCropGuides` into two instances to break loop cycle · `audit_workflows.py` |
| `apply_lanczos_init_preprocess.py` | Two-stage lanczos init preprocess · `debug_tools.md` |

### Apply scripts — IC-LoRA / ID-LoRA / amplification (5)

*Add here if you're wiring LoRA conditioning paths, LoRA loaders, or amplification.*

| Script | Purpose · callers |
|---|---|
| `apply_iclora_initial_render.py` | Phase 0a: wire IC-LoRA on initial-render path · `apply_loop_cropguides_symmetry.py`, `apply_strip_dead_lora_loaders.py`, `apply_loop_guide_preprocess_symmetry.py` |
| `apply_id_lora_initial_render.py` | Stage ID-LoRA / style-LoRA variant (single LoraLoaderModelOnly splice) · CLI; cited from private per-render action-item ladders (`internal/` only) |
| `apply_strip_dead_lora_loaders.py` | Strip dead LoRA loaders from canonical · `debug_tools.md`, `audit_workflows.py`, `f_pair_convention.md` |
| `apply_ttc_iclora_amplification_poc.py` | POC: amplify IC-LoRA contribution at inference (CFG-analog) · root CLAUDE.md, `cfg_analog_amplification.md`, `apply_ttc_init_guide_amplification_poc.py` |
| `apply_ttc_init_guide_amplification_poc.py` | Stage non-IC-LoRA variant of TTC amplification · `experiments/`, `debug_tools.md` |

### Apply scripts — sage / attention (1)

*Add here if you're tuning AudioLoopHelperSageAttention parameters.*

| Script | Purpose · callers |
|---|---|
| `apply_skip_under_seq_len.py` | Wire `skip_under_seq_len=1024` on `AudioLoopHelperSageAttention` (skip int8 quant overhead on short-Q) · CLI-only |

### Apply scripts — retake (2)

*Add here if you're building retake variants or retake-specific edits.*

| Script | Purpose · callers |
|---|---|
| `apply_audio_loop_retake.py` | Build retake workflow by forking production · `retake_guide.md`, `_helpers/_apply_helpers.py`, `audit_workflows.py` |
| `apply_p3_retake_edit_lora.py` | Wire IC-LoRA edit-anything pattern into retake · `debug_tools.md` |

### Apply scripts — layout + variants (2)

*Add here if you're polishing canvas layout, group structure, or workflow-variant scaffolding.*

| Script | Purpose · callers |
|---|---|
| `apply_intro_workflow.py` | Layout-maintenance for canonical latent variant · `apply_initial_render_audio_duration_autowire.py`, `apply_audio_latent_slice_*` |
| `apply_layout_polish_audio_loop_latent.py` | Stage polished tier-grouped layout for `audio-loop-music-video_latent.json` · CLI; uses `_helpers/_layout_grid.py` |

### Apply scripts — other / one-off (3)

*Add here if it's a true one-off: feature toggles, retired migrations, or experimental builds. If two scripts share a pattern, claim a new section instead of letting this one drift.*

| Script | Purpose · callers |
|---|---|
| `apply_melband_default_off.py` | Disable MelBand vocal separation by default across workflows · scripts/CLAUDE.md, `architecture_overview.md` |
| `apply_spectrogram_iclora_minimal.py` | Build experimental spectrogram-IC-LoRA workflow · `_helpers/_apply_helpers.py`, `debug_tools.md`, `spectrogram_iclora_tutorial.md` |
| `apply_vae_and_cleanup.py` | One-shot VAE cleanup applied to LATENT-variant workflows (2026-04-23) · `compare-workflows` skill — archive candidate (one-time migration, fully baked) |

### Apply scripts — bench / IC-LoRA bench (2)

*Add here if you're profiling bench workflows or sage arm variants.*

| Script | Purpose · callers |
|---|---|
| `apply_iclora_bench_profiling.py` | Wire `Profile*` nodes around audio loop in bench workflow · `apply_iclora_bench_sage_arm.py`, `bench_workflow_guide.md` |
| `apply_iclora_bench_sage_arm.py` | Sage-attention arm variant of iclora bench · `bench_compare_runs.py`, `bench_workflow_guide.md` |

### Profiling tools (3 — insert / remove / summarize triplet)

*Add here if you're inserting/removing Profile nodes or analyzing perf traces.*

| Script | Purpose · callers |
|---|---|
| `apply_profiling_nodes.py` | Insert `ProfileBegin` / `ProfileIterStep` / `ProfileEnd` · `remove_profiling_nodes.py`, `profiling_guide.md` |
| `remove_profiling_nodes.py` | Remove same nodes · `profiling_guide.md` |
| `profile_summary.py` | Categorized text summary from torch.profiler chrome trace · `profiling_guide.md` |

### Bench / telemetry summary (3)

*Add here if you're comparing bench runs or summarizing execution logs.*

| Script | Purpose · callers |
|---|---|
| `bench_compare_runs.py` | Side-by-side run comparator · `bench_workflow_guide.md`, `apply_iclora_bench_sage_arm.py` |
| `exec_log_summary.py` | Aggregate ComfyUI `exec.jsonl` into per-node-class bottleneck report · `bench_compare_runs.py`, `bench_workflow_guide.md` |
| `sage_telemetry_summary.py` | Aggregate sage tracer JSONL into per-mode summary · `exec_log_summary.py`, `debug_tools.md`, README |

### Audio analysis (3)

*Add here if you're extracting BPM/key/structure/F0 or rendering spectrograms.*

| Script | Purpose · callers |
|---|---|
| `analyze_audio.py` | ffmpeg-only energy/structure detection (zero Python deps) · `audio_analysis_guide.md`, `prompt-schedule` skill |
| `analyze_audio_features.py` | librosa: BPM, key, F0, structure, JSON for LLM · README, `spectrogram_to_reference.py`, `apply_ttc_init_guide_amplification_poc.py` |
| `preprocess_audio_for_ltx.py` | Audio preprocessing for LTX 2.3 V2A · `debugging_guide.md` |

### IC-LoRA reference asset prep (2)

*Add here if you're aligning ref videos or rendering spectrograms for IC-LoRA.*

| Script | Purpose · callers |
|---|---|
| `align_ref_video.py` | Align driving reference video to audio-loop IC-LoRA workflow params (F12) · CLI-only |
| `spectrogram_to_reference.py` | Render Mel spectrogram as PNG frame sequence (IC-LoRA spectrogram-as-reference, Phase 2.0) · scripts/CLAUDE.md, `spectrogram_iclora_tutorial.md` |

### Sage trace verification (1)

*Add here if you're auditing sage attention iteration behavior.*

| Script | Purpose · callers |
|---|---|
| `verify_sage_iteration_trace.sh` | Verify sage override is firing on every loop iteration · `debug_tools.md`, `bench_workflow_guide.md` |

## Duplication & merge opportunities

Reviewed during the 2026-05-05 cleanup; recording here so future curation passes don't re-derive.

### Worth considering

- **Profiling triplet** (`apply_profiling_nodes.py` / `remove_profiling_nodes.py` / `profile_summary.py`) could collapse into a single `profile.py` with `--insert` / `--remove` / `--summarize` subcommands. Net win: one entry-point, smaller mental surface for `profiling_guide.md`. Net cost: subcommand parsing + the existing scripts already work. **Verdict: defer** — value is cosmetic, not load-bearing.
- **Slot-dict `from-scratch` builders** (`build_keyframe_workflow.py` / `build_seam_refinement_workflow.py` / `build_upscale_workflow.py`) already share `WorkflowEditor.io_in/widget_in/out` helpers per the §"WorkflowEditor" note. Duplication is minimal; further factoring would create premature abstractions.
- **Audio analysis pair** (`analyze_audio.py` zero-dep + `analyze_audio_features.py` librosa): **keep separate**. The zero-dep version is the fast-path for environments without librosa; merging would force the full dependency chain on simple energy queries.

### Confirmed not worth merging

- **Strip-dead-node trio** (`apply_strip_alc_control_after_generate.py` / `apply_strip_dead_lora_loaders.py` / `apply_strip_sd3_shift_node.py`): different node types, different signatures, all tied to specific F-pair audits. A generic `apply_strip_dead.py --type X` would obscure the audit-pair coupling that makes these scripts traceable.
- **Symmetry pair** (`apply_loop_guide_preprocess_symmetry.py` F2 + `apply_loop_cropguides_symmetry.py` F3): adjacent rules, but each maps to its own audit invariant. Splitting matches the F-pair convention.
- **Autowire scripts** (`apply_iterations_autowire.py` / `apply_initial_render_audio_duration_autowire.py` / `apply_audio_latent_slice_source_seconds_autowire.py`): all "wire X to Y", but X and Y differ per script and each pairs with its own audit-check. Pattern is uniform; subjects are not.

### Naming irregularities (low-priority cosmetic)

- `_apply_helpers.py` (leading underscore) signals "not a CLI script" — distinct from peers. Convention is good; flagged here for visibility.
- `apply_canonical_resolution_fix.py` is a *fixer*, not a *generator* — name pattern matches other `apply_*` scripts, no change needed.
- Some apply scripts include a docstring `Last updated:` line; others don't. Not enforced; the file mtime is authoritative.

### Scripts with no callers (CLI-only utilities — by design)

- `align_ref_video.py` — F12 IC-LoRA video-ref user tool
- `apply_audio_vae_fix.py` — emergency fallback
- `apply_id_lora_initial_render.py` — staging tool, driven from private per-render action-item ladders
- `apply_skip_under_seq_len.py` — sage perf knob
- `build_seam_refinement_workflow.py` — just shipped (2026-05-05)
- `calc_ltx_resolution.py` — offline aspect-ratio CLI

These are intentionally leaf-only — invoked by humans via uv, not chained. Don't add fake callers.

## When `WorkflowEditor` itself is suspect

Rare but possible: a bug in the editor that produces malformed output the audit doesn't catch. In that case fall back to `_apply_helpers.py`'s RAW-orjson fork-and-strip primitives. Don't paper over by writing a hand-rolled traversal in a feature apply-script — fix the editor.

## References

- `../CLAUDE.md` — root project rules
- `../docs/reference/debug_tools.md` — full debug + apply-script inventory
- `../.claude/CLAUDE.md` — CLAUDE.md governance policy
- `templates/README.md` — apply-script template overview
- `../docs/guides/audio_analysis_guide.md` — audio-analysis script usage
