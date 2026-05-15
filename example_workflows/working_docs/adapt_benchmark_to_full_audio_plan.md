Last updated: 2026-05-15

# Migration plan: adapt `fml2v_var_d_audio_input.json` (B) into a full-audio-length looped workflow

**Status**: read-only research. This document describes the edit plan; the user implements (or commissions a follow-up apply script). No JSON has been mutated.

**Premise**: B produces higher per-iteration video quality than A (`audio-loop-music-video_latent.json`). B is structurally locked to ~4s of audio because it lacks A's loop machinery. Goal is to wrap B's quality-axis decisions in A's loop spine so it can process arbitrary-length music tracks.

**Sister docs**:
- `workflow_quality_delta_analysis.md` (this directory) — the axis-by-axis structural diff this plan is acting on.
- `internal/design/benchmark_ablation_test_plan.md` (private clone only) — defines the per-knob attribution that should run BEFORE this migration; tells you which of B's quality levers are load-bearing vs cosmetic.
- `scripts/CLAUDE.md` — apply-script conventions if/when this plan gets turned into an `apply_*.py` follow-up.

**Recommended sequencing**: run the ablation plan Phase 1 first. If the per-knob attribution says "refine alone carries the win, multi-keyframe is decorative," the migration recipe below shrinks to the two-pass-refine port only. If the attribution says "wins compound multiplicatively," port the full stack as described.

---

## Part 1 — Inventory: A's loop primitives (what we're transplanting)

For each node, the canonical role and the node ID in A (`example_workflows/audio-loop-music-video_latent.json`). Refer to root CLAUDE.md → Architecture for the docstring-level role descriptions.

### Loop spine (top-level in A)

| Node | ID in A | Role |
|---|---|---|
| `TensorLoopOpen` | `#1539` | Loop entry; `iterations_in` slot is the iteration count |
| `TensorLoopClose` | `#1540` | Loop exit; `should_stop` slot wired from `AudioLoopController` |
| `AudioLoopPlanner` | `#1560` | Computes `total_iterations` from full audio + planner settings; output slot 1 → `TensorLoopOpen.iterations_in` (the **`iterations_autowired` (F5)** invariant) |
| `AudioLoopController` | `#1582` | Per-iteration: derives `audio_start`, `audio_end`, `stride_seconds`, `should_stop`, seed |
| `LTXFramePlanner` | `#1634` | Dimension SSoT (width/height/frames/fps) — feeds AudioLoopPlanner and AudioLoopController |

### Pre-encoded prompt schedule (CLIP MUST be outside loop body)

| Node | ID in A | Role |
|---|---|---|
| `TimestampPromptScheduleBatchEncode` | `#1615` | Reads schedule + `CLIP` → emits a batch of CONDITIONING (one per schedule entry). Stamps `frame_rate` on each entry. **Outside loop body.** |
| `ConditioningSelectByIteration` | `#1616` (positive), `#2021` (negative) | Inside loop body. Selects the right CONDITIONING from the pre-encoded batch using `TensorLoopOpen` iteration index |

### Loop-body interior (inside subgraph `#843` in A)

| Node | ID in A | Role |
|---|---|---|
| `LoopIterationStamp` | `#1618` | Stamps iteration index onto MODEL for downstream iter-aware nodes |
| `IterationCleanup` | `#2007` (subgraph) | End-of-iteration tensor lifecycle |
| `LatentContextExtract` | `#2004` (subgraph) | Extracts overlap context from previous-iteration output (strips `noise_mask` automatically — don't substitute `LTXVSelectLatents`) |
| `LatentOverlapTrim` | `#2005` (subgraph) | Trims overlap region from current iteration output before concat |
| `LTXVAdainLatent` | `#2006` (subgraph) | Color/statistics normalization across iteration boundary |
| `LTXVAudioVideoMask` | `#606` (subgraph) | Iteration-aware audio/video mask (`audio_start_time = audio_end_time = window_size` keeps audio frozen — **do not change**) |
| `LTXVAddLatentGuide` | `#1519` (subgraph) | The single-keyframe init anchor; `latent_idx=-1`, strength from `FloatConstant #1269` |
| `AudioLatentSlice` | `#2012` (subgraph) | Per-iter audio latent slicing |
| `LTXVCropGuides` | `#2008` (subgraph), `#381` (top) | Strips guides from CONDITIONING before final sampling (F3 symmetry rule) |

### Output trimming + clustering (top-level in A)

| Node | ID in A | Role |
|---|---|---|
| `TrimImageBatchToAudio` | `#2029` | F14: clips image batch to audio length (prevents silence at end) |
| `TrimVideoLatentToAudio` | `#2028` | F14: clips video latent to audio length |
| `RunIdPrefix` | `#2026` | F15: per-render folder clustering on `VHS_VideoCombine.filename_prefix` |

### Currently absent in A (but available in repo)

| Node | Module | Role | When to use |
|---|---|---|---|
| `KeyframeLatentScheduleBatchEncode` | `nodes.py` | Pre-VAE-encodes a batch of keyframe images (analogous to `TimestampPromptScheduleBatchEncode` for prompts) | If porting B's three-keyframe pattern into the loop |
| `LatentSelectByIteration` | `nodes.py` | Selects per-iter keyframe latent inside loop | Pair with `KeyframeLatentScheduleBatchEncode` |

---

## Part 2 — B's quality-axis values to preserve

Source: `workflow_quality_delta_analysis.md`. Listed in priority order from the hypothesis ranking.

### Tier 1 — load-bearing (do not drop)

1. **Two-pass refine + spatial upsample topology**
   - Pass 1 latent at half-resolution (`width/2`, `height/2`), full 8-step distilled sigmas
   - `LTXVCropGuides` → `LTXVLatentUpsampler` between pass 1 output and pass 2 input (2× spatial)
   - Pass 2 refine: 3-step sigmas `0.85, 0.7250, 0.4219, 0.0`, separate `SamplerCustomAdvanced`
   - Pass 2 keyframe count reduced to 2 (first + last; drops middle)

2. **`euler_cfg_pp` / `euler_ancestral_cfg_pp` samplers**
   - Pass 1: `euler_ancestral_cfg_pp` (B's choice — but consider `euler_cfg_pp` to stay closer to canonical "no ancestral" rule; the ablation plan does not isolate this knob, so it's a judgment call)
   - Pass 2: `euler_cfg_pp`
   - **Both at `cfg=1`** — `_cfg_pp` activates the NAG negative path that canonical `euler` ignores at `cfg=1`

3. **Three-keyframe anchoring (pass 1) + two-keyframe anchoring (pass 2)**
   - `LTXVAddGuideMulti` pass 1 widgets: `['3', 0, 0.7, 0, 0.25, -1, 1]` (num=3, idx_0=0 str=0.7, idx_1=0(mid) str=0.25, idx_2=-1 str=1)
   - `LTXVAddGuideMulti` pass 2 widgets: `['2', 0, 1, -1, 1]`
   - Three keyframe images, each preprocessed via `LTXVPreprocess img_compression=18`

### Tier 2 — useful but porting-safe to defer

4. **`AudioLoopHelperSageAttention` mode `auto`** — B uses `auto`, shipped workflows use `auto_mask_aware`. The mask-aware path isn't exercised on audio-loop workflows (per CLAUDE.md "Pending review" note about `LTXVCropGuides`/`LTXVConcatAVLatent` stripping `guide_attention_entries`). **Recommend keeping `auto_mask_aware` for consistency with audit conventions.**

5. **`LTXVChunkFeedForward [2, 4096]`** — already matches A. No port needed.

### Tier 3 — perf housekeeping (skip on port)

6. `PathchSageAttentionKJ`, `LTX2MemoryEfficientSageAttentionPatch`, `LTX2AttentionTunerPatch` — all bypassed in B. Don't add them in the migrated workflow.

7. `Power Lora Loader (rgthree)` — no LoRAs configured in B. Don't add.

8. `UnetLoaderGGUF`, `DualCLIPLoaderGGUF` — bypassed alt loaders in B. Don't add.

### Knobs to align with A's conventions (not preserve from B)

- **Dimension SSoT**: replace B's SetNode/GetNode width/height plumbing with `LTXFramePlanner` (mandatory for `frame_planner_present` audit).
- **`LTXSmartImageResize`**: A's adaptive multi-stage lanczos resizer. Use it for all three keyframe images instead of B's `ImageResizeKJv2 + ResizeImagesByLongerEdge` chain — fixes aliasing risk at >2× downscale ratios (per CLAUDE.md `internal/analysis/smart_resize_quantization_postmortem.md` (private clone only)).
- **`TrimAudioDuration`**: drop B's 4.0-second cap. Use A's `[0, 600]` (or higher) for full audio.

---

## Part 3 — Migration recipe (ordered edits)

This is the conceptual recipe. Implementing it as an apply script would follow `scripts/CLAUDE.md` conventions (WorkflowEditor, `--dry-run`/`--revert`, idempotence, `require_nodes` pre-flight). Skipping a step or reordering 4 ↔ 5 likely breaks the loop topology.

### Step 0 — choose the starting point

Two valid starting points:
- **Start from A**: keep loop spine intact; transplant B's two-pass refine + multi-keyframe into the subgraph body. **Recommended** — A's loop machinery is large and well-tested; B's quality stack is smaller and easier to graft in.
- Start from B: keep two-pass stack intact; wrap in loop spine. Higher risk — every subgraph slot has to be designed from scratch.

Remainder of this recipe assumes "start from A."

### Step 1 — replace single-pass sampler with two-pass refine inside the subgraph

Inside subgraph `#843` in A, the current sampling pattern is:
- `LTXVSeparateAVLatent` ← `SamplerCustomAdvanced #573` ← (sampler, sigmas, guider, latent, noise)

Replace with two `SamplerCustomAdvanced` instances chained through `LTXVLatentUpsampler`:

```
[pass 1 input latent at half-res]
  → SamplerCustomAdvanced (pass1)
  → LTXVSeparateAVLatent (intermediate; discard audio half)
  → LTXVCropGuides
  → LTXVLatentUpsampler (2× spatial)
  → LTXVAddGuideMulti (pass 2 — 2 keyframes)
  → LTXVConcatAVLatent (re-attach audio)
  → SamplerCustomAdvanced (pass2)
  → LTXVSeparateAVLatent (final separate)
  → [existing LatentContextExtract / overlap trim path]
```

Pass 1 sigmas: `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`
Pass 1 sampler: `euler_ancestral_cfg_pp` (or `euler_cfg_pp` — see Tier 1 note)
Pass 2 sigmas: `0.85, 0.7250, 0.4219, 0.0`
Pass 2 sampler: `euler_cfg_pp`
Both CFGGuiders: `cfg=1`

### Step 2 — substitute multi-keyframe guides for single `LTXVAddLatentGuide`

In the subgraph, A currently has `LTXVAddLatentGuide #1519` with one image + one strength.

Two options:

**Option 2a — simple port (all iterations use the same 3 keyframe images)**:
- Add three new subgraph input slots: `keyframe_image_first`, `keyframe_image_mid`, `keyframe_image_last` (IMAGE type each).
- Top-level: three `LoadImage` + `LTXSmartImageResize` + `LTXVPreprocess(img_compression=18)` chains.
- Wire the three preprocessed images through the subgraph into a single `LTXVAddGuideMulti` (pass 1) replacing `LTXVAddLatentGuide #1519`.

**Option 2b — per-iter keyframe schedule (uses `KeyframeLatentScheduleBatchEncode` + `LatentSelectByIteration`)**:
- Author a schedule of keyframes (e.g. one set per timestamp range).
- Pre-VAE-encode all via `KeyframeLatentScheduleBatchEncode` (outside loop, analogous to how `TimestampPromptScheduleBatchEncode` handles prompts).
- Inside loop: `LatentSelectByIteration` plucks the per-iter keyframe latents → feed into `LTXVAddGuideMulti`.

Option 2a is simpler and matches B's behavior 1:1. Option 2b is the load-bearing improvement if you want narrative arcs across long songs. **Start with 2a; promote to 2b if Phase 3 of the ablation plan says the keyframe lever is real.**

### Step 3 — preserve and re-route the `LTXVImgToVideoInplaceKJ` initial-frame anchor

B replaces `LTXVImgToVideoInplaceKJ #531` with multi-keyframe guides. **Do not drop `LTXVImgToVideoInplaceKJ` from the migrated workflow.** It serves a different role: it writes the encoded init into frame 0 with `noise_mask=0` (locks the first frame against drift). This is A's F2 symmetry partner.

Keep `LTXVImgToVideoInplaceKJ #531` for the initial render path; layer `LTXVAddGuideMulti` on top for the per-iteration anchoring.

### Step 4 — keep `LTXFramePlanner` as the dimension SSoT; add half-res derivation

A's `LTXFramePlanner #1634` outputs full-res width/height. For two-pass refine, you need half-res for pass 1.

Add two `ComfyMathExpression 'a/2'` nodes (one for width, one for height) fed from `LTXFramePlanner` outputs. Wire them into the pass-1 `EmptyLTXVLatentVideo`. Pass 2's latent volume is determined by the upsampler, not a separate EmptyLatent.

### Step 5 — keep audio path untouched

A's audio chain (`LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode → LTXVConcatAVLatent`; optional `MelBandRoFormerModelLoader`/`MelBandRoFormerSampler` vocal-separation branch is bypassed by default per `apply_melband_default_off.py`) is sacred per CLAUDE.md. Do not modify. The two-pass sampler should consume the same encoded audio latent on both passes; `LTXVConcatAVLatent` re-attaches it before each pass.

### Step 6 — remove B-specific top-level cruft

When transplanting from B into A, do NOT carry over:
- `EmptyLTXVLatentVideo #32` (replaced by per-iter latent construction inside subgraph)
- Width/height SetNode/GetNode plumbing in B (replaced by `LTXFramePlanner`)
- B's prompt enhancer subgraph (`TextGenerateLTX2Prompt`, `ComfySwitchNode`, `easy showAnything`) — A has a curated prompt schedule via `TimestampPromptScheduleBatchEncode`
- Bypassed nodes from B (`UnetLoaderGGUF`, `PathchSageAttentionKJ`, etc.)

### Step 7 — verify F2/F3 symmetry holds

After the transplant:
- F2: both initial-render and loop-body branches must share the same `LTXVPreprocess(img_compression=18)` output (per `docs/reference/pipeline_flow_latent.md`).
- F3: loop `CFGGuider` positive/negative must flow through `LTXVCropGuides` before reaching the sampler.

The two-pass topology introduces a second `CFGGuider` and a second `LTXVCropGuides`. F3 must hold on BOTH passes' guider chains.

---

## Part 4 — Risks + footguns

Cross-referenced to CLAUDE.md rules. These are the rules most likely to be violated by a careless port.

### Critical

1. **"CLIP must not enter the loop body"** — root CLAUDE.md.
   - Risk: B's prompt enhancer subgraph uses CLIP directly. If you accidentally wire CLIP through the new loop body, every iteration re-runs Gemma encoding (slow + memory thrash).
   - Mitigation: pre-encode via `TimestampPromptScheduleBatchEncode` outside the loop, exactly as A does. `ConditioningSelectByIteration` is the only loop-body CLIP-product consumer.

2. **"Loop-body CONDITIONING must carry `frame_rate`"** — root CLAUDE.md.
   - Risk: any new CONDITIONING-producing node added to the loop body (e.g. if you wire a new conditioning blend) must stamp `frame_rate` via `node_helpers.conditioning_set_values`.
   - Mitigation: `TimestampPromptScheduleBatchEncode` already stamps it. The new `LTXVAddGuideMulti` nodes don't produce raw CONDITIONING — they wrap existing CONDITIONING — so the `frame_rate` stamp propagates. But if you add `ConditioningBlend` or similar, audit it.

3. **"`AudioLoopController` outputs are iteration-dependent in the executor DAG"** — root CLAUDE.md.
   - Risk: anything OUTSIDE the loop that needs `stride_seconds`/`audio_duration` (e.g. `TimestampPromptScheduleBatchEncode` inputs) must source them from `AudioLoopPlanner`, NOT `AudioLoopController`. Otherwise a cycle closes through `TensorLoopOpen` and `graph_acyclic` audit fails.
   - Mitigation: re-use A's existing wiring exactly. The `AudioLoopPlanner #1560` outputs feed `TimestampPromptScheduleBatchEncode` in A.

4. **"Don't name an INT widget exactly `seed` or `noise_seed`"** — root CLAUDE.md.
   - Risk: B's `RandomNoise #14`, `#15` use the default ComfyUI INT widget name. If you transplant these nodes verbatim, you inherit the `control_after_generate` auto-attach footgun.
   - Mitigation: audit `alc_seed_legacy_name` (F4) catches this. Verify by running the audit post-port.

### Important

5. **"`LTXVAudioVideoMask` (Node 606) wiring is intentional"** — root CLAUDE.md.
   - Risk: when you re-arrange the subgraph for two-pass, you may be tempted to "simplify" the `audio_start_time = audio_end_time = window_size` parameterization. Don't.
   - Mitigation: leave Node 606's existing wiring alone.

6. **"`AudioLoopHelperSageAttention` outputs are iteration-dependent"** — false; sage is a model patch, not iter-aware. But its placement order matters.
   - Mitigation: keep sage attention patch BEFORE the loop body (top-level patch chain on MODEL), as A does.

7. **Subgraph schema changes force a UI re-add** — root CLAUDE.md.
   - Risk: any change to subgraph input/output slots invalidates the slot-index baked into top-level wires. Users with the workflow open in ComfyUI must re-add the subgraph node after the migration.
   - Mitigation: document this in the apply script's `--help` output. Don't reorder existing slots; only append new ones.

8. **`first_frame_guide_strength`** — root CLAUDE.md.
   - Risk: A's `FloatConstant #1269 = 1.0` provides max identity stability. B's multi-keyframe pattern uses 0.7 for the first keyframe in pass 1. If you keep `LTXVImgToVideoInplaceKJ` for initial render AND add `LTXVAddGuideMulti` with strength 0.7 for the first frame, the two anchors may interact (additive? overriding? depends on the node).
   - Mitigation: empirically test on a 1-iter render first. If interactions are weird, set the `LTXVAddGuideMulti` first-keyframe strength to match `FloatConstant #1269` (1.0).

### Footguns CLAUDE.md flags but are unlikely on this port

9. "Don't ship two schema changes touching the same iteration-state plane in one session" — this migration is one structural change; not at risk.

10. "Bake new topology constraints into `audit_workflows.py`" — if you add new invariants (e.g. "two-pass workflow must have a `LTXVLatentUpsampler` between samplers"), document them as an F-pair following `docs/reference/f_pair_convention.md`.

---

## Part 5 — Validation steps

After porting, run these in order:

### 5.1 JSON validity (catches the most basic regression)

```bash
python3 -c "import json; json.load(open('example_workflows/<migrated_filename>.json'))"
```

### 5.2 Full audit sweep

```bash
uv run --group dev python scripts/audit_workflows.py
```

The migrated workflow MUST pass these audit IDs (per CLAUDE.md):

- `graph_acyclic` (ERR) — no cycle through `TensorLoopOpen`
- `iterations_autowired` (F5) — `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in`
- `frame_planner_present` (F8) — `LTXFramePlanner` is the dimension SSoT
- `planner_no_stride_input` (F7) — `AudioLoopPlanner` does NOT receive iteration-dependent inputs
- `alc_seed_legacy_name` (F4) — no INT widget named `seed`/`noise_seed`
- `alc_widget_drift` (F6) — `AudioLoopController` widget order matches schema
- F2 / F3 symmetry (initial-render preprocess + crop-guides chain)

### 5.3 1-iteration smoke test

Set `TrimAudioDuration` to 4 seconds (B's original cap) to force a single iteration; render. Compare output mp4 against B's output:
- Identical resolution, frame count, fps
- Identical visual quality (the port should be a no-op at 1 iteration)
- No NaN/black-frame artifacts

If 1-iter differs from B: the port introduced a subtle wire change. Diff the workflow JSON via `compare-workflows` against B.

### 5.4 Multi-iteration smoke test

Set `TrimAudioDuration` to ~10 seconds (forces 2–3 iterations at canonical stride). Render. Verify:
- All iterations complete (no crash mid-loop)
- Visual continuity across iteration boundaries (no flash/discontinuity at overlap zones)
- Audio plays through without gaps
- Final mp4 length matches audio length (F14 trim worked)

### 5.5 Full-song validation

Pick a 2-3 minute track. Render. Check:
- Wall-time is ~Nx the 1-iter wall-time where N = total_iterations
- Identity stability across the full song
- No memory leak (peak VRAM stays bounded — per CLAUDE.md, watch for offload pressure via `bench_aimdo_vram.py`)

---

## Part 6 — Is an apply script the right format?

This migration is a candidate for `scripts/apply_two_pass_quality.py` or similar, following `scripts/CLAUDE.md` conventions:

**Pros of an apply script**:
- Reproducible across the canonical + any user-customized derivative
- `--revert` lets users back out cleanly
- `--dry-run` shows planned edits before committing
- Pairs naturally with a new audit-check (`two_pass_refine_present`, perhaps F16)

**Cons / unknowns**:
- The two-pass topology change is much larger than typical apply scripts (most are ~50-line edits; this would be 200+ lines of WorkflowEditor calls)
- Subgraph schema changes (Step 1: add `keyframe_image_*` slots) force a UI re-add — users will be confused if the apply runs invisibly
- Per `scripts/CLAUDE.md` "Self-targeting apply scripts" gotcha: if this is in-place on canonical, it overwrites user manual edits

**Recommendation**: prototype as a staged variant (`scripts/templates/apply_script_staged_variant.py`) targeting `example_workflows/audio-loop-music-video_latent_two_pass.json` first. Ship as a sibling workflow rather than replacing canonical. Promote to in-place apply only after the Phase-1 ablation confirms the wins justify the wall-time cost and the new shape has been render-tested on multiple songs.

---

## Part 7 — Open questions the user must answer before implementation

1. **Per-iteration wall-time budget** — two-pass + upsample is roughly 2–2.5× per iteration. For a 3-minute song with stride ~3s, that's ~60 iterations. Is ~2× wall-time worth the per-clip quality lift? (Phase 3 of the ablation plan answers this empirically.)

2. **Keyframe schedule authoring** — does the user want option 2a (three fixed keyframes for all iterations, like B) or option 2b (per-iter keyframe schedule)? 2b is more complex but enables narrative arcs; 2a is the direct port.

3. **Sampler choice in pass 1** — keep B's `euler_ancestral_cfg_pp` (CLAUDE.md says no ancestral, but at `cfg=1` with `_cfg_pp` the ancestral noise dynamics may not be harmful) or swap to plain `euler_cfg_pp` (closer to canonical guidance)?

4. **Init image strategy** — B uses three independent init images. A uses one. For long songs, do you want the three keyframes to be different shots of the same subject (identity continuity) or different scenes (narrative)? Affects whether 2a or 2b is appropriate.

These should be answered before committing to the apply-script form factor.

---

## Summary

The migration is conceptually: "Wrap B's two-pass topology inside A's loop spine, keeping B's quality knobs (refine, multi-keyframe, CFG-pp samplers) and A's loop machinery (TensorLoop, AudioLoopPlanner, ConditioningSelectByIteration, F2/F3 symmetry)."

Largest risks: CLIP leaking into the loop body, F3 symmetry holding on both sampler passes, subgraph slot-index drift breaking saved workflows, and the iter-dependence cycle (`AudioLoopPlanner` vs `AudioLoopController` for outside-loop consumers).

Empirical step that must come first: run `internal/design/benchmark_ablation_test_plan.md` (private clone only) Phase 1 to confirm the multi-keyframe + two-pass-refine attribution. Skip this and you may port a stack where only one of the two structural changes actually carries the quality lift, paying full wall-time cost for half the benefit.

---

## Part 8 — Cosmetic / config decisions captured 2026-05-15

These came up during the planning discussion and are recorded here so they don't get lost in the actual migration edits.

### 8.1 — Move benchmark MarkdownNotes adjacent to the sage node

`fml2v_var_d_audio_input.json` has two `MarkdownNote` nodes (`#9000`, `#9001`) positioned far off-canvas at `[-1200, -800]` and `[-1200, -460]`. They were authored as a documentation header for the kernel-benchmark variant. For the adapted music-video workflow, move them next to the sage attention node:

| Node | Original pos | New pos (suggested) |
|---|---|---|
| `MarkdownNote #9000` ("Original workflow + credit") | `[-1200, -800]` | `[1500, -200]` |
| `MarkdownNote #9001` ("Adaptations + benchmark context") | `[-1200, -460]` | `[1500, 120]` |
| `AudioLoopHelperSageAttention #2296` (anchor) | `[1500, 800]` | unchanged |

This stacks the two notes vertically just above the sage node. If the whole sage-node cluster is being relocated as part of the migration to fit into A's canvas region, move the notes as a unit with it.

### 8.2 — Sage mode: `auto` → `auto_mask_aware`

`#2296` currently has widget `["auto", true, 1024]`. For the adapted music-video workflow, change widget[0] to `"auto_mask_aware"`. Rationale:

- `auto_mask_aware` routes masked cross-attn to fp16_triton (the kernel that handles LTX cross-attn cleanly) and unmasked self-attn to sage auto. Source: `nodes_sage.py:639-705` tooltip.
- `auto` delegates fully to sage's dispatch — that's correct for the kernel-benchmark variant (it lets sage's fp8 fork dispatch the fp8 masked path on sm89+), but the wrong default for a production music-video render.
- The canonical music-video workflow `audio-loop-music-video_latent.json` `#268` ships with `auto_mask_aware` — matching it is consistency-with-canonical.
- Caveat (root CLAUDE.md Pending-review note): audio-loop workflows cannot actually exercise LTX 2.3's masked self-attn path (something in `LTXVCropGuides` / `LTXVConcatAVLatent`'s NestedTensor packing strips `guide_attention_entries` before `_process_input` builds the mask). So `auto_mask_aware` is correct-and-safe but the mask routing is inert — it falls through to unmasked sage auto. Still the right default.

### 8.3 — `skip_under_seq_len`: keep at 1024

`#2296` widget[2] = `1024`. Keep as-is. Per `nodes_sage.py:686-700`: "Recommended: 1024 — sage's int8 quant + kernel-launch overhead dominates on short sequences (~0.45× torch_flash at seq=497/498)."

The audio-loop math gives seq_len ≈ 497 inside the window (25 latents/sec × ~19.88s window). That's exactly the regime the 1024 threshold targets — calls with `q.shape[1] < 1024` route to pytorch instead of sage. Both shipped workflows (`_latent.json` #268 and benchmark #2296) already use 1024. No change needed for the migration.

If the migration changes the window size significantly (e.g. shrinks window to <10s for faster iteration), revisit: a smaller window produces shorter sequences, and 1024 may need to come down proportionally. For window ≥ 15s, 1024 stays right.
