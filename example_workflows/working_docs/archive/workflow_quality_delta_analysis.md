Last updated: 2026-05-15

# Workflow quality delta analysis — A (`audio-loop-music-video_latent.json`) vs B (`fml2v_var_d_audio_input.json`)

Goal: enumerate the quality-affecting structural differences between the production audio-loop workflow A and the per-iteration benchmark workflow B. The user reports B produces visibly higher per-iteration video quality. This doc isolates which decisions in B are responsible so they can be ported into the looped form.

Prior art (read these before re-deriving): `internal/design/benchmark_ablation_test_plan.md` (private clone only) lays out a five-workflow ablation (`fml2v_var_{a,b,c,d}_*.json`) whose explicit purpose is attributing the same quality lift. B (`var_d`) is the audio-input port of that benchmark. The Phase-1 decision matrix in that plan turns subjective render comparisons into a "which knob carries the win" answer. This analysis is the structural pre-work that matrix references.

Source diff produced via `.claude/skills/compare-workflows/diff_workflows.py` plus targeted widget-value extraction.

---

## Executive summary

B is not "A with one knob flipped" — it is a structurally different topology:

- A: single-pass distilled (8 fixed sigmas, `cfg=1`, `euler`), one keyframe anchor (`LTXVAddLatentGuide` inside the loop subgraph), produces full-resolution latents in one shot, audio frozen via `noise_mask=0`.
- B: **two-pass**, half-resolution → spatial-upsample (`LTXVLatentUpsampler`) → refine; three-keyframe anchoring (`LTXVAddGuideMulti` with first/mid/last), audio frozen via `noise_mask=0`. CFG-pp ancestral sampler on pass 1, CFG-pp euler on pass 2.

The dominant quality-affecting deltas are stacked structural changes, not sampler/CFG tweaks. The hypothesis ordering at the bottom of this document predicts which deltas carry the lift; the ablation test plan provides the empirical answer.

---

## Per-axis comparison

Legend: **(Q)** quality-affecting, **(P)** perf-affecting, **(C)** cosmetic. A single delta can be both Q and P.

### 1. Sampler chain — **(Q, P)**

| Axis | A | B (pass 1, low-res) | B (pass 2, refine on upsampled) |
|---|---|---|---|
| `KSamplerSelect` | `euler` | `euler_ancestral_cfg_pp` | `euler_cfg_pp` |
| `ManualSigmas` | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` (canonical 8 fixed sigmas) | same 8 fixed sigmas | `0.85, 0.7250, 0.4219, 0.0` (3-step refine) |
| `CFGGuider` cfg | `1` | `1` | `1` |
| `ModelSamplingSD3` (flow shift) | absent — matches canonical | absent | absent |

**B violates root CLAUDE.md's "canonical distilled = euler + cfg=1 + no SD3 shift" rule on the sampler name only.** B uses `euler_ancestral_cfg_pp` for pass 1 and `euler_cfg_pp` for pass 2. CLAUDE.md says "No `euler_ancestral*`" — so B's pass-1 sampler is an explicit deviation from canonical.

Hypothesis: the `_cfg_pp` variant (Perp-Neg / CFG++) substantially changes how the negative prompt is integrated at `cfg=1` (where vanilla CFG is a no-op). With `cfg=1` and a non-zero NAG conditioning, `_cfg_pp` actually engages the negative path that vanilla `euler` ignores. **This is potentially the single biggest quality lever**, because the CLAUDE.md guidance about "no euler_ancestral" was written for `cfg≠1` workflows where ancestral noise can break distillation; at `cfg=1` with `_cfg_pp`, the math is different.

### 2. Two-pass refine + spatial upsample — **(Q, P)** — load-bearing

A: single `SamplerCustomAdvanced` (`#161`) → `LTXVSeparateAVLatent` → decode. One sampling pass at full resolution.

B: two passes wired in series.
- **Pass 1** (`SamplerCustomAdvanced #13`): runs the 8-step distilled chain at half resolution. Pass-1 latent volume is set by `EmptyLTXVLatentVideo #32` whose width/height come from `ComfyMathExpression 'a/2'` nodes (`#2191`, `#2192`) — explicit halving of the width/height SetNodes.
- **Spatial upsample**: pass-1 output goes through `LTXVCropGuides #2222 → LTXVLatentUpsampler #25` (2× spatial in latent space).
- **Pass 2** (`SamplerCustomAdvanced #21`): refines the upsampled latent for 3 sigma steps (`0.85, 0.7250, 0.4219, 0.0`) starting from a partially-denoised sigma. Output goes to final decode.

This is the same shape as Lightricks's reference T2V two-pass distilled workflow (referenced in `coderef/LTX-2/`). It is structurally absent from A.

Hypothesis: this is **the** quality lever. Half-resolution sampling lets the model commit to coarse structure cheaply; the spatial-upsample-then-refine pass adds fine texture without burning the whole compute budget at full res. Per CLAUDE.md's "TLDR distilled 8-step" rule, A was designed for single-pass throughput across many loop iterations; B trades per-iter wall time for per-iter quality.

### 3. Multi-keyframe anchoring — **(Q)**

A: a single `LTXVAddLatentGuide #1519` inside the loop subgraph anchors the init image at `latent_idx=-1` (last latent frame) with strength from `FloatConstant #1269` (default `1.0`). The image enters via `LTXVImgToVideoInplaceKJ #531` at frame 0 + this guide at the end.

B: two `LTXVAddGuideMulti` nodes, one per pass.
- `LTXVAddGuideMulti #2221` (pass 1): N=3 keyframes (`['3', 0, 0.7, 0, 0.25, -1, 1]` widget pattern: kf_count=3, idx_0=0 strength=0.7, idx_1=0 strength=0.25 (mid), idx_2=-1 strength=1). Three image inputs wired in.
- `LTXVAddGuideMulti #2182` (pass 2): N=2 keyframes for refine (`['2', 0, 1, -1, 1]`).

Three distinct keyframe images are loaded (`LoadImage #45`, `#47`, `#2172`) and preprocessed individually.

Hypothesis: the middle-keyframe anchor reduces identity drift and trajectory wandering across the ~97 latent-frame window. The ablation plan's Phase 1 directly attributes this — `var_a` drops the middle keyframe, `var_b` drops both middle and last; comparing A↔var_b on the keyframe axis isolates this lever.

### 4. Resolution + dimensions — **(Q, P)**

| Axis | A | B |
|---|---|---|
| Final width × height | 832 × 448 (`LTXFramePlanner #1634`) | 960 × 544 (`ImageResizeKJv2 #44`) |
| Frame count (pixel-frames) | 81 (LTXFramePlanner sets `frames` from latent count) | 97 (`EmptyLTXVLatentVideo length=97`) |
| Frame rate | 25 fps | 25 fps |
| Single-source-of-truth | `LTXFramePlanner` (canonical) | None — independent width/height SetNodes + GetNode lookups |

A uses `LTXFramePlanner` as the canonical dimension SSoT (per CLAUDE.md F8 audit). B does not — it threads width/height through SetNode/GetNode pairs (`#75`/`#220` for width, `#78`/`#219` for height) and computes the pass-1 half-res via `ComfyMathExpression a/2`. **B will fail the `frame_planner_present` audit on F8.**

Hypothesis: resolution itself is unlikely to be the quality driver (both are well above the model's minimum useful res), but it confounds wall-time comparisons. The dimension-SSoT difference matters for any port: B's pattern would need to be replaced with `LTXFramePlanner` to satisfy the audit.

### 5. Attention patches — **(P)** — neutral quality

| Axis | A | B (active stack) |
|---|---|---|
| Sage variant | `AudioLoopHelperSageAttention #268` `['auto', True, 1024]` (was `auto_mask_aware` pre-2026-05-15; unified to `auto` via `scripts/apply_sage_mode_auto.py`) | `AudioLoopHelperSageAttention #2296` `['auto', True, 1024]` |
| `PathchSageAttentionKJ` | absent | present but **bypassed** (mode=4) |
| `LTX2MemoryEfficientSageAttentionPatch` | absent | present but **bypassed** (mode=4) |
| `LTXVChunkFeedForward` | `#504` widgets `[2, 4096]` (active) | `#228` widgets `[2, 4096]` (active) |
| `LTX2AttentionTunerPatch` | `#1523` widgets `['', 1, 1, 1, 1, True]` (active) | `#229` widgets `['', 1, 1, 1, 1, True]` (bypassed, mode=4) |

As of 2026-05-15, `auto` is the unified default across all shipped workflows. Previously, audio-loop workflows shipped on `auto_mask_aware` and benchmark workflows on `auto` — the two-default split had no runtime payoff (the masked self-attn path doesn't fire on audio-loop workflows per root CLAUDE.md's "Pending review" note, so the modes are equivalent there), and `auto` is what benchmark workflows need. **Sage mode is no longer a delta axis between A and B.**

`LTX2AttentionTunerPatch` is active in A but bypassed in B — widgets are identity (`scale=1, gate=1, ...`), so bypassing is a no-op on math; this is housekeeping not behavior.

### 6. Conditioning routing + NAG — **(Q)**

| Axis | A | B |
|---|---|---|
| `LTXVConditioning frame_rate` | `25` | `25` |
| `LTX2_NAG` widgets `[nag_scale, nag_alpha, nag_tau, inplace]` | `[11, 0.25, 2.5, True]` | `[11, 0.25, 2.5, True]` |
| Negative prompt path | `CLIPTextEncode #507` → `LTX2_NAG #508` → `LTX2SamplingPreviewOverride #503` → LoRA → ... → CFGGuider | `CLIPTextEncode #11` → `LTX2_NAG #197` → `LTX2SamplingPreviewOverride #198` → SetNode → LoRA → ... → CFGGuider |
| Negative conditioning at CFG=1 | `ConditioningZeroOut #420` chain wired but inert at cfg=1 (per CLAUDE.md gotcha) | NAG negative routed directly; no `ConditioningZeroOut` |

`nag_scale=11` matches CLAUDE.md's flagged "aggressive" default — both workflows inherit this. Per CLAUDE.md guidance, dial 3–7 if initial render freezes. **Not a delta between A and B**.

**The interaction worth flagging**: A's negative path goes through `ConditioningZeroOut`, which CLAUDE.md notes is wired-correctly but runtime-inert at `cfg=1`. B routes the NAG-decorated negative directly into `CFGGuider.negative` — and uses `_cfg_pp` samplers (see axis 1). At `cfg=1` with `_cfg_pp`, the negative path is no longer inert. **This is the likely mechanism for the quality lift**: B's NAG is doing real work; A's NAG is effectively a no-op.

### 7. i2v init-image path — **(Q)**

| Axis | A | B |
|---|---|---|
| Init image count | 1 (`LoadImage #444`) | 3 (`LoadImage #45`, `#47`, `#2172`) |
| Resize pipeline | `LTXSmartImageResize #445` (832 × 448, top crop, adaptive multi-stage per CLAUDE.md postmortem) | `ImageResizeKJv2 #44` (960 × 544, lanczos, crop center) → `ResizeImagesByLongerEdge #2083` (1536) chain |
| `LTXVPreprocess img_compression` | `18` | `18` (×3) |
| Init-frame writer | `LTXVImgToVideoInplaceKJ #531` (writes encoded init into frame 0; `noise_mask=0` locks it) | None — replaced by multi-keyframe `LTXVAddGuideMulti` |
| `first_frame_guide_strength` | `FloatConstant #1269 = 1.0` (max identity anchor, per CLAUDE.md) | Replaced by per-keyframe widget strengths inside `LTXVAddGuideMulti` (pass 1: `[0.7, 0.25, 1]`; pass 2: `[1, 1]`) |

A uses `LTXSmartImageResize` (the adaptive multi-stage lanczos resizer documented in `internal/analysis/smart_resize_quantization_postmortem.md` (private clone only)). B uses naïve single-stage `ImageResizeKJv2` lanczos. **For >2× downscale ratios, CLAUDE.md flags single-pass lanczos as introducing aliasing that the model reads as motion cues** — but B's longer-edge=1536 followed by 960×544 crop is closer to 1.6× reduction, possibly within safe range. Quality-affecting in theory; magnitude likely small at B's resolutions.

The deeper change is that B replaces the init-frame inplace-write entirely (`LTXVImgToVideoInplaceKJ`) with `LTXVAddGuideMulti`. The CLAUDE.md F2/F3 symmetry rules ("both initial and loop branches share the same `LTXVPreprocess(img_compression=18)`") and the init-image inplace-write are A's identity-stability machinery. B substitutes multi-keyframe guide anchoring for the same job, which is a topologically different solution.

### 8. LoRA — **(Q)** — small delta

| Axis | A | B |
|---|---|---|
| `LoraLoaderModelOnly` (distilled) | `#2014` strength `0.5` — **bypassed** (mode=4) | `#186` strength `0.6` — **bypassed** (mode=4) |
| `Power Lora Loader (rgthree)` | absent | `#2107` present, no active LoRAs configured (empty widget dict) |

Both have the dynamic LoRA bypassed. **Not currently driving the quality delta**, but if the user enables it in B with strength 0.6 vs A with strength 0.5 they'd be a real delta to control for.

### 9. Model + CLIP — neutral

| Axis | A | B |
|---|---|---|
| UNET | `ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors` (default dtype) | same |
| GGUF alt | none | `UnetLoaderGGUF #191` Q4_K_S — **bypassed** (mode=4) |
| CLIP | `gemma_3_12B_it_fpmixed.safetensors` + `ltx-2.3_text_projection_bf16.safetensors` | same (active); GGUF Q2_K alt bypassed |

Identical active model + CLIP. No delta.

### 10. VAE decode — **(P)** — neutral

| Axis | A | B |
|---|---|---|
| Decoder | `LTXVTiledVAEDecode #1604 [1, 1, 1, True, 'auto', 'auto']` (single-tile, 24GB+ fast path) | `LTXVTiledVAEDecode #149 [1, 1, 1, True, 'auto', 'auto']` (same) |

Identical. Not a quality lever.

### 11. Audio path — **(Q)** — sacred per CLAUDE.md

| Axis | A | B |
|---|---|---|
| Audio encode chain | `LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode → SetLatentNoiseMask → LTXVConcatAVLatent` | same shape: `LoadAudio #2297 → TrimAudioDuration #2298 → LTXVAudioVAEEncode #2299 → SetLatentNoiseMask #2301 → LTXVConcatAVLatent` |
| `TrimAudioDuration` | `#567 [0, 600]` (10 min cap), `#601 [0, 10]` | `#2298 [0.0, 4.0]` (4-second clip) |
| `SolidMask` (noise mask source) | `#571 [0, 512, 512]` (value=0, full 512×512) | `#2300 [0, 512, 512]` (identical) |
| `LTXVAudioVideoMask` (Node 606 wiring) | active inside loop subgraph: `audio_start_time = audio_end_time = window_size` (per CLAUDE.md, intentional, keeps audio fixed) | not present — single-iter benchmark doesn't need iteration-aware audio masking |

Audio handling is **identical in shape**. The only delta is `TrimAudioDuration` clip length — B is artificially capped at 4s for benchmarking. This is the **load-bearing reason B is single-iter**: with 4s of audio and ~3.88s of LTX 2.3 latent capacity per pass, you fit exactly one iteration.

### 12. Loop structure — **(Q)** — the topological gap

A:
- `TensorLoopOpen #1539` / `TensorLoopClose #1540` wrap a subgraph (`#843`).
- `AudioLoopPlanner #1560` computes total iterations from full audio length; output 1 → `TensorLoopOpen.iterations_in`.
- `AudioLoopController #1582` derives per-iteration stride/start/end and the `should_stop` flag.
- `TimestampPromptScheduleBatchEncode #1615` pre-encodes the full prompt schedule (CLIP outside loop body, per CLAUDE.md "CLIP must not enter loop body" rule).
- `ConditioningSelectByIteration #1616` / `#2021` pluck the per-iter prompt inside the loop.
- `LoopIterationStamp #1618` and `IterationCleanup` (in subgraph) provide iter-aware stamping and tensor lifecycle.
- `LatentContextExtract` / `LatentOverlapTrim` (in subgraph) handle overlap-aware latent slicing.
- `TrimImageBatchToAudio #2029` / `TrimVideoLatentToAudio #2028` clip final output to audio length.
- `RunIdPrefix #2026` provides per-render folder clustering.

B:
- **None of the above are present at the top level.** B is a single-pass DAG with no `TensorLoopOpen/Close`, no `AudioLoopPlanner`, no `AudioLoopController`, no `TimestampPromptScheduleBatchEncode`, no iteration-aware nodes.
- B's "subgraph 0" is repurposed for prompt enhancement (`TextGenerateLTX2Prompt`, `ComfySwitchNode`), not for loop iteration — completely different role.

**This is THE load-bearing structural difference.** B as shipped cannot process more than ~3.88s of audio. Adapting B into a full-audio workflow means re-introducing every node in A's loop spine.

---

## Hypothesis ranking — what explains the quality gap

Ordered most → least likely to be the primary quality driver. The ablation plan (`internal/design/benchmark_ablation_test_plan.md` (private clone only)) Phase 1 is the empirical confirmation step.

1. **Two-pass refine + spatial upsample (axis 2)** — the largest structural lift. Half-res commit + upsample-refine is a well-known quality pattern; it's the only delta that does meaningful additional compute on the latent. **Predicted dominant.** Test via `var_c` (`fml2v_var_c_single_pass.json`): if `var_c ≈ var_b` and `source >> var_c`, refine is the lever.

2. **CFG-pp sampler interacting with NAG at cfg=1 (axes 1 + 6 together)** — at `cfg=1`, vanilla `euler` ignores the negative path, so A's NAG is effectively decorative; B's `euler_cfg_pp` makes the negative + NAG path live. **Predicted strong secondary.** Hard to ablate cleanly without a separate variant; would need a B-clone with `euler` swapped for `euler_cfg_pp` to isolate.

3. **Three-keyframe anchoring (axis 3)** — middle keyframe reduces identity/trajectory drift. **Predicted meaningful but smaller than 1 + 2.** Test via `var_a` (no middle) and `var_b` (first only): the slope `source → var_a → var_b` measures the keyframe-count contribution.

4. **Multi-image init resize pipeline (axis 7, image-quality side)** — B's three independently-preprocessed init images vs A's single `LTXSmartImageResize`. **Predicted small.** Could matter if B's init images are individually higher quality than A's.

5. **Everything else** — VAE, model, CLIP, audio shape, frame rate, NAG widget values — **identical or no-op-equivalent**. Not contributing to the delta.

The companion migration plan (`adapt_benchmark_to_full_audio_plan.md`) prioritizes the port order against this hypothesis ranking: refine + multi-keyframe first (biggest gain per port-effort unit), CFG-pp sampler swap as a one-line widget change, drift-resistant init only if needed.

---

## What's NOT a quality lever (despite looking like one)

- **`auto` vs `auto_mask_aware` sage mode** — mask path isn't exercised on audio-loop workflows per the "Pending review" note in root CLAUDE.md.
- **`LTX2AttentionTunerPatch` active vs bypassed** — identity widgets in both, math is a no-op.
- **Sigma string differences in pass 2** — pass 2 sigmas are unique to the two-pass refine pattern; comparing them to A's single-pass sigmas is a category error.
- **LoRA strength deltas (0.5 vs 0.6)** — both bypassed; not active.
- **GGUF model loaders in B** — bypassed; not active.

---

## Audit-IDs B will fail

If you ran `scripts/audit_workflows.py` against B today (it's currently excluded from the sweep), expected failures:

- `frame_planner_present` (F8) — B doesn't use `LTXFramePlanner`.
- `iterations_autowired` (F5) — no `AudioLoopPlanner → TensorLoopOpen`.
- `planner_no_stride_input` (F7) — no `AudioLoopPlanner` exists.
- `graph_acyclic` — likely passes (no loop = no cycle risk).
- `alc_seed_legacy_name` (F4), `alc_widget_drift` (F6) — likely pass (no `AudioLoopController` present, so nothing to validate).

These are exactly what the migration plan needs to address.
