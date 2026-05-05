# scripts/archive/ — retired apply-script inventory

Last updated: 2026-05-05

Apply scripts kept as design records of the topology each migration introduced. **Not for re-running.** Either:
- the migration is permanently baked into the canonical workflow (re-running is a no-op or destructive), or
- the migration was superseded by a different pattern (re-running would conflict with the current canonical shape), or
- the script is a generator of a shipped variant whose JSON now lives in `example_workflows/` as the source of truth.

Audit-script remediation pointers reference `scripts/archive/...` paths so a reader can still inspect the original migration when an audit-check fails.

## When to consult

- A current audit ERR cites a `scripts/archive/<X>.py` path → read that script + the entry below to understand the topology the audit is enforcing.
- Designing a new migration that touches similar ground → check if a related entry below documents a prior approach (and why it was retired).
- Wondering "did we ever try X?" → grep this file before re-deriving.

---

## Inventory

### `apply_audio_latent_pre_encode.py`

- **Originally**: 2026-05-01. Staged a workflow variant that encodes the full song's audio latent ONCE outside the loop and slices it per-iter in latent space (replacing the per-iter `LTXVAudioVAEEncode` + `TrimAudioDuration` chain inside the subgraph).
- **Why built**: per-iter audio re-encode plus repeated AudioVAE VRAM staging dominated render time. `AudioLatentSlice` (in `nodes_audio_latent_slice.py`) lifted the constraint that drove per-iter encoding.
- **Why archived (2026-05-04)**: the migration is baked into the canonical `audio-loop-music-video_latent.json` as part of the 2026-05-04 consolidation pass. The original input (`audio-loop-music-video_latent_iclora.json`) and staged output paths are no longer in tree.
- **Audit pin**: `audit_workflows.py` cites this archive path on the audio-pre-encode invariant when violated.

### `apply_iclora_video_reference.py`

- **Originally**: 2026-04-29. Wired video-reference IC-LoRA scaffolding into the audio-loop pipeline — `LTXICLoRALoaderModelOnly` on the top-level MODEL chain, `VHS_LoadVideo` + `ImageResizeKJv2` + `LTXVPreprocess(val=18)` for the reference clip, plus per-iter slot in the subgraph for `reference_video`.
- **Why built**: the F2/F3 symmetry rules around init-image preprocessing extend to ref-video preprocessing; the original IC-LoRA work shipped as a staged variant before being baked.
- **Why archived (2026-05-04)**: scaffolding is baked into the canonical (bypassed by default; un-bypass loader + guide + ref-video to enable). The sibling `audio-loop-music-video_latent_iclora.json` is no longer in tree.
- **Audit pin**: `audit_workflows.py` references this archive path on the F12 video-reference IC-LoRA invariant.

### `apply_sage_mode.py`

- **Originally**: pre-2026-04-23. Flipped the `sage_mode` widget on `PathchSageAttentionKJ` (node 268) across all shipping workflows — used to A/B between `auto`, `fp16`, `fp16_triton`, `fp8++`, etc.
- **Why built**: needed a fast switch when benchmarking sage modes against torch_flash; manual widget editing across 5+ workflows was error-prone.
- **Why archived (2026-05-05)**: the target node `PathchSageAttentionKJ` no longer exists in any shipped workflow — `apply_audioloophelper_sage.py` (also archived) swapped it out for `AudioLoopHelperSageAttention` on 2026-04-23. The current attention node manages mode internally (`auto_mask_aware` default); per-mode A/B now happens via the node widget directly or via `apply_skip_under_seq_len.py` for the seq-len threshold.

### `apply_audioloophelper_sage.py`

- **Originally**: 2026-04-23 (commit `7eb24bb`). Swapped node 268 `PathchSageAttentionKJ` → `AudioLoopHelperSageAttention` across the shipping workflows; default mode `auto_mask_aware` (routes masked cross-attn to fp16_triton, keeps fp8++ on self-attn).
- **Why built**: KJ's sage node has no mask-aware mode; LTX cross-attn passes a mask the fp8++ kernel can't handle cleanly. A node swap (rather than parallel wiring) was needed because only one node can own the `transformer_options["optimized_attention_override"]` slot.
- **Why archived (2026-05-05)**: canonical workflow already has `AudioLoopHelperSageAttention` and no `PathchSageAttentionKJ` — verified via `python3 -c "..."` topology check. Re-running the swap is a no-op. The current attention-perf knob shipped via `apply_skip_under_seq_len.py` (still in `scripts/`).
- **Citations to update**: `internal/design/polish_passes_design.md` (private clone only) cites this script as the "in-place-across-all pattern" reference. Citation still works (just points into archive).

### `apply_iteration_stamp.py`

- **Originally**: 2026-04-23 (commit `c46362c`). Inserted `LoopIterationStamp` between the patch-chain MODEL and the loop-body's MODEL input, plus wired `TensorLoopOpen.current_iteration` into the stamp.
- **Why built**: sage tracer needs a per-iteration stamp on `transformer_options["iteration"]` to group per-iter offload-asymmetry measurements. The stamp node is a MODEL passthrough that writes the iteration index without other side effects.
- **Why archived (2026-05-05)**: `LoopIterationStamp` is present in all four `audio-loop-music-video_latent*.json` workflows — verified via topology check. Re-running is idempotent. Node lives in `nodes.py`; tests in `tests/test_iteration_stamp.py` exercise the node directly without going through the apply script.
- **Citations to update**: `internal/design/sage_backlog.md` (private clone only) mentions this script as the prerequisite-machinery installer. Pointer still valid (now into archive).

### `apply_workflow_simplification.py`

- **Originally**: 2026-04-27 (commit `a0a4ab1`). Removed 11 verified-dead/redundant nodes from the latent workflow (notably `#1513 ModelSamplingSD3` after the canonical sigma migration left it orphaned).
- **Why built**: post-sigma-migration the canonical sampling path stopped feeding through `ModelSamplingSD3`; the node was orphaned but not stripped. Same pattern across other dead loaders.
- **Why archived (2026-05-05)**: `ModelSamplingSD3` is absent from the canonical (verified). Re-running is idempotent. Dead-loader stripping continues via `apply_strip_sd3_shift_node.py` and `apply_strip_dead_lora_loaders.py` (both still in `scripts/`).

### `apply_id_lora_runtime.py`

- **Originally**: 2026-04-27 (commit `722b285`). Added `LTXVReferenceAudio` runtime nodes + reference-slice trim for ID-LoRA (paper arxiv:2603.10256, model `AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K`). Three new nodes, all bypassed by default.
- **Why built**: ID-LoRA needs both LoRA weights AND a runtime "reference audio" injection + identity-guidance pass. The weights side shipped via `apply_lora_chain_bypassed.py`; this script added the runtime side. Two `LTXVReferenceAudio` instances were needed because the initial render and loop body have separately-rooted conditioning trees.
- **Why archived (2026-05-05)**: `LTXVReferenceAudio` is present in the canonical (bypassed). Re-running is idempotent. The active ID-LoRA staging path is now `apply_id_lora_initial_render.py` (still in `scripts/`), which handles the simpler MODEL-only LoRA splice without needing the runtime infrastructure for most use cases.

### `apply_lora_chain_bypassed.py`

- **Originally**: 2026-04-27 (commit `925c1ee`). Added a three-loader LoRA chain (ID-LoRA / IC-LoRA / Style-LoRA) to `audio-loop-music-video_latent.json`, all bypassed by default. Splice site: `#503 LTX2SamplingPreviewOverride → ID → IC → Style → #572 SetNode("model")`.
- **Why built**: gave users a single workflow with optional LoRA slots they could enable per-render without edits, with the patched MODEL flowing to both initial render and loop body via the existing Set/Get(654) → LoopIterationStamp(1618) chain.
- **Why archived (2026-05-05)**: **superseded**. The 2026-05-04 consolidation rationalized the chain to two loaders (`Distill LoRA` + `Style or ID LoRA`); IC-LoRA moved to its own `LTXICLoRALoaderModelOnly` chain in the F12 video-reference path. Re-running this script would produce a 3-loader shape inconsistent with the current canonical. Active ID-LoRA staging is `apply_id_lora_initial_render.py` (a simpler single-loader splice for staged variants).

### `patch_scheduling_wiring.py`

- **Originally**: 2026-04-12 (commit `1bc05bd`). Wired `TimestampPromptSchedule` + `ConditioningBlend` into all three audio-loop workflows: added a second `CLIPTextEncode` for next_prompt, `ConditioningBlend` for current/next blend, and rewired the subgraph positive conditioning input through the blend output.
- **Why built**: at the time, the prompt-schedule pattern was per-iter `CLIPTextEncode` + `ConditioningBlend(blend_factor)` driven by `TimestampPromptSchedule`. Solved per-section prompt switching with crossfade.
- **Why archived (2026-05-05)**: **superseded** by the `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` pattern (the current pre-encode-outside-loop pattern in root CLAUDE.md). The `9d71c3d` consolidation removed `ConditioningBlend` from the canonical entirely. Re-running this script would re-add a wiring shape that conflicts with the current pre-encode pattern (and would put CLIP back inside the loop body — explicitly forbidden by the root CLAUDE.md "CLIP must not enter the loop body" rule).

### `apply_perf_improvements.py`

- **Originally**: 2026-04-16 (commit `200e762`). Two in-place mods per workflow: (1) swap in-loop `CLIPTextEncode` → `CachedTextEncode_AudioLoop`, (2) insert `IterationCleanup` node inside subgraph between final `LatentOverlapTrim` and subgraph output.
- **Why built**: "Step 0" perf pass — `CachedTextEncode` cached the encoded conditioning across iterations (avoid re-encoding the same prompt); `IterationCleanup` freed per-iter intermediates to reduce VRAM pressure.
- **Why archived (2026-05-05)**: **superseded** by the BatchEncode pattern. The `9d71c3d` consolidation removed both `CachedTextEncode` and the per-iter `IterationCleanup` placement; the new pattern pre-encodes outside the loop entirely (eliminating the cache-need rationale) and `IterationCleanup` is now wired differently. Re-running this script would conflict with the current canonical. The `IterationCleanup` node itself remains live in `nodes.py`.

### `apply_config_validator.py`

- **Originally**: 2026-04-21 (commit `b46ad97`). Generates `audio-loop-music-video_latent_validator.json` by adding `LoopConfigValidator` + `PreviewAny` to the baseline latent workflow, wired to the same audio/window/length/resolution sources that feed `AudioLoopController` + `EmptyLTXVLatentVideo`. Also converts `AudioLoopController.overlap_seconds` from widget to linked input + creates a shared `FloatConstant` feeding both controller and validator.
- **Why built**: pre-render config diagnostics — catch overlap-seconds-vs-window-seconds inconsistencies, resolution-vs-latent-volume issues, etc. before sampler launch.
- **Why archived (2026-05-05)**: **generator of a shipped variant**. `audio-loop-music-video_latent_validator.json` is the shipped artifact and the source of truth. Global migrations (e.g. `apply_overlap_seconds_single_source.py`, `apply_initial_render_audio_duration_autowire.py`) target the variant directly. The `LoopConfigValidator` node lives in `nodes_validation.py` with tests in `tests/test_config_validator.py`.

### `apply_stg_hybrid_package.py`

- **Originally**: 2026-04-20 (commit `18ee568`). Generates `audio-loop-music-video_latent_stg.json` from the baseline. Keeps the authoritative distilled-1.1 sigma chain; swaps `CFGGuider` for `MultimodalGuider` + two `GuiderParameters` (AUDIO/VIDEO both cfg=2, stg=1, modality=1). Bypasses `LTX2_NAG`.
- **Why built**: STG (spatio-temporal guidance) is the primary quality lift on LTX 2.3. Mild CFG forces the guider's `noise_pred_neg` branch to run (cfg=1.0 hits an unbound-variable bug in KJ's multimodal_guider at `ComfyUI-LTXVideo/guiders/multimodal_guider.py:269`).
- **Why archived (2026-05-05)**: **generator of a shipped variant**. `audio-loop-music-video_latent_stg.json` is the shipped artifact (canonical with `MultimodalGuider` present, verified). Re-running the generator on a baseline that's drifted since 2026-04-20 would not produce the current variant cleanly anyway.

---

## Resurrection procedure

If you need to actually re-run an archived script (e.g. to migrate a new workflow fork that hasn't been baked yet):

1. Read this entry's "Why archived" — does the canonical state still match what the script expects? If not, the script's `require_nodes` pre-flight will refuse and that's correct.
2. Move the script back to `scripts/` (`git mv scripts/archive/<X>.py scripts/`).
3. Adapt input/output paths to current canonical state.
4. Run with `--dry-run` first.
5. Re-add an audit-check pair (per `docs/reference/f_pair_convention.md`) if the migration introduces a new invariant.

## Pointers

- Charter (one-pager): `./README.md`
- Active scripts inventory: `../CLAUDE.md`
- Audit-pair convention: `../../docs/reference/f_pair_convention.md`
- Debug-tool inventory: `../../docs/reference/debug_tools.md`
