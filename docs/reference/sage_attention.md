Last updated: 2026-04-23 (mask-aware routing shipped as default)

# AudioLoopHelperSageAttention -- Reference

## What it is

A sage-attention patch node, first-party to AudioLoopHelper. Drop-in
alternative to KJNodes' `PathchSageAttentionKJ` (node 268 in the
shipping example workflows). Same patch surface, three added properties
the KJ node lacks:

1. **Try/except pytorch fallback** on any sage exception. Never crashes
   a multi-hour iteration-loop run because of one bad shape.
2. **`CallbacksMP.ON_CLEANUP` handler** so the override is removed from
   the model clone on unload instead of persisting silently.
3. **Opt-in JSONL telemetry** gated by an env var. Per-call records
   (shape, mask presence, mode, fallback flag, elapsed_us, iteration
   stamp) for post-hoc forensic analysis. Zero overhead when unset.

## Source files

- Node + helpers: `nodes_sage.py` (~420 lines, single flat file)
- Tests: `tests/test_sage_node.py` (15 tests, GPU-free via `FakeModel`)
- Registered in: `nodes.py::AudioLoopHelperExtension.get_node_list`

## Parameters

### model (MODEL)

The model to patch. The node calls `model.clone()` first, so the
input model is never mutated.

### mode (Combo)

Default: `"auto_mask_aware"`. The combo is **arch-filtered at
node-import time** via `sageattention.core.get_cuda_arch_versions()` —
you only see modes that can actually run on the detected GPU. No
Blackwell-only options on Ada, no sm90 options on Ampere.

Arch mappings in `nodes_sage.py::_MODES_BY_ARCH`:

| Detected arch | Options shown (in combo order) |
|---|---|
| sm80 / sm86 / sm87 (Ampere) | disabled, **auto_mask_aware**, auto, `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp16_triton` |
| **sm89 (Ada / RTX 40xx)** | disabled, **auto_mask_aware**, auto, `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda++`, `sageattn_qk_int8_pv_fp16_triton` |
| sm90 (Hopper) | disabled, **auto_mask_aware**, auto, `sageattn_qk_int8_pv_fp8_cuda_sm90`, `sageattn_qk_int8_pv_fp16_triton` |
| sm100 / sm120 / sm121 (Blackwell) | disabled, **auto_mask_aware**, auto, `sageattn3`, `sageattn3_per_block_mean`, `sageattn_qk_int8_pv_fp16_triton` |
| unknown / sage not importable | disabled, **auto_mask_aware**, auto, `sageattn_qk_int8_pv_fp16_triton` |

Semantics:

- **`auto_mask_aware`** (default) — per-call routing: masked paths
  (LTX cross-attn carries a text-padding mask) dispatch to
  `sageattn_qk_int8_pv_fp16_triton`; unmasked paths (self-attn) delegate
  to sage's `auto`. Correct default because sage's INT8-QK-FP8/FP16-PV
  CUDA kernels do not implement mask support — `MaskMode` is
  `{kNone, kCausal}` only, and `attn_mask` passed via kwargs is silently
  dropped. Padded positions then contribute as if unmasked, contaminating
  the attention. The LTX-shape sweep measures rtol 0.26–0.94 vs SDPA
  across seq_kv = 32–1024 (more padding → worse rtol, as expected from
  "mask ignored"). Triton is the only sage kernel that implements
  masked attention (rtol ≈ 0.039 across the same range). Stateless
  per-call decision: no closure caches, no offload-survival risk beyond
  the base override. Full characterization:
  `internal/design/sage_backlog.md` item 2.
- **`disabled`** — no-op, returns the input model unchanged.
- **`auto`** — calls `sageattention.sageattn()` and lets sage's own
  dispatch pick the best kernel. On sm89 + CUDA >= 12.8 this lands on
  `sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`
  (SageAttention2++). **Not recommended for LTX workflows that hit
  masked cross-attn** — use `auto_mask_aware` instead. Kept for
  non-LTX workflows and for A/B comparison.
- **`sageattn_qk_int8_pv_fp16_cuda`** — INT8 QK + FP16 PV, fp32
  accumulator. No mask support (silently dropped). Safe for self-attn
  only; do not feed masked cross-attn calls to this mode.
- **`sageattn_qk_int8_pv_fp8_cuda++`** — INT8 QK + FP8 PV, fp32+fp16
  accumulator (SageAttention2++). Fastest on Ada for self-attn. Also
  no mask support.
- **`sageattn_qk_int8_pv_fp16_triton`** — JIT Triton; always
  available. Only mask-clean kernel on any arch. Small (~100ms–1s)
  JIT-compile cost on first use of a new shape visible as
  `elapsed_us` spikes in the trace; warm-shape performance is 2.4–3.9×
  SDPA depending on shape.
- **`sageattn3*`** — Blackwell-only. Not shown on Ada.

### fallback_on_error (BOOLEAN, default True)

When True, if the sage kernel raises (unsupported shape, dtype, mask
format, CUDA error, etc.), the override falls back to
`attention_pytorch` for that single call and logs the failure. A
shape-dedup'd log ensures long runs emit one line per distinct
`(shape, mode, error-class)` tuple, not per invocation.

When False, sage exceptions propagate and the workflow crashes
immediately. Useful for debugging — you want the raw stack trace.

## How it patches the model

The node writes to
`model_clone.model_options["transformer_options"]["optimized_attention_override"]`
— the supported ComfyUI hook consumed by `wrap_attn` in
`comfy/ldm/modules/attention.py:125-141`. This is NOT a torch
monkey-patch. The override wins over any global
`optimized_attention` set by `--use-sage-attention`, but it only
applies to models that pass through this node.

**Stacking with other KJNodes LTX-2 patches.** The shipping workflows
stack sage (node 268) → `LTXVChunkFeedForward` → `LTX2AttentionTunerPatch`
→ `LTX2_NAG` → preview override. These all attach to different
surfaces and compose cleanly. See
`internal/analysis/sage_attention_analysis.md` for the detailed
patch-chain analysis.

## Telemetry

Sage-specific notes only here. Full privacy/transparency reference
(what's captured, where, retention, on/off, the end-to-end workflow with
order of operations, and the related exec logger):
[`telemetry_and_tracing.md`](telemetry_and_tracing.md).

In short: set `AUDIOLOOPHELPER_SAGE_TRACE=auto` (or an explicit path)
**before launching ComfyUI**; per-call records appear under
`internal/analysis/runs/sage/sage_<timestamp>.jsonl`. The sage tracer
captures tensor shapes, kernel mode, and timing only — never prompt text.

The `mode` / `effective_mode` distinction is the sage-routing-specific
detail worth flagging here: `mode` is the configured widget value;
`effective_mode` is the kernel that actually dispatched. For
non-routing modes they're identical. For `auto_mask_aware` they
diverge — `effective_mode=sageattn_qk_int8_pv_fp16_triton` on masked
calls, `effective_mode=auto` on unmasked. A trace where
`mode=auto_mask_aware` but `effective_mode` never changes means the
routing isn't firing — file a bug.

`iter` is pulled from `transformer_options.get("iteration")` with a
fallback to `.get("step")`. The stamp comes from `LoopIterationStamp`
(see `nodes.py`), auto-inserted into shipping workflows by
`scripts/apply_iteration_stamp.py`. For workflows without the stamp
(e.g. a custom build), `iter` falls back to sampler step, so traces are
still groupable — just by step, not loop pass.

### When to use it

- **Verifying the override actually runs.** A clean run with the node
  active produces N non-null rows; a run with no rows means the
  override wasn't hit.
- **Checking for silent disengagement across iterations.** If per-iter
  call counts drop to 0 after iteration 0, the override is being
  dropped by model offload (NAG-asymmetry sibling risk). With
  `LoopIterationStamp` wired, group trace rows by `iter` and confirm
  counts are stable across iterations.
- **Measuring fallback rate.** `fallback_count / total_calls` tells
  you how much of your run is quietly running on pytorch SDPA. Non-zero
  indicates shapes sage can't handle — worth investigating.
- **Shape distribution for sizing micro-bench work.** `distinct_shapes`
  tells you how many unique shape tuples the model uses (usually
  small: self-attn + cross-attn with a handful of resolutions).

### Overhead

Zero when unset. When enabled, one syscall per attention call
(`buffering=1` line-buffered writes) — forensic-only, not
production-grade. ~22k syscalls per 5-iter LTX run.

## Differences from `PathchSageAttentionKJ`

| Property | `PathchSageAttentionKJ` | `AudioLoopHelperSageAttention` |
|---|---|---|
| Patch surface | `transformer_options["optimized_attention_override"]` | same |
| Mode combo | fixed 8-mode list including Blackwell modes | arch-filtered |
| On sage exception | crashes the run | falls back to `attention_pytorch` (dedup'd log) |
| On model unload | override persists | `CallbacksMP.ON_CLEANUP` pops it |
| Per-call telemetry | none | opt-in JSONL |
| `allow_compile` widget | yes | no (deferred; sage's torch.compile support is still thin) |

Both nodes are safe to keep installed. You can A/B by wiring the new
node into your workflow in parallel, bypassing node 268, running, then
flipping back if anything regresses.

## Related

- `internal/analysis/sage_attention_analysis.md` — patch-chain
  analysis with file:line references for the 5-node LTX-2 patch
  chain, why sage + tuner compose correctly, where LTX-Video actually
  calls attention, and the offload-asymmetry sibling risk.
- `internal/design/sage_backlog.md` — 8 deferred items with
  measurement gates. Top items: accuracy baselining (PSNR/SSIM/LPIPS
  fp8++ vs SDPA), mask-aware mode split (fp16 for cross-attn where
  masks live, fp8++ for self-attn), offload-asymmetry verification.
- `<sage_fork_repo>/README.md` — the sage build used by this node.
  Fork of `woct0rdho/SageAttention` (which forks `thu-ml/SageAttention`).
  Includes a local `build.sh` hardened to install into an explicit
  `VIRTUAL_ENV`. **We own this fork**, so routing policy here and
  future kernel-side work (e.g. adding masked-attention support to the
  CUDA kernels) can be co-optimized rather than blocked on upstream
  review. Measurement artifact:
  `<sage_fork_repo>/tests/test_sageattn_ltx_shapes.py` (the seq_kv sweep
  + synthetic-wide-V shape characterizing the CUDA kernels' dropped-mask
  behavior vs triton's honored-mask path). Sage-fork CHANGELOG's "Open
  work" section carries the mask-support feature-add triage.
