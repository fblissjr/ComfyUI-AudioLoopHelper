"""LTX-2.3 Regional Compile (spike): torch.compile per-block FFN only.

Compiles `transformer_blocks[i].ff` via torch.compile across all 48 LTX
blocks, leaving attention paths in eager dispatching to sage's
`optimized_attention_override` hook. This is the canonical PyTorch +
Diffusers pattern for diffusion DiTs (https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/):
compile the static-shape compute modules, exclude attention dispatchers
(sage's pybind kernels graph-break Inductor and produce rtol drift on
torch 2.11 per N5 spike 2026-05-01).

Why this works for LTX-2.3 audio-loop where compile-the-denoiser doesn't:
- FFN has static shape per (block, sequence-length); LTX-2.3 has only
  2 distinct seq lengths across the 6 sampler invocations (init=22932,
  loop iters=23296), so cudagraph_trees caches at most 2 graphs/block
- Sage attention runs eager on the unwrapped attn1/attn2 paths
- NAG cond/uncond both go through the same compiled FFN — same graph,
  no recapture
- Norm operations are inline `comfy.ldm.common_dit.rms_norm()` calls
  in BasicTransformerBlock.forward, NOT submodule attributes — so
  they're not directly compilable via setattr replacement (they would
  be compilable as part of a wrapper-around-forward, but that's a
  bigger surface; FFN-only is the tight spike scope).

Hooks:
- Mutates `model.model.diffusion_model.transformer_blocks[i].ff` in
  place (the underlying model is SHARED across clones — see "Cleanup
  semantics" below). Replaces with `torch.compile(original_ff, mode)`.
- Cleanup callback (CallbacksMP.ON_CLEANUP) restores originals.

Cleanup semantics — IMPORTANT:
- `ModelPatcher.clone()` does a shallow clone; `clone.model.diffusion_model`
  is the SAME object as `original.model.diffusion_model`. So our
  `block.ff = compiled` mutation affects the underlying model that
  every clone references.
- The ON_CLEANUP callback restores originals when the patched clone is
  unloaded. If cleanup is missed (e.g., process crash mid-render),
  next render starts with compiled `block.ff` from the prior session.
  Re-applying this node refreshes; bypassing it leaves stale compiled
  modules. Mitigation: track a sentinel attribute on the model so we
  detect and restore even on stale state.

This is a SPIKE node (`is_experimental=True`). If validation render
shows wins, harden into a proper wrapper-based version that doesn't
mutate shared state.
"""

from __future__ import annotations

import torch

try:
    from comfy_api.latest import io
    from typing_extensions import override
except ImportError:
    # pytest stubs: importable without ComfyUI runtime. Same pattern as
    # nodes_sage.py and nodes_easycache.py.
    class _Passthrough:
        def __getattr__(self, _name): return _Passthrough()
        def __call__(self, *_args, **_kwargs): return _Passthrough()

    class _IOStub(_Passthrough):
        class ComfyNode: pass

        @staticmethod
        def NodeOutput(*args): return args

    io = _IOStub()  # type: ignore[assignment]

    def override(fn):  # type: ignore[no-redef]
        return fn

try:
    from comfy.patcher_extension import CallbacksMP
    _ON_CLEANUP = CallbacksMP.ON_CLEANUP
except ImportError:
    _ON_CLEANUP = "on_cleanup"


# Single key so re-applying this node overwrites cleanly rather than
# stacking compiled wrappers.
PATCH_KEY = "ltxv_regional_compile"

# Sentinel attribute name on the FF module to detect already-compiled state.
_SENTINEL_ATTR = "_audioloophelper_regional_compile_orig"


def _patch_blocks(diffusion_model: torch.nn.Module, mode: str) -> dict[int, torch.nn.Module]:
    """Replace `block.ff` with `torch.compile(block.ff, mode=mode)` on
    every transformer block. Returns `{block_idx: original_ff}` so we
    can restore on cleanup. Idempotent: if `_SENTINEL_ATTR` is already
    set on a block.ff, we restore-then-recompile to refresh.
    """
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError(
            "diffusion_model has no `transformer_blocks` attribute; "
            "this node targets LTX-2.3-style models only."
        )
    originals: dict[int, torch.nn.Module] = {}
    for i, block in enumerate(blocks):
        ff = getattr(block, "ff", None)
        if ff is None:
            continue
        # If we've already compiled this block (stale state from prior render),
        # restore the original first so we don't double-wrap.
        if hasattr(ff, _SENTINEL_ATTR):
            ff = getattr(ff, _SENTINEL_ATTR)
            block.ff = ff
        originals[i] = ff
        compiled = torch.compile(ff, mode=mode)
        # Stash the original on the compiled wrapper so future re-applies
        # can find it even if our state dict is lost.
        setattr(compiled, _SENTINEL_ATTR, ff)
        block.ff = compiled
    return originals


def _restore_blocks(diffusion_model: torch.nn.Module, originals: dict[int, torch.nn.Module]) -> None:
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        return
    for i, original in originals.items():
        if i < len(blocks):
            blocks[i].ff = original
    originals.clear()


class LTXVideoRegionalCompile(io.ComfyNode):
    """torch.compile per-block FFN on the LTX-2.3 transformer.

    Compiles `transformer_blocks[i].ff` via `torch.compile` across all
    blocks. Leaves attention paths in eager dispatching to sage's
    `optimized_attention_override` hook (sage's pybind kernels would
    graph-break Inductor — N5 spike 2026-05-01 confirmed rtol drift).

    Connect AFTER the LTX checkpoint loader and BEFORE the next patch
    (typically `AudioLoopHelperSageAttention` or KJ patches). Order
    relative to sage doesn't matter (sage hooks attention, this hooks
    FFN — orthogonal surfaces).

    Mode trade-offs:
    - "default" — kernel fusion only, no graph capture. Lowest VRAM
      delta, conservative wins (~5-10% e2e estimate).
    - "reduce-overhead" — also enables `cudagraph_trees` per-block
      per-shape graph cache. Bigger wins (~13-25% e2e estimate per
      research) but ~1-2 GB extra VRAM for the graph pool. Recommended
      for 24GB cards once "default" is validated; risk of OOM if total
      stack pushes against the limit.

    EXPERIMENTAL: this node mutates shared model state in place
    (clone.model.diffusion_model is shared across patcher clones).
    Cleanup restores originals on unload. If a crash leaves stale
    compiled state, re-applying this node refreshes it via the
    `_SENTINEL_ATTR` detection.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVideoRegionalCompile",
            display_name="LTX Video Regional Compile (FFN only, experimental)",
            category="AudioLoopHelper/experimental",
            description=(
                "torch.compile per-block FFN on the LTX denoiser. Skips "
                "attention (sage path remains eager). Reduces kernel-launch "
                "overhead (the 42% bucket per 2026-05-01 bench). Mode default "
                "= fuse only; reduce-overhead = also cudagraph_trees (faster "
                "+ more VRAM)."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode",
                    options=["default", "reduce-overhead"],
                    default="default",
                    tooltip=(
                        "default = kernel fusion only (~5-10% e2e). "
                        "reduce-overhead = also cudagraph_trees per-shape "
                        "(~13-25% e2e, +1-2GB VRAM). Validate default first."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    @override
    def execute(cls, model, mode) -> io.NodeOutput:  # type: ignore[override]
        return io.NodeOutput(cls._patch_impl(model, mode=mode))

    @classmethod
    def _patch_impl(cls, model, *, mode: str):
        """Testable seam. Returns the patched clone."""
        clone = model.clone()
        diffusion_model = clone.model.diffusion_model
        originals = _patch_blocks(diffusion_model, mode)
        n_patched = len(originals)
        print(
            f"[RegionalCompile] patched {n_patched} transformer_blocks[i].ff "
            f"with torch.compile(mode='{mode}')"
        )

        def _cleanup(*_args, **_kwargs):
            _restore_blocks(diffusion_model, originals)
            print(f"[RegionalCompile] restored {n_patched} originals on cleanup")

        if hasattr(clone, "add_callback_with_key"):
            clone.add_callback_with_key(_ON_CLEANUP, PATCH_KEY, _cleanup)
        else:
            # Fallback path for older ModelPatcher revs / test stubs
            try:
                clone.add_callback(_ON_CLEANUP, _cleanup)
            except AttributeError:
                pass  # test environment without the API; cleanup runs on next manual call

        return clone


def comfy_entrypoint() -> dict:
    """Allow ComfyUI to discover this node module independently for spike use."""
    return {"node_list": [LTXVideoRegionalCompile]}
