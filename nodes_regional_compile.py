"""LTX-2.3 Regional Compile (spike): torch.compile per-block FFN.

Compiles `transformer_blocks[i].ff` across all 48 LTX blocks via
`torch.compile`, leaves attention in eager dispatching to sage's
`optimized_attention_override` hook (sage's pybind kernels graph-break
Inductor — N5 spike 2026-05-01). Targets the 42% launch-overhead bucket
from clean chrome trace 2026-05-01.

SPIKE: mutates shared `transformer_blocks[i].ff` in place (clones share
the underlying model). ON_CLEANUP callback restores originals. Stale
state detected via `isinstance(ff, OptimizedModule)` + the official
`._orig_mod` API. Full rationale + research links in CHANGELOG.

PLACEMENT: insert this node AFTER any node that calls `model.state_dict()`
(notably `LTXICLoRALoaderModelOnly`). Compiled `OptimizedModule` wrappers
in the diffusion model break LoRA-key matching during `state_dict()`
traversal. Order: `UNETLoader → ... → LTXICLoRALoaderModelOnly →
LTXVideoRegionalCompile → SetNode "model"`.
"""

from __future__ import annotations

from typing import Literal

import torch

CompileMode = Literal["default", "reduce-overhead"]

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


def _unwrap_compiled(ff: torch.nn.Module) -> torch.nn.Module:
    """Return the original module if `ff` is a torch.compile wrapper,
    else `ff` unchanged. Uses the official OptimizedModule._orig_mod
    API rather than a custom sentinel attribute (custom attrs on a
    Module wrapper get auto-registered as submodules by
    nn.Module.__setattr__, which created a state_dict cycle in the
    first spike — caught 2026-05-01 against the IC-LoRA loader chain)."""
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        return ff
    if isinstance(ff, OptimizedModule):
        return ff._orig_mod
    return ff


def _patch_blocks(diffusion_model: torch.nn.Module, mode: CompileMode) -> dict[int, torch.nn.Module]:
    """Replace `block.ff` with `torch.compile(block.ff, mode=mode)` on
    every transformer block. Returns `{block_idx: original_ff}` so we
    can restore on cleanup. Idempotent via `_unwrap_compiled`: re-runs
    unwrap-then-recompile to refresh, never double-wraps.
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
        ff = _unwrap_compiled(ff)
        block.ff = ff
        originals[i] = ff
        block.ff = torch.compile(ff, mode=mode)
    return originals


def _restore_blocks(diffusion_model: torch.nn.Module, originals: dict[int, torch.nn.Module]) -> None:
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        return
    for i, original in originals.items():
        blocks[i].ff = original
    originals.clear()


class LTXVideoRegionalCompile(io.ComfyNode):
    """torch.compile per-block FFN on the LTX-2.3 transformer.

    Compiles `transformer_blocks[i].ff` via `torch.compile` across all
    blocks. Leaves attention paths in eager dispatching to sage's
    `optimized_attention_override` hook (sage's pybind kernels would
    graph-break Inductor — N5 spike 2026-05-01 confirmed rtol drift).

    PLACEMENT: insert AFTER any node that calls `model.state_dict()` —
    most importantly, AFTER `LTXICLoRALoaderModelOnly`. Compiled
    `OptimizedModule` wrappers don't preserve the LoRA-key naming
    convention; LoRA loader's `model.state_dict()` walk fails when
    blocks are already compiled. Canonical order:
    `UNETLoader → ... → LTXICLoRALoaderModelOnly → LTXVideoRegionalCompile
    → SetNode "model"`.

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
    Cleanup restores originals on unload. Re-applying detects existing
    `OptimizedModule` wrappers via `._orig_mod` and refreshes cleanly.
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
    def _patch_impl(cls, model, *, mode: CompileMode):
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
