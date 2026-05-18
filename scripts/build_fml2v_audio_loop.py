"""Build the flat-canvas audio-loop variant of fml2v_var_d.

Source: ``example_workflows/benchmark_workflows/fml2v_var_d_audio_input.json``
        (the two-pass single-render benchmark)
Output: ``example_workflows/experimental/fml2v_var_d_audio_loop.json``
        (flat-canvas tensor-loop variant; subgraph-free)

Design doc: ``example_workflows/working_docs/fml2v_audio_loop_v1_design.md``

Build runs in phases:
    1. Strip / bypass benchmark structures we don't need.
    2. Add canonical loop math + globals.
    3. Conditioning path.
    4. Initial render path.
    5. Loop body (flat canvas, between TLO/TLC).
    6. Output assembly + LoopConfigValidator.

Each phase is idempotent on its own outputs — re-running rebuilds from the
benchmark source.

Usage::

    uv run --group dev python scripts/build_fml2v_audio_loop.py
    uv run --group dev python scripts/build_fml2v_audio_loop.py --dry-run
    uv run --group dev python scripts/build_fml2v_audio_loop.py --revert
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "example_workflows" / "benchmark_workflows" / "fml2v_var_d_audio_input.json"
OUTPUT = REPO_ROOT / "example_workflows" / "experimental" / "fml2v_var_d_audio_loop.json"


# ---------------------------------------------------------------------------
# Phase 1: strip / bypass benchmark structures we don't need
# ---------------------------------------------------------------------------

# Pass2 sampler chain + upscaler — BYPASS (mode=4), not strip (user kept these
# for future re-enable into Option C per-iter or Option B deferred workflow).
BYPASS_PASS2 = [
    4,     # KSamplerSelect euler_cfg_pp (refine)
    8,     # CFGGuider (refine)
    14,    # RandomNoise (refine)
    21,    # SamplerCustomAdvanced (refine)
    25,    # LTXVLatentUpsampler (2x spatial)
    34,    # LTXVConcatAVLatent (pre-refine)
    146,   # LTXVSeparateAVLatent (post-refine)
    150,   # LTXVAudioVAEDecode (refine audio out)
    149,   # LTXVTiledVAEDecode (refine video out — pass2 decode)
    216,   # ManualSigmas (refine tail)
    2156,  # LTXVCropGuides (post-refine)
    2182,  # LTXVAddGuideMulti (refine-input)
    2222,  # LTXVCropGuides (pre-upscale)
    182,   # LatentUpscaleModelLoader (kept loaded, just no consumer)
]

# Orphan / dead nodes — STRIP entirely
STRIP_ORPHANS = [
    2,     # LTXVScheduler (links empty in benchmark)
    5,     # ManualSigmas (links empty)
    9,     # LTXVEmptyLatentAudio (mode=4 bypassed; replaced by pre-encode)
]

# Bypassed sage variants we don't use (we use AudioLoopHelperSageAttention)
STRIP_SAGE_VARIANTS = [
    226,   # PathchSageAttentionKJ (bypassed)
    227,   # LTX2MemoryEfficientSageAttentionPatch (bypassed)
]

# Hardcoded 4-second audio chain (replaced by full-song pre-encode + AudioLatentSlice)
STRIP_HARDCODED_AUDIO = [
    2297,  # LoadAudio "your_audio.mp3"
    2298,  # TrimAudioDuration [0, 4.0]
    2299,  # LTXVAudioVAEEncode (single-window)
    2300,  # SolidMask (audio freeze)
    2301,  # SetLatentNoiseMask
    2215,  # SetNode "latent_audio"
]

# LTX2SamplingPreviewOverride — drop (preview during iter is slow + adds VRAM)
STRIP_PREVIEW_OVERRIDE = [
    198,   # LTX2SamplingPreviewOverride
]

# Top-level LTX2_NAG — drop (NAG moves inside loop body downstream of LoopIterationStamp)
STRIP_TOP_LEVEL_NAG = [
    197,   # LTX2_NAG (top-level)
]

# Static positive CLIPTextEncode — drop from positive flow (replaced by schedule
# encoder + bypassed parallel CLIPTextEncode added in Phase 3)
STRIP_STATIC_POSITIVE = [
    16,    # CLIPTextEncode positive
]

# SetNode namespace collapses — keep these for now, may strip later if unused
# - Set_model_with_lora #230 (kept as alias for Set_model since LoRA bypassed)
# - Set_model_nag #199 (NAG moves inside loop; drop this bus)
STRIP_NAMESPACE = [
    199,   # Set_model_nag (NAG moved inside loop body — bus no longer needed)
]


def _set_mode(wf: dict, node_id: int, mode: int) -> bool:
    """Set node.mode. Returns True if node found + updated."""
    for n in wf.get("nodes", []):
        if n.get("id") == node_id:
            n["mode"] = mode
            return True
    return False


def _strip_node_and_links(ed: WorkflowEditor, node_id: int) -> bool:
    """Remove a node + all its inbound + outbound links. Idempotent."""
    if not ed.has_node(node_id):
        return False
    ed.remove_node_and_links(node_id)
    return True


def phase1_strip_and_bypass(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 1: strip orphans + bypassed-sage + hardcoded audio + top-level NAG.
    Bypass pass2 sampler chain + upscaler (kept for re-enable later)."""

    wf = ed.wf
    log = (lambda *a: print("  ", *a)) if verbose else (lambda *a: None)

    print("[Phase 1] Bypassing pass2 sampler chain + spatial upscaler")
    for nid in BYPASS_PASS2:
        if _set_mode(wf, nid, 4):
            log(f"BYPASS #{nid}")

    print("[Phase 1] Stripping orphan / dead nodes")
    for nid in STRIP_ORPHANS:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")

    print("[Phase 1] Stripping bypassed sage variants we don't use")
    for nid in STRIP_SAGE_VARIANTS:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")

    print("[Phase 1] Stripping hardcoded 4s audio chain")
    for nid in STRIP_HARDCODED_AUDIO:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")

    print("[Phase 1] Stripping LTX2SamplingPreviewOverride")
    for nid in STRIP_PREVIEW_OVERRIDE:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")

    print("[Phase 1] Stripping top-level LTX2_NAG (moves inside loop body)")
    for nid in STRIP_TOP_LEVEL_NAG:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")

    print("[Phase 1] Stripping static positive CLIPTextEncode #16")
    for nid in STRIP_STATIC_POSITIVE:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")

    print("[Phase 1] Stripping Set_model_nag namespace bus")
    for nid in STRIP_NAMESPACE:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")


def phase2_loop_math_and_audio(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 2: add LTXFramePlanner + AudioLoopController + AudioLoopPlanner +
    FloatConstants. Fix fps=25 globally. Replace audio chain with full-song
    pre-encode. Swap ImageResizeKJv2 -> LTXSmartImageResize. Add LTXVPreprocess.

    TODO: implement.
    """
    print("[Phase 2] (not yet implemented)")


def phase3_conditioning(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 3: TimestampPromptScheduleBatchEncode + bypassed parallel CLIPTextEncode
    + nag_cond_video CLIPTextEncode + two ConditioningSelectByIteration.

    TODO: implement.
    """
    print("[Phase 3] (not yet implemented)")


def phase4_initial_render(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 4: EmptyLTXVLatentVideo + LTXVImgToVideoInplaceKJ + LTXVAddGuideMulti
    (multi-frame) + LTXVConcatAVLatent + init CFGGuider + RandomNoise + sampler
    -> Set_initial_latent + Set_reference_latent.

    TODO: implement.
    """
    print("[Phase 4] (not yet implemented)")


def phase5_loop_body(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 5: TensorLoopOpen + flat-canvas loop body + TensorLoopClose.

    Loop body model chain (downstream of LoopIterationStamp):
        Get_model -> LoopIterationStamp -> LTXVChunkFeedForward
                                       -> LTX2AttentionTunerPatch (bypassed)
                                       -> LTX2_NAG
                                       -> AudioLoopHelperSageAttention
                                       -> CFGGuider

    Loop body data path:
        AudioLatentSlice -> LatentContextExtract -> LatentOverlapTrim
                       -> LTXVAudioVideoMask -> LTXVAddLatentGuide
                       -> LTXVCropGuidesNoLatent (-> CFGGuider)
                       -> LTXVCropGuides (with latent, -> LTXVAdainLatent)
                       -> LTXVConcatAVLatent -> SamplerCustomAdvanced
                       -> LTXVSeparateAVLatent -> LTXVAdainLatent
                       -> IterationCleanup -> TLC.processed

    TODO: implement.
    """
    print("[Phase 5] (not yet implemented)")


def phase6_assembly(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 6: LatentConcat + TrimVideoLatentToAudio + LTXVTiledVAEDecode +
    TrimImageBatchToAudio + RunIdPrefix + VHS_VideoCombine. Add LoopConfigValidator.

    TODO: implement.
    """
    print("[Phase 6] (not yet implemented)")


def build(*, dry_run: bool = False, verbose: bool = True) -> None:
    """Run all phases on a fresh copy of the source workflow."""
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source workflow not found: {SOURCE}")

    # Reset output to a fresh copy of the benchmark source (idempotence).
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE, OUTPUT)
    print(f"Reset {OUTPUT.relative_to(REPO_ROOT)} from benchmark source")

    ed = WorkflowEditor(OUTPUT)

    phase1_strip_and_bypass(ed, verbose=verbose)
    phase2_loop_math_and_audio(ed, verbose=verbose)
    phase3_conditioning(ed, verbose=verbose)
    phase4_initial_render(ed, verbose=verbose)
    phase5_loop_body(ed, verbose=verbose)
    phase6_assembly(ed, verbose=verbose)

    if dry_run:
        print("[dry-run] Not writing output")
        return

    ed.save()
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({OUTPUT.stat().st_size} bytes)")


def revert() -> None:
    """Revert: restore output to a clean copy of the benchmark source."""
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source workflow not found: {SOURCE}")
    shutil.copy(SOURCE, OUTPUT)
    print(f"Reverted {OUTPUT.relative_to(REPO_ROOT)} to benchmark source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build but don't write output")
    parser.add_argument("--revert", action="store_true", help="Restore output to benchmark source")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-node logs")
    args = parser.parse_args()

    if args.revert:
        revert()
        return

    build(dry_run=args.dry_run, verbose=not args.quiet)


if __name__ == "__main__":
    main()
