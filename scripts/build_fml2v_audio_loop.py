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
TEMPLATES_PATH = Path(__file__).parent / "_node_templates_fml2v.json"

# Cached on first load; templates are static JSON shapes copied from canonical workflows.
_TEMPLATES: dict[str, dict] | None = None


def _templates() -> dict[str, dict]:
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = json.loads(TEMPLATES_PATH.read_text())
    return _TEMPLATES


def _add_from_template(
    ed: WorkflowEditor,
    type_name: str,
    pos: tuple[int, int],
    *,
    widget_values: list | None = None,
    title: str | None = None,
    size: tuple[int, int] = (270, 100),
    mode: int = 0,
) -> int:
    """Insert a new top-level node from the saved template JSON.

    Slot dicts copy from template, but every input ``link`` is reset to None
    and every output ``links`` to an empty list — caller wires after.
    """
    tmpl = _templates()[type_name]
    node_id = ed.next_node_id()
    inputs = []
    for inp in tmpl.get("inputs", []):
        inp_copy = {k: v for k, v in inp.items() if k != "link"}
        inp_copy["link"] = None
        inputs.append(inp_copy)
    outputs = []
    for out in tmpl.get("outputs", []):
        out_copy = {k: v for k, v in out.items() if k != "links"}
        out_copy["links"] = []
        outputs.append(out_copy)
    node = {
        "id": node_id,
        "type": type_name,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": 0,
        "mode": mode,
        "inputs": inputs,
        "outputs": outputs,
        "properties": dict(tmpl.get("properties", {})),
        "widgets_values": list(widget_values) if widget_values is not None else (
            list(tmpl.get("widgets_values") or [])
        ),
    }
    if title:
        node["title"] = title
    ed.add_node(node)
    return node_id


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
    """Phase 2: add loop-math infrastructure + audio pre-encode + image preprocess.

    Nodes added at top level (not yet wired to consumers — Phase 5 wires them
    into the loop body):

    - LTXFramePlanner (fps=25, target_seconds=19.88, w=960, h=544) — dim SSoT
    - AudioLoopController (current_iteration wires from TLO in Phase 5)
    - AudioLoopPlanner (iter-INDEPENDENT outputs feed TLO + schedule encoder)
    - FloatConstant ``overlap_seconds=2.0`` (shared between controller + planner)
    - FloatConstant ``first_frame_guide_strength=0.7`` (per-iter trailing anchor)
    - LoadAudio + TrimAudioDuration + LTXVAudioVAEEncode (full-song pre-encode)
    - LTXSmartImageResize (replaces benchmark's ImageResizeKJv2, fixes quantization
      aliasing per smart_resize_quantization_postmortem)
    - LTXVPreprocess(img_compression=18) (canonical F2/F3 init-image preprocess)

    Also fixes ``PrimitiveFloat #2076`` (fps global) widget 24 -> 25 to match
    the canonical inference fps.
    """
    log = (lambda *a: print("  ", *a)) if verbose else (lambda *a: None)

    # --- Loop math infrastructure ---
    fp_id = _add_from_template(
        ed, "LTXFramePlanner", (-3000, 3000),
        widget_values=[960, 544, 19.88, 25],
        title="LTXFramePlanner (dim SSoT)",
        size=(270, 250),
    )
    log(f"+ LTXFramePlanner #{fp_id}  [w=960, h=544, target_seconds=19.88, fps=25]")

    overlap_id = _add_from_template(
        ed, "FloatConstant", (-3000, 3300),
        widget_values=[2.0],
        title="overlap_seconds",
    )
    log(f"+ FloatConstant #{overlap_id}  [overlap_seconds=2.0]")

    init_strength_id = _add_from_template(
        ed, "FloatConstant", (-3000, 3400),
        widget_values=[0.7],
        title="first_frame_guide_strength",
    )
    log(f"+ FloatConstant #{init_strength_id}  [first_frame_guide_strength=0.7]")

    alc_id = _add_from_template(
        ed, "AudioLoopController", (-2700, 3000),
        # widgets: [current_iteration, window_seconds, overlap_seconds, base_seed, fps]
        widget_values=[1, 19.88, 2.0, 42, 25],
        title="AudioLoopController (iter-DEPENDENT outputs)",
        size=(270, 200),
    )
    log(f"+ AudioLoopController #{alc_id}  [window=19.88, overlap=2.0, seed=42, fps=25]")

    alp_id = _add_from_template(
        ed, "AudioLoopPlanner", (-2700, 3250),
        # widgets: [window_seconds, overlap_seconds, fps, max_iterations, schedule]
        widget_values=[19.88, 2.0, 25, 0, ""],
        title="AudioLoopPlanner (iter-INDEPENDENT outputs)",
        size=(270, 200),
    )
    log(f"+ AudioLoopPlanner #{alp_id}  [window=19.88, overlap=2.0, fps=25]")

    # --- Audio path: full-song pre-encode (replaces benchmark's hardcoded 4s) ---
    load_audio_id = _add_from_template(
        ed, "LoadAudio", (-3000, 3700),
        widget_values=["your_audio.mp3"],
        title="LoadAudio (full song)",
        size=(270, 100),
    )
    log(f"+ LoadAudio #{load_audio_id}")

    trim_id = _add_from_template(
        ed, "TrimAudioDuration", (-2700, 3700),
        widget_values=[0, 600],  # full song, capped at 10 min
        title="TrimAudioDuration (full-song clamp)",
        size=(270, 80),
    )
    log(f"+ TrimAudioDuration #{trim_id}  [start=0, duration=600]")

    audio_vae_id = _add_from_template(
        ed, "LTXVAudioVAEEncode", (-2400, 3700),
        widget_values=[],
        title="LTXVAudioVAEEncode (full-song)",
        size=(270, 80),
    )
    log(f"+ LTXVAudioVAEEncode #{audio_vae_id}")

    # --- Image preprocess: replace ImageResizeKJv2 with LTXSmartImageResize ---
    smart_resize_id = _add_from_template(
        ed, "LTXSmartImageResize", (-3000, 3900),
        # widgets: [target_width, target_height, keep_proportion, crop_position]
        widget_values=[960, 544, True, "top"],
        title="LTXSmartImageResize (multi-stage, anti-alias)",
        size=(270, 150),
    )
    log(f"+ LTXSmartImageResize #{smart_resize_id}")

    preprocess_id = _add_from_template(
        ed, "LTXVPreprocess", (-2700, 3900),
        widget_values=[18],
        title="LTXVPreprocess (img_compression=18)",
        size=(270, 80),
    )
    log(f"+ LTXVPreprocess #{preprocess_id}")

    # --- Fix benchmark fps global (PrimitiveFloat #2076 widget 24 -> 25) ---
    for n in ed.wf["nodes"]:
        if n.get("id") == 2076 and n.get("type") == "PrimitiveFloat":
            wv = n.get("widgets_values") or []
            if wv and wv[0] != 25:
                old = wv[0]
                n["widgets_values"][0] = 25
                log(f"= PrimitiveFloat #2076 (Set_fps source): {old} -> 25")
            break

    # Stash the new node IDs in the workflow's `properties` so later phases /
    # next-session Claude can find them without re-grepping by title.
    ed.wf.setdefault("properties", {})["build_fml2v_phase2"] = {
        "frame_planner": fp_id,
        "overlap_seconds": overlap_id,
        "first_frame_guide_strength": init_strength_id,
        "audio_loop_controller": alc_id,
        "audio_loop_planner": alp_id,
        "load_audio": load_audio_id,
        "trim_audio": trim_id,
        "audio_vae_encode": audio_vae_id,
        "smart_resize": smart_resize_id,
        "preprocess": preprocess_id,
    }
    log(f"= Stashed Phase 2 node IDs in wf['properties']['build_fml2v_phase2']")


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
