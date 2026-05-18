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
import functools
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


@functools.lru_cache(maxsize=1)
def _templates() -> dict[str, dict]:
    return json.loads(TEMPLATES_PATH.read_text())


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

# Pass2 sampler chain + upscaler — BYPASS (mode=4) in Phase 1.
# Phase 5 (loop body) will rewire these to live inside the loop body per
# Option B (two-pass refine inside loop). Keeping them on canvas with their
# widgets intact means Phase 5 just needs to re-wire connections, not re-add
# the nodes. See design doc "Sampler chain — two-pass inside the loop body".
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


_DEFAULT_POSITIVE_PROMPT = "video of a man dancing and singing"
_DEFAULT_NAG_PROMPT = (
    "still image with no motion, subtitles, deformed facial features, "
    "extra limbs, disfigured hands, duplicate character, twin, clone, microphone"
)


def _find_get_node(ed: WorkflowEditor, name: str) -> int:
    """Return id of the active GetNode whose first widget value is ``name``.

    Raises ValueError if not found — surfacing benchmark restructure loudly
    instead of silently wiring to the wrong slot.
    """
    for n in ed.wf.get("nodes", []):
        if n.get("type") != "GetNode":
            continue
        wv = n.get("widgets_values") or []
        if wv and wv[0] == name:
            return n["id"]
    raise ValueError(f"No GetNode found with widget value {name!r}")


def phase3_conditioning(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 3: top-level conditioning nodes (CLIP must NOT enter loop body).

    Adds:
      - Bypassed parallel ``CLIPTextEncode`` (positive static fallback). User
        can swap into CFGGuider.positive manually for A/B against the schedule.
      - Active ``TimestampPromptScheduleBatchEncode`` — stride/duration sourced
        from ``AudioLoopPlanner`` (NOT ``AudioLoopController``; sourcing from
        the controller closes a cycle through ``TensorLoopOpen``, caught by
        the ``graph_acyclic`` audit).
      - Active ``CLIPTextEncode`` for ``nag_cond_video`` (negative concepts);
        output left floating — Phase 5 wires into in-loop ``LTX2_NAG``.
      - Two ``ConditioningSelectByIteration`` instances:
          * INIT: ``current_iteration`` unwired (defaults to 0; Phase 4 uses it).
          * LOOP: ``current_iteration`` unwired (Phase 5 wires from TLO).

    Wiring this phase does:
      - ``Get_clip #124`` → all 3 CLIP encoders' ``clip`` inputs.
      - ``AudioLoopPlanner #2306`` slot 2 (``stride_seconds``) →
        batch_encoder.stride_seconds.
      - ``AudioLoopPlanner #2306`` slot 3 (``audio_duration``) →
        batch_encoder.audio_duration.
      - batch_encoder.conditioning_list → both selectors' ``conditioning_list``.

    Wiring deferred:
      - Selectors → CFGGuider positives (Phase 4 init, Phase 5 loop body).
      - LOOP selector's ``current_iteration`` ← TLO output (Phase 5).
      - nag_cond_video output → in-loop LTX2_NAG (Phase 5).
    """
    log = (lambda *a: print("  ", *a)) if verbose else (lambda *a: None)

    phase2 = ed.wf.get("properties", {}).get("build_fml2v_phase2", {})
    alp_id = phase2["audio_loop_planner"]
    get_clip_id = _find_get_node(ed, "clip")

    # --- Bypassed static-fallback CLIPTextEncode (positive) ---
    static_pos_id = _add_from_template(
        ed, "CLIPTextEncode", (-2100, 3000),
        widget_values=[_DEFAULT_POSITIVE_PROMPT],
        title="CLIPTextEncode (positive static fallback — bypassed)",
        size=(400, 88),
        mode=4,
    )
    ed.add_link(get_clip_id, 0, static_pos_id, 0, "CLIP")
    log(f"+ CLIPTextEncode #{static_pos_id}  [BYPASSED static positive fallback]")

    # --- Active schedule encoder (stamps frame_rate=25 on every CONDITIONING) ---
    # widgets: [schedule, stride_seconds_default, audio_duration_default,
    #           prefix_with_anchor, frame_rate]
    batch_id = _add_from_template(
        ed, "TimestampPromptScheduleBatchEncode", (-2100, 3200),
        widget_values=[
            f"0:00+: {_DEFAULT_POSITIVE_PROMPT}",
            19.88,   # stride_seconds default (overridden by Planner wire)
            600,     # audio_duration default (overridden by Planner wire)
            True,    # prefix_with_anchor
            25,      # frame_rate (load-bearing — see CLAUDE.md frame_rate=25)
        ],
        title="TimestampPromptScheduleBatchEncode (schedule)",
        size=(400, 200),
    )
    ed.add_link(get_clip_id, 0, batch_id, 0, "CLIP")
    # Slot 2 = stride_seconds, slot 3 = audio_duration on AudioLoopPlanner.
    # Sourcing from AudioLoopPlanner (iter-INDEPENDENT) — NOT Controller (cycle).
    ed.add_link(alp_id, 2, batch_id, 1, "FLOAT")
    ed.add_link(alp_id, 3, batch_id, 2, "FLOAT")
    log(f"+ TimestampPromptScheduleBatchEncode #{batch_id}  [schedule, frame_rate=25]")

    # --- Active nag_cond_video encoder (output floating, Phase 5 consumes) ---
    nag_cond_id = _add_from_template(
        ed, "CLIPTextEncode", (-2100, 3450),
        widget_values=[_DEFAULT_NAG_PROMPT],
        title="CLIPTextEncode (nag_cond_video)",
        size=(400, 88),
    )
    ed.add_link(get_clip_id, 0, nag_cond_id, 0, "CLIP")
    log(f"+ CLIPTextEncode #{nag_cond_id}  [nag_cond_video — output floats until Phase 5]")

    # --- Two ConditioningSelectByIteration (INIT + LOOP). ---
    # INIT: current_iteration defaults to 0 (Phase 4 consumes for initial render).
    # LOOP: current_iteration wired from TLO in Phase 5; meanwhile widget=0.
    sel_init_id = _add_from_template(
        ed, "ConditioningSelectByIteration", (-1700, 3200),
        widget_values=[0],
        title="ConditioningSelectByIteration (INIT — iter=0 default)",
        size=(290, 80),
    )
    ed.add_link(batch_id, 0, sel_init_id, 0, "*")
    log(f"+ ConditioningSelectByIteration #{sel_init_id}  [INIT]")

    sel_loop_id = _add_from_template(
        ed, "ConditioningSelectByIteration", (-1700, 3320),
        widget_values=[0],
        title="ConditioningSelectByIteration (LOOP — wired to TLO in Phase 5)",
        size=(290, 80),
    )
    ed.add_link(batch_id, 0, sel_loop_id, 0, "*")
    log(f"+ ConditioningSelectByIteration #{sel_loop_id}  [LOOP — current_iteration wired in Phase 5]")

    # Stash IDs for Phase 4/5 to find without re-grepping.
    ed.wf.setdefault("properties", {})["build_fml2v_phase3"] = {
        "static_positive": static_pos_id,
        "batch_encoder": batch_id,
        "nag_cond_video": nag_cond_id,
        "selector_init": sel_init_id,
        "selector_loop": sel_loop_id,
    }
    log("= Stashed Phase 3 node IDs in wf['properties']['build_fml2v_phase3']")


# Benchmark node IDs Phase 4 rewires (not stripped — many have other purposes).
_BENCH_EMPTY_LATENT = 32           # EmptyLTXVLatentVideo (pass1)
_BENCH_ADD_GUIDE_MULTI = 2221      # LTXVAddGuideMulti (pass1; 3-frame init)
_BENCH_CONCAT_AV = 24              # LTXVConcatAVLatent (pass1)
_BENCH_LTXV_CONDITIONING = 10      # LTXVConditioning (frame_rate=25)
_BENCH_SEPARATE_AV = 18            # LTXVSeparateAVLatent (pass1)
_BENCH_LOAD_IMAGE_FIRST = 45       # LoadImage "FIRST FRAME"
_BENCH_SET_WIDTH = 2073            # SetNode "width"
_BENCH_SET_HEIGHT = 2072           # SetNode "height"
_BENCH_SET_FRAMES = 2075           # SetNode "frames"
_BENCH_SET_FPS = 2074              # SetNode "fps"
_BENCH_SET_FIRSTFRAME = 75         # SetNode "firstframe"

# Orphaned by Phase 4 dim-SSoT + image-bus rewires; strip after rewiring.
_PHASE4_STRIP_AFTER_REWIRE = [
    2076,  # PrimitiveFloat fps (replaced by FramePlanner.fps_float)
    2077,  # SimpleCalculatorKJ frames calc (replaced by FramePlanner.frames)
    2079,  # INTConstant height (replaced by FramePlanner.height)
    2080,  # INTConstant width  (replaced by FramePlanner.width)
    2083,  # ResizeImagesByLongerEdge (replaced by LTXSmartImageResize chain)
    2084,  # LTXVPreprocess (redundant once Set_firstframe carries preprocessed)
]


def _add_setnode(ed: WorkflowEditor, bus_name: str, pos: tuple[int, int], dtype: str = "LATENT") -> int:
    """Add a KJNodes-shape SetNode (single typed input + '*' passthrough output)."""
    node_id = ed.next_node_id()
    node = {
        "id": node_id,
        "type": "SetNode",
        "pos": list(pos),
        "size": [210, 60],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"name": dtype, "type": dtype, "link": None}],
        "outputs": [{"name": "*", "type": "*", "links": []}],
        "properties": {
            "Node name for S&R": "SetNode",
            "aux_id": "kijai/ComfyUI-KJNodes",
            "previousName": "",
        },
        "widgets_values": [bus_name],
        "title": f"Set_{bus_name}",
    }
    ed.add_node(node)
    return node_id


def phase4_initial_render(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 4: bring benchmark's pass1 chain into the canonical init-render
    topology and add Set_initial_latent + Set_reference_latent buses for Phase 5.

    Reuses benchmark pass1 nodes (EmptyLatent #32, AddGuideMulti #2221,
    ConcatAV #24, Conditioning #10, CFGGuider #36, RandomNoise #15, Sampler #13,
    Separate #18, KSamplerSelect #1 = euler_ancestral_cfg_pp, ManualSigmas
    #215 = canonical 9-value) rather than rebuilding from scratch.

    Surgical changes:
      - Dim SSoT: rewire Set_width / Set_height / Set_frames / Set_fps inputs
        to source from LTXFramePlanner #2302 outputs. EmptyLatent #32 dims wire
        directly from FramePlanner (skipping the ``a/2`` ComfyMathExpressions
        which are reserved for Phase 5 pass1 half-res).
      - Audio: wire LoadAudio → TrimAudio → AudioVAEEncode chain (Phase 2
        added these as orphans). Rewire ConcatAV #24.audio_latent ← #2309
        (replacing dangling Get_latent_audio whose Set was stripped in Phase 1).
      - Image bus: LoadImage #45 → LTXSmartImageResize #2310 → LTXVPreprocess
        #2311 → Set_firstframe. Strip dead #2083 + redundant #2084. Rewire
        AddGuideMulti.image_1 ← Get_firstframe directly (preprocess now happens
        upstream of the bus, satisfying F2 symmetry for Phase 5).
      - Conditioning: wire LTXVConditioning #10.positive ← selector_init #2315
        (was unwired since Phase 1 stripped the static positive encoder).
      - Insert LTXVImgToVideoInplaceKJ between EmptyLatent and AddGuideMulti
        for the frame-0 anchor at strength=1 (canonical noise_mask=0 lock).
      - Add post-sample LTXVCropGuides + Set_initial_latent + Set_reference_latent
        buses (Phase 5 loop body consumes both: initial_latent → TLO.previous_value,
        reference_latent → per-iter LTXVAdainLatent).

    Strips: 2076 (PrimitiveFloat fps), 2077 (SimpleCalc frames), 2079/2080
    (INTConstants benchmark dims), 2083 (ResizeImagesByLongerEdge), 2084
    (redundant LTXVPreprocess).
    """
    log = (lambda *a: print("  ", *a)) if verbose else (lambda *a: None)

    phase2 = ed.wf.get("properties", {}).get("build_fml2v_phase2", {})
    phase3 = ed.wf.get("properties", {}).get("build_fml2v_phase3", {})
    fp_id = phase2["frame_planner"]              # #2302
    load_audio_id = phase2["load_audio"]         # #2307
    trim_audio_id = phase2["trim_audio"]         # #2308
    audio_vae_id = phase2["audio_vae_encode"]    # #2309
    smart_resize_id = phase2["smart_resize"]     # #2310
    preprocess_id = phase2["preprocess"]         # #2311
    sel_init_id = phase3["selector_init"]        # #2315

    get_vae_id = _find_get_node(ed, "vae")
    get_vae_audio_id = _find_get_node(ed, "vae_audio")
    get_firstframe_id = _find_get_node(ed, "firstframe")

    # --- 1. Dim SSoT: Set_width / Set_height / Set_frames / Set_fps ← FramePlanner ---
    for set_node_id, fp_slot, dtype, label in [
        (_BENCH_SET_WIDTH, 0, "INT", "Set_width"),
        (_BENCH_SET_HEIGHT, 1, "INT", "Set_height"),
        (_BENCH_SET_FRAMES, 2, "INT", "Set_frames"),
        (_BENCH_SET_FPS, 5, "FLOAT", "Set_fps"),
    ]:
        old_link = ed.find_node(set_node_id)["inputs"][0].get("link")
        if old_link is not None:
            ed.remove_link(old_link)
        ed.add_link(fp_id, fp_slot, set_node_id, 0, dtype)
        log(f"  rewire {label} ← FramePlanner slot {fp_slot}")

    # --- 2. EmptyLatent dims direct from FramePlanner (full res, not /2) ---
    for slot_idx, fp_slot, dtype in [(0, 0, "INT"), (1, 1, "INT"), (2, 2, "INT")]:
        empty_node = ed.find_node(_BENCH_EMPTY_LATENT)
        old_link = empty_node["inputs"][slot_idx].get("link")
        if old_link is not None:
            ed.remove_link(old_link)
        ed.add_link(fp_id, fp_slot, _BENCH_EMPTY_LATENT, slot_idx, dtype)
    log(f"  rewire EmptyLatent #{_BENCH_EMPTY_LATENT} dims ← FramePlanner (full res)")

    # --- 3. Audio chain: LoadAudio → Trim → VAEEncode ---
    ed.add_link(load_audio_id, 0, trim_audio_id, 0, "AUDIO")
    ed.add_link(trim_audio_id, 0, audio_vae_id, 0, "AUDIO")
    ed.add_link(get_vae_audio_id, 0, audio_vae_id, 1, "VAE")
    log(f"  wire LoadAudio #{load_audio_id} → TrimAudio #{trim_audio_id} → AudioVAEEncode #{audio_vae_id}")

    # Rewire ConcatAV.audio_latent ← AudioVAEEncode (Get_latent_audio was orphaned by Phase 1)
    concat_node = ed.find_node(_BENCH_CONCAT_AV)
    old_link = concat_node["inputs"][1].get("link")
    if old_link is not None:
        ed.remove_link(old_link)
    ed.add_link(audio_vae_id, 0, _BENCH_CONCAT_AV, 1, "LATENT")
    log(f"  rewire ConcatAV #{_BENCH_CONCAT_AV}.audio_latent ← AudioVAEEncode #{audio_vae_id}")

    # --- 4. Image bus: LoadImage → SmartResize → Preprocess → Set_firstframe ---
    ed.add_link(_BENCH_LOAD_IMAGE_FIRST, 0, smart_resize_id, 0, "IMAGE")
    ed.add_link(smart_resize_id, 0, preprocess_id, 0, "IMAGE")
    set_ff = ed.find_node(_BENCH_SET_FIRSTFRAME)
    old_link = set_ff["inputs"][0].get("link")
    if old_link is not None:
        ed.remove_link(old_link)
    ed.add_link(preprocess_id, 0, _BENCH_SET_FIRSTFRAME, 0, "IMAGE")
    log(f"  wire LoadImage #{_BENCH_LOAD_IMAGE_FIRST} → SmartResize #{smart_resize_id} → Preprocess #{preprocess_id} → Set_firstframe #{_BENCH_SET_FIRSTFRAME}")

    # Rewire AddGuideMulti.image_1 ← Get_firstframe (skip dead #2084 preprocess)
    addgm = ed.find_node(_BENCH_ADD_GUIDE_MULTI)
    old_link = addgm["inputs"][4].get("link")  # num_guides.image_1
    if old_link is not None:
        ed.remove_link(old_link)
    ed.add_link(get_firstframe_id, 0, _BENCH_ADD_GUIDE_MULTI, 4, "IMAGE")
    log(f"  rewire AddGuideMulti #{_BENCH_ADD_GUIDE_MULTI}.image_1 ← Get_firstframe #{get_firstframe_id}")

    # --- 5. Conditioning: LTXVConditioning.positive ← selector_init ---
    ed.add_link(sel_init_id, 0, _BENCH_LTXV_CONDITIONING, 0, "CONDITIONING")
    log(f"  wire LTXVConditioning #{_BENCH_LTXV_CONDITIONING}.positive ← selector_init #{sel_init_id}")

    # --- 6. Insert LTXVImgToVideoInplaceKJ between EmptyLatent and AddGuideMulti ---
    inplace_id = _add_from_template(
        ed, "LTXVImgToVideoInplaceKJ", (-2700, 2400),
        widget_values=["1", 1, 0],  # [num_images=1, strength=1, frame_idx=0]
        title="LTXVImgToVideoInplaceKJ (frame-0 anchor)",
        size=(290, 130),
    )
    ed.add_link(get_vae_id, 0, inplace_id, 0, "VAE")
    ed.add_link(_BENCH_EMPTY_LATENT, 0, inplace_id, 1, "LATENT")
    ed.add_link(get_firstframe_id, 0, inplace_id, 2, "IMAGE")
    # Rewire AddGuideMulti.latent (slot 3) ← InplaceKJ (was ← EmptyLatent directly)
    old_link = addgm["inputs"][3].get("link")
    if old_link is not None:
        ed.remove_link(old_link)
    ed.add_link(inplace_id, 0, _BENCH_ADD_GUIDE_MULTI, 3, "LATENT")
    log(f"+ LTXVImgToVideoInplaceKJ #{inplace_id}  [frame-0 anchor, strength=1]")

    # --- 7. Post-sample LTXVCropGuides + Set_initial_latent + Set_reference_latent ---
    cropguides_id = _add_from_template(
        ed, "LTXVCropGuides", (-1700, 2200),
        widget_values=[],
        title="LTXVCropGuides (init render — post-sample F2/F3)",
        size=(290, 80),
    )
    # pos/neg flow through AddGuideMulti (where guides were added). Use those outputs.
    ed.add_link(_BENCH_ADD_GUIDE_MULTI, 0, cropguides_id, 0, "CONDITIONING")  # positive
    ed.add_link(_BENCH_ADD_GUIDE_MULTI, 1, cropguides_id, 1, "CONDITIONING")  # negative
    ed.add_link(_BENCH_SEPARATE_AV, 0, cropguides_id, 2, "LATENT")            # video_latent
    log(f"+ LTXVCropGuides #{cropguides_id}  [pos/neg from AddGuideMulti, latent from SeparateAV]")

    set_initial_id = _add_setnode(ed, "initial_latent", (-1400, 2200))
    set_reference_id = _add_setnode(ed, "reference_latent", (-1400, 2280))
    ed.add_link(cropguides_id, 2, set_initial_id, 0, "LATENT")
    ed.add_link(cropguides_id, 2, set_reference_id, 0, "LATENT")
    log(f"+ Set_initial_latent #{set_initial_id}, Set_reference_latent #{set_reference_id}")

    # --- 8. Strip nodes orphaned by the above rewires ---
    for nid in _PHASE4_STRIP_AFTER_REWIRE:
        if _strip_node_and_links(ed, nid):
            log(f"STRIP  #{nid}")

    # Stash IDs for Phase 5 to find.
    ed.wf.setdefault("properties", {})["build_fml2v_phase4"] = {
        "img_to_video_inplace": inplace_id,
        "post_sample_cropguides": cropguides_id,
        "set_initial_latent": set_initial_id,
        "set_reference_latent": set_reference_id,
    }
    log("= Stashed Phase 4 node IDs in wf['properties']['build_fml2v_phase4']")


def phase5_loop_body(ed: WorkflowEditor, *, verbose: bool = True) -> None:  # noqa: ARG001
    """Phase 5: TensorLoopOpen + flat-canvas TWO-PASS loop body + TensorLoopClose.

    Option B scope (two-pass refine inside loop body). Full topology spec
    lives in ``example_workflows/working_docs/fml2v_audio_loop_v1_design.md``
    "Sampler chain — two-pass inside the loop body" section — use that doc
    as the wire-level recipe. F2/F3 symmetry must hold on BOTH passes'
    CFGGuiders (load-bearing audit invariants).

    TODO: implement.
    """
    print("[Phase 5] (Option B two-pass topology, not yet implemented)")


def phase6_assembly(ed: WorkflowEditor, *, verbose: bool = True) -> None:  # noqa: ARG001
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
