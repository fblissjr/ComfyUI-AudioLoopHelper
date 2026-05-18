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
    2214,  # GetNode "latent_audio" — orphan after Set_latent_audio strip;
           # KJNodes would throw "No SetNode found for latent_audio(GetNode)" at load.
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
    # Resolution: 960x512 (NOT 960x544) — LTX two-pass refine requires the
    # base resolution to be div-64 so the half-res pass1 stays div-32 and the
    # 2x upsampler returns to a dim that matches the init-render output (which
    # Phase 6's LatentConcat then welds across iterations).
    fp_id = _add_from_template(
        ed, "LTXFramePlanner", (-3000, 3000),
        widget_values=[960, 512, 19.88, 25],
        title="LTXFramePlanner (dim SSoT)",
        size=(270, 250),
    )
    log(f"+ LTXFramePlanner #{fp_id}  [w=960, h=512, target_seconds=19.88, fps=25]")

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
        # Matches FramePlanner 960x512 (two-pass div-64 base).
        widget_values=[960, 512, True, "top"],
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
_BENCH_INIT_CFG_GUIDER = 36        # CFGGuider (init render, pass1)
_BENCH_TOP_LEVEL_CHUNK_FFN = 228   # benchmark's top-level LTXVChunkFeedForward (model chain head)

# Orphaned by Phase 4 dim-SSoT + image-bus rewires; strip after rewiring.
_PHASE4_STRIP_AFTER_REWIRE = [
    2076,  # PrimitiveFloat fps (replaced by FramePlanner.fps_float)
    2077,  # SimpleCalculatorKJ frames calc (replaced by FramePlanner.frames)
    2079,  # INTConstant height (replaced by FramePlanner.height)
    2080,  # INTConstant width  (replaced by FramePlanner.width)
    2083,  # ResizeImagesByLongerEdge (replaced by LTXSmartImageResize chain)
    2084,  # LTXVPreprocess (redundant once Set_firstframe carries preprocessed)
    200,   # GetNode "model_nag" — dead bus (Phase 1 stripped Set_model_nag #199);
    201,   #   #36 init CFGGuider rewired to Get_model below, leaving these orphan.
           #   Stripping them prevents KJNodes' load-time bus-resolution from
           #   throwing "No SetNode found for model_nag(GetNode)".
]


def _add_getnode(ed: WorkflowEditor, bus_name: str, pos: tuple[int, int], dtype: str) -> int:
    """Add a KJNodes-shape GetNode via the canonical WorkflowEditor factory."""
    node_id = ed.next_node_id()
    ed.add_node(WorkflowEditor.make_get_node(node_id, bus_name, dtype, list(pos)))
    return node_id


def _add_setnode(ed: WorkflowEditor, bus_name: str, pos: tuple[int, int], dtype: str) -> int:
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
        ed.rewire_input(set_node_id, 0, fp_id, fp_slot, dtype)
        log(f"  rewire {label} ← FramePlanner slot {fp_slot}")

    # --- 2. EmptyLatent dims direct from FramePlanner (full res, not /2) ---
    for slot_idx, fp_slot in [(0, 0), (1, 1), (2, 2)]:
        ed.rewire_input(_BENCH_EMPTY_LATENT, slot_idx, fp_id, fp_slot, "INT")
    log(f"  rewire EmptyLatent #{_BENCH_EMPTY_LATENT} dims ← FramePlanner (full res)")

    # --- 3. Audio chain: LoadAudio → Trim → VAEEncode ---
    ed.add_link(load_audio_id, 0, trim_audio_id, 0, "AUDIO")
    ed.add_link(trim_audio_id, 0, audio_vae_id, 0, "AUDIO")
    ed.add_link(get_vae_audio_id, 0, audio_vae_id, 1, "VAE")
    log(f"  wire LoadAudio #{load_audio_id} → TrimAudio #{trim_audio_id} → AudioVAEEncode #{audio_vae_id}")

    # Rewire ConcatAV.audio_latent ← AudioVAEEncode (Get_latent_audio was orphaned by Phase 1)
    ed.rewire_input(_BENCH_CONCAT_AV, 1, audio_vae_id, 0, "LATENT")
    log(f"  rewire ConcatAV #{_BENCH_CONCAT_AV}.audio_latent ← AudioVAEEncode #{audio_vae_id}")

    # --- 4. Image bus: LoadImage → SmartResize → Preprocess → Set_firstframe ---
    ed.add_link(_BENCH_LOAD_IMAGE_FIRST, 0, smart_resize_id, 0, "IMAGE")
    ed.add_link(smart_resize_id, 0, preprocess_id, 0, "IMAGE")
    ed.rewire_input(_BENCH_SET_FIRSTFRAME, 0, preprocess_id, 0, "IMAGE")
    log(f"  wire LoadImage #{_BENCH_LOAD_IMAGE_FIRST} → SmartResize #{smart_resize_id} → Preprocess #{preprocess_id} → Set_firstframe #{_BENCH_SET_FIRSTFRAME}")

    # Rewire AddGuideMulti.image_1 ← Get_firstframe (skip dead #2084 preprocess)
    ed.rewire_input(_BENCH_ADD_GUIDE_MULTI, 4, get_firstframe_id, 0, "IMAGE")
    log(f"  rewire AddGuideMulti #{_BENCH_ADD_GUIDE_MULTI}.image_1 ← Get_firstframe #{get_firstframe_id}")

    # --- 5. Conditioning: LTXVConditioning.positive ← selector_init ---
    ed.add_link(sel_init_id, 0, _BENCH_LTXV_CONDITIONING, 0, "CONDITIONING")
    log(f"  wire LTXVConditioning #{_BENCH_LTXV_CONDITIONING}.positive ← selector_init #{sel_init_id}")

    # --- 5b. Init CFGGuider.model ← Get_model (was Get_model_nag, dead bus). ---
    # Phase 1 stripped Set_model_nag (top-level NAG moved into loop body in
    # Phase 5). The init render gets the UN-patched model directly; the loop
    # body's CFGGuiders get the patched model via Get_loop_patched_model.
    get_model_id = _find_get_node(ed, "model")
    ed.rewire_input(_BENCH_INIT_CFG_GUIDER, 0, get_model_id, 0, "MODEL")
    log(f"  rewire init CFGGuider #{_BENCH_INIT_CFG_GUIDER}.model ← Get_model #{get_model_id} (was dead Get_model_nag)")

    # --- 5c. Wire benchmark's top-level model patch chain head to UNETLoader. ---
    # The benchmark workflow had the chain UNETLoader → ChunkFFN(#228) →
    # AttnTuner(#229, bypassed) → PowerLoraLoader → Set_model(#192) but the
    # head link (UNETLoader.MODEL → #228.model) is absent in the source JSON,
    # leaving #228 unwired and ComfyUI rejecting the prompt at validation.
    # Set_model is consumed by init CFGGuider above, so this chain IS live —
    # wire it. Loop-body model patches live on a separate (Phase 5) chain
    # downstream of Get_model, so this only affects top-level setup.
    unet_loaders = [n["id"] for n in ed.wf.get("nodes", [])
                    if n.get("type") == "UNETLoader" and n.get("mode", 0) != 4]
    if not unet_loaders:
        raise SystemExit("No active UNETLoader found — benchmark workflow shape changed")
    if ed.has_node(_BENCH_TOP_LEVEL_CHUNK_FFN):
        ed.rewire_input(_BENCH_TOP_LEVEL_CHUNK_FFN, 0, unet_loaders[0], 0, "MODEL")
        log(f"  wire benchmark ChunkFFN #{_BENCH_TOP_LEVEL_CHUNK_FFN}.model ← UNETLoader #{unet_loaders[0]} (was unwired)")

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
    ed.rewire_input(_BENCH_ADD_GUIDE_MULTI, 3, inplace_id, 0, "LATENT")
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

    set_initial_id = _add_setnode(ed, "initial_latent", (-1400, 2200), dtype="LATENT")
    set_reference_id = _add_setnode(ed, "reference_latent", (-1400, 2280), dtype="LATENT")
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


# Benchmark bypassed pass2 nodes Phase 5 unbypasses + rewires.
_BENCH_PASS2_KSAMPLER = 4              # KSamplerSelect euler_cfg_pp
_BENCH_PASS2_CFG_GUIDER = 8            # CFGGuider (pass2)
_BENCH_PASS2_SAMPLER = 21              # SamplerCustomAdvanced (pass2)
_BENCH_PASS2_SIGMAS = 216              # ManualSigmas 4-value canonical
_BENCH_PASS2_UPSAMPLER = 25            # LTXVLatentUpsampler 2x spatial
_BENCH_PRE_PASS2_CONCAT_AV = 34        # LTXVConcatAVLatent (re-attach audio)
_BENCH_POST_PASS2_SEPARATE = 146       # LTXVSeparateAVLatent (post-pass2)
_BENCH_PRE_PASS2_GUIDE_MULTI = 2182    # LTXVAddGuideMulti N=2 first+last
_BENCH_BETWEEN_CROPGUIDES = 2222       # LTXVCropGuides (pre-upsample)

# Existing benchmark nodes Phase 5 fans-out from (already feeding init render).
_BENCH_PASS1_KSAMPLER = 1              # KSamplerSelect euler_ancestral_cfg_pp (fan-out)
_BENCH_PASS1_SIGMAS = 215              # ManualSigmas 9-value canonical (fan-out)
_BENCH_NEGATIVE_ENCODER = 11           # CLIPTextEncode (benchmark negative, fan-out)
_BENCH_GET_MODEL = 122                 # GetNode "model" (benchmark's model bus)

# Half-res ComfyMathExpressions for pass1 latent dims (benchmark had these,
# Phase 4 left them wired to Get_width/Get_height which now flow from
# FramePlanner via the dim-SSoT setters — so they automatically yield
# FramePlanner.width/2 + FramePlanner.height/2 with no Phase 5 rewire needed).
_BENCH_WIDTH_HALF_EXPR = 2191          # ComfyMathExpression "a/2" for width
_BENCH_HEIGHT_HALF_EXPR = 2192         # ComfyMathExpression "a/2" for height


def phase5_loop_body(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 5: flat-canvas two-pass loop body between TensorLoopOpen/Close.

    Option B topology (per design doc): every iteration runs pass1 (half-res
    denoise) → upsample → pass2 (full-res refine), all inside the TLO/TLC
    boundary. The model patch chain (LoopIterationStamp → ChunkFFN → AttnTuner
    → NAG → Sage) lives downstream of TLO so per-iter patches survive any
    comfy-aimdo offload, feeding BOTH passes' CFGGuiders via the
    ``loop_patched_model`` bus.

    Substages:
      5a. AudioLoopController/Planner input wiring + TLO/TLC boundaries +
          model patch chain + Set_loop_patched_model bus.
      5b. Pass 1 half-res denoise chain (EmptyLatent_p1 + AudioLatentSlice +
          LatentContextExtract + LatentOverlapTrim + LTXVAudioVideoMask +
          LTXVAddLatentGuide + F3 cropguides dual + AdaIN + ConcatAV +
          Sampler). Reuses init render's KSamplerSelect #1 + ManualSigmas
          #215 via fan-out (both iter-independent).
      5c. Between passes + Pass 2 (unbypass + rewire benchmark's bypassed
          pass2 chain: #4/#8/#21/#25/#34/#146/#216/#2182/#2222). Adds final
          AdaIN + IterationCleanup.
      5d. TLC wiring (processed ← post-pass2 chain output, stop ←
          AudioLoopController.should_stop).

    Containment rule: every iter-dependent node lies on a path TLO → TLC so
    ComfyUI-NativeLooping's ``_WhileLoopClose._explore_dependencies`` clones
    it per iter. Iter-independent nodes (KSamplerSelect, ManualSigmas, model
    patch chain feeding via SetNode bus) execute ONCE statically; their
    cached output is reused per iter.
    """
    log = (lambda *a: print("  ", *a)) if verbose else (lambda *a: None)

    phase2 = ed.wf.get("properties", {}).get("build_fml2v_phase2", {})
    phase3 = ed.wf.get("properties", {}).get("build_fml2v_phase3", {})
    fp_id = phase2["frame_planner"]
    overlap_const_id = phase2["overlap_seconds"]
    init_strength_id = phase2["first_frame_guide_strength"]
    alc_id = phase2["audio_loop_controller"]
    alp_id = phase2["audio_loop_planner"]
    trim_audio_id = phase2["trim_audio"]
    audio_vae_id = phase2["audio_vae_encode"]
    nag_cond_id = phase3["nag_cond_video"]
    sel_loop_id = phase3["selector_loop"]
    # Phase 4 added Set_initial_latent + Set_reference_latent; loop body
    # consumes them via fresh GetNodes (added below per consumer site).
    assert "build_fml2v_phase4" in ed.wf.get("properties", {}), "Phase 4 must run before Phase 5"

    get_vae_id = _find_get_node(ed, "vae")
    get_model_id = _BENCH_GET_MODEL  # benchmark's #122 Get_model

    # ====================================================================
    # 5a — Controller wiring + TLO/TLC + model patch chain
    # ====================================================================
    log("[Phase 5a] Controller + TLO/TLC + model patch chain")

    # Wire AudioLoopController + Planner inputs: audio + shared overlap_seconds.
    # The single shared FloatConstant clears the overlap_seconds_single_source ERR.
    ed.rewire_input(alc_id, 0, trim_audio_id, 0, "AUDIO")           # audio
    ed.rewire_input(alc_id, 5, overlap_const_id, 0, "FLOAT")        # overlap_seconds
    ed.rewire_input(alp_id, 0, trim_audio_id, 0, "AUDIO")           # audio
    ed.rewire_input(alp_id, 3, overlap_const_id, 0, "FLOAT")        # overlap_seconds
    log(f"  wire Controller#{alc_id} + Planner#{alp_id} inputs (audio + shared overlap_seconds)")

    # Add TLO + TLC boundaries.
    tlo_id = _add_from_template(
        ed, "TensorLoopOpen", (-2400, 1200),
        widget_values=["iterations", 50, 0],
        title="TensorLoopOpen (loop start)",
        size=(280, 140),
    )
    tlc_id = _add_from_template(
        ed, "TensorLoopClose", (1400, 1200),
        widget_values=[True, "disabled"],
        title="TensorLoopClose (loop end)",
        size=(280, 100),
    )
    # initial_value ← Get_initial_latent (consumes Phase 4 bus)
    get_initial_id = _add_getnode(ed, "initial_latent", (-2700, 1200), "LATENT")
    ed.add_link(get_initial_id, 0, tlo_id, 0, "LATENT")
    # iterations_in ← AudioLoopPlanner.total_iterations (F5 audit invariant)
    ed.add_link(alp_id, 1, tlo_id, 1, "INT")
    log(f"+ TensorLoopOpen #{tlo_id}, TensorLoopClose #{tlc_id}  (iterations_in ← Planner.total_iterations, F5)")

    # Wire AudioLoopController.current_iteration ← TLO.current_iteration (slot 3).
    ed.rewire_input(alc_id, 1, tlo_id, 3, "INT")
    # Wire selector_loop.current_iteration ← TLO.current_iteration.
    ed.rewire_input(sel_loop_id, 1, tlo_id, 3, "INT")
    log(f"  wire Controller#{alc_id}.current_iteration + selector_loop#{sel_loop_id}.current_iteration ← TLO.current_iteration")

    # Model patch chain (downstream of LoopIterationStamp; feeds BOTH CFGGuiders
    # via SetNode bus). Patch order per CLAUDE.md: Stamp → ChunkFFN → AttnTuner
    # → NAG → Sage (Sage LAST so ON_CLEANUP drains first).
    stamp_id = _add_from_template(
        ed, "LoopIterationStamp", (-2100, 1500),
        widget_values=[0],
        title="LoopIterationStamp",
        size=(280, 80),
    )
    ed.add_link(get_model_id, 0, stamp_id, 0, "MODEL")              # model
    ed.add_link(tlo_id, 3, stamp_id, 1, "INT")                      # current_iteration ← TLO

    chunk_id = _add_from_template(
        ed, "LTXVChunkFeedForward", (-1800, 1500),
        widget_values=[2, 4096],
        title="LTXVChunkFeedForward",
        size=(280, 80),
    )
    ed.add_link(stamp_id, 0, chunk_id, 0, "MODEL")

    attn_id = _add_from_template(
        ed, "LTX2AttentionTunerPatch", (-1500, 1500),
        widget_values=["", 1, 1, 1, 1, True],
        title="LTX2AttentionTunerPatch (bypassed default)",
        size=(280, 130),
        mode=4,
    )
    ed.add_link(chunk_id, 0, attn_id, 0, "MODEL")

    nag_id = _add_from_template(
        ed, "LTX2_NAG", (-1200, 1500),
        widget_values=[11, 0.25, 2.5, True],
        title="LTX2_NAG (nag_cond_video from Phase 3)",
        size=(280, 130),
    )
    ed.add_link(attn_id, 0, nag_id, 0, "MODEL")
    ed.add_link(nag_cond_id, 0, nag_id, 1, "CONDITIONING")          # nag_cond_video
    # nag_cond_audio (slot 2) left unwired — widget-only

    sage_id = _add_from_template(
        ed, "AudioLoopHelperSageAttention", (-900, 1500),
        widget_values=["auto", True, 1024],
        title="AudioLoopHelperSageAttention (Sage LAST)",
        size=(280, 100),
    )
    ed.add_link(nag_id, 0, sage_id, 0, "MODEL")

    # BYPASSED-by-default diagnostic: pass-through that logs per-call patch
    # state. Toggle mode=0 in the UI before the first live render to verify
    # whether ComfyUI-NativeLooping's _explore_dependencies walks the
    # SetNode/GetNode bus (= patches re-apply per iter) or treats it as
    # opaque (= patches freeze at iter 0). Source: nodes.py::IterPatchInspector.
    inspector_id = _add_from_template(
        ed, "IterPatchInspector", (-600, 1400),
        widget_values=["patches_loop_body", False],
        title="IterPatchInspector (bypassed; toggle to diagnose per-iter patch survival)",
        size=(290, 100),
        mode=4,
    )
    ed.add_link(sage_id, 0, inspector_id, 0, "MODEL")

    set_patched_id = _add_setnode(ed, "loop_patched_model", (-300, 1500), dtype="MODEL")
    ed.add_link(inspector_id, 0, set_patched_id, 0, "MODEL")
    log(f"+ model patch chain: Stamp#{stamp_id} → ChunkFFN#{chunk_id} → AttnTuner#{attn_id} (bypassed) → NAG#{nag_id} → Sage#{sage_id} → IterPatchInspector#{inspector_id} (bypassed) → Set_loop_patched_model#{set_patched_id}")

    # ====================================================================
    # 5b — Pass 1 half-res denoise chain
    # ====================================================================
    log("[Phase 5b] Pass 1 half-res denoise chain")

    # EmptyLatent_p1: width/height from ComfyMathExpression "a/2" (fed from
    # FramePlanner via existing Set_width/Set_height bus — already SSoT after
    # Phase 4); length = FramePlanner.frames stride-sized chunk.
    # ComfyMathExpression has FLOAT(slot 0) + INT(slot 1) outputs; INT for dims.
    empty_p1_id = _add_from_template(
        ed, "EmptyLTXVLatentVideo", (-2100, 1800),
        widget_values=[480, 256, 49, 1],  # widget defaults (wired inputs win); matches 960x512 / 2
        title="EmptyLTXVLatentVideo (loop pass1 half-res)",
        size=(290, 110),
    )
    ed.add_link(_BENCH_WIDTH_HALF_EXPR, 1, empty_p1_id, 0, "INT")
    ed.add_link(_BENCH_HEIGHT_HALF_EXPR, 1, empty_p1_id, 1, "INT")
    ed.add_link(fp_id, 2, empty_p1_id, 2, "INT")
    log(f"+ EmptyLatent_p1 #{empty_p1_id}  (half-res, length ← FramePlanner.frames)")

    # Half-res init image chain for pass1's AddLatentGuide. LTXVAddLatentGuide
    # asserts ``latent.shape[3,4] % guide.shape[3,4] == 0`` — the guide latent
    # spatial dims must divide the sampler latent dims. Pass1 samples at
    # half-res (480x256 → latent 60x32); a full-res firstframe encode would
    # produce a 120x64 guide which fails the divisibility check at runtime
    # (60 % 120 != 0). Build a dedicated half-res resize+preprocess+encode
    # so the guide matches pass1's sampler dims (60 % 60 == 0).
    smart_resize_p1_id = _add_from_template(
        ed, "LTXSmartImageResize", (-2400, 1950),
        widget_values=[480, 256, True, "top"],  # widget defaults; inputs win
        title="LTXSmartImageResize (half-res for pass1 guide)",
        size=(290, 150),
    )
    ed.add_link(_BENCH_LOAD_IMAGE_FIRST, 0, smart_resize_p1_id, 0, "IMAGE")
    ed.add_link(_BENCH_WIDTH_HALF_EXPR, 1, smart_resize_p1_id, 1, "INT")
    ed.add_link(_BENCH_HEIGHT_HALF_EXPR, 1, smart_resize_p1_id, 2, "INT")

    preprocess_p1_id = _add_from_template(
        ed, "LTXVPreprocess", (-2400, 2120),
        widget_values=[18],
        title="LTXVPreprocess (half-res; img_compression=18)",
        size=(270, 80),
    )
    ed.add_link(smart_resize_p1_id, 0, preprocess_p1_id, 0, "IMAGE")

    vae_encode_id = _add_from_template(
        ed, "VAEEncode", (-2100, 1950),
        widget_values=[],
        title="VAEEncode (half-res init image → pass1 guide_latent)",
        size=(290, 80),
    )
    ed.add_link(preprocess_p1_id, 0, vae_encode_id, 0, "IMAGE")
    ed.add_link(get_vae_id, 0, vae_encode_id, 1, "VAE")

    # AudioLatentSlice: per-iter audio window from full pre-encoded audio.
    audio_slice_id = _add_from_template(
        ed, "AudioLatentSlice", (-1800, 1800),
        widget_values=[300.0, 0.0, 19.88],  # widgets defaults (inputs win)
        title="AudioLatentSlice (per-iter window)",
        size=(290, 110),
    )
    ed.add_link(audio_vae_id, 0, audio_slice_id, 0, "LATENT")
    ed.add_link(alc_id, 2, audio_slice_id, 1, "FLOAT")              # source_seconds ← Controller.audio_duration
    ed.add_link(alc_id, 0, audio_slice_id, 2, "FLOAT")              # start_seconds ← Controller.start_index
    ed.add_link(alc_id, 4, audio_slice_id, 3, "FLOAT")              # duration_seconds ← Controller.stride_seconds

    # NOTE: canonical single-pass workflows wire
    # ``LatentContextExtract(TLO.prev_value) → mask.video_latent`` so each
    # iter inherits the prev iter's overlap region as a context. Two-pass
    # refine here samples pass1 at half-res while TLO.previous_value is
    # full-res (post-pass2 upsample), so direct ContextExtract → mask wiring
    # would shape-mismatch. Cross-iter continuity in this build comes
    # instead from the fixed init-image anchor (LTXVAddLatentGuide trailing
    # frame) + frozen audio + the prompt schedule. LatentOverlapTrim is
    # still required, but on the OUTPUT side (post-AdaIN_final) — it trims
    # the overlap region from each iter's output so adjacent windows don't
    # double-up when Phase 6's LatentConcat welds them.

    # LTXVAudioVideoMask: canonical wiring leaves audio frozen via
    # audio_start_time = audio_end_time (widget defaults [10, 10] give empty
    # mask range → audio preserved). Don't wire those inputs.
    mask_id = _add_from_template(
        ed, "LTXVAudioVideoMask", (-900, 1800),
        widget_values=[25, 1, 10, 10, 10, "pad", "add"],
        title="LTXVAudioVideoMask (audio frozen)",
        size=(290, 180),
    )
    ed.add_link(empty_p1_id, 0, mask_id, 0, "LATENT")               # video_latent ← empty (half-res)
    ed.add_link(audio_slice_id, 0, mask_id, 1, "LATENT")            # audio_latent

    # LTXVAddLatentGuide: per-iter trailing init anchor (latent_idx=-1).
    add_guide_id = _add_from_template(
        ed, "LTXVAddLatentGuide", (-600, 1800),
        widget_values=[-1, 0.7],  # [latent_idx=-1, strength_widget_default]
        title="LTXVAddLatentGuide (trailing init anchor)",
        size=(290, 180),
    )
    ed.add_link(get_vae_id, 0, add_guide_id, 0, "VAE")
    ed.add_link(sel_loop_id, 0, add_guide_id, 1, "CONDITIONING")    # positive ← selector_loop
    ed.add_link(_BENCH_NEGATIVE_ENCODER, 0, add_guide_id, 2, "CONDITIONING")  # negative
    ed.add_link(mask_id, 0, add_guide_id, 3, "LATENT")              # latent ← mask.video
    ed.add_link(vae_encode_id, 0, add_guide_id, 4, "LATENT")        # guiding_latent
    ed.add_link(init_strength_id, 0, add_guide_id, 5, "FLOAT")      # strength ← FloatConstant 0.7

    # F3 dual cropguides: NoLatent for cond path; with-latent for AdaIN path.
    nocrop_id = _add_from_template(
        ed, "LTXVCropGuidesNoLatent", (-300, 1700),
        widget_values=[],
        title="LTXVCropGuidesNoLatent (cond path)",
        size=(290, 80),
    )
    ed.add_link(add_guide_id, 0, nocrop_id, 0, "CONDITIONING")
    ed.add_link(add_guide_id, 1, nocrop_id, 1, "CONDITIONING")

    crop_p1_id = _add_from_template(
        ed, "LTXVCropGuides", (-300, 1900),
        widget_values=[],
        title="LTXVCropGuides (latent path for AdaIN)",
        size=(290, 100),
    )
    ed.add_link(add_guide_id, 0, crop_p1_id, 0, "CONDITIONING")
    ed.add_link(add_guide_id, 1, crop_p1_id, 1, "CONDITIONING")
    ed.add_link(add_guide_id, 2, crop_p1_id, 2, "LATENT")

    # AdaIN with reference from initial-render bus.
    get_ref_id = _add_getnode(ed, "reference_latent", (0, 1950), "LATENT")
    adain_p1_id = _add_from_template(
        ed, "LTXVAdainLatent", (0, 1800),
        widget_values=[0.2, False],
        title="LTXVAdainLatent (pass1 reference)",
        size=(290, 100),
    )
    ed.add_link(crop_p1_id, 2, adain_p1_id, 0, "LATENT")
    ed.add_link(get_ref_id, 0, adain_p1_id, 1, "LATENT")

    # Re-attach audio for sampler input.
    concat_p1_id = _add_from_template(
        ed, "LTXVConcatAVLatent", (300, 1800),
        widget_values=[],
        title="LTXVConcatAVLatent (pass1 pre-sample)",
        size=(290, 80),
    )
    ed.add_link(adain_p1_id, 0, concat_p1_id, 0, "LATENT")
    ed.add_link(mask_id, 1, concat_p1_id, 1, "LATENT")              # audio_latent ← mask.audio

    # Sampler with patched model from bus.
    get_patched_p1_id = _add_getnode(ed, "loop_patched_model", (300, 1600), "MODEL")

    cfg_p1_id = _add_from_template(
        ed, "CFGGuider", (600, 1700),
        widget_values=[1],
        title="CFGGuider (pass1)",
        size=(290, 100),
    )
    ed.add_link(get_patched_p1_id, 0, cfg_p1_id, 0, "MODEL")
    ed.add_link(nocrop_id, 0, cfg_p1_id, 1, "CONDITIONING")
    ed.add_link(nocrop_id, 1, cfg_p1_id, 2, "CONDITIONING")

    noise_p1_id = _add_from_template(
        ed, "RandomNoise", (600, 1850),
        widget_values=[42, "fixed"],
        title="RandomNoise (pass1, seed ← Controller.iteration_seed)",
        size=(290, 80),
    )
    ed.add_link(alc_id, 3, noise_p1_id, 0, "INT")                   # noise_seed ← Controller.iteration_seed

    sampler_p1_id = _add_from_template(
        ed, "SamplerCustomAdvanced", (900, 1800),
        widget_values=[],
        title="SamplerCustomAdvanced (pass1)",
        size=(290, 130),
    )
    ed.add_link(noise_p1_id, 0, sampler_p1_id, 0, "NOISE")
    ed.add_link(cfg_p1_id, 0, sampler_p1_id, 1, "GUIDER")
    ed.add_link(_BENCH_PASS1_KSAMPLER, 0, sampler_p1_id, 2, "SAMPLER")  # fan-out from #1
    ed.add_link(_BENCH_PASS1_SIGMAS, 0, sampler_p1_id, 3, "SIGMAS")     # fan-out from #215
    ed.add_link(concat_p1_id, 0, sampler_p1_id, 4, "LATENT")
    log(f"+ Pass 1 chain (Empty#{empty_p1_id} → ... → Sampler#{sampler_p1_id}); KSamplerSelect#{_BENCH_PASS1_KSAMPLER}+Sigmas#{_BENCH_PASS1_SIGMAS} fan-out from init render")

    # ====================================================================
    # 5c — Between passes + Pass 2 (unbypass + rewire benchmark chain)
    # ====================================================================
    log("[Phase 5c] Between-passes + Pass 2 (unbypass benchmark chain)")

    # Post-pass1 SeparateAV (strip audio; upsampler is video-only).
    sep_p1_post_id = _add_from_template(
        ed, "LTXVSeparateAVLatent", (1200, 1800),
        widget_values=[],
        title="LTXVSeparateAVLatent (post-pass1, pre-upsample)",
        size=(290, 100),
    )
    ed.add_link(sampler_p1_id, 0, sep_p1_post_id, 0, "LATENT")

    # Unbypass and rewire benchmark's between+pass2 chain.
    for nid in [_BENCH_BETWEEN_CROPGUIDES, _BENCH_PASS2_UPSAMPLER,
                _BENCH_PRE_PASS2_GUIDE_MULTI, _BENCH_PRE_PASS2_CONCAT_AV,
                _BENCH_PASS2_KSAMPLER, _BENCH_PASS2_SIGMAS,
                _BENCH_PASS2_CFG_GUIDER, _BENCH_PASS2_SAMPLER,
                _BENCH_POST_PASS2_SEPARATE]:
        ed.find_node(nid)["mode"] = 0
    log(f"  unbypassed #{_BENCH_BETWEEN_CROPGUIDES}, #{_BENCH_PASS2_UPSAMPLER}, #{_BENCH_PRE_PASS2_GUIDE_MULTI}, #{_BENCH_PRE_PASS2_CONCAT_AV}, #{_BENCH_PASS2_KSAMPLER}, #{_BENCH_PASS2_SIGMAS}, #{_BENCH_PASS2_CFG_GUIDER}, #{_BENCH_PASS2_SAMPLER}, #{_BENCH_POST_PASS2_SEPARATE}")

    # Between cropguides (#2222) currently has latent ← #18 (init render) +
    # pos/neg ← benchmark namespace GetNodes. Rewire to loop-body sources.
    ed.rewire_input(_BENCH_BETWEEN_CROPGUIDES, 0, sel_loop_id, 0, "CONDITIONING")
    ed.rewire_input(_BENCH_BETWEEN_CROPGUIDES, 1, _BENCH_NEGATIVE_ENCODER, 0, "CONDITIONING")
    ed.rewire_input(_BENCH_BETWEEN_CROPGUIDES, 2, sep_p1_post_id, 0, "LATENT")
    log(f"  rewire CropGuides#{_BENCH_BETWEEN_CROPGUIDES} (latent ← post-pass1 separate.video)")

    # Pre-pass2 ConcatAV (#34) currently has audio ← #18.audio (init render);
    # rewire to loop body pass1's mask.audio (same audio across both passes).
    ed.rewire_input(_BENCH_PRE_PASS2_CONCAT_AV, 1, mask_id, 1, "LATENT")
    log(f"  rewire ConcatAV#{_BENCH_PRE_PASS2_CONCAT_AV}.audio_latent ← LTXVAudioVideoMask#{mask_id}.audio")

    # Pass2 CFGGuider model ← Get_loop_patched_model (was dangling Get_model_nag).
    get_patched_p2_id = _add_getnode(ed, "loop_patched_model", (300, 1300), "MODEL")
    ed.rewire_input(_BENCH_PASS2_CFG_GUIDER, 0, get_patched_p2_id, 0, "MODEL")
    log(f"  rewire CFGGuider#{_BENCH_PASS2_CFG_GUIDER}.model ← Get_loop_patched_model")

    # Pass2 RandomNoise: benchmark's #14 is widget-only (no input slot). Add a
    # NEW RandomNoise with iteration_seed wire; rewire Sampler.noise to it.
    noise_p2_id = _add_from_template(
        ed, "RandomNoise", (900, 1200),
        widget_values=[42, "fixed"],
        title="RandomNoise (pass2, seed ← Controller.iteration_seed)",
        size=(290, 80),
    )
    ed.add_link(alc_id, 3, noise_p2_id, 0, "INT")
    ed.rewire_input(_BENCH_PASS2_SAMPLER, 0, noise_p2_id, 0, "NOISE")
    log(f"+ RandomNoise #{noise_p2_id} (pass2); rewire Sampler#{_BENCH_PASS2_SAMPLER}.noise")

    # ====================================================================
    # 5d — Post-pass2 AdaIN + IterationCleanup + TLC wiring
    # ====================================================================
    log("[Phase 5d] Post-pass2 AdaIN + IterationCleanup + TLC")

    # Final AdaIN: refines pass2 video latent against reference_latent.
    get_ref_p2_id = _add_getnode(ed, "reference_latent", (1500, 1900), "LATENT")
    adain_final_id = _add_from_template(
        ed, "LTXVAdainLatent", (1500, 1800),
        widget_values=[0.2, False],
        title="LTXVAdainLatent (post-pass2 final)",
        size=(290, 100),
    )
    ed.add_link(_BENCH_POST_PASS2_SEPARATE, 0, adain_final_id, 0, "LATENT")  # latents ← post-pass2 video
    ed.add_link(get_ref_p2_id, 0, adain_final_id, 1, "LATENT")

    # LatentOverlapTrim: drop the overlap region from this iter's output so
    # adjacent windows don't double-up when Phase 6's LatentConcat welds them.
    # overlap_latent_frames sourced from AudioLoopController (iter-dependent
    # alongside the controller's stride math, so it stays on the TLO→TLC path).
    overlap_trim_id = _add_from_template(
        ed, "LatentOverlapTrim", (1800, 1700),
        widget_values=[4],
        title="LatentOverlapTrim (drop overlap from output)",
        size=(290, 80),
    )
    ed.add_link(adain_final_id, 0, overlap_trim_id, 0, "LATENT")
    ed.add_link(alc_id, 6, overlap_trim_id, 1, "INT")               # overlap_latent_frames

    # IterationCleanup drains per-iter caches.
    cleanup_id = _add_from_template(
        ed, "IterationCleanup", (2100, 1800),
        widget_values=["always"],
        title="IterationCleanup",
        size=(290, 80),
    )
    ed.add_link(overlap_trim_id, 0, cleanup_id, 0, "LATENT")

    # TLC wires: flow_control pairs the close to its open (the loop-boundary
    # signal NativeLooping reads to drive iteration), processed carries the
    # per-iter output, stop carries the controller's terminate flag.
    ed.add_link(tlo_id, 0, tlc_id, 0, "FLOW_CONTROL")               # flow_control ← TLO
    ed.add_link(cleanup_id, 0, tlc_id, 1, "LATENT")
    ed.add_link(alc_id, 1, tlc_id, 2, "BOOLEAN")                    # should_stop
    log(f"+ AdaIN_final #{adain_final_id} → Cleanup #{cleanup_id} → TLC #{tlc_id}.processed; TLC.flow_control ← TLO #{tlo_id}; TLC.stop ← Controller.should_stop")

    # Stash IDs for Phase 6 to find.
    ed.wf.setdefault("properties", {})["build_fml2v_phase5"] = {
        "tlo": tlo_id,
        "tlc": tlc_id,
        "stamp": stamp_id,
        "chunk_ffn": chunk_id,
        "attn_tuner": attn_id,
        "nag": nag_id,
        "sage": sage_id,
        "iter_patch_inspector": inspector_id,
        "set_patched_model": set_patched_id,
        "empty_latent_p1": empty_p1_id,
        "smart_resize_p1": smart_resize_p1_id,
        "preprocess_p1": preprocess_p1_id,
        "vae_encode_init_guide": vae_encode_id,
        "audio_slice": audio_slice_id,
        "overlap_trim_output": overlap_trim_id,
        "av_mask": mask_id,
        "add_latent_guide": add_guide_id,
        "crop_guides_no_latent": nocrop_id,
        "crop_guides_p1": crop_p1_id,
        "adain_p1": adain_p1_id,
        "concat_av_p1": concat_p1_id,
        "cfg_guider_p1": cfg_p1_id,
        "noise_p1": noise_p1_id,
        "sampler_p1": sampler_p1_id,
        "separate_av_p1_post": sep_p1_post_id,
        "noise_p2": noise_p2_id,
        "adain_final": adain_final_id,
        "iteration_cleanup": cleanup_id,
    }
    log("= Stashed Phase 5 node IDs in wf['properties']['build_fml2v_phase5']")


# Benchmark output-chain nodes Phase 6 rewires.
_BENCH_VAE_DECODE = 149                # LTXVTiledVAEDecode (bypassed pass2 decoder)
_BENCH_VHS_COMBINE = 43                # VHS_VideoCombine


def phase6_assembly(ed: WorkflowEditor, *, verbose: bool = True) -> None:
    """Phase 6: assemble per-iter latents into final video + audio output.

    Topology (top-level):
      Get_initial_latent + TLC.output → LatentConcat (dim='t')
        → TrimVideoLatentToAudio (snap-UP to LTX boundary; video ≥ audio)
        → LTXVTiledVAEDecode [1,1,1,True,'auto','auto'] (24GB+ single-tile)
        → TrimImageBatchToAudio (clips overshoot to exact int(audio*fps))
        → Set_final_video → existing VHS_VideoCombine.images
      TrimAudio #2308 → Set_final_audio (Option B: skip LTXVAudioVAEDecode;
        audio frozen via mask=0 throughout the loop, so source is
        bit-identical to a decode of the latent).
      RunIdPrefix → VHS_VideoCombine.filename_prefix (F15 — converts the
        widget to an input slot if needed).
      LoopConfigValidator: optional config-consistency check fed from
        AudioLoopPlanner + FramePlanner outputs.

    Reuses benchmark's bypassed #149 LTXVTiledVAEDecode (widgets already
    match canonical 24GB+ `[1, 1, 1, True, 'auto', 'auto']`); unbypass +
    rewire its latents input to the new trim chain. Leaves benchmark's
    bypassed LTXVAudioVAEDecode #150 alone (dead under Option B).
    """
    log = (lambda *a: print("  ", *a)) if verbose else (lambda *a: None)

    phase2 = ed.wf.get("properties", {}).get("build_fml2v_phase2", {})
    phase5 = ed.wf.get("properties", {}).get("build_fml2v_phase5", {})
    fp_id = phase2["frame_planner"]
    overlap_const_id = phase2["overlap_seconds"]
    trim_audio_id = phase2["trim_audio"]
    tlc_id = phase5["tlc"]

    # --- 1. LatentConcat: init + TLC, weld across temporal dim ---
    get_initial_id = _add_getnode(ed, "initial_latent", (2000, 1800), "LATENT")
    concat_id = _add_from_template(
        ed, "LatentConcat", (2300, 1800),
        widget_values=["t"],
        title="LatentConcat (init + TLC across t)",
        size=(290, 80),
    )
    ed.add_link(get_initial_id, 0, concat_id, 0, "LATENT")
    ed.add_link(tlc_id, 0, concat_id, 1, "LATENT")

    # --- 2. TrimVideoLatentToAudio (snap-UP to LTX 8-frame boundary) ---
    trim_latent_id = _add_from_template(
        ed, "TrimVideoLatentToAudio", (2600, 1800),
        widget_values=[25],
        title="TrimVideoLatentToAudio (snap-UP)",
        size=(290, 100),
    )
    ed.add_link(concat_id, 0, trim_latent_id, 0, "LATENT")
    ed.add_link(trim_audio_id, 0, trim_latent_id, 1, "AUDIO")
    ed.add_link(fp_id, 4, trim_latent_id, 2, "INT")                 # fps_int

    # --- 3. Unbypass #149 LTXVTiledVAEDecode; rewire latents ← TrimVideoLatent ---
    ed.find_node(_BENCH_VAE_DECODE)["mode"] = 0
    ed.rewire_input(_BENCH_VAE_DECODE, 1, trim_latent_id, 0, "LATENT")
    log(f"  unbypass + rewire LTXVTiledVAEDecode #{_BENCH_VAE_DECODE}.latents ← TrimVideoLatentToAudio #{trim_latent_id}")

    # --- 4. TrimImageBatchToAudio (exact-frame clip post-decode) ---
    trim_img_id = _add_from_template(
        ed, "TrimImageBatchToAudio", (3200, 1800),
        widget_values=[25],
        title="TrimImageBatchToAudio (exact-frame clip)",
        size=(290, 100),
    )
    ed.add_link(_BENCH_VAE_DECODE, 0, trim_img_id, 0, "IMAGE")
    ed.add_link(trim_audio_id, 0, trim_img_id, 1, "AUDIO")
    ed.add_link(fp_id, 4, trim_img_id, 2, "INT")                    # fps_int

    # --- 5. Rewire VHS_VideoCombine direct from the trim chain. F14 audit
    #        requires VHS.images to come directly from TrimImageBatchToAudio
    #        (not via a SetNode bus). Option B: audio also direct from
    #        TrimAudio (skip the LTXVAudioVAEDecode round-trip; audio frozen
    #        via mask=0 throughout, so source is bit-identical). Benchmark's
    #        Set_final_video / Set_final_audio + their Get-side consumers
    #        become orphan after this rewire; left on canvas as no-ops
    #        rather than stripped (cosmetic-only).
    ed.rewire_input(_BENCH_VHS_COMBINE, 0, trim_img_id, 0, "IMAGE")
    ed.rewire_input(_BENCH_VHS_COMBINE, 1, trim_audio_id, 0, "AUDIO")
    log(f"  rewire VHS_VideoCombine #{_BENCH_VHS_COMBINE}.images ← TrimImageBatchToAudio (F14); .audio ← TrimAudio (Option B)")

    # --- 6. RunIdPrefix → VHS_VideoCombine.filename_prefix (F15) ---
    run_id_id = _add_from_template(
        ed, "RunIdPrefix", (2300, 2200),
        widget_values=["fml2v_var_d_audio_loop", "%Y%m%d_%H%M%S"],
        title="RunIdPrefix",
        size=(290, 100),
    )
    # Convert filename_prefix widget → input on VHS_VideoCombine (idempotent).
    vhs = ed.find_node(_BENCH_VHS_COMBINE)
    try:
        prefix_slot = WorkflowEditor.find_input_slot(vhs, "filename_prefix")
    except ValueError:
        vhs.setdefault("inputs", []).append({
            "name": "filename_prefix",
            "type": "STRING",
            "widget": {"name": "filename_prefix"},
            "link": None,
        })
        prefix_slot = len(vhs["inputs"]) - 1
    ed.add_link(run_id_id, 0, _BENCH_VHS_COMBINE, prefix_slot, "STRING")
    log(f"+ RunIdPrefix #{run_id_id} → VHS_VideoCombine #{_BENCH_VHS_COMBINE}.filename_prefix (slot {prefix_slot})")

    # --- 7. LoopConfigValidator (config-consistency sanity check) ---
    validator_id = _add_from_template(
        ed, "LoopConfigValidator", (2600, 2200),
        # widgets: [window_seconds, overlap_seconds, fps, length_widget,
        #          width_widget, height_widget, schedule, dim_rule,
        #          fudge_factor, audio_path, kf_batch]
        widget_values=[19.88, 2.0, 25, 0, 0, 0, "", "div_by_32", 0.2, "", 0],
        title="LoopConfigValidator",
        size=(290, 240),
    )
    ed.add_link(trim_audio_id, 0, validator_id, 0, "AUDIO")
    ed.add_link(overlap_const_id, 0, validator_id, 5, "FLOAT")      # overlap_seconds
    ed.add_link(fp_id, 2, validator_id, 2, "INT")                   # length ← FramePlanner.frames
    ed.add_link(fp_id, 0, validator_id, 3, "INT")                   # width
    ed.add_link(fp_id, 1, validator_id, 4, "INT")                   # height

    # Stash IDs.
    ed.wf.setdefault("properties", {})["build_fml2v_phase6"] = {
        "get_initial_for_concat": get_initial_id,
        "latent_concat": concat_id,
        "trim_video_latent_to_audio": trim_latent_id,
        "trim_image_batch_to_audio": trim_img_id,
        "run_id_prefix": run_id_id,
        "loop_config_validator": validator_id,
    }
    log("= Stashed Phase 6 node IDs in wf['properties']['build_fml2v_phase6']")


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
