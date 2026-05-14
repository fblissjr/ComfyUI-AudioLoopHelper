"""Build four variants of the benchmark workflow.

Last updated: 2026-05-14

Source: example_workflows/benchmark_workflows/fml2v_sage_masked_attn_benchmark.json
(adapted from RuneXX_LTX-2.3-Workflows; produces noticeably better video +
motion quality than our canonical looped workflow).

The benchmark stacks three structural choices on top of our canonical:
  - 3 keyframe guides per generation (first/middle/last via LTXVAddGuideMulti)
  - Two-pass distilled sampling (8-step euler_ancestral_cfg_pp -> spatial
    upsample -> 4-step euler_cfg_pp refine at sigma 0.85 -> 0)
  - LTXVLatentUpsampler 2x between passes

But the benchmark generates audio (LTXVEmptyLatentAudio -> joint t2v+t2a)
rather than consuming it. Var D is the audio-input adaptation.

Variants here:

  Var A (no-middle-keyframe)
      Pass 1: 3 guides -> 2 guides (drop middle). Pass 2 unchanged
      (already 2 guides). Tests: does the middle anchor matter?

  Var B (first-keyframe-only)
      Both passes: -> 1 guide (only first frame). Closest match to our
      canonical's single-anchor pattern but with two-pass refine intact.
      Tests: how much of the quality lift comes from multi-keyframe?

  Var C (single-pass)
      Bypass the upsampler + pass-2 chain. Decoder reads pass-1 crop
      output directly. Same 3 keyframes, single 8-step pass.
      Tests: how much comes from the refine pass + spatial upsample?

  Var D (audio-input)
      Replace LTXVEmptyLatentAudio with LoadAudio -> TrimAudioDuration ->
      LTXVAudioVAEEncode -> SetLatentNoiseMask(mask=0) chain. Audio is
      now an input (frozen via noise_mask=0) like our canonical, instead
      of co-generated with video. Tests the benchmark's video stack on
      the audio-driven path that our canonical targets.

All four variants also swap the slow generic VAEDecodeTiled[512,64,4096,8]
to LTXVTiledVAEDecode[1,1,1,True,'auto','auto'] (~3x faster cold-pass on
24GB+; see scripts/apply_no_tile_vae_decode.py for empirical timings).

A/B render workflow:
  Source benchmark vs A vs B vs C vs D — same keyframe images, same seed
  (42 on RandomNoise #15, 43 on RandomNoise #14). Var D additionally
  needs an audio file path set on its LoadAudio node.

Usage:
    uv run --group dev python scripts/build_benchmark_ablations.py
    uv run --group dev python scripts/build_benchmark_ablations.py --revert
    uv run --group dev python scripts/build_benchmark_ablations.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_ltx_decoder import _LTX_TYPE, _swap_to_ltx
from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "example_workflows" / "benchmark_workflows" / "fml2v_sage_masked_attn_benchmark.json"
OUT_DIR = REPO_ROOT / "example_workflows" / "benchmark_workflows"

VAR_A = OUT_DIR / "fml2v_var_a_no_middle_keyframe.json"
VAR_B = OUT_DIR / "fml2v_var_b_first_keyframe_only.json"
VAR_C = OUT_DIR / "fml2v_var_c_single_pass.json"
VAR_D = OUT_DIR / "fml2v_var_d_audio_input.json"

# Node IDs from the source benchmark
GUIDEMULTI_PASS1 = 2221   # LTXVAddGuideMulti, 3 guides (first, middle, last)
GUIDEMULTI_PASS2 = 2182   # LTXVAddGuideMulti, 2 guides (first, last)
UPSAMPLER = 25            # LTXVLatentUpsampler (spatial 2x)
CONCAT_PASS2 = 34         # LTXVConcatAVLatent (pass 2 input)
SAMPLER_PASS2 = 21        # SamplerCustomAdvanced (pass 2)
SEPARATE_PASS2 = 146      # LTXVSeparateAVLatent (post-pass-2)
CROP_PASS2 = 2156         # LTXVCropGuides (post-pass-2)
SEPARATE_PASS1 = 18       # LTXVSeparateAVLatent (post-pass-1)  -> audio source for Var C
CROP_PASS1 = 2222         # LTXVCropGuides (post-pass-1)        -> video source for Var C
VAE_DECODE_VIDEO = 149    # VAEDecodeTiled  (samples input rewired in Var C)
VAE_DECODE_AUDIO = 150    # LTXVAudioVAEDecode (samples input rewired in Var C)

# Audio nodes for Var D
EMPTY_AUDIO_LATENT = 9    # LTXVEmptyLatentAudio  (bypassed in Var D)
SET_LATENT_AUDIO = 2215   # SetNode 'latent_audio' (rewired in Var D)
GET_VAE_AUDIO = 148       # GetNode 'vae_audio'    (reused for new encode chain)

VAR_D_DEFAULT_AUDIO_FILE = "your_audio.mp3"   # user replaces in UI

REQUIRED_SOURCE_NODES = (
    GUIDEMULTI_PASS1, GUIDEMULTI_PASS2, UPSAMPLER, CONCAT_PASS2,
    SAMPLER_PASS2, SEPARATE_PASS2, CROP_PASS2, SEPARATE_PASS1,
    CROP_PASS1, VAE_DECODE_VIDEO, VAE_DECODE_AUDIO,
    EMPTY_AUDIO_LATENT, SET_LATENT_AUDIO, GET_VAE_AUDIO,
)


# ---------------------------------------------------------------------------
# DynamicCombo (LTXVAddGuideMulti) helpers
# ---------------------------------------------------------------------------

def _input_name_for_guide(field: str, idx: int) -> str:
    return f"num_guides.{field}_{idx}"


def _split_guide_input_name(name: str) -> tuple[str, int] | None:
    """Return (field, index) for an input like 'num_guides.image_3' -> ('image', 3),
    or None for non-guide inputs."""
    if not name.startswith("num_guides."):
        return None
    rest = name.split(".", 1)[1]  # e.g. "image_3", "frame_idx_3", "strength_3"
    parts = rest.rsplit("_", 1)
    if len(parts) != 2:
        return None
    field, num_str = parts
    try:
        return field, int(num_str)
    except ValueError:
        return None


def _drop_links_for_inputs(ed: WorkflowEditor, node: dict, names_to_drop: set[str]) -> None:
    """Remove top-level links connected to any input whose name is in names_to_drop."""
    for inp in node.get("inputs", []):
        if inp.get("name") in names_to_drop and inp.get("link") is not None:
            ed.remove_link(inp["link"])


def _renumber_guide_input(inp: dict, new_idx: int) -> None:
    """In-place rename of a guide input slot to a new index (e.g. image_3 -> image_2)."""
    split = _split_guide_input_name(inp["name"])
    if split is None:
        return
    field, _ = split
    inp["name"] = _input_name_for_guide(field, new_idx)
    # The schema also stores 'label' for image inputs (e.g. 'image_3') and a
    # nested {'widget': {'name': ...}} for the float/int slots. Keep them in sync.
    if "label" in inp:
        inp["label"] = f"{field}_{new_idx}"
    widget = inp.get("widget")
    if isinstance(widget, dict) and "name" in widget:
        widget["name"] = inp["name"]


def _surviving_widget_tuples(old_widgets: list, keep_positions: list[int]) -> list[tuple]:
    """Pick (frame_idx, strength) pairs at the kept 1-based positions from
    the LTXVAddGuideMulti widgets_values shape [num_str, f1, s1, f2, s2, ...].

    Derives from the source workflow rather than re-typing literals so a
    benchmark widget tweak doesn't silently diverge in the variants.
    """
    return [(old_widgets[1 + (p - 1) * 2], old_widgets[2 + (p - 1) * 2]) for p in keep_positions]


def _reduce_guide_multi(
    ed: WorkflowEditor,
    node_id: int,
    keep_positions: list[int],
) -> None:
    """Reduce an LTXVAddGuideMulti node to the guides in `keep_positions`
    (1-based, original indices). Surviving frame_idx + strength widget
    values are read from the source node so variants track the source."""
    node = ed.find_node(node_id)
    inputs = node.get("inputs", [])
    surviving_widgets = _surviving_widget_tuples(node.get("widgets_values", []), keep_positions)

    drop_names: set[str] = set()
    rename_map: dict[str, int] = {}  # input-name -> new index

    for inp in inputs:
        split = _split_guide_input_name(inp["name"])
        if split is None:
            continue
        _, old_idx = split
        if old_idx not in keep_positions:
            drop_names.add(inp["name"])
        else:
            new_idx = keep_positions.index(old_idx) + 1
            if new_idx != old_idx:
                rename_map[inp["name"]] = new_idx

    # Drop dangling links FIRST (before we mutate inputs)
    _drop_links_for_inputs(ed, node, drop_names)

    new_inputs: list = []
    for inp in inputs:
        if inp.get("name") in drop_names:
            continue
        if inp["name"] in rename_map:
            _renumber_guide_input(inp, rename_map[inp["name"]])
        new_inputs.append(inp)
    node["inputs"] = new_inputs

    new_widgets: list = [str(len(keep_positions))]
    for f_idx, strength in surviving_widgets:
        new_widgets.extend([f_idx, strength])
    node["widgets_values"] = new_widgets


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------

def _build_var_a(ed: WorkflowEditor) -> None:
    """Drop middle keyframe from pass-1 LTXVAddGuideMulti (3 -> 2 guides).
    Pass 2 (#2182) untouched — already 2 guides."""
    _reduce_guide_multi(ed, GUIDEMULTI_PASS1, keep_positions=[1, 3])


def _build_var_b(ed: WorkflowEditor) -> None:
    """First-frame-only: both LTXVAddGuideMulti nodes -> 1 guide."""
    _reduce_guide_multi(ed, GUIDEMULTI_PASS1, keep_positions=[1])
    _reduce_guide_multi(ed, GUIDEMULTI_PASS2, keep_positions=[1])


def _swap_decoder_to_ltx_single_tile(ed: WorkflowEditor) -> None:
    """VAEDecodeTiled[512,64,4096,8] -> LTXVTiledVAEDecode single-tile.
    Single-tile is ~3x faster cold-pass on 24GB+ (see apply_no_tile_vae_decode.py
    for empirical timings). Reuses the swap helper from apply_ltx_decoder.py
    so the slot/link rewrite logic only lives in one place; overrides the
    widget defaults from [2,2,1] to [1,1,1] for the 24GB+ path."""
    node = ed.find_node(VAE_DECODE_VIDEO)
    if node.get("type") == _LTX_TYPE:
        return
    if _swap_to_ltx(node, ed.wf["links"]):
        node["widgets_values"] = [1, 1, 1, True, "auto", "auto"]


def _build_var_c(ed: WorkflowEditor) -> None:
    """Single-pass: bypass upsampler + pass-2 chain, rewire decoders to pass-1."""
    # Rewire the video decoder's samples input from pass-2 crop -> pass-1 crop.
    # Both outputs are slot 2 (LATENT).
    samples_slot = WorkflowEditor.find_input_slot(ed.find_node(VAE_DECODE_VIDEO), "samples")
    ed.rewire_input(VAE_DECODE_VIDEO, samples_slot, CROP_PASS1, 2, "LATENT")

    # Rewire the audio decoder's samples input from post-pass-2 separator (audio
    # slot 1) -> post-pass-1 separator (audio slot 1).
    audio_samples_slot = WorkflowEditor.find_input_slot(ed.find_node(VAE_DECODE_AUDIO), "samples")
    ed.rewire_input(VAE_DECODE_AUDIO, audio_samples_slot, SEPARATE_PASS1, 1, "LATENT")

    # Bypass the entire pass-2 chain + upsampler. Active mode is 0; bypass is 4.
    for nid in (UPSAMPLER, GUIDEMULTI_PASS2, CONCAT_PASS2, SAMPLER_PASS2,
                SEPARATE_PASS2, CROP_PASS2):
        ed.find_node(nid)["mode"] = 4


def _build_var_d(ed: WorkflowEditor) -> None:
    """Audio-input: replace LTXVEmptyLatentAudio with a LoadAudio chain
    that freezes audio via noise_mask=0 (mirrors our canonical's audio
    handling + Rune's Custom-Audio reference workflow pattern)."""
    # 1. LoadAudio
    load_audio_id = ed.add_top_level_node(
        node_type="LoadAudio",
        pos=[4500, 4500],
        size=[300, 124],
        inputs=[],
        outputs=[WorkflowEditor.out("AUDIO", "AUDIO")],
        widgets_values=[VAR_D_DEFAULT_AUDIO_FILE, None, None],
        properties={
            "cnr_id": "comfy-core",
            "Node name for S&R": "LoadAudio",
        },
        title="LoadAudio (user-supplied song)",
    )

    # 2. TrimAudioDuration -- match the video timeline (4s buffer over 3.88s)
    trim_id = ed.add_top_level_node(
        node_type="TrimAudioDuration",
        pos=[4500, 4660],
        size=[270, 82],
        inputs=[
            WorkflowEditor.io_in("audio", "AUDIO"),
        ],
        outputs=[WorkflowEditor.out("AUDIO", "AUDIO")],
        widgets_values=[0.0, 4.0],   # 97 latent-frames / 25fps = 3.88s + 0.12s buffer
        properties={
            "cnr_id": "comfy-core",
            "Node name for S&R": "TrimAudioDuration",
        },
    )

    # 3. LTXVAudioVAEEncode
    audio_enc_id = ed.add_top_level_node(
        node_type="LTXVAudioVAEEncode",
        pos=[4500, 4780],
        size=[270, 78],
        inputs=[
            WorkflowEditor.io_in("audio", "AUDIO"),
            WorkflowEditor.io_in("audio_vae", "VAE"),
        ],
        outputs=[WorkflowEditor.out("Audio Latent", "LATENT")],
        widgets_values=[],
        properties={
            "cnr_id": "comfy-core",
            "Node name for S&R": "LTXVAudioVAEEncode",
        },
    )

    # 4. SolidMask (mask value 0 = freeze; ComfyUI broadcasts to latent dims)
    solid_mask_id = ed.add_top_level_node(
        node_type="SolidMask",
        pos=[4810, 4780],
        size=[270, 106],
        inputs=[],
        outputs=[WorkflowEditor.out("MASK", "MASK")],
        widgets_values=[0, 512, 512],
        properties={
            "cnr_id": "comfy-core",
            "Node name for S&R": "SolidMask",
        },
        title="SolidMask (mask=0 -> freeze audio)",
    )

    # 5. SetLatentNoiseMask (wraps audio latent with the freeze mask)
    set_mask_id = ed.add_top_level_node(
        node_type="SetLatentNoiseMask",
        pos=[4810, 4660],
        size=[270, 80],
        inputs=[
            WorkflowEditor.io_in("samples", "LATENT"),
            WorkflowEditor.io_in("mask", "MASK"),
        ],
        outputs=[WorkflowEditor.out("LATENT", "LATENT")],
        widgets_values=[],
        properties={
            "cnr_id": "comfy-core",
            "Node name for S&R": "SetLatentNoiseMask",
        },
    )

    # Wire the new chain: LoadAudio -> Trim -> Encode -> SetMask
    ed.add_link(load_audio_id, 0, trim_id, 0, "AUDIO")
    ed.add_link(trim_id, 0, audio_enc_id, 0, "AUDIO")
    ed.add_link(GET_VAE_AUDIO, 0, audio_enc_id, 1, "VAE")
    ed.add_link(audio_enc_id, 0, set_mask_id, 0, "LATENT")
    ed.add_link(solid_mask_id, 0, set_mask_id, 1, "MASK")

    # Rewire Set_latent_audio (#2215) input from #9 LTXVEmptyLatentAudio
    # to the new SetLatentNoiseMask output.
    set_node = ed.find_node(SET_LATENT_AUDIO)
    set_slot = WorkflowEditor.find_input_slot(set_node, "LATENT")
    ed.rewire_input(SET_LATENT_AUDIO, set_slot, set_mask_id, 0, "LATENT")

    # Bypass the now-orphaned LTXVEmptyLatentAudio so the UI shows clearly
    # that the audio source has changed (keeping it in the graph lets
    # users toggle back by un-bypassing + rewiring).
    ed.find_node(EMPTY_AUDIO_LATENT)["mode"] = 4


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

VARIANTS = [
    ("Var A (no-middle-keyframe)", VAR_A, _build_var_a),
    ("Var B (first-keyframe-only)", VAR_B, _build_var_b),
    ("Var C (single-pass)", VAR_C, _build_var_c),
    ("Var D (audio-input)", VAR_D, _build_var_d),
]


def _assert_required_nodes(ed: WorkflowEditor) -> None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to build: required source node(s) missing from "
            f"{SRC}: {missing}. The benchmark workflow shape may have changed."
        )


def _build_one(label: str, output: Path, mutate, dry_run: bool) -> None:
    if dry_run:
        print(f"would build {label} -> {output}")
        ed = WorkflowEditor(SRC)
        mutate(ed)
        _swap_decoder_to_ltx_single_tile(ed)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, output)
    ed = WorkflowEditor(output)
    mutate(ed)
    _swap_decoder_to_ltx_single_tile(ed)
    ed.save()
    print(f"  wrote {output}")


def _revert() -> None:
    for _, output, _ in VARIANTS:
        if output.exists():
            output.unlink()
            print(f"removed {output}")
        else:
            print(f"{output} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true",
                    help=f"Delete the {len(VARIANTS)} variant files (does not touch source).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be built without writing.")
    args = ap.parse_args()

    if args.revert:
        _revert()
        return

    if not SRC.exists():
        raise SystemExit(f"Source benchmark workflow missing: {SRC}")

    # Pre-flight: validate source once before doing any disk work so a
    # broken source benchmark fails fast without leaving orphan copies.
    _assert_required_nodes(WorkflowEditor(SRC))

    print(f"source: {SRC}")
    for label, output, mutate in VARIANTS:
        _build_one(label, output, mutate, dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print("Next steps:")
        variant_paths = ", ".join(f"'{p}'" for _, p, _ in VARIANTS)
        print(f"  1. Validate JSON: python3 -c \"import json; [json.load(open(p)) for p in ({variant_paths})]\"")
        print("  2. Render each on identical audio + keyframes + seeds; A/B against source.")


if __name__ == "__main__":
    main()
