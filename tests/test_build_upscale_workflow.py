"""Structural tests for scripts/build_upscale_workflow.py.

Last updated: 2026-05-10

Pins two correctness invariants the audit can't catch:

1. No `LTXVImgToVideoConditionOnly` in the refine path (Issue A — would
   freeze every latent frame and defeat the refine).
2. Model chain carries the three canonical perf/VRAM patches with
   widget values byte-equal to the loop (Issue B — would OOM on 24 GB).

Full postmortem: `internal/analysis/i2v_v5_workflow_assessment.md`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import build_upscale_workflow as buw
from workflow_utils import WorkflowEditor


@pytest.fixture
def built_wf(tmp_path: Path) -> tuple[dict, dict[str, int]]:
    """Build the workflow in memory; return (wf_dict, role_to_id_map)."""
    ed = WorkflowEditor.from_scratch(tmp_path / "draft.json")
    ids = buw.build(ed)
    return ed.wf, ids


def _node_by_id(wf: dict) -> dict[int, dict]:
    return {n["id"]: n for n in wf["nodes"]}


def _types(wf: dict) -> set[str]:
    return {n.get("type") for n in wf["nodes"]}


def _link_into(wf: dict, tgt_id: int, tgt_slot: int) -> tuple | None:
    """Return the link tuple feeding (tgt_id, tgt_slot), or None."""
    for L in wf.get("links", []):
        # link: [id, src_node, src_slot, tgt_node, tgt_slot, dtype]
        if L[3] == tgt_id and L[4] == tgt_slot:
            return L
    return None


# ---------------------------------------------------------------------------
# Issue A: ConditionOnly must be absent
# ---------------------------------------------------------------------------


def test_no_ltxv_img_to_video_condition_only(built_wf):
    """The refine path must not include LTXVImgToVideoConditionOnly.

    With strength=1 and a multi-frame image batch, the node freezes the
    entire latent (noise_mask=0 everywhere) and overwrites the
    upsampler's output with a re-encoded bilinear of the source. See
    the docstring at module top.
    """
    wf, _ = built_wf
    assert "LTXVImgToVideoConditionOnly" not in _types(wf), (
        "LTXVImgToVideoConditionOnly must not appear in the upscale workflow. "
        "Drop it; the upsampled latent feeds the sampler directly. "
        "See internal/analysis/i2v_v5_workflow_assessment.md Issue A."
    )


def test_sampler_latent_input_comes_from_av_concat(built_wf):
    """SamplerCustomAdvanced.latent_image must be wired straight from
    LTXVConcatAVLatent — confirms there's no ConditionOnly-shaped node
    intercepting between the upsampler/concat and the sampler."""
    wf, ids = built_wf
    by_id = _node_by_id(wf)
    sampler_id = ids["sampler"]
    sampler = by_id[sampler_id]
    # latent_image is the 5th input on SamplerCustomAdvanced
    slot = next(
        i for i, s in enumerate(sampler["inputs"]) if s["name"] == "latent_image"
    )
    link = _link_into(wf, sampler_id, slot)
    assert link is not None, "sampler.latent_image has no incoming link"
    src_id = link[1]
    assert by_id[src_id]["type"] == "LTXVConcatAVLatent", (
        f"sampler.latent_image must come from LTXVConcatAVLatent, "
        f"got {by_id[src_id]['type']} (#{src_id})"
    )


def test_av_concat_video_input_comes_from_upsampler(built_wf):
    """LTXVConcatAVLatent.video_latent must come straight from
    LTXVLatentUpsampler — no node intercepts the upsampler output."""
    wf, ids = built_wf
    by_id = _node_by_id(wf)
    av_id = ids["av_concat"]
    # video_latent is slot 0
    link = _link_into(wf, av_id, 0)
    assert link is not None, "av_concat.video_latent has no incoming link"
    src_id = link[1]
    assert by_id[src_id]["type"] == "LTXVLatentUpsampler", (
        f"av_concat.video_latent must come from LTXVLatentUpsampler, "
        f"got {by_id[src_id]['type']} (#{src_id})"
    )


# ---------------------------------------------------------------------------
# Issue B: model chain carries the canonical perf/VRAM patches
# ---------------------------------------------------------------------------


CANONICAL_PATCH_WIDGETS = {
    "AudioLoopHelperSageAttention": ["auto_mask_aware", True, 1024],
    "LTXVChunkFeedForward": [2, 4096],
    "LTX2AttentionTunerPatch": ["", 1, 1, 1, 1, True],
}


@pytest.mark.parametrize("patch_type", list(CANONICAL_PATCH_WIDGETS))
def test_model_patch_present(built_wf, patch_type):
    """Each of the three perf/VRAM model-chain patches must appear once."""
    wf, _ = built_wf
    matches = [n for n in wf["nodes"] if n.get("type") == patch_type]
    assert len(matches) == 1, (
        f"Expected exactly one {patch_type}, found {len(matches)}. "
        f"See internal/analysis/i2v_v5_workflow_assessment.md Issue B."
    )


@pytest.mark.parametrize(
    "patch_type,expected_widgets",
    list(CANONICAL_PATCH_WIDGETS.items()),
)
def test_model_patch_widgets_match_canonical(built_wf, patch_type, expected_widgets):
    """Patch widget values must equal those in the canonical loop
    workflow so model behavior is identical."""
    wf, _ = built_wf
    node = next(n for n in wf["nodes"] if n.get("type") == patch_type)
    assert node.get("widgets_values") == expected_widgets, (
        f"{patch_type} widgets {node.get('widgets_values')!r} != "
        f"canonical {expected_widgets!r}"
    )


def test_model_chain_order_unet_to_cfg_guider(built_wf):
    """Model chain must walk: UNETLoader → AudioLoopHelperSageAttention
    → LTXVChunkFeedForward → LTX2AttentionTunerPatch → CFGGuider.

    Order matches the canonical loop. The `model.state_dict()` warning
    in root CLAUDE.md applies — module-mutating patches go between the
    loader and the guider, in this order.
    """
    wf, ids = built_wf
    by_id = _node_by_id(wf)

    expected_chain = [
        "CFGGuider",
        "LTX2AttentionTunerPatch",
        "LTXVChunkFeedForward",
        "AudioLoopHelperSageAttention",
        "UNETLoader",
    ]

    cur_id = ids["cfg_guider"]
    walked: list[str] = []
    for _ in range(len(expected_chain)):
        n = by_id[cur_id]
        walked.append(n["type"])
        # Find the MODEL input link and follow it back
        model_slot = next(
            (i for i, s in enumerate(n["inputs"]) if s.get("type") == "MODEL"),
            None,
        )
        if model_slot is None:
            break
        link = _link_into(wf, cur_id, model_slot)
        if link is None:
            break
        cur_id = link[1]

    assert walked == expected_chain, (
        f"Model chain order mismatch.\n  expected: {expected_chain}\n"
        f"  got:      {walked}"
    )


# ---------------------------------------------------------------------------
# Ready-to-run: dynamic inputs auto-track the loaded video
# ---------------------------------------------------------------------------


def test_empty_audio_frames_number_autowired_from_loaded_video(built_wf):
    """LTXVEmptyLatentAudio.frames_number must be wired from
    VHS_LoadVideo.frame_count (output slot 1).

    Hardcoding the widget defeats the workflow on any input video whose
    frame count differs from the default — AV concat produces a
    mismatched-shape latent and the sampler errors out. Autowiring
    makes the workflow accept any loaded video without per-run UI
    edits.
    """
    wf, ids = built_wf
    by_id = _node_by_id(wf)
    empty_audio_id = ids["empty_audio"]
    empty_audio = by_id[empty_audio_id]
    slot = next(
        i for i, s in enumerate(empty_audio["inputs"]) if s["name"] == "frames_number"
    )
    link = _link_into(wf, empty_audio_id, slot)
    assert link is not None, (
        "LTXVEmptyLatentAudio.frames_number has no incoming link — "
        "would lock the workflow to a single input video length."
    )
    src_id, src_slot = link[1], link[2]
    src = by_id[src_id]
    assert src["type"] == "VHS_LoadVideo", (
        f"frames_number must come from VHS_LoadVideo, got {src['type']}"
    )
    src_slot_name = src["outputs"][src_slot]["name"]
    assert src_slot_name == "frame_count", (
        f"frames_number must come from VHS_LoadVideo.frame_count, "
        f"got VHS_LoadVideo.{src_slot_name}"
    )


# ---------------------------------------------------------------------------
# Build is deterministic (idempotence supports `--revert` + re-run)
# ---------------------------------------------------------------------------


def test_build_is_deterministic(tmp_path: Path):
    """Two builds back-to-back produce identical node count + link count
    + ordered node-type sequence. Pins idempotence-on-rebuild — `--revert`
    + re-run must produce a byte-equivalent (modulo workflow uuid) draft.
    """
    ed1 = WorkflowEditor.from_scratch(tmp_path / "a.json")
    ed2 = WorkflowEditor.from_scratch(tmp_path / "b.json")
    buw.build(ed1)
    buw.build(ed2)

    assert len(ed1.wf["nodes"]) == len(ed2.wf["nodes"])
    assert len(ed1.wf["links"]) == len(ed2.wf["links"])
    assert [n["id"] for n in ed1.wf["nodes"]] == [n["id"] for n in ed2.wf["nodes"]]
    assert [n["type"] for n in ed1.wf["nodes"]] == [n["type"] for n in ed2.wf["nodes"]]
