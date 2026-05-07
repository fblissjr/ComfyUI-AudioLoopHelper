"""Tests for scripts/apply_smart_image_resize.py.

Last updated: 2026-05-07

Swaps the canonical init-image resize node (Node 445, ImageResizeKJv2)
for our LTXSmartImageResize. Tests assert: type swap, slot remap on
incoming wires (width/height preserved, mask dropped), output IMAGE
consumer untouched, idempotence, --revert restores original shape.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "apply_smart_image_resize.py"
CANONICAL_DIR = REPO_ROOT / "example_workflows"

NODE_445_RESIZE = 445
LEGACY_TYPE = "ImageResizeKJv2"
NEW_TYPE = "LTXSmartImageResize"


def _run(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd, check=False,
    )


@pytest.fixture
def staged_workflows(tmp_path: Path) -> Path:
    """Copy example_workflows into tmp_path and revert to pre-apply state."""
    dst = tmp_path / "example_workflows"
    dst.mkdir()
    for jp in sorted(CANONICAL_DIR.glob("*.json")):
        shutil.copy2(jp, dst / jp.name)
    # Normalize: revert if anything is already migrated.
    subprocess.run(
        [sys.executable, str(SCRIPT), "--workflows-dir", str(dst), "--revert"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    )
    return dst


def test_apply_swaps_node_445_in_loop_workflow(staged_workflows):
    target = staged_workflows / "audio-loop-music-video_latent.json"
    r = _run("--workflows-dir", str(staged_workflows))
    assert r.returncode == 0, r.stderr
    ed = WorkflowEditor(target)
    n = ed.find_node(NODE_445_RESIZE)
    assert n.get("type") == NEW_TYPE, f"expected {NEW_TYPE}, got {n.get('type')}"


def test_apply_remaps_width_and_height_links(staged_workflows):
    target = staged_workflows / "audio-loop-music-video_latent.json"
    _run("--workflows-dir", str(staged_workflows))
    ed = WorkflowEditor(target)
    n = ed.find_node(NODE_445_RESIZE)
    # New schema: image=0, width=1, height=2, keep_proportion=3, crop_position=4
    name_to_slot = {inp.get("name"): i for i, inp in enumerate(n.get("inputs", []))}
    assert name_to_slot.get("image") == 0
    assert name_to_slot.get("width") == 1
    assert name_to_slot.get("height") == 2

    # The incoming width/height links must reach the new slot indices.
    for input_name, expected_slot in (("width", 1), ("height", 2)):
        link = ed.find_link_to_slot(NODE_445_RESIZE, expected_slot)
        assert link is not None, f"input {input_name!r} not wired to slot {expected_slot}"
        # Sanity: source is LTXFramePlanner (id 1634 in canonical)
        assert link[1] == 1634, (
            f"width/height not from LTXFramePlanner; got source #{link[1]}"
        )


def test_apply_drops_mask_input(staged_workflows):
    """After apply, Node 445 must not expose a mask input slot.

    Whether a mask wire was dropped depends on the workflow's pre-state
    (some canonical clones may have lost the wire historically). The
    load-bearing post-state assertion is: the new schema has no mask
    input. If a mask link existed pre-apply, it must also be absent
    from the global links array (no dangling references).
    """
    target = staged_workflows / "audio-loop-music-video_latent.json"
    pre_ed = WorkflowEditor(target)
    pre_445 = pre_ed.find_node(NODE_445_RESIZE)
    pre_mask_link = next(
        (inp.get("link") for inp in pre_445.get("inputs", []) if inp.get("name") == "mask"),
        None,
    )

    _run("--workflows-dir", str(staged_workflows))
    ed = WorkflowEditor(target)

    # Post-state: no mask slot in inputs.
    n = ed.find_node(NODE_445_RESIZE)
    names = {inp.get("name") for inp in n.get("inputs", [])}
    assert "mask" not in names, "new schema should not expose 'mask' input"

    # If the pre-state had a mask wire, it must be gone from the global
    # links list (no dangling references).
    if pre_mask_link is not None:
        link_ids = {l[0] for l in ed.wf.get("links", []) if isinstance(l, list)}
        assert pre_mask_link not in link_ids, (
            f"mask link {pre_mask_link} should have been removed"
        )


def test_apply_preserves_image_output_consumer(staged_workflows):
    target = staged_workflows / "audio-loop-music-video_latent.json"
    pre_ed = WorkflowEditor(target)
    pre_445 = pre_ed.find_node(NODE_445_RESIZE)
    pre_out_links = list(pre_445.get("outputs", [])[0].get("links") or [])
    assert pre_out_links, "fixture invalid: pre-state IMAGE output had no consumers"

    _run("--workflows-dir", str(staged_workflows))
    ed = WorkflowEditor(target)
    n = ed.find_node(NODE_445_RESIZE)
    out_links = list(n.get("outputs", [])[0].get("links") or [])
    # Same set of consumer link IDs.
    assert set(out_links) == set(pre_out_links), (
        f"IMAGE consumers changed: {pre_out_links!r} -> {out_links!r}"
    )


def test_apply_widgets_in_new_format(staged_workflows):
    target = staged_workflows / "audio-loop-music-video_latent.json"
    _run("--workflows-dir", str(staged_workflows))
    ed = WorkflowEditor(target)
    n = ed.find_node(NODE_445_RESIZE)
    wv = n.get("widgets_values", [])
    assert len(wv) == 4, f"expected 4 widgets, got {wv}"
    assert isinstance(wv[0], int) and isinstance(wv[1], int), (
        f"width/height should be ints: {wv}"
    )
    assert wv[2] is True, f"keep_proportion should default True: {wv}"
    assert wv[3] in ("center", "top", "bottom", "left", "right")


def test_apply_idempotent(staged_workflows):
    _run("--workflows-dir", str(staged_workflows))
    ed1 = WorkflowEditor(staged_workflows / "audio-loop-music-video_latent.json")
    n1_inputs = ed1.find_node(NODE_445_RESIZE).get("inputs", [])

    _run("--workflows-dir", str(staged_workflows))
    ed2 = WorkflowEditor(staged_workflows / "audio-loop-music-video_latent.json")
    n2_inputs = ed2.find_node(NODE_445_RESIZE).get("inputs", [])
    assert n1_inputs == n2_inputs, "second apply changed inputs"


def test_revert_restores_legacy_type(staged_workflows):
    target = staged_workflows / "audio-loop-music-video_latent.json"
    _run("--workflows-dir", str(staged_workflows))
    ed_mid = WorkflowEditor(target)
    assert ed_mid.find_node(NODE_445_RESIZE).get("type") == NEW_TYPE

    _run("--workflows-dir", str(staged_workflows), "--revert")
    ed_post = WorkflowEditor(target)
    n = ed_post.find_node(NODE_445_RESIZE)
    assert n.get("type") == LEGACY_TYPE
    # Width/height links restored to old slot indices (slots 2 + 3).
    for expected_slot, expected_name in ((2, "width"), (3, "height")):
        slot_name = n["inputs"][expected_slot].get("name") if expected_slot < len(n["inputs"]) else None
        assert slot_name == expected_name, (
            f"after revert, slot {expected_slot} should be {expected_name!r}, got {slot_name!r}"
        )


def test_apply_skips_workflows_without_node_445(staged_workflows):
    # The retake variant has no Node 445; the apply script should skip cleanly.
    r = _run("--workflows-dir", str(staged_workflows))
    assert r.returncode == 0
    retake = staged_workflows / "audio-loop-music-video_retake.json"
    if retake.exists():
        ed = WorkflowEditor(retake)
        # If 445 is absent in the source, post-apply it should still be absent
        # (or, if present with a non-target type, untouched).
        assert not ed.has_node(NODE_445_RESIZE) or \
               ed.find_node(NODE_445_RESIZE).get("type") != NEW_TYPE


def test_audit_passes_after_apply(staged_workflows):
    """Migrated workflows must still pass the structural audit."""
    _run("--workflows-dir", str(staged_workflows))
    audit = REPO_ROOT / "scripts" / "audit_workflows.py"
    for jp in sorted(staged_workflows.glob("*.json")):
        r = subprocess.run(
            [sys.executable, str(audit), str(jp)],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )
        assert r.returncode == 0, f"audit failed on {jp.name}:\n{r.stdout}\n{r.stderr}"
