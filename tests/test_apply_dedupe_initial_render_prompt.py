"""Tests for scripts/apply_dedupe_initial_render_prompt.py.

Last updated: 2026-05-07

Phase 1 of the workflow-organization rework. Removes the
duplicate-prompt footgun where Node 169 (CLIPTextEncode for the
initial render) repeats the schedule's 0:00 entry verbatim. Replaces
it with a top-level `ConditioningSelectByIteration` reading
`conditioning_list[0]` from the existing batch encoder (Node 1615),
so the schedule string is the single source of truth.

Tests assert:
  - new ConditioningSelectByIteration node present, current_iteration=0
  - new selector wired from Node 1615 output 0 (conditioning_list)
  - new selector's CONDITIONING fans out to:
      Node 164.positive (was link 379)
      Node 420.conditioning (was link 1201)
  - Node 169 removed
  - links 379, 1201, 1256 removed
  - Node 1615 output 0 still wires to its prior consumer (link 3046 intact)
  - Node 416 (DualCLIPLoader) still feeds Node 507 + Node 1615 (unchanged)
  - source workflow unchanged
  - idempotent, --dry-run, --revert work
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "apply_dedupe_initial_render_prompt.py"
SOURCE = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

NODE_169_CLIP_TEXT_ENCODE = 169
NODE_164_LTXV_CONDITIONING = 164
NODE_420_ZERO_OUT = 420
NODE_416_DUAL_CLIP = 416
NODE_507_CLIP_TEXT_ENCODE = 507
NODE_1615_BATCH_ENCODER = 1615
NODE_1582_LOOP_CONTROLLER = 1582
NODE_1560_LOOP_PLANNER = 1560

REMOVED_LINK_169_TO_164_POSITIVE = 379
REMOVED_LINK_169_TO_420 = 1201
REMOVED_LINK_416_TO_169 = 1256
PRESERVED_LINK_1615_TO_LOOP_SUBGRAPH = 3046

NEW_NODE_TITLE = "Initial render conditioning (from schedule[0])"
NEW_NODE_TYPE = "ConditioningSelectByIteration"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


@pytest.fixture
def tmp_paths(tmp_path: Path) -> tuple[Path, Path]:
    in_dir = tmp_path / "example_workflows"
    in_dir.mkdir()
    in_path = in_dir / SOURCE.name
    shutil.copy2(SOURCE, in_path)

    out_dir = tmp_path / "internal" / "workflows"
    out_dir.mkdir(parents=True)
    out_path = out_dir / "loop_dedupe_initial_prompt.draft.json"
    return in_path, out_path


def _apply(in_path: Path, out_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run("--input", str(in_path), "--output", str(out_path), *extra)


def _find_new_selector(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf["nodes"]:
        if n.get("type") == NEW_NODE_TYPE and n.get("title") == NEW_NODE_TITLE:
            return n
    return None


def _link_present(ed: WorkflowEditor, link_id: int) -> bool:
    return any(lk[0] == link_id for lk in ed.wf["links"])


def test_apply_creates_selector_with_current_iteration_zero(tmp_paths):
    in_path, out_path = tmp_paths
    r = _apply(in_path, out_path)
    assert r.returncode == 0, f"apply failed: {r.stdout}\n{r.stderr}"
    ed = WorkflowEditor(out_path)
    sel = _find_new_selector(ed)
    assert sel is not None, "selector node not present"

    # current_iteration widget value = 0
    wv = sel.get("widgets_values", [])
    assert len(wv) >= 1 and wv[0] == 0, (
        f"expected widgets_values[0]==0 (current_iteration), got {wv!r}"
    )


def test_selector_wired_from_batch_encoder(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    ed = WorkflowEditor(out_path)
    sel = _find_new_selector(ed)
    assert sel is not None

    cond_list_slot = WorkflowEditor.find_input_slot(sel, "conditioning_list")
    link = ed.find_link_to_slot(sel["id"], cond_list_slot)
    assert link is not None, "conditioning_list input not wired"
    assert link[1] == NODE_1615_BATCH_ENCODER, (
        f"conditioning_list wired from #{link[1]}, expected #{NODE_1615_BATCH_ENCODER}"
    )
    assert link[2] == 0, f"wired from output slot {link[2]}, expected 0"


def test_selector_fans_out_to_both_node169_consumers(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    ed = WorkflowEditor(out_path)
    sel = _find_new_selector(ed)
    assert sel is not None

    n164 = ed.find_node(NODE_164_LTXV_CONDITIONING)
    n420 = ed.find_node(NODE_420_ZERO_OUT)

    # Node 164 positive must now be wired from the selector
    pos_slot = WorkflowEditor.find_input_slot(n164, "positive")
    pos_link = ed.find_link_to_slot(NODE_164_LTXV_CONDITIONING, pos_slot)
    assert pos_link is not None, "Node 164 positive is not wired"
    assert pos_link[1] == sel["id"], (
        f"Node 164 positive wired from #{pos_link[1]}, expected new selector #{sel['id']}"
    )

    # Node 420 conditioning must now be wired from the selector
    zo_slot = WorkflowEditor.find_input_slot(n420, "conditioning")
    zo_link = ed.find_link_to_slot(NODE_420_ZERO_OUT, zo_slot)
    assert zo_link is not None, "Node 420 conditioning is not wired"
    assert zo_link[1] == sel["id"], (
        f"Node 420 conditioning wired from #{zo_link[1]}, expected new selector #{sel['id']}"
    )


def test_node_169_removed(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    ed = WorkflowEditor(out_path)
    assert not ed.has_node(NODE_169_CLIP_TEXT_ENCODE), "Node 169 was not removed"


def test_dead_links_removed(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    ed = WorkflowEditor(out_path)
    assert not _link_present(ed, REMOVED_LINK_169_TO_164_POSITIVE), \
        "stale link 379 (Node 169 -> Node 164.positive) was not removed"
    assert not _link_present(ed, REMOVED_LINK_169_TO_420), \
        "stale link 1201 (Node 169 -> Node 420) was not removed"
    assert not _link_present(ed, REMOVED_LINK_416_TO_169), \
        "stale link 1256 (Node 416 -> Node 169) was not removed"


def test_other_clip_consumers_intact(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    ed = WorkflowEditor(out_path)
    n416 = ed.find_node(NODE_416_DUAL_CLIP)
    consumers = {lk[3] for lk in ed.wf["links"] if lk[1] == NODE_416_DUAL_CLIP}
    assert NODE_507_CLIP_TEXT_ENCODE in consumers, "Node 507 lost its CLIP wire"
    assert NODE_1615_BATCH_ENCODER in consumers, "Node 1615 lost its CLIP wire"
    # Make sure Node 416's outputs[0].links does not still mention link 1256
    out_links = n416["outputs"][0].get("links") or []
    assert REMOVED_LINK_416_TO_169 not in out_links, (
        "Node 416 outputs[0].links still references stale link 1256"
    )


def test_batch_encoder_existing_consumer_preserved(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    ed = WorkflowEditor(out_path)
    n1615 = ed.find_node(NODE_1615_BATCH_ENCODER)
    out_links = n1615["outputs"][0].get("links") or []
    assert PRESERVED_LINK_1615_TO_LOOP_SUBGRAPH in out_links, (
        f"Node 1615 lost its prior consumer link {PRESERVED_LINK_1615_TO_LOOP_SUBGRAPH}"
    )
    # And the new selector also reads from output 0
    assert len(out_links) >= 2, (
        f"expected at least 2 consumers on Node 1615 output 0, got {out_links!r}"
    )


def test_apply_does_not_mutate_source(tmp_paths):
    in_path, out_path = tmp_paths
    before = in_path.read_bytes()
    _apply(in_path, out_path)
    after = in_path.read_bytes()
    assert before == after, "source workflow was mutated"


def test_apply_is_idempotent(tmp_paths):
    in_path, out_path = tmp_paths
    r1 = _apply(in_path, out_path)
    assert r1.returncode == 0
    ed1 = WorkflowEditor(out_path)
    selector_count_1 = sum(1 for n in ed1.wf["nodes"]
                           if n.get("type") == NEW_NODE_TYPE
                           and n.get("title") == NEW_NODE_TITLE)
    assert selector_count_1 == 1

    r2 = _apply(in_path, out_path)
    assert r2.returncode == 0
    ed2 = WorkflowEditor(out_path)
    selector_count_2 = sum(1 for n in ed2.wf["nodes"]
                           if n.get("type") == NEW_NODE_TYPE
                           and n.get("title") == NEW_NODE_TITLE)
    assert selector_count_2 == 1, "second apply added a duplicate selector"


def test_dry_run_does_not_write_draft(tmp_paths):
    in_path, out_path = tmp_paths
    r = _apply(in_path, out_path, "--dry-run")
    assert r.returncode == 0
    assert not out_path.exists()


def test_revert_removes_draft(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    assert out_path.exists()
    r = _apply(in_path, out_path, "--revert")
    assert r.returncode == 0
    assert not out_path.exists()


def test_batch_encoder_stride_and_duration_now_from_planner(tmp_paths):
    """Cycle break: batch encoder's stride_seconds + audio_duration must
    source from AudioLoopPlanner (no current_iteration dependency), not
    AudioLoopController (which transitively pulls current_iteration).
    """
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    ed = WorkflowEditor(out_path)

    encoder = ed.find_node(NODE_1615_BATCH_ENCODER)
    for input_name in ("stride_seconds", "audio_duration"):
        slot = WorkflowEditor.find_input_slot(encoder, input_name)
        link = ed.find_link_to_slot(NODE_1615_BATCH_ENCODER, slot)
        assert link is not None, f"batch encoder.{input_name} is not wired"
        assert link[1] == NODE_1560_LOOP_PLANNER, (
            f"batch encoder.{input_name} wired from #{link[1]}, "
            f"expected planner #{NODE_1560_LOOP_PLANNER}"
        )

    # Planner must have outputs[2]+[3] in the saved JSON for ComfyUI's
    # loader to surface them on workflow load.
    planner = ed.find_node(NODE_1560_LOOP_PLANNER)
    outs = planner.get("outputs", [])
    assert len(outs) >= 4, (
        f"planner outputs[] should be extended to >=4 slots, has {len(outs)}"
    )
    assert outs[2].get("name") == "stride_seconds"
    assert outs[3].get("name") == "audio_duration"


def test_audit_passes_on_draft(tmp_paths):
    """Sanity: audit runs without ERR on the draft."""
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    audit_script = REPO_ROOT / "scripts" / "audit_workflows.py"
    r = subprocess.run(
        [sys.executable, str(audit_script), str(out_path)],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert r.returncode == 0, (
        f"audit reported ERR(s) on the draft:\n{r.stdout}\n{r.stderr}"
    )
