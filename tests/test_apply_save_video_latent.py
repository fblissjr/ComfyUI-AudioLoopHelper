"""Tests for scripts/apply_save_video_latent.py.

Last updated: 2026-05-07

Phase A enabler for the seam-zone diagnostic (P5 Phase A,
`internal/design/polish_passes_design.md`). The apply script stages a
loop-workflow variant with a `SaveLatent` node wired to the top-level
`LTXVSeparateAVLatent.video_latent` output (Node 245 in canonical), so
a render emits a `.latent` file consumable by
`scripts/diagnose_overlap_seams.py`.

Tests assert:
  - apply produces an `internal/workflows/loop_with_save_latent.draft.json`
    containing a `SaveLatent` node wired from Node 245 output 0
  - widget shape: filename_prefix string default
  - idempotence: re-apply is a no-op
  - --dry-run does not write the draft
  - --revert removes the draft
  - source workflow at `example_workflows/audio-loop-music-video_latent.json`
    is never mutated
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "apply_save_video_latent.py"
SOURCE = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

# Canonical loop-workflow node id for the top-level LTXVSeparateAVLatent.
SEPARATE_AV_NODE_ID = 245
SAVE_LATENT_TYPE = "SaveLatent"
NEW_NODE_TITLE = "Save video_latent (seam diagnostic)"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


@pytest.fixture
def tmp_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Copy source workflow + return (input_path, output_path) under tmp_path."""
    in_dir = tmp_path / "example_workflows"
    in_dir.mkdir()
    in_path = in_dir / SOURCE.name
    shutil.copy2(SOURCE, in_path)

    out_dir = tmp_path / "internal" / "workflows"
    out_dir.mkdir(parents=True)
    out_path = out_dir / "loop_with_save_latent.draft.json"
    return in_path, out_path


def _apply(in_path: Path, out_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run(
        "--input", str(in_path.relative_to(REPO_ROOT) if in_path.is_relative_to(REPO_ROOT) else in_path),
        "--output", str(out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path),
        *extra,
    )


def _find_save_latent_node(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf["nodes"]:
        if n.get("type") == SAVE_LATENT_TYPE and n.get("title") == NEW_NODE_TITLE:
            return n
    return None


def test_apply_creates_draft_with_save_latent_wired(tmp_paths):
    in_path, out_path = tmp_paths
    result = _apply(in_path, out_path)
    assert result.returncode == 0, f"apply failed: {result.stdout}\n{result.stderr}"
    assert out_path.exists(), "draft was not written"

    ed = WorkflowEditor(out_path)
    node = _find_save_latent_node(ed)
    assert node is not None, "SaveLatent node not present in draft"

    # Wired to Node 245 video_latent output.
    samples_slot = WorkflowEditor.find_input_slot(node, "samples")
    link = ed.find_link_to_slot(node["id"], samples_slot)
    assert link is not None, "samples input is not wired"
    src_node_id, src_slot = link[1], link[2]
    assert src_node_id == SEPARATE_AV_NODE_ID, (
        f"samples wired from #{src_node_id} (expected #{SEPARATE_AV_NODE_ID})"
    )
    assert src_slot == 0, f"wired from output slot {src_slot} (expected 0=video_latent)"

    # Widget shape: one string (filename_prefix).
    wv = node.get("widgets_values", [])
    assert len(wv) == 1 and isinstance(wv[0], str), (
        f"expected widgets_values=[<filename_prefix:str>], got {wv!r}"
    )


def test_apply_does_not_mutate_source(tmp_paths):
    in_path, out_path = tmp_paths
    before = in_path.read_bytes()
    result = _apply(in_path, out_path)
    assert result.returncode == 0
    after = in_path.read_bytes()
    assert before == after, "source workflow was mutated"


def test_apply_is_idempotent(tmp_paths):
    in_path, out_path = tmp_paths
    r1 = _apply(in_path, out_path)
    assert r1.returncode == 0
    ed1 = WorkflowEditor(out_path)
    # Count only ACTIVE (mode=0) SaveLatents — the canonical source workflow
    # carries a bypassed SaveLatent toggle from apply_run_id_layout.py
    # (the assembled-latent capture point), which we must not double-count
    # against the per-iter SaveLatent this apply script adds.
    n_save = sum(
        1 for n in ed1.wf["nodes"]
        if n.get("type") == SAVE_LATENT_TYPE and n.get("mode", 0) == 0
    )
    assert n_save == 1

    r2 = _apply(in_path, out_path)
    assert r2.returncode == 0
    ed2 = WorkflowEditor(out_path)
    n_save_2 = sum(
        1 for n in ed2.wf["nodes"]
        if n.get("type") == SAVE_LATENT_TYPE and n.get("mode", 0) == 0
    )
    assert n_save_2 == 1, "second apply added a duplicate SaveLatent"


def test_dry_run_does_not_write_draft(tmp_paths):
    in_path, out_path = tmp_paths
    assert not out_path.exists()
    result = _apply(in_path, out_path, "--dry-run")
    assert result.returncode == 0, f"dry-run failed: {result.stderr}"
    assert not out_path.exists(), "dry-run wrote the draft anyway"


def test_revert_removes_draft(tmp_paths):
    in_path, out_path = tmp_paths
    _apply(in_path, out_path)
    assert out_path.exists()

    result = _apply(in_path, out_path, "--revert")
    assert result.returncode == 0
    assert not out_path.exists(), "revert did not remove the draft"


def test_revert_when_no_draft_is_noop(tmp_paths):
    in_path, out_path = tmp_paths
    assert not out_path.exists()
    result = _apply(in_path, out_path, "--revert")
    assert result.returncode == 0, f"revert on absent draft failed: {result.stderr}"
