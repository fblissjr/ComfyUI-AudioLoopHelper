"""Tests for scripts/apply_promote_schedule_to_inputs.py.

Last updated: 2026-05-07

Layout-only change to the canonical loop workflow: moves Node 1615
(the TimestampPromptScheduleBatchEncode prompt-schedule node) into
the "1. Inputs" group at the top, and expands the group's horizontal
bounding box to fit. Behavioral no-op — only `pos` (node) and
`bounding` (group) are mutated. Tests assert exactly that scope.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "apply_promote_schedule_to_inputs.py"
CANONICAL = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

NODE_1615_SCHEDULE = 1615
INPUTS_GROUP_TITLE = "1. Inputs"

# Target layout (kept in test as the contract; script reads from same constants).
TARGET_NODE_POS = [440, 260]
TARGET_NODE_SIZE = [420, 360]
TARGET_GROUP_MIN_WIDTH = 880  # must be >= node_x + node_w to encompass


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


@pytest.fixture
def staged_canonical(tmp_path: Path) -> Path:
    """Stage a copy of the canonical workflow at a *pre-apply* state.

    The shipped canonical may already be in post-apply state. Running
    --revert on the staged copy normalizes it to pre-apply regardless,
    so each test can rely on its starting point. Same lockstep pattern
    as test_apply_strip_sd3_shift_node uses.
    """
    dst_dir = tmp_path / "example_workflows"
    dst_dir.mkdir()
    dst = dst_dir / CANONICAL.name
    shutil.copy2(CANONICAL, dst)
    # Normalize: revert if migration metadata is present (canonical may
    # already be in post-apply state). --revert is a no-op on un-migrated
    # input, so it's safe to run unconditionally.
    subprocess.run(
        [sys.executable, str(SCRIPT), "--workflow", str(dst), "--revert"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    )
    return dst


def _apply(workflow_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run("--workflow", str(workflow_path), *extra)


def _find_inputs_group(ed: WorkflowEditor) -> dict | None:
    for g in ed.wf.get("groups", []):
        if g.get("title") == INPUTS_GROUP_TITLE:
            return g
    return None


def test_apply_moves_node_1615_to_target_pos(staged_canonical):
    r = _apply(staged_canonical)
    assert r.returncode == 0, f"apply failed: {r.stdout}\n{r.stderr}"
    ed = WorkflowEditor(staged_canonical)
    n = ed.find_node(NODE_1615_SCHEDULE)
    assert n["pos"] == TARGET_NODE_POS, (
        f"Node 1615 pos={n['pos']}, expected {TARGET_NODE_POS}"
    )
    assert n["size"] == TARGET_NODE_SIZE, (
        f"Node 1615 size={n['size']}, expected {TARGET_NODE_SIZE}"
    )


def test_apply_expands_inputs_group_to_fit(staged_canonical):
    _apply(staged_canonical)
    ed = WorkflowEditor(staged_canonical)
    g = _find_inputs_group(ed)
    assert g is not None, f"group {INPUTS_GROUP_TITLE!r} missing"
    bx, _by, bw, _bh = g["bounding"]
    assert bx <= TARGET_NODE_POS[0], (
        f"group origin x={bx} must be <= node x={TARGET_NODE_POS[0]}"
    )
    assert bx + bw >= TARGET_NODE_POS[0] + TARGET_NODE_SIZE[0], (
        f"group right edge ({bx + bw}) does not encompass node right edge "
        f"({TARGET_NODE_POS[0] + TARGET_NODE_SIZE[0]})"
    )
    assert bw >= TARGET_GROUP_MIN_WIDTH, (
        f"group width {bw} < target minimum {TARGET_GROUP_MIN_WIDTH}"
    )


def test_apply_does_not_change_links_or_node_count(staged_canonical):
    """Behavioral no-op assertion: only pos / bounding mutate. Link
    array, node count, and every other node's pos must be byte-stable.
    """
    import orjson
    before = orjson.loads(staged_canonical.read_bytes())
    _apply(staged_canonical)
    after = orjson.loads(staged_canonical.read_bytes())

    assert len(before["nodes"]) == len(after["nodes"]), "node count changed"
    assert before["links"] == after["links"], "links array changed"

    # Every node other than 1615 must keep its pos/size.
    by_id_before = {n["id"]: n for n in before["nodes"]}
    by_id_after = {n["id"]: n for n in after["nodes"]}
    for nid, n_before in by_id_before.items():
        if nid == NODE_1615_SCHEDULE:
            continue
        n_after = by_id_after[nid]
        assert n_before.get("pos") == n_after.get("pos"), (
            f"#{nid} pos changed unexpectedly: {n_before.get('pos')} -> {n_after.get('pos')}"
        )

    # Group titles + count unchanged; only "1. Inputs" bounding may differ.
    assert len(before.get("groups", [])) == len(after.get("groups", []))
    for g_before, g_after in zip(before.get("groups", []), after.get("groups", [])):
        assert g_before.get("title") == g_after.get("title")
        if g_before.get("title") != INPUTS_GROUP_TITLE:
            assert g_before.get("bounding") == g_after.get("bounding"), (
                f"non-target group {g_before.get('title')!r} bounding changed"
            )


def test_apply_is_idempotent(staged_canonical):
    _apply(staged_canonical)
    ed1 = WorkflowEditor(staged_canonical)
    pos1 = list(ed1.find_node(NODE_1615_SCHEDULE)["pos"])
    g1 = _find_inputs_group(ed1)
    bounding1 = list(g1["bounding"]) if g1 else None

    _apply(staged_canonical)
    ed2 = WorkflowEditor(staged_canonical)
    pos2 = list(ed2.find_node(NODE_1615_SCHEDULE)["pos"])
    g2 = _find_inputs_group(ed2)
    bounding2 = list(g2["bounding"]) if g2 else None

    assert pos1 == pos2, f"second apply moved node again: {pos1} -> {pos2}"
    assert bounding1 == bounding2, (
        f"second apply changed group again: {bounding1} -> {bounding2}"
    )


def test_dry_run_does_not_mutate(staged_canonical):
    import orjson
    before = staged_canonical.read_bytes()
    r = _apply(staged_canonical, "--dry-run")
    assert r.returncode == 0
    after = staged_canonical.read_bytes()
    assert before == after, "dry-run mutated the file"


def test_revert_restores_original(staged_canonical):
    import orjson
    before = orjson.loads(staged_canonical.read_bytes())
    _apply(staged_canonical)
    after_apply = orjson.loads(staged_canonical.read_bytes())
    assert after_apply["nodes"] != before["nodes"] or \
           after_apply["groups"] != before["groups"], "no-op apply?"

    r = _apply(staged_canonical, "--revert")
    assert r.returncode == 0
    after_revert = orjson.loads(staged_canonical.read_bytes())

    n_before_pos = next(n for n in before["nodes"] if n["id"] == NODE_1615_SCHEDULE)["pos"]
    n_after_pos = next(n for n in after_revert["nodes"] if n["id"] == NODE_1615_SCHEDULE)["pos"]
    assert n_before_pos == n_after_pos, (
        f"revert did not restore node pos: {n_before_pos} -> {n_after_pos}"
    )

    g_before = next((g for g in before.get("groups", []) if g.get("title") == INPUTS_GROUP_TITLE), None)
    g_after = next((g for g in after_revert.get("groups", []) if g.get("title") == INPUTS_GROUP_TITLE), None)
    assert g_before is not None and g_after is not None
    assert g_before["bounding"] == g_after["bounding"], (
        f"revert did not restore group bounding: "
        f"{g_before['bounding']} -> {g_after['bounding']}"
    )


def test_audit_passes_after_apply(staged_canonical):
    """Layout change must not break any audit invariant."""
    _apply(staged_canonical)
    audit = REPO_ROOT / "scripts" / "audit_workflows.py"
    r = subprocess.run(
        [sys.executable, str(audit), str(staged_canonical)],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert r.returncode == 0, f"audit failed:\n{r.stdout}\n{r.stderr}"
