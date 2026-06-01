"""Tests for WorkflowEditor.prune_orphan_output_links().

ComfyUI's UI leaves stale link ids in a node's denormalized
`outputs[].links` cache when a connection is rewired and re-saved: the
top-level `links` array (source of truth) drops the old link, but the
source node's output cache keeps it. These orphans are inert on load but
fail strict link-integrity validation (audit `link_integrity` /
`scripts/test_workflow_integrity.py`).

The normalization removes ONLY ids that reference no actual link record,
so graph topology is provably unchanged.
"""

from __future__ import annotations

import orjson
import pytest

from workflow_utils import WorkflowEditor


def _editor(tmp_path, wf: dict) -> WorkflowEditor:
    p = tmp_path / "wf.json"
    p.write_bytes(orjson.dumps(wf))
    return WorkflowEditor(str(p))


def _wf(nodes, links) -> dict:
    return {"nodes": nodes, "links": links, "last_node_id": 0, "last_link_id": 0}


def test_prunes_orphan_output_link_id(tmp_path):
    # Output caches links 10 (real) and 99 (orphan); array has only 10.
    wf = _wf(
        nodes=[
            {"id": 1, "outputs": [{"name": "x", "type": "INT", "links": [10, 99]}], "inputs": []},
            {"id": 2, "outputs": [], "inputs": [{"name": "x", "type": "INT", "link": 10}]},
        ],
        links=[[10, 1, 0, 2, 0, "INT"]],
    )
    ed = _editor(tmp_path, wf)
    removed = ed.prune_orphan_output_links()
    assert removed == 1
    assert ed.find_node(1)["outputs"][0]["links"] == [10]


def test_returns_zero_when_clean(tmp_path):
    wf = _wf(
        nodes=[{"id": 1, "outputs": [{"name": "x", "type": "INT", "links": [10]}], "inputs": []}],
        links=[[10, 1, 0, 2, 0, "INT"]],
    )
    ed = _editor(tmp_path, wf)
    assert ed.prune_orphan_output_links() == 0
    assert ed.find_node(1)["outputs"][0]["links"] == [10]


def test_idempotent(tmp_path):
    wf = _wf(
        nodes=[{"id": 1, "outputs": [{"name": "x", "type": "INT", "links": [10, 99]}], "inputs": []}],
        links=[[10, 1, 0, 2, 0, "INT"]],
    )
    ed = _editor(tmp_path, wf)
    assert ed.prune_orphan_output_links() == 1
    assert ed.prune_orphan_output_links() == 0


def test_preserves_order_of_remaining(tmp_path):
    wf = _wf(
        nodes=[{"id": 1, "outputs": [{"name": "x", "type": "INT", "links": [10, 99, 20]}], "inputs": []}],
        links=[[10, 1, 0, 2, 0, "INT"], [20, 1, 0, 3, 0, "INT"]],
    )
    ed = _editor(tmp_path, wf)
    assert ed.prune_orphan_output_links() == 1
    assert ed.find_node(1)["outputs"][0]["links"] == [10, 20]


def test_does_not_touch_array_or_input_links(tmp_path):
    wf = _wf(
        nodes=[
            {"id": 1, "outputs": [{"name": "x", "type": "INT", "links": [10, 99]}], "inputs": []},
            {"id": 2, "outputs": [], "inputs": [{"name": "x", "type": "INT", "link": 10}]},
        ],
        links=[[10, 1, 0, 2, 0, "INT"]],
    )
    ed = _editor(tmp_path, wf)
    ed.prune_orphan_output_links()
    assert ed.wf["links"] == [[10, 1, 0, 2, 0, "INT"]]
    assert ed.find_node(2)["inputs"][0]["link"] == 10


def test_handles_none_and_empty_links(tmp_path):
    wf = _wf(
        nodes=[
            {"id": 1, "outputs": [{"name": "x", "type": "INT", "links": None}], "inputs": []},
            {"id": 2, "outputs": [{"name": "y", "type": "INT", "links": []}], "inputs": []},
        ],
        links=[],
    )
    ed = _editor(tmp_path, wf)
    assert ed.prune_orphan_output_links() == 0  # no crash
