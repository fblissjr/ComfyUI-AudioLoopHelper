"""Tests for WorkflowEditor.add_subgraph_input().

Appending a boundary input to a subgraph requires mirroring the slot on
the top-level invoker node, in lockstep. Both apply scripts that did this
hand-rolled the two dict literals (apply_keyframe_endanchor_variant.py's
END + MID inputs; apply_audio_latent_slice_source_seconds_autowire.py).
Promoting to a method at the 3rd call site per root CLAUDE.md.

Two invariants the method must hold:
  - APPEND-ONLY: never reorder/remove existing inputs (subgraph slot
    indices bake at save time; removal shifts higher slots).
  - DETERMINISTIC boundary id via uuid5(NAMESPACE_OID, name) so re-running
    a generator script stays byte-stable (md5 regen discipline).
"""

from __future__ import annotations

import uuid

import orjson
import pytest

from workflow_utils import WorkflowEditor

SG_ID = "abc-subgraph-uuid"


def _editor(tmp_path, wf: dict) -> WorkflowEditor:
    p = tmp_path / "wf.json"
    p.write_bytes(orjson.dumps(wf))
    return WorkflowEditor(str(p))


def _wf_with_subgraph(sg_inputs=None, invoker_inputs=None) -> dict:
    """Workflow with one subgraph (id SG_ID) and a top-level invoker node
    whose `type` == SG_ID."""
    return {
        "nodes": [
            {
                "id": 100,
                "type": SG_ID,  # invoker: type matches the subgraph id
                "inputs": list(invoker_inputs or []),
                "outputs": [],
            },
        ],
        "links": [],
        "last_node_id": 100,
        "last_link_id": 0,
        "definitions": {
            "subgraphs": [
                {
                    "id": SG_ID,
                    "nodes": [],
                    "links": [],
                    "inputs": list(sg_inputs or []),
                    "outputs": [],
                },
            ],
        },
    }


def test_returns_new_slot_index_equal_to_prior_len(tmp_path):
    ed = _editor(tmp_path, _wf_with_subgraph(
        sg_inputs=[{"name": "existing", "type": "INT"}],
        invoker_inputs=[{"name": "existing", "type": "INT", "link": None}],
    ))
    slot = ed.add_subgraph_input("new_in", "LATENT")
    assert slot == 1  # prior len of sg["inputs"]


def test_appends_does_not_reorder(tmp_path):
    ed = _editor(tmp_path, _wf_with_subgraph(
        sg_inputs=[{"name": "a", "type": "INT"}, {"name": "b", "type": "FLOAT"}],
        invoker_inputs=[
            {"name": "a", "type": "INT", "link": None},
            {"name": "b", "type": "FLOAT", "link": None},
        ],
    ))
    slot = ed.add_subgraph_input("c", "LATENT")
    assert slot == 2
    sg = ed.get_subgraph(0)
    assert [i["name"] for i in sg["inputs"]] == ["a", "b", "c"]
    inv = ed.find_node(100)
    assert [i["name"] for i in inv["inputs"]] == ["a", "b", "c"]


def test_boundary_dict_shape_and_key_order(tmp_path):
    ed = _editor(tmp_path, _wf_with_subgraph())
    ed.add_subgraph_input("kf", "LATENT", label="my label", pos=[1, 2])
    sg = ed.get_subgraph(0)
    entry = sg["inputs"][0]
    # Exact key set + order (byte-stability depends on insertion order).
    assert list(entry.keys()) == [
        "id", "name", "type", "linkIds", "localized_name", "label", "pos",
    ]
    assert entry["name"] == "kf"
    assert entry["type"] == "LATENT"
    assert entry["linkIds"] == []
    assert entry["localized_name"] == "kf"
    assert entry["label"] == "my label"
    assert entry["pos"] == [1, 2]


def test_boundary_id_is_deterministic_uuid5(tmp_path):
    ed = _editor(tmp_path, _wf_with_subgraph())
    ed.add_subgraph_input("end_guide_latent", "LATENT")
    entry = ed.get_subgraph(0)["inputs"][0]
    assert entry["id"] == str(uuid.uuid5(uuid.NAMESPACE_OID, "end_guide_latent"))


def test_boundary_id_stable_across_fresh_editors(tmp_path):
    ed1 = _editor(tmp_path, _wf_with_subgraph())
    ed1.add_subgraph_input("same_name", "LATENT")
    id1 = ed1.get_subgraph(0)["inputs"][0]["id"]

    p2 = tmp_path / "wf2.json"
    p2.write_bytes(orjson.dumps(_wf_with_subgraph()))
    ed2 = WorkflowEditor(str(p2))
    ed2.add_subgraph_input("same_name", "LATENT")
    id2 = ed2.get_subgraph(0)["inputs"][0]["id"]

    assert id1 == id2


def test_invoker_mirror_shape_and_key_order(tmp_path):
    ed = _editor(tmp_path, _wf_with_subgraph())
    ed.add_subgraph_input("kf", "LATENT", label="my label")
    inv = ed.find_node(100)
    mirror = inv["inputs"][0]
    assert list(mirror.keys()) == ["label", "name", "type", "link"]
    assert mirror["label"] == "my label"
    assert mirror["name"] == "kf"
    assert mirror["type"] == "LATENT"
    assert mirror["link"] is None


def test_label_defaults_to_name(tmp_path):
    ed = _editor(tmp_path, _wf_with_subgraph())
    ed.add_subgraph_input("kf", "LATENT")
    sg_entry = ed.get_subgraph(0)["inputs"][0]
    mirror = ed.find_node(100)["inputs"][0]
    assert sg_entry["label"] == "kf"
    assert mirror["label"] == "kf"


def test_pos_defaults_to_zero_zero(tmp_path):
    ed = _editor(tmp_path, _wf_with_subgraph())
    ed.add_subgraph_input("kf", "LATENT")
    assert ed.get_subgraph(0)["inputs"][0]["pos"] == [0, 0]


def test_raises_when_subgraph_missing(tmp_path):
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
    ed = _editor(tmp_path, wf)
    with pytest.raises(ValueError):
        ed.add_subgraph_input("kf", "LATENT")


def test_raises_when_invoker_missing(tmp_path):
    # Subgraph exists but no top-level node has type == SG_ID.
    wf = _wf_with_subgraph()
    wf["nodes"] = []  # drop the invoker
    ed = _editor(tmp_path, wf)
    with pytest.raises(ValueError):
        ed.add_subgraph_input("kf", "LATENT")
