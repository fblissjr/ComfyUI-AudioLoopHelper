"""Tests for the `layout_no_orphans` generic audit invariant.

A non-Note node at pos=[0, 0] is the silent-failure mode for apply scripts
that insert nodes without classifying them in their layout-grid table.
The invariant catches it before the workflow ships.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"


def _load_audit_module():
    spec = spec_from_file_location(
        "audit_workflows", REPO_ROOT / "scripts" / "audit_workflows.py",
    )
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_canonical_workflow_no_orphans_ok():
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(CANONICAL)
    matches = [f for f in findings if f.check == "layout_no_orphans"]
    assert any(f.status == "OK" for f in matches), (
        f"expected OK layout_no_orphans on canonical, got {matches}"
    )


def test_orphan_node_at_origin_errs(tmp_path):
    """Inject a non-Note node at pos=[0, 0] -> audit must flag ERR."""
    audit_mod = _load_audit_module()
    broken = tmp_path / "audio-loop-music-video_latent.json"
    data = orjson.loads(CANONICAL.read_bytes())
    for node in data["nodes"]:
        if node.get("type") != "Note":
            node["pos"] = [0, 0]
            break
    broken.write_bytes(orjson.dumps(data))

    findings = audit_mod._audit_one(broken)
    matches = [f for f in findings if f.check == "layout_no_orphans"]
    errs = [f for f in matches if f.status == "ERR"]
    assert errs, f"expected ERR when a node sits at pos=[0, 0], got {matches}"
    assert "apply_layout_" in errs[0].message or "apply_intro_workflow" in errs[0].message


def test_note_at_origin_does_not_err(tmp_path):
    """Notes are author-positioned; pos=[0, 0] on a Note is acceptable."""
    audit_mod = _load_audit_module()
    fixture = tmp_path / "audio-loop-music-video_latent.json"
    data = orjson.loads(CANONICAL.read_bytes())
    note_seen = False
    for node in data["nodes"]:
        if node.get("type") == "Note":
            node["pos"] = [0, 0]
            note_seen = True
            break
    assert note_seen, "canonical should have at least one Note for this fixture"
    fixture.write_bytes(orjson.dumps(data))

    findings = audit_mod._audit_one(fixture)
    matches = [f for f in findings if f.check == "layout_no_orphans"]
    assert not any(f.status == "ERR" for f in matches), (
        f"Note at pos=[0, 0] should not trigger ERR, got {matches}"
    )
