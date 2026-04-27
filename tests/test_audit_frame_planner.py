"""Tests for the `frame_planner_present` audit invariant.

Production workflows must wire LTXFramePlanner as the single source of truth
for dimension config (width/height/length/fps/window_seconds/frame_rate).
Retake variant lacks the audio-loop spine and is intentionally exempt.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example_workflows"
CANONICAL = EXAMPLE_DIR / "audio-loop-music-video_latent.json"
RETAKE = EXAMPLE_DIR / "audio-loop-music-video_retake.json"


def _load_audit_module():
    spec = spec_from_file_location(
        "audit_workflows", REPO_ROOT / "scripts" / "audit_workflows.py",
    )
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_canonical_workflow_has_frame_planner_ok():
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(CANONICAL)
    matches = [f for f in findings if f.check == "frame_planner_present"]
    assert any(f.status == "OK" for f in matches), (
        f"expected OK frame_planner_present on canonical, got {matches}"
    )


def test_retake_workflow_skipped():
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(RETAKE)
    matches = [f for f in findings if f.check == "frame_planner_present"]
    # Retake intentionally has no LTXFramePlanner — should produce no
    # ERR finding for this check (either OK n/a or no record at all).
    assert not any(f.status == "ERR" for f in matches), (
        f"retake should be exempt, got {matches}"
    )


def test_workflow_missing_planner_errs(tmp_path):
    """Strip LTXFramePlanner from the canonical workflow → audit must flag ERR."""
    audit_mod = _load_audit_module()
    broken = tmp_path / "audio-loop-music-video_latent.json"
    data = orjson.loads(CANONICAL.read_bytes())
    data["nodes"] = [
        n for n in data["nodes"] if n.get("type") != "LTXFramePlanner"
    ]
    broken.write_bytes(orjson.dumps(data))
    findings = audit_mod._audit_one(broken)
    matches = [f for f in findings if f.check == "frame_planner_present"]
    errs = [f for f in matches if f.status == "ERR"]
    assert errs, f"expected ERR when LTXFramePlanner stripped, got {matches}"
    # Remediation pointer must mention the apply script
    assert "apply_frame_planner_consolidation" in errs[0].message
