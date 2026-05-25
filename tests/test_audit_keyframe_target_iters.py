"""Tests for the `keyframe_target_iters_set` audit invariant.

`LTXIterKeyframeSchedule` picks a pre-encoded keyframe latent per loop
iteration by matching `current_iteration` against each row's `target_iters`
(comma-separated, 1-based). When EVERY row's `target_iters` is empty, no row
ever matches, so the selector returns its `fallback_latent` (the init image)
on every iteration — the wired keyframes silently never fire and the render is
bit-identical to the no-keyframe canonical. No error, no warning at runtime; it
just looks like "only one image is in use."

This WARN-level check catches that regression. The shipped keyframe workflows
default `target_iters` to firing values (1,2,3) via
`scripts/apply_keyframe_iter_anchor.py`; this guards against a row being cleared
back to empty.

Widget layout: [current_iteration, num_keyframes_combo, target_iters_1..N].
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example_workflows"
KEYFRAME = EXAMPLE_DIR / "audio-loop-music-video_latent_keyframe.json"
DEFAULT = EXAMPLE_DIR / "audio-loop-music-video_latent.json"

CHECK = "keyframe_target_iters_set"


def _load_audit_module():
    spec = spec_from_file_location(
        "audit_workflows", REPO_ROOT / "scripts" / "audit_workflows.py",
    )
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shipped_keyframe_target_iters_set_ok():
    """Shipped keyframe workflow ships target_iters pre-filled (1,2,3) → OK."""
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(KEYFRAME)
    matches = [f for f in findings if f.check == CHECK]
    assert any(f.status == "OK" for f in matches), (
        f"expected OK {CHECK} on shipped keyframe workflow, got {matches}"
    )


def test_workflow_without_selector_skipped():
    """The default loop workflow has no LTXIterKeyframeSchedule — check must not fire."""
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(DEFAULT)
    matches = [f for f in findings if f.check == CHECK]
    assert not matches, f"check should not fire without a selector, got {matches}"


def test_all_empty_target_iters_warns(tmp_path):
    """Clear every target_iters row → WARN (keyframes silently fall back to init)."""
    audit_mod = _load_audit_module()
    broken = tmp_path / "audio-loop-music-video_latent_keyframe.json"
    data = orjson.loads(KEYFRAME.read_bytes())

    sel = next(n for n in data["nodes"] if n["type"] == "LTXIterKeyframeSchedule")
    wv = sel["widgets_values"]  # [current_iteration, num_combo, t1..tN]
    num = int(wv[1])
    for i in range(num):
        wv[2 + i] = ""
    broken.write_bytes(orjson.dumps(data))

    findings = audit_mod._audit_one(broken)
    matches = [f for f in findings if f.check == CHECK]
    warns = [f for f in matches if f.status == "WARN"]
    assert warns, f"expected WARN when all target_iters empty, got {matches}"
    assert "target_iters" in warns[0].message.lower()
