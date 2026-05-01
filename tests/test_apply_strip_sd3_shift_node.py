"""Tests for scripts/apply_strip_sd3_shift_node.py.

Last updated: 2026-04-30

Phase 1b of the post-Numa-render plan. Lightricks's distilled inference
applies NO flow-matching shift (verified against
coderef/ID-LoRA/ID-LoRA-2.3/.../distilled.py:106-112 and their official
2.3 distilled example workflows). The `ModelSamplingSD3 shift=13` node
present in 8 of our shipped workflows is a borrowed-from-SD3 holdover
and is, in fact, currently DEAD (its output links to nothing — verified
across all 8 instances). Strip is pure cleanup: removes a misleading
"we have a sigma shift configured" appearance.

Tests assert the strip migration:
  - removes node #1513 (`ModelSamplingSD3`, mode=0, widgets=[13]) from
    workflows that match the canonical scaffolding signature
  - cleans the inbound link from #503 to #1513
  - is idempotent (re-run is a no-op)
  - skips user-customized shifts (different shift value, bypassed mode,
    different node id) — preserves any deliberate user opt-in
  - audit semantics flip: `model_sampling_shift` fires WARN if
    `ModelSamplingSD3` is PRESENT (the inverse of the prior check)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import orjson
import pytest

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "apply_strip_sd3_shift_node.py"
CANONICAL_VARIANT = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent_keyframe.json"

NODE_SD3_ID = 1513         # canonical id across all 8 shipped variants
UPSTREAM_SOURCE_ID = 503   # LTX2SamplingPreviewOverride feeding link 2794


def _run_script(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd, check=False,
    )


@pytest.fixture
def variant_copy(tmp_path: Path) -> Path:
    """Copy a variant workflow + restore pre-strip ModelSamplingSD3 state.

    The shipped variant is post-strip (clean). For these tests we need
    pre-strip state. Use the apply script's own `--revert` to recreate
    the canonical scaffolding so the fixture stays in lockstep with the
    script's understanding of "before".
    """
    dst_dir = tmp_path / "example_workflows"
    dst_dir.mkdir()
    dst = dst_dir / "audio-loop-music-video_latent_keyframe.json"
    shutil.copy2(CANONICAL_VARIANT, dst)
    # Restore the SD3 scaffolding so tests have pre-strip state to operate on
    result = _run_against_dir(dst_dir, "--revert")
    assert result.returncode == 0, (
        f"could not restore SD3 scaffolding for test fixture:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    ed = WorkflowEditor(dst)
    assert ed.has_node(NODE_SD3_ID), "fixture failed to restore #1513"
    return dst


def _run_against_dir(workflows_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run_script("--workflows-dir", str(workflows_dir), *extra)


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"script failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------- happy path ----------

def test_dry_run_does_not_write(variant_copy):
    before = variant_copy.read_bytes()
    result = _run_against_dir(variant_copy.parent, "--dry-run")
    _assert_ok(result)
    after = variant_copy.read_bytes()
    assert before == after, "dry-run must not modify files"
    assert "would" in result.stdout.lower()


def test_apply_removes_sd3_node(variant_copy):
    _assert_ok(_run_against_dir(variant_copy.parent))
    ed = WorkflowEditor(variant_copy)
    assert not ed.has_node(NODE_SD3_ID), "#1513 should have been removed"
    assert not ed.find_nodes_by_type("ModelSamplingSD3"), \
        "no ModelSamplingSD3 nodes should remain"


def test_apply_removes_inbound_link(variant_copy):
    """Link 2794 (#503.0 -> #1513.0) should no longer reference #1513."""
    _assert_ok(_run_against_dir(variant_copy.parent))
    ed = WorkflowEditor(variant_copy)
    for link in ed.wf["links"]:
        if not isinstance(link, list) or len(link) < 6:
            continue
        _, src, _, tgt, _, _ = link
        assert tgt != NODE_SD3_ID, f"link still targets removed #{NODE_SD3_ID}"
        assert src != NODE_SD3_ID, f"link still originates from removed #{NODE_SD3_ID}"


def test_apply_is_idempotent(variant_copy):
    _assert_ok(_run_against_dir(variant_copy.parent))
    first = variant_copy.read_bytes()
    result = _run_against_dir(variant_copy.parent)
    _assert_ok(result)
    assert variant_copy.read_bytes() == first, "second run must be a no-op"
    assert (
        "no change" in result.stdout.lower()
        or "already" in result.stdout.lower()
    ), f"expected no-change report: {result.stdout}"


def test_apply_no_change_when_already_stripped(tmp_path):
    """A canonical-shaped workflow without ModelSamplingSD3 is a no-op."""
    dst_dir = tmp_path / "example_workflows"
    dst_dir.mkdir()
    dst = dst_dir / "audio-loop-music-video_latent.json"
    shutil.copy2(REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json", dst)

    result = _run_against_dir(dst_dir)
    _assert_ok(result)
    assert (
        "no change" in result.stdout.lower()
        or "already" in result.stdout.lower()
        or "skip" in result.stdout.lower()
    )
    assert dst.read_bytes() == (REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json").read_bytes()


# ---------- skip-paths (preserve user customization) ----------

def test_apply_skips_user_customized_shift(variant_copy):
    """If user changed the shift widget value (e.g., shift=7), preserve."""
    ed = WorkflowEditor(variant_copy)
    ed.find_node(NODE_SD3_ID)["widgets_values"] = [7]
    ed.save()

    result = _run_against_dir(variant_copy.parent)
    _assert_ok(result)
    ed2 = WorkflowEditor(variant_copy)
    assert ed2.has_node(NODE_SD3_ID), \
        f"node was stripped despite user-customized shift value"


def test_apply_skips_bypassed_node(variant_copy):
    """If user bypassed the node (mode=4), don't touch it."""
    ed = WorkflowEditor(variant_copy)
    ed.find_node(NODE_SD3_ID)["mode"] = 4
    ed.save()

    result = _run_against_dir(variant_copy.parent)
    _assert_ok(result)
    ed2 = WorkflowEditor(variant_copy)
    assert ed2.has_node(NODE_SD3_ID), \
        "node was stripped despite being bypassed (user customization)"


# ---------- audit integration ----------

def _load_audit_module():
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "audit_workflows", REPO_ROOT / "scripts" / "audit_workflows.py",
    )
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_audit_no_warn_after_strip(variant_copy):
    """After the strip, model_sampling_shift should not WARN/ERR on the
    presence-of-SD3 axis (the new semantics)."""
    _assert_ok(_run_against_dir(variant_copy.parent))
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(variant_copy)
    sd3_findings = [
        f for f in findings if f.check == "model_sampling_shift"
    ]
    assert sd3_findings, \
        "audit check 'model_sampling_shift' missing — semantic flip not applied"
    # Post-strip: no SD3 should mean OK or no-WARN; the flipped semantic
    # WARNs only when SD3 is present.
    assert not any(f.status in ("WARN", "ERR") for f in sd3_findings), \
        f"strip didn't satisfy audit check: {sd3_findings}"


def test_audit_warns_when_sd3_present(variant_copy):
    """Pre-strip variant has ModelSamplingSD3 → audit should WARN."""
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(variant_copy)
    sd3_warnings = [
        f for f in findings
        if f.check == "model_sampling_shift" and f.status in ("WARN", "ERR")
    ]
    assert sd3_warnings, (
        f"audit should WARN/ERR when ModelSamplingSD3 present (post-flip semantic). "
        f"Got: {[f for f in findings if f.check == 'model_sampling_shift']}"
    )
