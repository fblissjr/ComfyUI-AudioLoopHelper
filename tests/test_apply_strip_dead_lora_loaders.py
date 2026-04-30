"""Tests for scripts/apply_strip_dead_lora_loaders.py.

Last updated: 2026-04-30

The canonical baseline workflow ships three bypassed (mode=4) LoRA-loader
nodes that form dead scaffolding (never wired to a guide, one points at a
placeholder filename). These tests assert that the strip migration:

  - removes nodes #1625, #1626, #1627 when they match the canonical shape
  - rebridges the MODEL chain (link 3080 source -> link 3083 consumer
    becomes a single direct link, byte-equivalent to manual rebridge)
  - is idempotent (re-run is a no-op)
  - skips workflows where a user has un-bypassed or renamed any of the
    three nodes (don't trample user customizations)
  - skips workflows that lack the scaffolding (already stripped, or never
    inherited it like the experimental siblings)
  - tolerates id collisions (e.g. _latent_keyframe.json reuses id 1627
    for KeyframeLatentScheduleBatchEncode — strip must not touch it)
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
SCRIPT = REPO_ROOT / "scripts" / "apply_strip_dead_lora_loaders.py"
CANONICAL = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

# Canonical scaffolding node IDs the script keys off.
ID_ID_LORA = 1625      # LoraLoaderModelOnly "ID-LoRA File (audio-conditioned identity)"
ID_IC_LORA = 1626      # LTXICLoRALoaderModelOnly "IC-LoRA File (visual reference adapter)"
ID_STYLE = 1627        # LoraLoaderModelOnly "Style/Generic LoRA"

UPSTREAM_SOURCE_ID = 503  # LTX2SamplingPreviewOverride feeding link 3080
DOWNSTREAM_SINK_ID = 572  # SetNode "model" consuming link 3083


def _run_script(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd, check=False,
    )


@pytest.fixture
def canonical_copy(tmp_path: Path) -> Path:
    """Copy of the canonical workflow on a tmp dir, with the dead
    scaffolding RESTORED so tests can validate the strip behavior.

    The shipped canonical is post-strip (clean); for these tests we need
    a copy in pre-strip state. We use the script's own `--revert` to
    add the three scaffolding nodes back through the canonical bridge
    link, exactly mirroring what the canonical looked like before the
    strip migration ran.
    """
    dst_dir = tmp_path / "example_workflows"
    dst_dir.mkdir()
    dst = dst_dir / "audio-loop-music-video_latent.json"
    shutil.copy2(CANONICAL, dst)
    # Restore scaffolding so tests have pre-strip state to operate on.
    result = _run_against_dir(dst_dir, "--revert")
    assert result.returncode == 0, (
        f"could not restore scaffolding for test fixture:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Sanity: scaffolding is now back
    ed = WorkflowEditor(dst)
    assert ed.has_node(ID_ID_LORA), "fixture failed to restore #1625"
    return dst


def _run_against_dir(workflows_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the apply script with --workflows-dir override."""
    return _run_script("--workflows-dir", str(workflows_dir), *extra)


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"script failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------- happy path ----------

def test_dry_run_does_not_write(canonical_copy):
    before = canonical_copy.read_bytes()
    result = _run_against_dir(canonical_copy.parent, "--dry-run")
    _assert_ok(result)
    after = canonical_copy.read_bytes()
    assert before == after, "dry-run must not modify files"
    # And it should report what it WOULD do
    assert "would" in result.stdout.lower(), (
        f"dry-run should report 'would ...': {result.stdout}"
    )


def test_apply_removes_three_dead_nodes(canonical_copy):
    _assert_ok(_run_against_dir(canonical_copy.parent))
    ed = WorkflowEditor(canonical_copy)
    for nid in (ID_ID_LORA, ID_IC_LORA, ID_STYLE):
        assert not ed.has_node(nid), (
            f"node #{nid} should have been removed but is still present"
        )


def test_apply_rebridges_model_chain(canonical_copy):
    _assert_ok(_run_against_dir(canonical_copy.parent))
    ed = WorkflowEditor(canonical_copy)
    # SetNode(572).model must now read from LTX2SamplingPreviewOverride(503).0
    sink_link = ed.find_link_to_slot(DOWNSTREAM_SINK_ID, 0)
    assert sink_link is not None, (
        "SetNode(572).model is unwired after strip — chain not rebridged"
    )
    assert sink_link[1] == UPSTREAM_SOURCE_ID, (
        f"SetNode(572).model should read directly from "
        f"LTX2SamplingPreviewOverride({UPSTREAM_SOURCE_ID}), got src={sink_link[1]}"
    )
    assert sink_link[2] == 0, (
        f"SetNode(572).model should read from slot 0 of upstream, got {sink_link[2]}"
    )


def test_apply_no_orphan_links_to_dead_nodes(canonical_copy):
    """After strip, no link should reference the deleted node ids."""
    _assert_ok(_run_against_dir(canonical_copy.parent))
    ed = WorkflowEditor(canonical_copy)
    dead_ids = {ID_ID_LORA, ID_IC_LORA, ID_STYLE}
    for link in ed.wf["links"]:
        if not isinstance(link, list) or len(link) < 6:
            continue
        _, src, _, tgt, _, _ = link
        assert src not in dead_ids, f"link still references deleted src {src}"
        assert tgt not in dead_ids, f"link still references deleted tgt {tgt}"


def test_apply_is_idempotent(canonical_copy):
    _assert_ok(_run_against_dir(canonical_copy.parent))
    first_bytes = canonical_copy.read_bytes()
    result = _run_against_dir(canonical_copy.parent)
    _assert_ok(result)
    second_bytes = canonical_copy.read_bytes()
    assert first_bytes == second_bytes, "second run must be a no-op"
    assert (
        "no change" in result.stdout.lower()
        or "already" in result.stdout.lower()
    ), f"second-run status should report no-change: {result.stdout}"


# ---------- skip-paths (preserve user customization) ----------

def test_apply_skips_when_user_unbypassed_node(canonical_copy):
    """If user activated #1626 (mode != 4), the script must not strip ANY of the three."""
    ed = WorkflowEditor(canonical_copy)
    ed.find_node(ID_IC_LORA)["mode"] = 0  # un-bypass
    ed.save()

    result = _run_against_dir(canonical_copy.parent)
    _assert_ok(result)
    ed2 = WorkflowEditor(canonical_copy)
    for nid in (ID_ID_LORA, ID_IC_LORA, ID_STYLE):
        assert ed2.has_node(nid), (
            f"node #{nid} was stripped despite user un-bypassing #{ID_IC_LORA}"
        )


def test_apply_skips_when_node_title_customized(canonical_copy):
    """If user renamed any of the three, the script must skip."""
    ed = WorkflowEditor(canonical_copy)
    ed.find_node(ID_ID_LORA)["title"] = "My custom ID-LoRA"
    ed.save()

    result = _run_against_dir(canonical_copy.parent)
    _assert_ok(result)
    ed2 = WorkflowEditor(canonical_copy)
    for nid in (ID_ID_LORA, ID_IC_LORA, ID_STYLE):
        assert ed2.has_node(nid), (
            f"node #{nid} was stripped despite user renaming #{ID_ID_LORA}"
        )


# ---------- no-op paths (already stripped, never had it) ----------

def test_apply_no_change_when_already_stripped(tmp_path):
    """A workflow without any of the three nodes is a no-op. Uses the
    real shipped canonical (which is now post-strip)."""
    dst_dir = tmp_path / "example_workflows"
    dst_dir.mkdir()
    dst = dst_dir / "audio-loop-music-video_latent.json"
    shutil.copy2(CANONICAL, dst)  # already-stripped canonical

    result = _run_against_dir(dst_dir)
    _assert_ok(result)
    # Should report no-change
    assert (
        "no change" in result.stdout.lower()
        or "already" in result.stdout.lower()
        or "skip" in result.stdout.lower()
    ), f"expected no-change report: {result.stdout}"
    # File unchanged
    assert dst.read_bytes() == CANONICAL.read_bytes()


def test_apply_tolerates_id_collision_with_different_type(tmp_path):
    """A workflow that reuses id 1627 for a non-LoraLoaderModelOnly node
    (e.g. _latent_keyframe.json's KeyframeLatentScheduleBatchEncode) must
    not be touched."""
    # Build a synthetic minimal workflow with id 1627 as a different type.
    wf_dir = tmp_path / "example_workflows"
    wf_dir.mkdir()
    wf_path = wf_dir / "synthetic.json"
    wf = {
        "revision": 0,
        "last_node_id": 1700,
        "last_link_id": 100,
        "nodes": [
            {
                "id": 1627,
                "type": "KeyframeLatentScheduleBatchEncode",
                "pos": [0, 0],
                "size": [200, 60],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {"Node name for S&R": "KeyframeLatentScheduleBatchEncode"},
                "widgets_values": [],
                "title": "Keyframe Latent Schedule (Batch Encode)",
            },
        ],
        "links": [],
        "groups": [],
    }
    wf_path.write_bytes(orjson.dumps(wf))

    _assert_ok(_run_against_dir(wf_dir))
    ed = WorkflowEditor(wf_path)
    assert ed.has_node(1627), "unrelated node #1627 was incorrectly stripped"
    n = ed.find_node(1627)
    assert n["type"] == "KeyframeLatentScheduleBatchEncode", (
        f"node #1627 type mutated to {n['type']!r}"
    )


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


def test_audit_dead_scaffolding_check_passes_after_strip(canonical_copy):
    _assert_ok(_run_against_dir(canonical_copy.parent))
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(canonical_copy)
    relevant = [f for f in findings if f.check == "dead_lora_loader_scaffolding_absent"]
    assert relevant, "audit check 'dead_lora_loader_scaffolding_absent' is missing"
    assert all(f.status != "ERR" for f in relevant), (
        f"strip didn't satisfy audit check: {relevant}"
    )


def test_audit_dead_scaffolding_check_fires_when_present(canonical_copy):
    """Pre-strip canonical fails the audit check (proves the check fires)."""
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(canonical_copy)
    errs = [
        f for f in findings
        if f.check == "dead_lora_loader_scaffolding_absent" and f.status == "ERR"
    ]
    assert errs, (
        "audit check should fire ERR when scaffolding is present, "
        f"got: {[f for f in findings if f.check == 'dead_lora_loader_scaffolding_absent']}"
    )
