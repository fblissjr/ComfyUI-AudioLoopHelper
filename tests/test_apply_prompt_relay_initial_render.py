"""Tests for scripts/apply_prompt_relay_initial_render.py (Phase 1).

Covers:
  - --dry-run produces no file writes
  - apply creates the staging file, is idempotent on re-apply
  - --revert deletes the staging file
  - post-apply topology: MODEL fork isolates PromptRelay patches to the
    initial render branch (CFGGuider(153).model) and does NOT leak onto
    the loop subgraph MODEL input
  - post-apply topology: LTXVConditioning(164).positive is re-sourced from
    PromptRelayEncode, not the original CLIPTextEncode(169)
  - F2 preprocess symmetry + F3 loop cropguides symmetry still valid
  - required-source-node guard raises when canonical node IDs are missing
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
SCRIPT = REPO_ROOT / "scripts" / "apply_prompt_relay_initial_render.py"
# Frozen pre-dedupe baseline. PromptRelay's apply script splices into
# the post-Node-169 conditioning chain, which is mutually exclusive with
# the dedupe migration that's now baked into the canonical
# (apply_dedupe_initial_render_prompt.py removes Node 169). The frozen
# baseline isolates this script's tests from canonical drift; the
# script itself remains usable on any pre-dedupe input.
CANONICAL = REPO_ROOT / "tests" / "fixtures" / "canonical_loop_pre_dedupe.json"


# Node IDs the apply script keys off. Mirrors the constants in the script.
CFGGUIDER_ID = 153
LTXVCONDITIONING_ID = 164
CLIPTEXTENCODE_POS_ID = 169
LTX2_NAG_ID = 508
SETNODE_MODEL_ID = 572
ATTENTION_TUNER_ID = 1523
EMPTY_LATENT_ID = 344
DUAL_CLIP_LOADER_ID = 416


def _run_script(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd, check=False,
    )


@pytest.fixture
def tmp_workflow(tmp_path: Path) -> tuple[Path, Path]:
    """Copy canonical input + return (input, staging-output) paths."""
    input_path = tmp_path / "canonical.json"
    output_path = tmp_path / "scratch" / "staged.json"
    shutil.copy2(CANONICAL, input_path)
    return input_path, output_path


def _apply(input_path: Path, output_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run_script(
        "--input", str(input_path),
        "--output", str(output_path),
        *extra,
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"script failed (rc={result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_dry_run_does_not_write(tmp_workflow):
    input_path, output_path = tmp_workflow
    result = _apply(input_path, output_path, "--dry-run")
    _assert_ok(result)
    assert not output_path.exists(), "dry-run must not touch disk"


def test_apply_creates_staging_file(tmp_workflow):
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    assert output_path.exists()
    # Valid JSON
    data = orjson.loads(output_path.read_bytes())
    assert "nodes" in data


def test_apply_is_idempotent(tmp_workflow):
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    first_bytes = output_path.read_bytes()
    _assert_ok(_apply(input_path, output_path))
    second_bytes = output_path.read_bytes()
    assert first_bytes == second_bytes, "re-running apply must be a no-op"


def test_revert_deletes_staging_file(tmp_workflow):
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    assert output_path.exists()
    _assert_ok(_run_script("--output", str(output_path), "--revert"))
    assert not output_path.exists()


def test_post_apply_adds_prompt_relay_encode(tmp_workflow):
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    ed = WorkflowEditor(output_path)
    matches = ed.find_nodes_by_type("PromptRelayEncode")
    assert len(matches) == 1, "exactly one PromptRelayEncode should be inserted"


def test_cfg_guider_model_sources_from_prompt_relay(tmp_workflow):
    """CFGGuider(153).model must be fed by PromptRelayEncode, not SetNode(572)."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    ed = WorkflowEditor(output_path)
    relay = ed.find_nodes_by_type("PromptRelayEncode")[0]
    model_link = ed.find_link_to_slot(CFGGUIDER_ID, 0)
    assert model_link is not None, "CFGGuider(153).model has no inbound link"
    assert model_link[1] == relay["id"], (
        f"CFGGuider(153).model should read from PromptRelayEncode ({relay['id']}), "
        f"got src={model_link[1]}"
    )


def test_prompt_relay_model_does_not_reach_subgraph(tmp_workflow):
    """PromptRelay-patched MODEL must not leak into the loop subgraph."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    ed = WorkflowEditor(output_path)
    relay_id = ed.find_nodes_by_type("PromptRelayEncode")[0]["id"]
    invoker = ed.find_subgraph_invoker()
    assert invoker is not None, "subgraph invoker not found"
    invoker_model_link = ed.find_link_to_slot(invoker["id"], 2)  # slot 2 = model
    assert invoker_model_link is not None, "subgraph invoker model slot is unwired"
    # The immediate source must NOT be PromptRelayEncode. It is allowed to be
    # any node in the pre-fork chain (LoopIterationStamp / GetNode(654) / ...).
    # We walk one hop only: that covers the common leak pattern.
    assert invoker_model_link[1] != relay_id, (
        "PromptRelayEncode MODEL leaked into the subgraph invoker model input"
    )


def test_ltxvconditioning_positive_sources_from_prompt_relay(tmp_workflow):
    """LTXVConditioning(164).positive must be fed by PromptRelayEncode, not
    the original CLIPTextEncode(169)."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    ed = WorkflowEditor(output_path)
    relay_id = ed.find_nodes_by_type("PromptRelayEncode")[0]["id"]
    pos_link = ed.find_link_to_slot(LTXVCONDITIONING_ID, 0)
    assert pos_link is not None
    assert pos_link[1] == relay_id, (
        f"LTXVConditioning(164).positive should source from PromptRelayEncode "
        f"({relay_id}), got src={pos_link[1]}"
    )


def test_prompt_relay_required_inputs_wired(tmp_workflow):
    """PromptRelayEncode needs model, clip, latent wired; strings come from widgets."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    ed = WorkflowEditor(output_path)
    relay = ed.find_nodes_by_type("PromptRelayEncode")[0]
    relay_id = relay["id"]
    # MODEL fork: PromptRelayEncode.model comes from AttentionTunerPatch(1523)
    model_link = ed.find_link_to_slot(relay_id, WorkflowEditor.find_input_slot(relay, "model"))
    assert model_link is not None
    assert model_link[1] == ATTENTION_TUNER_ID, (
        f"PromptRelayEncode.model should fork from AttentionTunerPatch({ATTENTION_TUNER_ID}), "
        f"got src={model_link[1]}"
    )
    # CLIP: same DualCLIPLoader that feeds Node 169
    clip_link = ed.find_link_to_slot(relay_id, WorkflowEditor.find_input_slot(relay, "clip"))
    assert clip_link is not None
    assert clip_link[1] == DUAL_CLIP_LOADER_ID
    # LATENT: EmptyLTXVLatentVideo for shape inference
    latent_link = ed.find_link_to_slot(relay_id, WorkflowEditor.find_input_slot(relay, "latent"))
    assert latent_link is not None
    assert latent_link[1] == EMPTY_LATENT_ID


def test_nag_remains_on_loop_branch(tmp_workflow):
    """LTX2_NAG(508) still feeds the existing chain into SetNode(572); loop
    subgraph continues to see NAG-patched MODEL."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    ed = WorkflowEditor(output_path)
    # NAG input still attention tuner
    nag_in = ed.find_link_to_slot(LTX2_NAG_ID, 0)
    assert nag_in is not None
    assert nag_in[1] == ATTENTION_TUNER_ID
    # Something still feeds the SetNode that the loop subgraph pulls from
    set_model_in = ed.find_link_to_slot(SETNODE_MODEL_ID, 0)
    assert set_model_in is not None


def test_f2_f3_audit_checks_still_pass(tmp_workflow):
    """Phase 1 must not break preprocess_symmetry or loop_cropguides_symmetry."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    # Import audit module and run against the produced file.
    # audit_workflows is a script, not a package. Load it by file path.
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "audit_workflows",
        REPO_ROOT / "scripts" / "audit_workflows.py",
    )
    assert spec is not None and spec.loader is not None
    audit_mod = module_from_spec(spec)
    spec.loader.exec_module(audit_mod)
    findings = audit_mod._audit_one(output_path)
    errs = [f for f in findings if f.status == "ERR" and f.check in {
        "preprocess_symmetry", "loop_cropguides_symmetry",
    }]
    assert not errs, f"Phase 1 broke F2/F3: {errs}"


def test_missing_source_nodes_raises(tmp_path):
    """If the canonical input is missing required anchor nodes, abort."""
    broken = tmp_path / "broken.json"
    data = orjson.loads(CANONICAL.read_bytes())
    data["nodes"] = [n for n in data["nodes"] if n["id"] != LTXVCONDITIONING_ID]
    broken.write_bytes(orjson.dumps(data))
    result = _apply(broken, tmp_path / "out.json")
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "missing" in combined or "refus" in combined


def _load_audit_module():
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "audit_workflows", REPO_ROOT / "scripts" / "audit_workflows.py",
    )
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_audit_reports_prompt_relay_wiring_ok(tmp_workflow):
    """Audit check `prompt_relay_wiring` should be OK on a correctly-staged file."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(output_path)
    relay_findings = [f for f in findings if f.check == "prompt_relay_wiring"]
    assert any(f.status == "OK" for f in relay_findings), (
        f"expected OK prompt_relay_wiring, got {relay_findings}"
    )
    assert not any(f.status == "ERR" for f in relay_findings)


def test_audit_reports_err_if_prompt_relay_leaks_into_subgraph(tmp_workflow):
    """Simulate a leak: add a link from PromptRelayEncode.MODEL to the
    subgraph invoker's MODEL slot. Audit must flag ERR."""
    input_path, output_path = tmp_workflow
    _assert_ok(_apply(input_path, output_path))

    # Find PromptRelayEncode id and subgraph invoker id, then splice a leaky link.
    ed = WorkflowEditor(output_path)
    relay_id = ed.find_nodes_by_type("PromptRelayEncode")[0]["id"]
    invoker = ed.find_subgraph_invoker()
    assert invoker is not None
    ed.add_link(relay_id, 0, invoker["id"], 2, "MODEL")
    ed.save()

    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(output_path)
    relay_errs = [
        f for f in findings
        if f.check == "prompt_relay_wiring" and f.status == "ERR"
    ]
    assert relay_errs, "audit should flag MODEL leak into subgraph invoker"
    assert "leak" in relay_errs[0].message.lower()
