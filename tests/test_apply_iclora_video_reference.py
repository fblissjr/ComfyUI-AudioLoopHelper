"""Tests for scripts/apply_iclora_video_reference.py.

Last updated: 2026-04-30

Wires LTXICLoRALoaderModelOnly + LTXAddVideoICLoRAGuide (in subgraph)
+ VHS_LoadVideo + ref-clip preprocessing chain into the canonical
audio-loop workflow. Static-mode wiring (durable foundation; sliding
mode is a follow-up flag extension on the same script).

Tests cover:
  - --dry-run and --revert plumbing
  - top-level: LTXICLoRALoaderModelOnly spliced into MODEL chain
  - top-level: VHS_LoadVideo -> ImageResizeKJv2 -> LTXVPreprocess(18)
    chain wired into subgraph invoker's new IMAGE slot
  - subgraph: new `reference_video` IMAGE input slot present
  - subgraph: LTXAddVideoICLoRAGuide inserted between #1519 outputs
    and existing consumers
  - subgraph: GetImageRangeFromBatch slicer wired to new input slot
  - F3 symmetry preserved: IC-LoRA guide outputs reach CFGGuider only
    via LTXVCropGuidesNoLatent (matching the existing init-image path)
  - pre-flight: refuses to run when canonical still has dead scaffolding
  - pre-flight: refuses to run when --reference-video file missing
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
SCRIPT = REPO_ROOT / "scripts" / "apply_iclora_video_reference.py"
CANONICAL = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

# Top-level node ids the script keys off.
LTX2_PREVIEW_OVERRIDE_ID = 503        # MODEL output -> SetNode(572) "model"
SETNODE_MODEL_ID = 572                # Set "model"
LTXFRAME_PLANNER_ID = None            # discovered; not hardcoded

# Subgraph internal node ids the script splices around.
SUBGRAPH_LATENT_GUIDE_ID = 1519       # LTXVAddLatentGuide
SUBGRAPH_CROPGUIDES_NOLATENT_ID = 655 # LTXVCropGuidesNoLatent (F3 path to CFGGuider)
SUBGRAPH_CROPGUIDES_ID = 2008         # LTXVCropGuides (LATENT path)
SUBGRAPH_CFGGUIDER_ID = 644           # CFGGuider
SUBGRAPH_CONCAT_AV_ID = 583           # LTXVConcatAVLatent

# IC-LoRA file used in tests. The Cseti cameraman LoRA on the dev box;
# the script only checks that the path exists, doesn't load it.
DEFAULT_IC_LORA_REL = (
    "Cseti_LTX2.3-22B_IC-LoRA-Cameraman_v1/balanced15/"
    "lora_weights_step_10500.safetensors"
)


def _run_script(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=cwd, check=False,
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"script failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.fixture
def stub_ref_video(tmp_path: Path) -> Path:
    p = tmp_path / "ref.mp4"
    p.write_bytes(b"\x00")  # placeholder; script checks existence, not content
    return p


@pytest.fixture
def stub_lora(tmp_path: Path) -> Path:
    """Create a stub LoRA file. Test passes the absolute path so we don't
    depend on the dev box's models directory layout."""
    p = tmp_path / "lora_weights_step_10500.safetensors"
    p.write_bytes(b"\x00")
    return p


@pytest.fixture
def staged_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Return (input_path, output_path) — input is a copy of the
    canonical (already post-strip), output is the staging target."""
    input_path = tmp_path / "canonical.json"
    output_path = tmp_path / "scratch" / "staged.json"
    shutil.copy2(CANONICAL, input_path)
    return input_path, output_path


def _apply(
    input_path: Path, output_path: Path,
    ref_video: Path, ic_lora: Path,
    *extra: str,
) -> subprocess.CompletedProcess:
    return _run_script(
        "--input", str(input_path),
        "--output", str(output_path),
        "--reference-video", str(ref_video),
        "--ic-lora-file", str(ic_lora),
        *extra,
    )


# ---------- plumbing ----------

def test_dry_run_does_not_write(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    result = _apply(input_path, output_path, stub_ref_video, stub_lora, "--dry-run")
    _assert_ok(result)
    assert not output_path.exists(), "dry-run must not write"


def test_apply_creates_staging_file(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    assert output_path.exists()
    data = orjson.loads(output_path.read_bytes())
    assert "nodes" in data


def test_apply_is_idempotent(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    first = output_path.read_bytes()
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    assert output_path.read_bytes() == first, "idempotent re-run must not modify"


def test_revert_deletes_staging_file(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    assert output_path.exists()
    _assert_ok(_run_script("--output", str(output_path), "--revert"))
    assert not output_path.exists()


# ---------- top-level wiring ----------

def test_top_level_iclora_loader_inserted(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    loaders = ed.find_nodes_by_type("LTXICLoRALoaderModelOnly")
    assert len(loaders) == 1, (
        f"expected exactly one LTXICLoRALoaderModelOnly, found {len(loaders)}"
    )
    # Loader is spliced between #503 and #572: #503.0 -> loader.0 -> #572.0
    setnode_link = ed.find_link_to_slot(SETNODE_MODEL_ID, 0)
    assert setnode_link is not None
    assert setnode_link[1] == loaders[0]["id"], (
        f"SetNode(572).model should read from the new loader, "
        f"got src={setnode_link[1]}"
    )
    # Loader's own input reads from #503.0
    loader_input_link = ed.find_link_to_slot(loaders[0]["id"], 0)
    assert loader_input_link is not None
    assert loader_input_link[1] == LTX2_PREVIEW_OVERRIDE_ID


def test_top_level_lora_filename_widget_set(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    loaders = ed.find_nodes_by_type("LTXICLoRALoaderModelOnly")
    widgets = loaders[0].get("widgets_values") or []
    assert widgets, "loader has no widgets_values"
    assert widgets[0] == str(stub_lora), (
        f"loader widget[0] (lora_file) = {widgets[0]!r}, expected {str(stub_lora)!r}"
    )


def test_top_level_ref_video_load_chain(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    # The three preprocessing nodes
    loaders = ed.find_nodes_by_type("VHS_LoadVideo")
    resizers = ed.find_nodes_by_type("ImageResizeKJv2")
    preprocs = ed.find_nodes_by_type("LTXVPreprocess")
    assert loaders, "VHS_LoadVideo not added by apply script"
    # The canonical already has its own ImageResizeKJv2 / LTXVPreprocess for the
    # init image; we must add a SECOND of each for the ref video. Asserts >= 2.
    assert len(resizers) >= 2, "expected >=2 ImageResizeKJv2 (init + ref video)"
    assert len(preprocs) >= 2, "expected >=2 LTXVPreprocess (init + ref video)"


# ---------- subgraph wiring ----------

def test_subgraph_has_reference_video_image_input(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    sg = ed.get_subgraph(0)
    assert sg is not None, "subgraph missing"
    image_inputs = [
        i for i in sg.get("inputs", [])
        if i.get("type") == "IMAGE"
    ]
    matching = [i for i in image_inputs if i.get("name") == "reference_video"]
    assert matching, (
        f"new IMAGE input 'reference_video' not added to subgraph schema. "
        f"existing IMAGE inputs: {[i.get('name') for i in image_inputs]}"
    )


def test_subgraph_has_iclora_guide(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    sg = ed.get_subgraph(0)
    assert sg is not None
    guides = [
        n for n in sg.get("nodes", [])
        if n.get("type") == "LTXAddVideoICLoRAGuide"
    ]
    assert len(guides) == 1, (
        f"expected exactly 1 LTXAddVideoICLoRAGuide in subgraph, found {len(guides)}"
    )


def test_subgraph_has_image_range_slicer(staged_paths, stub_ref_video, stub_lora):
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    sg = ed.get_subgraph(0)
    assert sg is not None
    slicers = [
        n for n in sg.get("nodes", [])
        if n.get("type") == "GetImageRangeFromBatch"
    ]
    assert len(slicers) == 1, (
        f"expected exactly 1 GetImageRangeFromBatch in subgraph, found {len(slicers)}"
    )


def test_subgraph_iclora_guide_outputs_pass_through_cropguides(
    staged_paths, stub_ref_video, stub_lora,
):
    """F3 symmetry: IC-LoRA guide CONDITIONING outputs must reach CFGGuider
    only via LTXVCropGuidesNoLatent (the established F3 path)."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    sg = ed.get_subgraph(0)
    assert sg is not None
    guide = next(
        (n for n in sg.get("nodes", []) if n.get("type") == "LTXAddVideoICLoRAGuide"),
        None,
    )
    assert guide is not None
    guide_id = guide["id"]

    # Find any link from guide CONDITIONING outputs (slot 0 = positive,
    # slot 1 = negative). Each should go to LTXVCropGuidesNoLatent or
    # LTXVCropGuides — never directly to CFGGuider.
    for slot in (0, 1):
        cond_consumers = [
            l for l in sg.get("links", [])
            if l.get("origin_id") == guide_id and l.get("origin_slot") == slot
        ]
        assert cond_consumers, (
            f"IC-LoRA guide CONDITIONING output slot {slot} has no consumers"
        )
        for link in cond_consumers:
            tgt_id = link.get("target_id")
            tgt_node = next(
                (n for n in sg.get("nodes", []) if n.get("id") == tgt_id),
                None,
            )
            assert tgt_node is not None, (
                f"link to dangling subgraph node {tgt_id}"
            )
            # IC-LoRA guide CONDITIONING must NOT feed CFGGuider directly
            assert tgt_node.get("type") != "CFGGuider", (
                f"IC-LoRA guide slot {slot} -> CFGGuider directly "
                f"(F3 symmetry violated; should pass through CropGuides)"
            )


# ---------- pre-flight ----------

def test_preflight_fails_when_ref_video_missing(staged_paths, stub_lora, tmp_path):
    input_path, output_path = staged_paths
    nonexistent = tmp_path / "does_not_exist.mp4"
    result = _apply(input_path, output_path, nonexistent, stub_lora)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "reference" in combined or "ref" in combined or "video" in combined or "missing" in combined


def test_preflight_fails_when_ic_lora_missing(staged_paths, stub_ref_video, tmp_path):
    input_path, output_path = staged_paths
    nonexistent = tmp_path / "does_not_exist.safetensors"
    result = _apply(input_path, output_path, stub_ref_video, nonexistent)
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "lora" in combined or "missing" in combined


def test_preflight_fails_when_canonical_has_dead_scaffolding(
    staged_paths, stub_ref_video, stub_lora, tmp_path,
):
    """If the input still has #1625/#1626/#1627, refuse — Step 0 must run first."""
    input_path, output_path = staged_paths
    # Restore scaffolding via the strip script's --revert in-place.
    # (We use the strip script here; cleanest way to re-add scaffolding.)
    strip_script = REPO_ROOT / "scripts" / "apply_strip_dead_lora_loaders.py"
    workflows_dir = input_path.parent
    # The strip script expects a directory, so move input there and rename
    # to canonical filename
    canonical_named = workflows_dir / "audio-loop-music-video_latent.json"
    if canonical_named != input_path:
        input_path.rename(canonical_named)
        input_path = canonical_named
    revert_result = subprocess.run(
        [sys.executable, str(strip_script),
         "--workflows-dir", str(workflows_dir), "--revert"],
        capture_output=True, text=True, check=False,
    )
    assert revert_result.returncode == 0, revert_result.stderr

    # Now the input has scaffolding restored. Apply should refuse.
    result = _apply(input_path, output_path, stub_ref_video, stub_lora)
    assert result.returncode != 0, (
        f"apply should refuse when canonical still has dead scaffolding\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "scaffolding" in combined or "strip" in combined or "1625" in combined or "1626" in combined


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


def test_audit_clean_after_apply(staged_paths, stub_ref_video, stub_lora):
    """Post-apply staged file should have zero ERR-level findings beyond
    pre-existing canonical issues unrelated to IC-LoRA."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(output_path)
    new_iclora_errs = [
        f for f in findings
        if f.status == "ERR" and "iclora" in f.check.lower()
    ]
    assert not new_iclora_errs, (
        f"apply produced IC-LoRA-related ERR findings: {new_iclora_errs}"
    )


def test_audit_fires_err_when_guide_bypasses_cropguides(
    staged_paths, stub_ref_video, stub_lora,
):
    """If we splice the guide.positive directly into CFGGuider (skipping
    cropguides), the audit must catch it."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))

    # Mutate: redirect guide CONDITIONING outputs straight to CFGGuider
    ed = WorkflowEditor(output_path)
    sg = ed.get_subgraph(0)
    assert sg is not None
    guide = next(n for n in sg["nodes"] if n["type"] == "LTXAddVideoICLoRAGuide")
    cfg_id = SUBGRAPH_CFGGUIDER_ID
    # Add a leaky link guide.0 -> CFGGuider.1
    ed.add_subgraph_link(guide["id"], 0, cfg_id, 1, "CONDITIONING")
    ed.save()

    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(output_path)
    errs = [
        f for f in findings
        if f.check == "iclora_video_reference_guide_in_loop_with_cropguides"
        and f.status == "ERR"
    ]
    assert errs, "audit should ERR when guide bypasses cropguides"


def test_audit_fires_err_when_loader_missing(
    staged_paths, stub_ref_video, stub_lora,
):
    """If we remove the IC-LoRA loader but keep the guide, audit must ERR."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))

    ed = WorkflowEditor(output_path)
    loaders = ed.find_nodes_by_type("LTXICLoRALoaderModelOnly")
    assert loaders
    ed.remove_node_and_links(loaders[0]["id"])
    ed.save()

    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(output_path)
    errs = [
        f for f in findings
        if f.check == "iclora_loader_present_when_guide_present"
        and f.status == "ERR"
    ]
    assert errs, "audit should ERR when loader is missing but guide present"


def test_audit_fires_err_when_preprocess_val_wrong(
    staged_paths, stub_ref_video, stub_lora,
):
    """If LTXVPreprocess val drifts from 18, audit must ERR (F2 symmetry)."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))

    # Mutate ALL LTXVPreprocess to use val=0 (skip preprocessing).
    # That breaks F2 for both init and ref-video chains.
    ed = WorkflowEditor(output_path)
    for n in ed.find_nodes_by_type("LTXVPreprocess"):
        wv = n.get("widgets_values") or []
        if wv:
            n["widgets_values"][0] = 0
    ed.save()

    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(output_path)
    errs = [
        f for f in findings
        if f.check == "iclora_ref_video_preprocess_symmetry"
        and f.status == "ERR"
    ]
    assert errs, "audit should ERR when no LTXVPreprocess has val=18"


# ---------- sliding mode (Phase 2) ----------

def test_ref_mode_static_default_has_no_calculator(
    staged_paths, stub_ref_video, stub_lora,
):
    """Default (no --ref-mode flag) is static. No SimpleCalculatorKJ
    should be added; GetImageRangeFromBatch.start_index stays a widget-
    driven INT, not a wired input."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(input_path, output_path, stub_ref_video, stub_lora))
    ed = WorkflowEditor(output_path)
    sg = ed.get_subgraph(0)
    assert sg is not None
    calcs = [n for n in sg.get("nodes", []) if n.get("type") == "SimpleCalculatorKJ"]
    assert calcs == [], (
        f"static mode (default) must not add SimpleCalculatorKJ; found {len(calcs)}"
    )
    slicer = next(
        (n for n in sg.get("nodes", []) if n.get("type") == "GetImageRangeFromBatch"),
        None,
    )
    assert slicer is not None, "GetImageRangeFromBatch missing"
    start_idx_input = next(
        (i for i in slicer.get("inputs", []) if i.get("name") == "start_index"),
        None,
    )
    assert start_idx_input is not None
    assert start_idx_input.get("link") is None, (
        "static mode: start_index must remain widget-driven (link=None)"
    )
    assert "widget" in start_idx_input, (
        "static mode: start_index must keep its widget field"
    )


def test_ref_mode_sliding_inserts_calculator_and_wires_start_index(
    staged_paths, stub_ref_video, stub_lora,
):
    """--ref-mode sliding inserts SimpleCalculatorKJ in the subgraph,
    bakes ref_fps (default 25) into the expression as `round(a * 25)`,
    wires its `a` input from the subgraph's video_start_time (FLOAT),
    and rewires GetImageRangeFromBatch.start_index from widget to a
    wired INT input fed by the calculator's Int output (slot 1)."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(
        input_path, output_path, stub_ref_video, stub_lora,
        "--ref-mode", "sliding",
    ))
    ed = WorkflowEditor(output_path)
    sg = ed.get_subgraph(0)
    assert sg is not None

    calcs = [n for n in sg.get("nodes", []) if n.get("type") == "SimpleCalculatorKJ"]
    assert len(calcs) == 1, (
        f"sliding mode must add exactly one SimpleCalculatorKJ; found {len(calcs)}"
    )
    calc = calcs[0]
    expr = (calc.get("widgets_values") or [None])[0]
    assert expr == "round(a * 25)", (
        f"calculator expression must bake force_rate (default 25) into "
        f'the formula; got widgets_values[0]={expr!r}'
    )

    # Calculator's `a` input must be wired (from video_start_time via -10)
    a_input = next((i for i in calc.get("inputs", []) if i.get("name") == "a"), None)
    assert a_input is not None, "SimpleCalculatorKJ missing 'a' input"
    assert a_input.get("link") is not None, (
        f"calculator.a must be wired; got link={a_input.get('link')}"
    )

    # Slicer.start_index must now be wired (link != None) AND have no widget field
    slicer = next(
        (n for n in sg.get("nodes", []) if n.get("type") == "GetImageRangeFromBatch"),
        None,
    )
    assert slicer is not None
    start_idx = next(
        (i for i in slicer.get("inputs", []) if i.get("name") == "start_index"),
        None,
    )
    assert start_idx is not None
    assert start_idx.get("link") is not None, (
        "sliding mode: GetImageRangeFromBatch.start_index must be wired"
    )
    assert "widget" not in start_idx, (
        "sliding mode: start_index input must NOT carry a widget field "
        "(widget→wire conversion incomplete)"
    )

    # The link feeding start_index must originate from the calculator's
    # Int output (slot 1, not Float slot 0).
    link_id = start_idx["link"]
    link = next((l for l in sg["links"] if l.get("id") == link_id), None)
    assert link is not None
    assert link.get("origin_id") == calc["id"], (
        f"start_index link must originate from SimpleCalculatorKJ; "
        f"got origin_id={link.get('origin_id')}"
    )
    assert link.get("origin_slot") == 1, (
        f"start_index link must read from calculator's Int output (slot 1); "
        f"got origin_slot={link.get('origin_slot')}"
    )


def test_ref_mode_sliding_with_custom_ref_fps_bakes_into_expression(
    staged_paths, stub_ref_video, stub_lora,
):
    """--ref-fps overrides the default 25 baked into BOTH the calculator
    expression AND VHS_LoadVideo.force_rate (single source of truth)."""
    input_path, output_path = staged_paths
    _assert_ok(_apply(
        input_path, output_path, stub_ref_video, stub_lora,
        "--ref-mode", "sliding", "--ref-fps", "30",
    ))
    ed = WorkflowEditor(output_path)
    # Calculator expression must reference 30
    sg = ed.get_subgraph(0)
    calc = next((n for n in sg["nodes"] if n.get("type") == "SimpleCalculatorKJ"), None)
    assert calc is not None
    expr = calc.get("widgets_values", [None])[0]
    assert expr == "round(a * 30)", (
        f"calculator expression must bake --ref-fps=30; got {expr!r}"
    )
    # VHS_LoadVideo.force_rate must also be 30 (single SoT)
    vhs = ed.find_nodes_by_type("VHS_LoadVideo")[0]
    fr = vhs.get("widgets_values", {}).get("force_rate")
    assert fr == 30, (
        f"VHS_LoadVideo.force_rate must match --ref-fps; got {fr}"
    )
