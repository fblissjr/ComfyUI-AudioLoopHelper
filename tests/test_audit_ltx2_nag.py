"""Tests for the `ltx2_nag_reaches_loop` audit invariant.

LTX2_NAG patches the MODEL via `add_object_patch`. Unlike PromptRelay
patches (CLIP-driven, evicted on offload), NAG patches survive the loop
offload/reload IF AND ONLY IF the patched MODEL is visible to the loop
subgraph. The canonical wiring routes LTX2_NAG.MODEL → ... → SetNode(572)
(=  loop entry via Get 654). If a future edit forks the chain so that
LTX2_NAG only reaches the initial CFGGuider, the loop body sees an
unpatched MODEL and the NAG negative prompt silently disengages from
iteration 1 onward.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example_workflows"
CANONICAL = EXAMPLE_DIR / "audio-loop-music-video_latent.json"
RETAKE = EXAMPLE_DIR / "audio-loop-music-video_retake.json"

LTX2_NAG_ID = 508
SETNODE_MODEL_ID = 572


def _load_audit_module():
    spec = spec_from_file_location(
        "audit_workflows", REPO_ROOT / "scripts" / "audit_workflows.py",
    )
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_canonical_workflow_nag_reaches_loop_ok():
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(CANONICAL)
    matches = [f for f in findings if f.check == "ltx2_nag_reaches_loop"]
    assert any(f.status == "OK" for f in matches), (
        f"expected OK ltx2_nag_reaches_loop on canonical, got {matches}"
    )


def test_retake_workflow_skipped():
    """Retake has no LTX2_NAG and no loop — check should not fire."""
    audit_mod = _load_audit_module()
    findings = audit_mod._audit_one(RETAKE)
    matches = [f for f in findings if f.check == "ltx2_nag_reaches_loop"]
    assert not any(f.status == "ERR" for f in matches), (
        f"retake should be exempt, got {matches}"
    )


def test_workflow_with_severed_nag_chain_errs(tmp_path):
    """Sever the MODEL chain between LTX2_NAG and SetNode(572): rewire
    SetNode(572) input to bypass NAG entirely (e.g., directly from a
    pre-NAG anchor). Audit must flag ERR — NAG won't reach the loop."""
    audit_mod = _load_audit_module()
    broken = tmp_path / "audio-loop-music-video_latent.json"
    data = orjson.loads(CANONICAL.read_bytes())

    # Find the link that feeds SetNode(572).slot[0] and reroute it from
    # an upstream node that isn't downstream of NAG. In the canonical
    # chain: NAG(508)→503→1625→1626→1627→572(SetNode).  We'll point
    # 572's MODEL input directly at AttentionTuner/LTX2BasicSampler
    # (anything pre-NAG). Easiest: grab the LTX2_NAG.MODEL INPUT source
    # — that's pre-NAG by construction.
    links = data["links"]
    nag_input_link = next(
        lk for lk in links
        if isinstance(lk, list) and lk[3] == LTX2_NAG_ID and lk[4] == 0
    )
    pre_nag_src = nag_input_link[1]
    pre_nag_slot = nag_input_link[2]

    set_input_link = next(
        lk for lk in links
        if isinstance(lk, list) and lk[3] == SETNODE_MODEL_ID and lk[4] == 0
    )
    # Rewire 572's MODEL input to come from pre-NAG source
    set_input_link[1] = pre_nag_src
    set_input_link[2] = pre_nag_slot

    broken.write_bytes(orjson.dumps(data))
    findings = audit_mod._audit_one(broken)
    matches = [f for f in findings if f.check == "ltx2_nag_reaches_loop"]
    errs = [f for f in matches if f.status == "ERR"]
    assert errs, f"expected ERR when NAG severed from loop, got {matches}"
    assert "loop" in errs[0].message.lower() or "subgraph" in errs[0].message.lower()
