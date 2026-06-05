"""apply_pre_decode_cleanup.

Last updated: 2026-06-05

Splices ``PreDecodeCleanup`` (free pinned staging + unload all models)
immediately upstream of ``TrimVideoLatentToAudio`` on the final-output decode
path of every full-song LOOP workflow.

Why: the full-song final VAE decode is a single-node RAM spike on top of
page-locked staging + offloaded models — a sum that can kernel-OOM the
process at the LAST step, after all sampling succeeded, regardless of launch
flags. By decode time the models are no longer needed; the cleanup removes
them from the decode profile. Mechanism + sizing:
``docs/reference/benchmarking_memory_pressure.md``.

Scope rules (explicit, not implicit-by-omission):
  - ELIGIBLE: workflows with an ACTIVE TensorLoopOpen (the full-song loop
    family) whose final-output decoder is fed by TrimVideoLatentToAudio.
  - EXEMPT: single-pass / short-clip workflows (no TensorLoop). No RAM spike
    to dodge there, and back-to-back battery renders would pay a full model
    cold-reload on every prompt.
  - Workflows missing the trim entirely: run
    ``scripts/apply_trim_video_latent_to_audio.py`` first (this script
    splices upstream of the trim so the trim-feeds-decoder audit invariant
    stays intact).

Audit pair: ``pre_decode_cleanup_present`` (WARN) in
``scripts/audit_workflows.py``.

Usage:
    uv run --group dev python scripts/apply_pre_decode_cleanup.py
    uv run --group dev python scripts/apply_pre_decode_cleanup.py --dry-run
    uv run --group dev python scripts/apply_pre_decode_cleanup.py --revert

Idempotent. ``--revert`` splices the node back out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402
from apply_trim_video_latent_to_audio import (  # noqa: E402
    _find_latent_decoder,
    _find_latent_input_slot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_GLOBS = (
    "example_workflows/*.json",
    "example_workflows/experimental/*.json",
)

CLEANUP_TYPE = "PreDecodeCleanup"


def _has_active_tensor_loop(ed: WorkflowEditor) -> bool:
    return any(
        n.get("type") == "TensorLoopOpen" and n.get("mode", 0) == 0
        for n in ed.wf["nodes"]
    )


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    if not _has_active_tensor_loop(ed):
        return "skip (no active TensorLoop — single-pass/short decode, deliberately exempt)"

    decoder = _find_latent_decoder(ed)
    if decoder is None:
        return "skip (no active LATENT decoder reachable from VHS_VideoCombine.images)"
    latent_slot = _find_latent_input_slot(decoder)
    if latent_slot is None:
        return f"skip (decoder #{decoder['id']} has no latents/samples LATENT input)"

    trim_link = ed.find_link_to_slot(decoder["id"], latent_slot)
    if trim_link is None:
        return f"skip (decoder #{decoder['id']} latent input unlinked)"
    _, trim_id, _, *_ = trim_link
    trim = ed.find_node(trim_id)
    if trim.get("type") != "TrimVideoLatentToAudio":
        return (f"skip (decoder fed by {trim.get('type')}, not TrimVideoLatentToAudio — "
                "run apply_trim_video_latent_to_audio.py first)")

    src_link = ed.find_link_to_slot(trim_id, 0)  # trim.latent
    if src_link is None:
        return f"skip (trim #{trim_id}.latent has no incoming link)"
    _, src_id, src_slot, *_ = src_link
    src_node = ed.find_node(src_id)

    if revert:
        if src_node.get("type") != CLEANUP_TYPE:
            return "already reverted"
        cleanup_in = ed.find_link_to_slot(src_id, 0)
        if cleanup_in is None:
            return f"skip (cleanup #{src_id}.latent has no incoming link)"
        _, orig_src, orig_slot, *_ = cleanup_in
        if dry_run:
            return f"would revert (remove {CLEANUP_TYPE} #{src_id})"
        ed.rewire_input(trim_id, 0, orig_src, orig_slot, "LATENT")
        ed.remove_node_and_links(src_id)
        ed.save()
        return f"reverted (removed {CLEANUP_TYPE} #{src_id})"

    if src_node.get("type") == CLEANUP_TYPE:
        return f"no change ({CLEANUP_TYPE} #{src_id} already wired)"

    if dry_run:
        return f"would update (splice {CLEANUP_TYPE} between #{src_id} and trim #{trim_id})"

    tx, ty = trim.get("pos", [0, 0])
    new_id = ed.add_top_level_node(
        node_type=CLEANUP_TYPE,
        pos=[tx - 300, ty + 180],
        size=[270, 82],
        inputs=[{"localized_name": "latent", "name": "latent", "type": "LATENT", "link": None}],
        outputs=[WorkflowEditor.out("latent", "LATENT")],
        widgets_values=["always"],
        properties={"Node name for S&R": CLEANUP_TYPE,
                    "aux_id": "fblissjr/ComfyUI-AudioLoopHelper"},
        title="Pre-Decode Cleanup (unload models)",
    )
    ed.rewire_input(trim_id, 0, new_id, 0, "LATENT")
    ed.add_link(src_id, src_slot, new_id, 0, "LATENT")
    ed.prune_orphan_output_links()
    ed.save()
    return f"updated (spliced {CLEANUP_TYPE} #{new_id} upstream of trim #{trim_id})"


def apply(revert: bool, dry_run: bool) -> int:
    action = ("Would " if dry_run else "") + ("revert" if revert else "apply").capitalize()
    print(f"{action} {CLEANUP_TYPE} across loop workflows...")
    fail = 0
    for pattern in TARGET_GLOBS:
        for wf_path in sorted(REPO_ROOT.glob(pattern)):
            status = _apply_one(wf_path, revert, dry_run)
            print(f"  {wf_path.relative_to(REPO_ROOT)}: {status}")
            if status.startswith("load error"):
                fail += 1
    return fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
