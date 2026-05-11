"""apply_trim_video_latent_to_audio.

Last updated: 2026-05-10

Production apply: splice ``TrimVideoLatentToAudio`` between the
assembled video latent and the final VAE decode. Replaces the
image-level F14 (``TrimImageBatchToAudio``) on the architectural
fix path — trim at the producer (pre-decode) instead of the
consumer (post-decode). Same user-visible output (verified via
A/B render on 2026-05-10) at lower decode VRAM/time.

Topology:
  Before: <decoder>.latents <- LatentConcat (or LTXVCropGuides)
  After:  <decoder>.latents <- TrimVideoLatentToAudio
          TrimVideoLatentToAudio.latent <- (original source)

Audio source: the same source feeding ``VHS_VideoCombine.audio``
(traced through F14 if F14 is still wired). fps source:
``LTXFramePlanner.fps_int`` when present, else widget=25.

Idempotent. ``--revert`` splices it out and restores the direct
latent → decode wiring. ``--dry-run`` shows what would change.

Migration note: this script runs BEFORE F14 revert in the safe
migration order (see CHANGELOG / postmortem). Workflows briefly
carry BOTH F14 and the latent trim — that's fine (F14 becomes a
no-op pass-through on correctly-trimmed latent batches).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import DECODER_TYPES, WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_FILES = {
    "edit_anything_v2v_reference.json",
    "upscale_3pass_reference.json",
}


def _find_latent_decoder(ed: WorkflowEditor) -> dict | None:
    """Pick the active LATENT-decoder whose output reaches an active
    VHS_VideoCombine.images (possibly via TrimImageBatchToAudio)."""
    combines = [
        n for n in ed.wf["nodes"]
        if n.get("type") == "VHS_VideoCombine" and n.get("mode", 0) == 0
    ]
    if not combines:
        return None
    combine = combines[0]
    images_link = ed.find_link_to_slot(combine["id"], 0)
    if images_link is None:
        return None

    cur_id = images_link[1]
    seen: set[int] = set()
    while cur_id not in seen:
        seen.add(cur_id)
        node = ed.find_node(cur_id)
        ntype = node.get("type")
        if ntype in DECODER_TYPES:
            return node
        # Step backwards through pass-throughs that take IMAGE in / out.
        # F14 sits between decoder and combine.
        if ntype == "TrimImageBatchToAudio":
            link = ed.find_link_to_slot(cur_id, 0)
            if link is None:
                return None
            cur_id = link[1]
            continue
        # Unknown intermediate — give up rather than guess.
        return None
    return None


def _find_latent_input_slot(decoder: dict) -> int | None:
    """Decoder LATENT input is named ``latents`` (LTX) or ``samples``
    (core VAEDecode). Return the slot index of whichever exists."""
    for name in ("latents", "samples"):
        for i, s in enumerate(decoder.get("inputs", [])):
            if s.get("name") == name and s.get("type") == "LATENT":
                return i
    return None


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    if wf_path.name in SKIP_FILES:
        return "skip (excluded by SKIP_FILES)"
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    decoder = _find_latent_decoder(ed)
    if decoder is None:
        return "skip (no active LATENT decoder reachable from VHS_VideoCombine.images)"
    latent_slot = _find_latent_input_slot(decoder)
    if latent_slot is None:
        return f"skip (decoder #{decoder['id']} has no latents/samples LATENT input)"

    incoming = ed.find_link_to_slot(decoder["id"], latent_slot)
    if incoming is None:
        return f"skip (decoder #{decoder['id']}.latents has no incoming link)"
    _, src_id, src_slot, *_ = incoming
    src_node = ed.find_node(src_id)

    if revert:
        if src_node.get("type") != "TrimVideoLatentToAudio":
            return "already reverted"
        # Splice out: rewire decoder.latent ← original upstream of trim.
        trim_in = ed.find_link_to_slot(src_id, 0)
        if trim_in is None:
            return f"skip (trim #{src_id}.latent has no incoming link)"
        _, orig_src, orig_slot, *_ = trim_in
        if dry_run:
            return f"would revert (remove TrimVideoLatentToAudio #{src_id}, restore #{orig_src}.{orig_slot} -> decoder)"
        ed.rewire_input(decoder["id"], latent_slot, orig_src, orig_slot, "LATENT")
        ed.remove_node_and_links(src_id)
        ed.save()
        return f"reverted (removed TrimVideoLatentToAudio #{src_id})"

    if src_node.get("type") == "TrimVideoLatentToAudio":
        return f"no change (TrimVideoLatentToAudio #{src_id} already wired)"

    # Need audio source — trace from VHS_VideoCombine.audio.
    combine = next(
        n for n in ed.wf["nodes"]
        if n.get("type") == "VHS_VideoCombine" and n.get("mode", 0) == 0
    )
    audio_link = ed.find_link_to_slot(combine["id"], 1)
    if audio_link is None:
        return f"skip (VHS_VideoCombine #{combine['id']}.audio has no link — can't size trim)"
    _, aud_src, aud_src_slot, *_ = audio_link

    # fps source: LTXFramePlanner.fps_int when present (output slot 4).
    fp = next(
        (n for n in ed.wf["nodes"] if n.get("type") == "LTXFramePlanner"),
        None,
    )

    if dry_run:
        fps_note = "fps from LTXFramePlanner" if fp else "fps widget=25"
        return f"would update (splice TrimVideoLatentToAudio between #{src_id} and decoder #{decoder['id']}, {fps_note})"

    dx, dy = decoder.get("pos", [0, 0])
    trim_id = ed.add_top_level_node(
        node_type="TrimVideoLatentToAudio",
        pos=[dx - 320, dy], size=[300, 100],
        inputs=[
            WorkflowEditor.io_in("latent", "LATENT"),
            WorkflowEditor.io_in("audio", "AUDIO"),
            WorkflowEditor.widget_in("fps", "INT"),
        ],
        outputs=[WorkflowEditor.out("latent", "LATENT")],
        widgets_values=[25],
        title="Trim latent to audio",
    )

    # Rewire: decoder.latent ← trim.output
    ed.rewire_input(decoder["id"], latent_slot, trim_id, 0, "LATENT")
    # trim.latent ← original source
    ed.add_link(src_id, src_slot, trim_id, 0, "LATENT")
    # trim.audio ← combine.audio source
    ed.add_link(aud_src, aud_src_slot, trim_id, 1, "AUDIO")
    # trim.fps ← LTXFramePlanner.fps_int when present
    if fp is not None:
        ed.add_link(fp["id"], 4, trim_id, 2, "INT")

    ed.save()
    fps_note = "fps from LTXFramePlanner" if fp else "fps widget=25"
    return f"updated (added TrimVideoLatentToAudio #{trim_id} before decoder #{decoder['id']}, {fps_note})"


def _iter_workflows() -> list[Path]:
    paths: list[Path] = []
    for d in (REPO_ROOT / "example_workflows", REPO_ROOT / "internal" / "workflows"):
        if not d.exists():
            continue
        paths.extend(sorted(d.rglob("*.json")))
    return paths


def apply(revert: bool, dry_run: bool) -> int:
    action = ("Would " if dry_run else "") + ("revert" if revert else "apply").capitalize()
    print(f"{action} TrimVideoLatentToAudio across workflows...")
    fail = 0
    for wf_path in _iter_workflows():
        rel = wf_path.relative_to(REPO_ROOT)
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {rel}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true", help="Undo the change.")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
