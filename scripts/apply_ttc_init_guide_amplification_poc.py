"""apply_ttc_init_guide_amplification_poc.

Last updated: 2026-04-25

Stages a non-IC-LoRA variant of the canonical latent workflow that wires
TTC1 CFG-analog amplification on the **init-frame `LTXVAddLatentGuide`**
contribution inside the loop subgraph. Pairs the IC-LoRA-targeted POC
(`apply_ttc_iclora_amplification_poc.py`) — same mechanism, different
conditional, no IC-LoRA required.

Output lives under `example_workflows/experimental/` (the same
sibling directory as `iclora_amplification_poc.json`) so the staged
JSON is reviewable cross-machine. `scripts/audit_workflows.py`
includes this file in its scan and recognizes the deliberate F3
asymmetry on the negative branch as intentional.

Mechanism (mirrors the IC-LoRA POC docstring; see also CLAUDE.md
"CFG-analog amplification of any conditional contribution"):

    eps_out = eps_without_init_guide
              + cfg * (eps_with_init_guide - eps_without_init_guide)

Two branches reach `CFGGuider`:

  - positive (with init guide):
      INPUT_DISTRIBUTOR(slot 6)
        -> LTXVAddLatentGuide(1519) -> LTXVCropGuides(655) -> CFGGuider(644).positive
  - negative (without init guide), after this script:
      INPUT_DISTRIBUTOR(slot 6) -> CFGGuider(644).negative

The negative branch deliberately skips both `LTXVAddLatentGuide(1519)` and
`LTXVCropGuides(655)`. CropGuides is symmetry-mandatory only when guide
keyframe metadata exists on the conditioning (F3 in CLAUDE.md); the
without-guide branch never has any to strip, so the asymmetry is
load-bearing for this experiment, not a bug.

`cfg` on `CFGGuider(644)` then sweeps the amplification factor:
    cfg = 0  -> no init-guide contribution
    cfg = 1  -> standard init-guide behavior (sanity check vs unmodified workflow)
    cfg > 1  -> amplified init-guide pull (untested for distilled LTX 2.3)
    cfg < 0  -> anti-init-guide (push the loop AWAY from init-frame anchor)

Use cases this enables a clean answer for:
- Identity-drift studies: does amplifying the init-frame guide reduce
  iter-over-iter drift, or does it just over-anchor and freeze motion?
- F2/F3 symmetry follow-up: with amplification at hand, does the
  CropGuides-symmetry path exhibit a genuinely different drift profile
  from the IC-LoRA-amplification path?
- Negative-cfg ablation: -1 deliberately repels the loop from the init
  frame — useful for diagnosing "is the init image holding the loop too
  tight" vs "is something else causing static frames?"

POC protocol (UI side, after running this script):
1. Open the staged workflow in ComfyUI:
   `example_workflows/experimental/init_guide_amplification_poc.json`
2. **Bypass `LTX2_NAG`** (in the loop subgraph). With NAG active, the
   negative-conditioning slot serves both NAG and the without-init-guide
   stream, which muddles the test. Same constraint as the IC-LoRA POC.
3. Set `LoadImage` to your init image (production loop already requires one).
4. Set the audio source on the audio path (untouched by this script).
5. Run a cfg sweep on `CFGGuider(644).cfg` with a fixed seed:
     cfg=1.0  -- sanity check, should match non-POC workflow byte-close
     cfg=2.0, 3.0, 5.0  -- amplification stress test
     cfg=0.0  -- baseline without init-guide contribution
     cfg=-1.0  -- anti-init-guide exploratory

After each run, decode + extract audio if A/V analysis is part of your
study (`ffmpeg -i <out.mp4> -vn -acodec pcm_s16le <out.wav>` ->
`scripts/analyze_audio_features.py`). For visual identity-drift
analysis, sample frames at iter boundaries and diff against iter 1.

Compatibility:
- Composes on top of F2 (preprocess symmetry) and F3 (cropguides
  symmetry) — both already present on the source workflow per
  `audit_workflows.py`. This script does not touch the F3-symmetric
  positive path (#1519 -> #655 -> #644.positive).
- Does NOT compose with the IC-LoRA POC (different rewire of the same
  CFGGuider.negative slot). Run them as separate sweeps.

Usage:
    uv run --group dev python scripts/apply_ttc_init_guide_amplification_poc.py
    uv run --group dev python scripts/apply_ttc_init_guide_amplification_poc.py --revert
    uv run --group dev python scripts/apply_ttc_init_guide_amplification_poc.py --dry-run

Idempotent on the OUTPUT path. `--revert` deletes the staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from workflow_utils import WorkflowEditor  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = REPO_ROOT / "example_workflows/experimental/init_guide_amplification_poc.json"

# Subgraph node IDs (validated against `example_workflows/audio-loop-music-video_latent.json`)
ADD_LATENT_GUIDE = 1519       # LTXVAddLatentGuide -- adds init-frame guide at latent_idx=-1
CROP_GUIDES = 655             # LTXVCropGuides -- F3 symmetry node
CFG_GUIDER = 644              # CFGGuider -- target of the rewire
SUBGRAPH_INPUT_DISTRIBUTOR = -10  # virtual; slot 6 = "positive" (CONDITIONING)
POSITIVE_INPUT_SLOT = 6

REQUIRED_SUBGRAPH_NODES = (ADD_LATENT_GUIDE, CROP_GUIDES, CFG_GUIDER)


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = [nid for nid in REQUIRED_SUBGRAPH_NODES if ed.find_subgraph_node(nid) is None]
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required subgraph node(s) missing: {missing}. "
            "This script assumes the canonical audio-loop latent workflow subgraph layout."
        )
    cfg = ed.find_subgraph_node(CFG_GUIDER)
    assert cfg is not None
    if cfg.get("type") != "CFGGuider":
        raise SystemExit(
            f"Node {CFG_GUIDER} is type {cfg.get('type')!r}, expected CFGGuider."
        )


def _already_migrated(ed: WorkflowEditor) -> bool:
    """Return True iff `CFGGuider(644).negative` already reads from the
    subgraph input distributor's positive slot (the rewire target)."""
    cfg = ed.find_subgraph_node(CFG_GUIDER)
    if cfg is None:
        return False
    neg_slot = ed.find_input_slot(cfg, "negative")
    link = ed.find_subgraph_link_to_slot(CFG_GUIDER, neg_slot)
    return (
        link is not None
        and link.get("origin_id") == SUBGRAPH_INPUT_DISTRIBUTOR
        and link.get("origin_slot") == POSITIVE_INPUT_SLOT
    )


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if dry_run:
        ed = WorkflowEditor(input_path)
        _assert_required_nodes_present(ed)
        print(f"would copy {input_path} -> {output_path}")
        print(
            f"would rewire CFGGuider({CFG_GUIDER}).negative "
            f"<- INPUT_DISTRIBUTOR(slot {POSITIVE_INPUT_SLOT}) [positive, before AddLatentGuide]"
        )
        return

    if output_path.exists() and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
        return

    if input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    _assert_required_nodes_present(ed)

    cfg = ed.find_subgraph_node(CFG_GUIDER)
    assert cfg is not None
    neg_slot = ed.find_input_slot(cfg, "negative")

    new_lid = ed.rewire_subgraph_input(
        tgt_node=CFG_GUIDER, tgt_slot=neg_slot,
        new_src=SUBGRAPH_INPUT_DISTRIBUTOR, new_src_slot=POSITIVE_INPUT_SLOT,
        dtype="CONDITIONING",
    )
    print(
        f"  rewired CFGGuider({CFG_GUIDER}).negative "
        f"<- INPUT_DISTRIBUTOR(slot {POSITIVE_INPUT_SLOT}) [positive, before AddLatentGuide] "
        f"(link {new_lid})"
    )

    cfg["title"] = "CFGGuider (TTC1: cfg = init-guide amplification w)"

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Run audit_workflows.py against {output_path.name} to confirm F2/F3 still hold")
    print(f"  3. Open in ComfyUI, bypass LTX2_NAG, sweep CFGGuider({CFG_GUIDER}).cfg per docstring")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied/changed without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return

    _migrate(Path(args.input), output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
