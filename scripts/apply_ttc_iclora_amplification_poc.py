"""POC for TTC1: amplifying the IC-LoRA contribution at inference time.

Inspired by the classifier-free-guidance formula -- an existing inference-time
technique that amplifies a signal by taking the difference between conditional
and unconditional model predictions and scaling it. We apply the same form with
(positive_with_iclora, positive_without_iclora) as the two inputs:

    eps_out = eps_without_iclora + w * (eps_with_iclora - eps_without_iclora)

Feeding these two conditionings to CFGGuider's two inputs lets the existing
sampler's math run the dual forward pass and blend, with w as the cfg widget.
No new sampler code required.

Sweep cfg on CFGGuider to explore amplification:
    cfg = 0  -> no IC-LoRA contribution (structural reference ignored)
    cfg = 1  -> standard IC-LoRA behavior (sanity check vs unmodified workflow)
    cfg > 1  -> amplified IC-LoRA (untested territory for distilled LTX 2.3)
    cfg < 0  -> anti-reference (push away from the structural reference)

Full analysis: internal/analysis/iclora_landscape_analysis.md section TTC1.

POC protocol:
1. Apply this script to stage the workflow.
2. In ComfyUI frontend:
   a. Un-bypass LTXAddVideoICLoRAGuide(1622) and LTXICLoRALoaderModelOnly(1619).
   b. Un-bypass LTXVImgToVideoInplaceKJ(531). An init is required -- R16 showed
      no-init with IC-LoRA active produces reference leakage because the
      IC-LoRA has no competing visual anchor.
   c. Bypass LTX2_NAG(508). POC variable isolation: with NAG active, the
      negative-conditioning slot serves both NAG and the without-iclora stream,
      which muddles the test.
   d. Set LoadImage(444) to your init image.
   e. Set LoadVideo(1620) to your color spectrogram.
3. Run the cfg sweep (same seed each time):
   a. cfg=1.0 -- sanity check, should match non-POC workflow byte-close.
   b. cfg=2.0 -- first amplification test.
   c. cfg=3.0, 5.0 -- stress test.
   d. cfg=0.0 -- baseline without IC-LoRA (should resemble no-IC-LoRA output).
   e. cfg=-1.0 -- anti-reference (exploratory).

Usage:
    uv run python scripts/apply_ttc_iclora_amplification_poc.py
    uv run python scripts/apply_ttc_iclora_amplification_poc.py --revert

Idempotent. Forks the spectrogram_iclora_minimal workflow; original is untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from workflow_utils import WorkflowEditor  # noqa: E402

SRC = REPO_ROOT / "example_workflows/experimental/spectrogram_iclora_minimal.json"
OUTPUT = REPO_ROOT / "example_workflows/experimental/iclora_amplification_poc.json"

LTXV_CONDITIONING = 164   # pos without IC-LoRA
IC_LORA_GUIDE = 1622      # pos with IC-LoRA
CFG_GUIDER = 153          # receives the rewire


def build(output_path: Path) -> None:
    ed = WorkflowEditor(SRC)

    missing = ed.require_nodes([LTXV_CONDITIONING, IC_LORA_GUIDE, CFG_GUIDER])
    if missing:
        raise SystemExit(f"Source workflow missing expected nodes: {missing}")

    cfg_guider = ed.find_node(CFG_GUIDER)
    neg_slot = ed.find_input_slot(cfg_guider, "negative")

    # Replace whatever feeds CFGGuider.negative with LTXVConditioning.positive
    # (the "positive without IC-LoRA" stream, taken before the IC-LoRA guide
    # node modifies the conditioning). CFGGuider then computes
    #     eps_without_iclora + cfg * (eps_with_iclora - eps_without_iclora)
    # on every denoising step.
    new_lid = ed.rewire_input(
        tgt_node=CFG_GUIDER, tgt_slot=neg_slot,
        new_src=LTXV_CONDITIONING, new_src_slot=0,
        dtype="CONDITIONING",
    )
    print(f"  rewired CFGGuider({CFG_GUIDER}).negative <- "
          f"LTXVConditioning({LTXV_CONDITIONING}).positive (link {new_lid})")

    cfg_guider["title"] = "CFGGuider (TTC1: cfg = IC-LoRA amplification w)"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ed.save(output_path)
    print(f"\nWrote {output_path}")
    print(f"  nodes: {len(ed.wf['nodes'])}, links: {len(ed.wf['links'])}")
    print("\nNext: open in ComfyUI frontend, follow protocol in script docstring.")


def revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"nothing to revert at {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revert", action="store_true", help="Delete the staged workflow")
    ap.add_argument("--output", default=OUTPUT, type=Path, help="Override output path")
    args = ap.parse_args()
    if args.revert:
        revert(args.output)
    else:
        build(args.output)


if __name__ == "__main__":
    main()
