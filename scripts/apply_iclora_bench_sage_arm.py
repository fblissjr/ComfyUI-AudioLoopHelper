"""apply_iclora_bench_sage_arm.

Last updated: 2026-05-01

Produces a sage-attention arm variant of the iclora bench workflow for
controlled A/B comparison of sage configurations. Reads from
`internal/scratch/audio-loop-music-video_latent_iclora_bench.json` (the
ProfileBegin/IterStep/End-wired bench variant from
`apply_iclora_bench_profiling.py`) and emits one of four arms:

  - ours    : AudioLoopHelperSageAttention active, no KJ patch
              (= the bench variant unchanged; produces a copy)
  - off     : AudioLoopHelperSageAttention bypassed (mode=4),
              no KJ patch — runs ComfyUI's default attention dispatch
              (likely attention_pytorch). Establishes the "no sage"
              floor.
  - kj      : AudioLoopHelperSageAttention bypassed (mode=4),
              LTX2MemoryEfficientSageAttentionPatch inserted on the
              MODEL chain — KJ's per-block + RoPE-fusion path
  - stacked : both active. LTX-2 transformer blocks use KJ's per-block
              patch; any leftover attention layers (e.g., text-encoder
              cross-attention if not part of patched blocks) fall through
              to ours. Telemetry from our node will be PARTIAL (only
              captures the leftover layers).

The KJ insertion splices LTX2MemoryEfficientSageAttentionPatch into the
MODEL chain between LTX2SamplingPreviewOverride(#503) and the existing
LTXICLoRALoaderModelOnly that the iclora wiring inserted. Order:

  Pre:   #503 -> #1635 (LTXICLoRALoaderModelOnly) -> #572 (SetNode "model")
  Post:  #503 -> #2012 (LTX2MemoryEfficientSageAttentionPatch)
                  -> #1635 -> #572

Putting the KJ patch BEFORE the IC-LoRA loader means the patch sees the
LoRA-modified MODEL. Both nodes preserve MODEL → MODEL identity, so
order is mostly irrelevant for correctness but conventionally the
attention-patch goes first.

Output staging suffix per arm:
  internal/scratch/audio-loop-music-video_latent_iclora_bench_<arm>.json

Suggested RUN_ID per arm (so artifacts in data/runs/ are clearly
labeled): RUN_ID=arm_<arm>. Pass to start_experiment.sh:
  RUN_ID=arm_ours ./start_experiment.sh

Compare across arms via scripts/bench_compare_runs.py once you've
rendered each.

Usage:
    uv run --group dev python scripts/apply_iclora_bench_sage_arm.py --arm ours
    uv run --group dev python scripts/apply_iclora_bench_sage_arm.py --arm off
    uv run --group dev python scripts/apply_iclora_bench_sage_arm.py --arm kj
    uv run --group dev python scripts/apply_iclora_bench_sage_arm.py --arm stacked

    uv run --group dev python scripts/apply_iclora_bench_sage_arm.py --arm <name> --revert
    uv run --group dev python scripts/apply_iclora_bench_sage_arm.py --arm <name> --dry-run

Idempotent on each arm's output path. `--revert` deletes that arm's staged file.

Pre-flight: requires the bench-variant workflow to exist (run
apply_iclora_bench_profiling.py first).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = REPO_ROOT / "internal" / "scratch" / "audio-loop-music-video_latent_iclora_bench.json"

# Anchor ids in the bench workflow (inherited from iclora wiring).
SAGE_NODE_ID = 268                      # AudioLoopHelperSageAttention
LTX_PREVIEW_OVERRIDE_ID = 503           # LTX2SamplingPreviewOverride (MODEL source)
ICLORA_LOADER_ID = 1635                 # LTXICLoRALoaderModelOnly (current consumer of #503)

_Arm = Literal["ours", "off", "kj", "stacked"]


def _output_path_for_arm(arm: _Arm, base_dir: Path) -> Path:
    return base_dir / "internal" / "scratch" / f"audio-loop-music-video_latent_iclora_bench_{arm}.json"


def _add_kj_patch(ed: WorkflowEditor) -> int:
    """Insert LTX2MemoryEfficientSageAttentionPatch between #503 and #1635."""
    nid = ed.add_top_level_node(
        node_type="LTX2MemoryEfficientSageAttentionPatch",
        pos=[-2300, 1000],
        size=[300, 90],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "triton_kernels", "type": "BOOLEAN",
             "widget": {"name": "triton_kernels"}, "link": None},
        ],
        outputs=[
            {"name": "model", "type": "MODEL", "links": []},
        ],
        widgets_values=[True],  # triton_kernels=True for fused RoPE
        properties={
            "Node name for S&R": "LTX2MemoryEfficientSageAttentionPatch",
            "cnr_id": "kjnodes",
        },
        title="KJ Sage Patch (LTX-2 per-block, fused RoPE)",
    )
    # Splice: #503 -> #1635 becomes #503 -> #nid -> #1635
    existing = ed.find_link_to_slot(ICLORA_LOADER_ID, 0)
    if existing is None:
        raise SystemExit(
            f"LTXICLoRALoaderModelOnly({ICLORA_LOADER_ID}).model has no inbound link"
        )
    if existing[1] != LTX_PREVIEW_OVERRIDE_ID:
        raise SystemExit(
            f"Expected #503 -> #{ICLORA_LOADER_ID}; got src={existing[1]}"
        )
    ed.remove_link(existing[0])
    ed.add_link(LTX_PREVIEW_OVERRIDE_ID, 0, nid, 0, "MODEL")
    ed.add_link(nid, 0, ICLORA_LOADER_ID, 0, "MODEL")
    return nid


def _bypass_our_sage_node(ed: WorkflowEditor) -> None:
    node = ed.find_node(SAGE_NODE_ID)
    node["mode"] = 4  # bypass


def _apply_arm(input_path: Path, output_path: Path, arm: _Arm, dry_run: bool) -> None:
    if input_path != output_path and output_path.exists():
        # Already-applied check: arm-specific node presence
        ed = WorkflowEditor(output_path)
        sage_node = ed.find_node(SAGE_NODE_ID)
        sage_active = sage_node.get("mode", 0) != 4
        kj_present = bool(ed.find_nodes_by_type("LTX2MemoryEfficientSageAttentionPatch"))
        expected = {
            "ours":    (True,  False),
            "off":     (False, False),
            "kj":      (False, True),
            "stacked": (True,  True),
        }[arm]
        if (sage_active, kj_present) == expected:
            print(f"{output_path.name}: already at arm '{arm}', skipping. Run --revert to reset.")
            return

    if not dry_run and input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed_target = output_path if (output_path.exists() and not dry_run) else input_path
    ed = WorkflowEditor(ed_target)

    # Pre-flight: bench-variant must exist (Profile nodes wired)
    if not ed.find_nodes_by_type("ProfileBegin_AudioLoop"):
        raise SystemExit(
            "Refusing to migrate: input lacks ProfileBegin_AudioLoop. "
            "Run scripts/apply_iclora_bench_profiling.py first."
        )
    if not ed.has_node(SAGE_NODE_ID):
        raise SystemExit(f"Refusing to migrate: missing AudioLoopHelperSageAttention(#{SAGE_NODE_ID})")
    if not ed.has_node(ICLORA_LOADER_ID):
        raise SystemExit(f"Refusing to migrate: missing LTXICLoRALoaderModelOnly(#{ICLORA_LOADER_ID})")

    if dry_run:
        print(f"would set arm={arm} on {output_path.name}")
        return

    # Reset the workflow to a known state: ensure our sage is active, no KJ patch.
    ed.find_node(SAGE_NODE_ID)["mode"] = 0
    for kj in ed.find_nodes_by_type("LTX2MemoryEfficientSageAttentionPatch"):
        ed.remove_node_and_links(kj["id"])
    # Restore the direct #503 -> #1635 link if KJ removal left a gap
    if ed.find_link_to_slot(ICLORA_LOADER_ID, 0) is None:
        ed.add_link(LTX_PREVIEW_OVERRIDE_ID, 0, ICLORA_LOADER_ID, 0, "MODEL")

    # Apply the chosen arm
    if arm == "ours":
        pass  # already in canonical state above
    elif arm == "off":
        _bypass_our_sage_node(ed)
    elif arm == "kj":
        _bypass_our_sage_node(ed)
        kj_id = _add_kj_patch(ed)
        print(f"  added LTX2MemoryEfficientSageAttentionPatch as node {kj_id}")
    elif arm == "stacked":
        kj_id = _add_kj_patch(ed)
        print(f"  added LTX2MemoryEfficientSageAttentionPatch as node {kj_id}")

    ed.save(output_path)
    print(f"  wrote {output_path} (arm={arm})")
    print()
    print(f"Next steps for arm '{arm}':")
    print(f"  1. RUN_ID=arm_{arm} ./start_experiment.sh")
    print(f"  2. Load {output_path.name} in ComfyUI; queue prompt with same inputs as other arms")
    print(f"  3. After completion, artifacts at data/runs/arm_{arm}/{{exec,sage}}.jsonl + profiler/")
    print()
    print( "Once you have 2+ arms rendered, compare via:")
    print( "  uv run --group dev python scripts/bench_compare_runs.py \\")
    print( "    --runs arm_ours arm_off arm_kj arm_stacked")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--arm", required=True, choices=("ours", "off", "kj", "stacked"),
                    help="Which sage configuration to produce.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="Source bench-variant workflow (default: the iclora_bench staged file).")
    ap.add_argument("--out", default=None,
                    help="Output path (default: per-arm path under internal/scratch/).")
    ap.add_argument("--revert", action="store_true",
                    help="Delete this arm's staged file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()

    output_path = Path(args.out) if args.out else _output_path_for_arm(args.arm, REPO_ROOT)
    if args.revert:
        _revert(output_path)
        return 0

    _apply_arm(Path(args.input), output_path, args.arm, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
