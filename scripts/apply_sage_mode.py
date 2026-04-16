"""Flip the SageAttention mode on PathchSageAttentionKJ (node 268) across
all example workflows in one command.

Usage:
    uv run python scripts/apply_sage_mode.py <mode>

Modes (shorthand accepted in parens):
    fp16_cuda    (fp16c)   INT8 QK + FP16 PV, fp32 accum. Requires _qattn_sm80
                           extension. Original conservative choice for audio-video
                           cross-attention quality. Use after ~/dev/sage-fork rebuild.
    fp16_triton  (triton)  INT8 QK + FP16 PV, fp32 accum. JIT Triton, no SM80
                           extension required. ~10-15% slower than fp16_cuda.
    fp8_cuda     (fp8)     INT8 QK + FP8 PV, fp32+fp32 accum. Native Ada fp8
                           tensor cores. Fastest on 4090. Slight precision step
                           down from fp16 PV.
    fp8_cuda++   (fp8pp)   FP8 PV with fp32+fp16 accum. Fastest non-Blackwell.
                           Risk for audio cross-attention.
    auto                   Let sageattention pick per hardware.
    disabled               Bypass sage entirely (falls back to PyTorch SDPA).

Also toggles the node's `mode` field (0 = active, 4 = bypassed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"
SAGE_NODE_ID = 268

MODE_ALIASES = {
    "fp16_cuda":    "sageattn_qk_int8_pv_fp16_cuda",
    "fp16c":        "sageattn_qk_int8_pv_fp16_cuda",
    "fp16_triton":  "sageattn_qk_int8_pv_fp16_triton",
    "triton":       "sageattn_qk_int8_pv_fp16_triton",
    "fp8_cuda":     "sageattn_qk_int8_pv_fp8_cuda",
    "fp8":          "sageattn_qk_int8_pv_fp8_cuda",
    "fp8_cuda++":   "sageattn_qk_int8_pv_fp8_cuda++",
    "fp8pp":        "sageattn_qk_int8_pv_fp8_cuda++",
    "auto":         "auto",
    "disabled":     "disabled",
}


def apply_mode(mode_value: str) -> None:
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        wf = orjson.loads(wf_path.read_bytes())
        changed = False
        for n in wf["nodes"]:
            if n.get("id") == SAGE_NODE_ID and n.get("type") == "PathchSageAttentionKJ":
                # mode=4 only when explicitly disabling the sage patch itself.
                target_node_mode = 4 if mode_value == "disabled" else 0
                target_widgets = [mode_value, False]  # [sage_mode, allow_compile]
                if n.get("mode") != target_node_mode:
                    n["mode"] = target_node_mode
                    changed = True
                if n.get("widgets_values") != target_widgets:
                    n["widgets_values"] = target_widgets
                    changed = True
        if changed:
            wf_path.write_bytes(orjson.dumps(wf, option=orjson.OPT_INDENT_2))
            print(f"  updated {wf_path.name}")
        else:
            print(f"  no change {wf_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=sorted(MODE_ALIASES.keys()),
        help="Sage mode to apply (see module docstring for details).",
    )
    args = parser.parse_args()

    resolved = MODE_ALIASES[args.mode]
    print(f"Setting PathchSageAttentionKJ (node {SAGE_NODE_ID}) to: {resolved}")
    if resolved == "disabled":
        print("  -> node will be bypassed (mode=4)")
    apply_mode(resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
