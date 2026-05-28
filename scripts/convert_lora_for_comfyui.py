#!/usr/bin/env python3
"""Re-key a trained ltx-trainer LoRA for ComfyUI-LTXVideo's expected naming.

Our trained LoRA has a MIXED key layout — an artifact of training-time
block-swap. The 12 kept-on-GPU blocks (0-11) serialize as direct submodule
paths; the 36 swapped blocks (12-47) serialize WRAPPED with our
`StreamingBlockWrapper.block` attribute injected into the path:

    blocks 0-11:   diffusion_model.transformer_blocks.<N>.<attn>.<attr>.lora_A/B.weight       (direct)
    blocks 12-47:  diffusion_model.transformer_blocks.<N>.block.<attn>.<attr>.lora_A/B.weight (wrapped)

ComfyUI-LTXVideo's model has NO `.block.` wrapper at inference time — all
blocks are direct submodules. So the 1728 wrapped keys (75% of our LoRA)
fail to attach, the LoRA loads "partially" with silently bad coverage, and
the render is effectively baseline. Caught when block 12-47 warnings
flooded the log on first attempt.

This script STRIPS the spurious `.block.` segment so all 2304 keys match
the unwrapped ComfyUI model paths. Idempotent: keys without `.block.` are
left alone.

Long-term fix lives in the trainer's checkpoint save path: should unwrap
StreamingBlockWrapper before serializing LoRA state-dict. Tracked
separately; this script is the in-the-meantime adapter.

Usage:
    uv run --group dev python scripts/convert_lora_for_comfyui.py \\
        <input.safetensors> <output.safetensors>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


# Match: diffusion_model.transformer_blocks.<N>.block.<rest>
# and STRIP the `.block` segment. Non-matching keys (no .block. or
# non-transformer paths like embeddings_connector) → unchanged.
_BLOCK_KEY_RE = re.compile(
    r"^(diffusion_model\.transformer_blocks\.\d+)\.block\."
)


def convert_key(key: str) -> str:
    """Strip the spurious `.block.` from transformer-block paths
    (artifact of training-time StreamingBlockWrapper). Idempotent: keys
    already without `.block.` are returned unchanged."""
    return _BLOCK_KEY_RE.sub(r"\1.", key)


def convert_state_dict(state_dict: dict) -> dict:
    return {convert_key(k): v for k, v in state_dict.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Trained LoRA safetensors from the fork")
    ap.add_argument("output", type=Path, help="Where to write the re-keyed LoRA")
    args = ap.parse_args()

    sd = load_file(str(args.input))
    n_total = len(sd)
    new_sd = convert_state_dict(sd)
    n_renamed = sum(1 for old, new in zip(sd.keys(), new_sd.keys()) if old != new)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_sd, str(args.output))

    print(f"keys total: {n_total}")
    print(f"keys renamed: {n_renamed} ({100 * n_renamed / max(n_total, 1):.1f}%)")
    print(f"wrote: {args.output}")
    if n_renamed == 0:
        print("WARN: no keys renamed — either already converted, or pattern didn't match.", file=sys.stderr)


if __name__ == "__main__":
    main()
