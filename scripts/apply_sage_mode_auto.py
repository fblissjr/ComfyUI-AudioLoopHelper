"""apply_sage_mode_auto.

Last updated: 2026-05-15

Symptom it fixes: AudioLoopHelperSageAttention sage mode drift across shipped
workflows. Audio-loop workflows historically defaulted to `auto_mask_aware`;
benchmark workflows defaulted to `auto`. Two defaults across the repo means
new variants inherit whichever they were forked from, and the split has no
runtime payoff on audio-loop workflows (the masked self-attn path is
effectively inert there per root CLAUDE.md's Pending-review note — something
in `LTXVCropGuides` / `LTXVConcatAVLatent`'s NestedTensor packing strips
`guide_attention_entries` before `_process_input` builds the mask, so
`auto_mask_aware` and `auto` produce equivalent runtime behavior).

Root cause: historical authoring choice — the `_mask_aware` suffix was
defensive for cross-attn that turned out not to fire on these workflows.
Benchmark workflows (FML2V multi-guide topology) DO exercise the masked
self-attn path and need `auto` so sage-fork's fp8++ mask CUDA kernel
dispatches correctly; standardizing every workflow on `auto` is the
union-safe default.

Fix: set `widgets_values[0]` to `"auto"` on every `AudioLoopHelperSageAttention`
node across `example_workflows/` (and `experimental/`).

Compatibility:
  - Does not touch `PathchSageAttentionKJ` nodes (KJNodes upstream) — those
    remain bypassed (mode=4) wherever they appear in benchmark workflows.
  - Does not touch `widgets_values[1]` (fallback_on_error) or `widgets_values[2]`
    (skip_under_seq_len). Some experimental workflows have only 2 widget values
    (missing the skip_under_seq_len third widget) — preserved as-is.
  - Companion: `apply_skip_under_seq_len.py` (sage perf knob, separate concern).

Usage:
    uv run --group dev python scripts/apply_sage_mode_auto.py
    uv run --group dev python scripts/apply_sage_mode_auto.py --revert
    uv run --group dev python scripts/apply_sage_mode_auto.py --dry-run

Idempotent. Already-`auto` nodes report "no change".
`--revert` restores `auto_mask_aware` on audio-loop workflows (the historical
default) — benchmark workflows that were always `auto` are left unchanged on
revert, matching their pre-script state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

SAGE_NODE_TYPE = "AudioLoopHelperSageAttention"
TARGET_MODE = "auto"
LEGACY_MODE = "auto_mask_aware"

# Benchmark workflows pre-date the audio-loop default and ship with `auto`
# already — `--revert` should NOT switch them to `auto_mask_aware` because
# that was never their state. Detect by parent directory.
BENCHMARK_SUBDIR = "benchmark_workflows"


def _is_benchmark(wf_path: Path) -> bool:
    return BENCHMARK_SUBDIR in wf_path.parts


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    sage_nodes = ed.find_nodes_by_type(SAGE_NODE_TYPE)
    if not sage_nodes:
        return "skip (no AudioLoopHelperSageAttention node)"

    target = TARGET_MODE
    if revert:
        # Benchmark workflows were always `auto` — preserve.
        target = TARGET_MODE if _is_benchmark(wf_path) else LEGACY_MODE

    changes: list[tuple[int, str, str]] = []
    for n in sage_nodes:
        wv = n.get("widgets_values") or []
        if not wv:
            return f"skip (node {n['id']} has empty widgets_values)"
        current = wv[0]
        if current == target:
            continue
        if not revert and current not in (LEGACY_MODE, TARGET_MODE):
            return f"skip (node {n['id']} has unexpected mode {current!r})"
        changes.append((n["id"], current, target))

    if not changes:
        return f"no change (already {target})"

    if dry_run:
        details = ", ".join(f"#{nid} {old}->{new}" for nid, old, new in changes)
        verb = "would revert" if revert else "would update"
        return f"{verb} ({details})"

    for nid, _, new in changes:
        n = ed.find_node(nid)
        n["widgets_values"][0] = new
    ed.save(wf_path)
    details = ", ".join(f"#{nid} {old}->{new}" for nid, old, new in changes)
    verb = "reverted" if revert else "updated"
    return f"{verb} ({details})"


def apply(revert: bool, dry_run: bool) -> int:
    action = (
        f"Would {'revert' if revert else 'apply'}"
        if dry_run
        else ("Reverting" if revert else "Applying")
    )
    print(f"{action} sage_mode_auto across example_workflows/...")
    fail = 0
    for wf_path in sorted(WORKFLOWS_DIR.rglob("*.json")):
        status = _apply_one(wf_path, revert, dry_run)
        rel = wf_path.relative_to(WORKFLOWS_DIR)
        print(f"  {rel}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--revert", action="store_true",
        help="Undo the change applied by this script.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what WOULD change without writing files.",
    )
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
