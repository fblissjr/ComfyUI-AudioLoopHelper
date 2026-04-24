"""Validate that every example workflow uses an LTX-2.3-compliant resolution.

LTX 2.3 spatial compression is 32x per dimension. Per `coderef/LTX-2/
packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:325`:
  - One-stage pipeline: height + width divisible by 32.
  - Two-stage distilled pipeline: height + width divisible by 64.

ComfyUI uses only the transformer (no Python pipeline wrapper), so the
assert_resolution check doesn't run at load time — a bad resolution
silently produces degraded output rather than erroring. This script
catches that.

Usage:
    uv run python scripts/validate_workflow_resolution.py
Exits non-zero on failure so CI can catch regressions.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import EXAMPLE_WORKFLOWS_DIR as EXAMPLE_DIR

DIV_STRICT = 64  # distilled two-stage requirement
DIV_PERMISSIVE = 32  # single-stage requirement


def _check_workflow(path: Path) -> list[str]:
    """Return list of human-readable issues for one workflow file."""
    issues: list[str] = []
    wf = json.loads(path.read_text())
    for node in wf["nodes"]:
        if node.get("type") == "ImageResizeKJv2":
            wv = node.get("widgets_values", [])
            if len(wv) < 2:
                issues.append(f"  node {node['id']} ImageResizeKJv2: missing width/height widgets")
                continue
            w, h = wv[0], wv[1]
            if not isinstance(w, int) or not isinstance(h, int):
                issues.append(f"  node {node['id']} ImageResizeKJv2: non-integer width/height {w!r}x{h!r}")
                continue
            if w % DIV_PERMISSIVE != 0 or h % DIV_PERMISSIVE != 0:
                issues.append(
                    f"  node {node['id']} ImageResizeKJv2: {w}x{h} not divisible by {DIV_PERMISSIVE} "
                    f"(hard requirement for LTX 2.3 single-stage)"
                )
                continue
            if w % DIV_STRICT != 0 or h % DIV_STRICT != 0:
                issues.append(
                    f"  node {node['id']} ImageResizeKJv2: {w}x{h} divisible by {DIV_PERMISSIVE} "
                    f"but NOT by {DIV_STRICT} (off-grid for distilled two-stage); consider nearest "
                    f"{DIV_STRICT}-aligned values"
                )
    return issues


def main() -> int:
    workflows = sorted(EXAMPLE_DIR.glob("*.json"))
    if not workflows:
        print(f"No workflows in {EXAMPLE_DIR}", file=sys.stderr)
        return 2

    any_failed = False
    any_warned = False
    for path in workflows:
        issues = _check_workflow(path)
        if not issues:
            print(f"OK   {path.name}")
            continue
        # Treat the "not divisible by 32" issues as hard failures, rest as warnings.
        hard = [i for i in issues if f"not divisible by {DIV_PERMISSIVE}" in i]
        soft = [i for i in issues if i not in hard]
        if hard:
            any_failed = True
            print(f"FAIL {path.name}")
            for msg in hard:
                print(msg)
        if soft:
            any_warned = True
            if not hard:
                print(f"WARN {path.name}")
            for msg in soft:
                print(msg)

    if any_failed:
        return 1
    if any_warned:
        print("\n(soft warnings only — workflows will run but may produce off-distribution output)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
