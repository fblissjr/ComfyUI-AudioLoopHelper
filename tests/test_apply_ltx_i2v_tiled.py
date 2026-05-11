"""Smoke tests for the LTX 2.3 I2V tiled-sampler apply scripts.

Last updated: 2026-05-11

Covers:
  - `scripts/apply_ltx_i2v_tiled_optimizations.py` (Arm 0 baseline)
  - `scripts/apply_ltx_i2v_tiled_ab_variants.py` (Arms 1-5 + no_rtx)

Structural tests (always run): each script imports cleanly, declares
the expected constants, and the variant script's dispatch covers
all advertised arms.

Functional tests (skip when the gitignored source workflow is
absent): exercise the apply scripts against the user's local
scratch source via subprocess. Validates that each arm produces a
JSON-valid output, that `--dry-run` writes nothing, that re-running
short-circuits via `_already_migrated`, and that `--revert` removes
the output.

Coverage gap: this file does NOT cover a freshly-cloned CI workflow.
Both scripts depend on a source workflow that lives under
`internal/scratch/` (gitignored). A future refactor could
parameterize the source path or ship a synthesized minimal fixture
for CI coverage; until then these tests catch local regressions
during apply-script edits but won't fire on CI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZE_SCRIPT = REPO_ROOT / "scripts" / "apply_ltx_i2v_tiled_optimizations.py"
VARIANTS_SCRIPT = REPO_ROOT / "scripts" / "apply_ltx_i2v_tiled_ab_variants.py"

# The end-to-end optimize-script test consumes a source workflow from
# `internal/scratch/` (gitignored). Path is filename-specific so it
# stays out of the tracked test code via env var. Set
# `LTX_I2V_TILED_SCRATCH_SOURCE=<abs path>` to enable the test;
# otherwise it skips.
_SCRATCH_ENV = "LTX_I2V_TILED_SCRATCH_SOURCE"
_scratch_env_value = os.environ.get(_SCRATCH_ENV, "").strip()
SCRATCH_SOURCE = Path(_scratch_env_value) if _scratch_env_value else None

ARM0_BASELINE = REPO_ROOT / "internal" / "workflows" / "ltx_i2v_tiled_optimized.draft.json"

# Match `_OUTPUTS` in the variants script.
EXPECTED_ARMS = ("arm3", "arm4", "arm5", "no_rtx")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


# --------------------------------------------------------------------------
# Structural tests -- always run
# --------------------------------------------------------------------------


def test_optimize_script_help_runs():
    """The optimize script's --help must work without ComfyUI loaded."""
    proc = _run(OPTIMIZE_SCRIPT, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--input" in proc.stdout
    assert "--revert" in proc.stdout
    assert "--dry-run" in proc.stdout


def test_variants_script_help_runs():
    """The variants script's --help must enumerate every dispatched arm."""
    proc = _run(VARIANTS_SCRIPT, "--help")
    assert proc.returncode == 0, proc.stderr
    for arm in EXPECTED_ARMS:
        assert arm in proc.stdout, f"--arm {arm} missing from help text"


def test_variants_script_rejects_unknown_arm(tmp_path: Path):
    """argparse `choices=` must enforce the arm allowlist."""
    proc = _run(VARIANTS_SCRIPT, "--arm", "bogus_arm", "--input", str(tmp_path / "x.json"))
    assert proc.returncode != 0
    assert "bogus_arm" in proc.stderr or "invalid choice" in proc.stderr


def test_variants_script_dispatch_table_matches_outputs():
    """`_DISPATCH` must include every key in `_OUTPUTS` and vice versa.

    Imports the module directly (no subprocess) so this runs even
    when the scratch source is absent.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import apply_ltx_i2v_tiled_ab_variants as mod  # type: ignore
    finally:
        sys.path.pop(0)
    assert set(mod._DISPATCH.keys()) == set(mod._OUTPUTS.keys())
    assert set(mod._DISPATCH.keys()) == set(EXPECTED_ARMS)
    assert set(mod.ARMS) == set(EXPECTED_ARMS)


# --------------------------------------------------------------------------
# Functional tests -- skip when the gitignored source is absent
# --------------------------------------------------------------------------


pytestmark_needs_baseline = pytest.mark.skipif(
    not ARM0_BASELINE.exists(),
    reason=(
        f"Arm 0 baseline {ARM0_BASELINE} absent. Run "
        f"`scripts/apply_ltx_i2v_tiled_optimizations.py --input {SCRATCH_SOURCE}` first."
    ),
)


@pytestmark_needs_baseline
@pytest.mark.parametrize("arm", EXPECTED_ARMS)
def test_each_arm_produces_valid_output(arm: str, tmp_path: Path):
    """Each arm writes a JSON-valid file with idempotence + revert."""
    output = tmp_path / f"variant_{arm}.draft.json"

    proc = _run(VARIANTS_SCRIPT, "--arm", arm, "--input", str(ARM0_BASELINE), "--output", str(output))
    assert proc.returncode == 0, proc.stderr
    assert output.exists(), f"Expected output {output} not produced"
    with output.open() as f:
        json.load(f)

    # Idempotence: second invocation reports already-migrated and leaves file intact.
    mtime_before = output.stat().st_mtime
    proc2 = _run(VARIANTS_SCRIPT, "--arm", arm, "--input", str(ARM0_BASELINE), "--output", str(output))
    assert proc2.returncode == 0, proc2.stderr
    assert "already migrated" in proc2.stdout, proc2.stdout
    assert output.stat().st_mtime == mtime_before

    # Revert removes the file.
    proc3 = _run(VARIANTS_SCRIPT, "--arm", arm, "--input", str(ARM0_BASELINE), "--output", str(output), "--revert")
    assert proc3.returncode == 0, proc3.stderr
    assert not output.exists()


@pytestmark_needs_baseline
def test_dry_run_writes_nothing(tmp_path: Path):
    """`--dry-run` must not produce or modify any output file."""
    output = tmp_path / "variant_arm3.draft.json"
    proc = _run(VARIANTS_SCRIPT, "--arm", "arm3", "--input", str(ARM0_BASELINE), "--output", str(output), "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "would copy" in proc.stdout
    assert "would apply" in proc.stdout
    assert not output.exists()


@pytestmark_needs_baseline
def test_revert_on_missing_output_is_noop(tmp_path: Path):
    """`--revert` against a non-existent output reports nothing-to-do but succeeds."""
    output = tmp_path / "never_created.draft.json"
    proc = _run(VARIANTS_SCRIPT, "--arm", "no_rtx", "--input", str(ARM0_BASELINE), "--output", str(output), "--revert")
    assert proc.returncode == 0, proc.stderr
    assert "does not exist" in proc.stdout or "nothing to revert" in proc.stdout


# --------------------------------------------------------------------------
# Optimize-script smoke test -- skip when scratch source is absent
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    SCRATCH_SOURCE is None or not SCRATCH_SOURCE.exists(),
    reason=f"Set ${_SCRATCH_ENV}=<abs path> to enable the optimize-script smoke test.",
)
def test_optimize_produces_baseline(tmp_path: Path):
    """End-to-end: optimize script consumes the scratch source and produces a JSON-valid baseline."""
    assert SCRATCH_SOURCE is not None  # guarded by skipif
    output = tmp_path / "baseline.draft.json"
    # Stage a writable copy of the source so we don't depend on its current state being modifiable.
    src_copy = tmp_path / "source.json"
    shutil.copy2(SCRATCH_SOURCE, src_copy)

    proc = _run(OPTIMIZE_SCRIPT, "--input", str(src_copy), "--output", str(output))
    assert proc.returncode == 0, proc.stderr
    assert output.exists()
    with output.open() as f:
        wf = json.load(f)

    # Post-apply structural assertions: each Phase-1 transformation visible.
    types = [n.get("type") for n in wf.get("nodes", [])]
    assert "AudioLoopHelperSageAttention" in types, "sage attention missing from optimized output"
    assert "LTXVTiledVAEDecode" in types, "tiled VAE decode missing"
    assert "LTXSmartImageResize" in types, "smart resize missing"
    assert "LTXVPreprocess" in types, "preprocess missing"
    # Phase-2 dead-branch strip: SolidMask + dead audio re-encode gone.
    assert "LTXVAudioVAEEncode" not in types, "dead audio re-encode should be stripped"

    # Idempotence on second apply.
    proc2 = _run(OPTIMIZE_SCRIPT, "--input", str(src_copy), "--output", str(output))
    assert proc2.returncode == 0, proc2.stderr
    assert "already migrated" in proc2.stdout, proc2.stdout
