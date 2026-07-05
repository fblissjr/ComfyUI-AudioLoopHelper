"""Tests for scripts/apply_spatial_inpaint.py (experimental spatial-inpaint retake).

Covers:
  - --dry-run writes no file
  - apply creates the experimental file, is idempotent on re-apply (byte-stable)
  - --revert deletes it
  - post-apply topology: the IC-LoRA loader is spliced between UNETLoader and
    the sage/patch chain (before the module-mutating nodes); the IC-LoRA guide
    feeds CFGGuider + CropGuides + the sampler's latent_image; the Laplacian
    blend sits between the decode and the image trim
  - the temporal-mask path (LatentTemporalMask, source VAEEncode) is gone

The output is a render-unvalidated experimental variant; these tests guard the
build's structural invariants against drift in the retake workflow it forks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import orjson
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "apply_spatial_inpaint.py"
OUTPUT = (REPO_ROOT / "example_workflows" / "experimental"
          / "audio-loop-music-video_spatial_inpaint.json")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def _types(wf: dict) -> set[str]:
    return {n["type"] for n in wf["nodes"]}


def _feeder(wf: dict, node_id: int, slot_name: str) -> tuple[int, int] | None:
    """(src_id, src_slot) feeding node_id.slot_name, or None if unlinked."""
    by_id = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}
    for inp in by_id[node_id].get("inputs", []):
        if inp["name"] == slot_name:
            lk = inp.get("link")
            if lk is None:
                return None
            _, src, src_slot, *_ = links[lk]
            return (src, src_slot)
    raise AssertionError(f"no slot {slot_name} on #{node_id}")


@pytest.fixture
def built(tmp_path):
    """Build into a temp path so the repo copy is untouched."""
    out = tmp_path / "spatial_inpaint.json"
    r = _run("--output", str(out))
    assert r.returncode == 0, r.stderr
    wf = orjson.loads(out.read_bytes())
    return out, wf


def test_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "dry.json"
    r = _run("--output", str(out), "--dry-run")
    assert r.returncode == 0
    assert not out.exists()


def test_apply_idempotent(tmp_path):
    out = tmp_path / "idem.json"
    _run("--output", str(out))
    first = out.read_bytes()
    _run("--output", str(out))
    assert out.read_bytes() == first


def test_revert_deletes(tmp_path):
    out = tmp_path / "rev.json"
    _run("--output", str(out))
    assert out.exists()
    _run("--output", str(out), "--revert")
    assert not out.exists()


def test_iclora_loader_before_patch_chain(built):
    _, wf = built
    assert "LTXICLoRALoaderModelOnly" in _types(wf)
    loader = next(n["id"] for n in wf["nodes"]
                  if n["type"] == "LTXICLoRALoaderModelOnly")
    # loader.model <- UNETLoader(414); Sage(268).model <- loader
    assert _feeder(wf, loader, "model") == (414, 0)
    assert _feeder(wf, 268, "model") == (loader, 0)


def test_guide_interposed(built):
    _, wf = built
    guide = next(n["id"] for n in wf["nodes"]
                 if n["type"] == "LTXAddVideoICLoRAGuideAdvanced")
    assert _feeder(wf, 153, "positive") == (guide, 0)   # CFGGuider
    assert _feeder(wf, 381, "positive") == (guide, 0)   # CropGuides
    assert _feeder(wf, 161, "latent_image") == (guide, 2)  # Sampler
    # guide fed by inpaint image + empty latent
    img_src = _feeder(wf, guide, "image")
    assert img_src is not None
    inpaint_id = next(n["id"] for n in wf["nodes"]
                      if n["type"] == "LTXVInpaintPreprocess")
    assert img_src[0] == inpaint_id


def test_laplacian_blend_before_trim(built):
    _, wf = built
    blend = next(n["id"] for n in wf["nodes"]
                 if n["type"] == "LTXVLaplacianPyramidBlend")
    # blend.image_a <- decode(1604); TrimImageBatchToAudio(1628).images <- blend
    assert _feeder(wf, blend, "image_a") == (1604, 0)
    assert _feeder(wf, 1628, "images") == (blend, 0)


def test_temporal_mask_path_removed(built):
    _, wf = built
    assert "LatentTemporalMask" not in _types(wf)
    # the source VAEEncode (encoded-source base) is gone; base is empty latent
    assert "EmptyLTXVLatentVideo" in _types(wf)
