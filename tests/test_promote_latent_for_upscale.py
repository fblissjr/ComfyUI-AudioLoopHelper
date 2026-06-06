"""Behavioral tests for scripts/promote_latent_for_upscale.py.

Last updated: 2026-06-06

Pins the file-discovery contract: given a workflow name and an output
dir, return the most recent banked ``.latent`` across BOTH layouts —

  - PreDecodeCleanup checkpoints (current; rotated):
    ``<output>/latents/checkpoints/<workflow_name>_NNNNN_.latent``
  - legacy standalone-SaveLatent per-render folders:
    ``<output>/<workflow_name>/<timestamp>/latents/segment_NNNNN_.latent``

newest mtime wins across the union.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from promote_latent_for_upscale import find_latest_assembled_latent


def _make_layout(root: Path, workflow_name: str, timestamps: list[str]) -> dict[str, Path]:
    """Create the legacy per-render output tree for a workflow + return the
    map of {timestamp -> .latent path} for assertions."""
    paths: dict[str, Path] = {}
    for ts in timestamps:
        latent_dir = root / workflow_name / ts / "latents"
        latent_dir.mkdir(parents=True)
        p = latent_dir / "segment_00001_.latent"
        p.write_bytes(b"")
        paths[ts] = p
    return paths


def _make_checkpoint(root: Path, workflow_name: str, counter: int) -> Path:
    ckpt_dir = root / "latents" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / f"{workflow_name}_{counter:05}_.latent"
    p.write_bytes(b"")
    return p


def test_returns_most_recently_modified_latent(tmp_path: Path):
    paths = _make_layout(
        tmp_path, "audio-loop-music-video_latent",
        ["20260510_120000", "20260510_143022", "20260510_100000"],
    )
    # Pin mtimes so the test doesn't depend on filesystem timestamp resolution
    base = time.time()
    _set_mtime(paths["20260510_100000"], base - 100)
    _set_mtime(paths["20260510_120000"], base - 50)
    _set_mtime(paths["20260510_143022"], base)

    found = find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent")
    assert found == paths["20260510_143022"]


def test_checkpoint_layout_found(tmp_path: Path):
    ckpt = _make_checkpoint(tmp_path, "audio-loop-music-video_latent", 3)
    found = find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent")
    assert found == ckpt


def test_newest_wins_across_layouts(tmp_path: Path):
    """A fresh checkpoint must beat a stale legacy SaveLatent file and
    vice versa — discovery is mtime across the union, not layout priority."""
    legacy = _make_layout(
        tmp_path, "audio-loop-music-video_latent", ["20260510_120000"],
    )["20260510_120000"]
    ckpt = _make_checkpoint(tmp_path, "audio-loop-music-video_latent", 1)
    base = time.time()
    _set_mtime(legacy, base - 100)
    _set_mtime(ckpt, base)
    assert find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent") == ckpt
    _set_mtime(legacy, base + 100)
    assert find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent") == legacy


def test_checkpoints_scoped_per_workflow(tmp_path: Path):
    """Another workflow's checkpoint in the shared folder must not match."""
    _make_checkpoint(tmp_path, "other_workflow", 1)
    with pytest.raises(FileNotFoundError, match="no banked .latent"):
        find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent")


def test_workflow_with_no_renders_raises(tmp_path: Path):
    (tmp_path / "audio-loop-music-video_latent").mkdir()
    with pytest.raises(FileNotFoundError, match="no banked .latent"):
        find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent")


def test_unknown_workflow_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no banked .latent"):
        find_latest_assembled_latent(tmp_path, "missing_workflow")


def test_skips_non_latent_files_in_latents_dir(tmp_path: Path):
    """The latents/ dir might end up with non-segment files (test artifacts,
    user-renamed copies). The finder should match the segment_*.latent
    pattern explicitly."""
    paths = _make_layout(
        tmp_path, "audio-loop-music-video_latent", ["20260510_143022"],
    )
    # Add a same-dir file that should be ignored
    other = paths["20260510_143022"].parent / "notes.txt"
    other.write_text("ignored")
    found = find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent")
    assert found == paths["20260510_143022"]


def _set_mtime(path: Path, when: float) -> None:
    import os
    os.utime(path, (when, when))
