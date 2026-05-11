"""Behavioral tests for scripts/promote_latent_for_upscale.py.

Last updated: 2026-05-10

Pins the file-discovery contract: given a workflow name and an output
dir, return the most recent ``segment_*.latent`` file written by the
loop's bypassed-SaveLatent toggle (added by
``scripts/apply_run_id_layout.py``).

Output layout produced by RunIdPrefix (``<workflow_name>/<timestamp>/latents/segment_NNNNN_.latent``):

    <output>/audio-loop-music-video_latent/
      20260510_120000/latents/segment_00001_.latent       ← older
      20260510_143022/latents/segment_00001_.latent       ← newer
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from promote_latent_for_upscale import find_latest_assembled_latent


def _make_layout(root: Path, workflow_name: str, timestamps: list[str]) -> dict[str, Path]:
    """Create the per-render output tree for a workflow + return the
    map of {timestamp -> .latent path} for assertions."""
    paths: dict[str, Path] = {}
    for ts in timestamps:
        latent_dir = root / workflow_name / ts / "latents"
        latent_dir.mkdir(parents=True)
        p = latent_dir / "segment_00001_.latent"
        p.write_bytes(b"")
        paths[ts] = p
    return paths


def test_returns_most_recently_modified_latent(tmp_path: Path):
    paths = _make_layout(
        tmp_path, "audio-loop-music-video_latent",
        ["20260510_120000", "20260510_143022", "20260510_100000"],
    )
    # Pin mtimes so the test doesn't depend on filesystem timestamp resolution
    base = time.time()
    Path(paths["20260510_100000"]).touch(); _set_mtime(paths["20260510_100000"], base - 100)
    Path(paths["20260510_120000"]).touch(); _set_mtime(paths["20260510_120000"], base - 50)
    Path(paths["20260510_143022"]).touch(); _set_mtime(paths["20260510_143022"], base)

    found = find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent")
    assert found == paths["20260510_143022"]


def test_workflow_with_no_renders_raises(tmp_path: Path):
    (tmp_path / "audio-loop-music-video_latent").mkdir()
    with pytest.raises(FileNotFoundError, match=r"no segment_\*\.latent files"):
        find_latest_assembled_latent(tmp_path, "audio-loop-music-video_latent")


def test_unknown_workflow_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no output folder"):
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
