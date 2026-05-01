"""Tests for scripts/apply_audio_latent_pre_encode.py.

Verifies the topology change: top-level full-song encode + SetNode +
GetNode + new subgraph LATENT input slot + AudioLatentSlice in the loop
body + bypass of the per-iter encode chain.

These tests do NOT touch GPU or torch.compile — purely structural
assertions on the produced workflow JSON.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import orjson
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "apply_audio_latent_pre_encode.py"
INPUT_DEFAULT = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent_iclora.json"

# Anchor IDs the script touches (mirror the script's constants).
SG_AUDIO_VAE_ENCODE_ID = 598
SG_TRIM_AUDIO_ID = 600
SG_AUDIO_VIDEO_MASK_ID = 606
SETNODE_NAME = "full_audio_latent"


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"script failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.fixture
def staged_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Return (input_path, output_path). Input is a copy of the iclora
    workflow (the script's default input)."""
    input_path = tmp_path / "iclora.json"
    output_path = tmp_path / "scratch" / "staged.json"
    shutil.copy2(INPUT_DEFAULT, input_path)
    return input_path, output_path


def _apply(input_path: Path, output_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run_script(
        "--input", str(input_path),
        "--output", str(output_path),
        *extra,
    )


# ---------- plumbing ----------

class TestPlumbing:
    def test_dry_run_does_not_write(self, staged_paths):
        input_path, output_path = staged_paths
        result = _apply(input_path, output_path, "--dry-run")
        _assert_ok(result)
        assert not output_path.exists()

    def test_apply_creates_staging_file(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        assert output_path.exists()
        wf = orjson.loads(output_path.read_bytes())
        assert "nodes" in wf

    def test_apply_is_idempotent(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        first = output_path.read_bytes()
        _assert_ok(_apply(input_path, output_path))
        assert output_path.read_bytes() == first

    def test_revert_deletes_staging_file(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        assert output_path.exists()
        _assert_ok(_run_script("--output", str(output_path), "--revert"))
        assert not output_path.exists()


# ---------- top-level wiring ----------

class TestTopLevelWiring:
    def test_full_song_encode_node_added(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        wf = orjson.loads(output_path.read_bytes())
        # Should have AT LEAST 2 LTXVAudioVAEEncode nodes (existing #566 + new full-song)
        encodes = [n for n in wf["nodes"] if n.get("type") == "LTXVAudioVAEEncode"]
        assert len(encodes) >= 2, f"expected ≥2 LTXVAudioVAEEncode (init + full-song); got {len(encodes)}"

    def test_setnode_full_audio_latent_added(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        wf = orjson.loads(output_path.read_bytes())
        setnodes = [
            n for n in wf["nodes"]
            if n.get("type") == "SetNode"
            and (n.get("widgets_values") or [None])[0] == SETNODE_NAME
        ]
        assert len(setnodes) == 1, f"expected exactly 1 SetNode '{SETNODE_NAME}'; got {len(setnodes)}"

    def test_getnode_full_audio_latent_added(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        wf = orjson.loads(output_path.read_bytes())
        getnodes = [
            n for n in wf["nodes"]
            if n.get("type") == "GetNode"
            and (n.get("widgets_values") or [None])[0] == SETNODE_NAME
        ]
        assert len(getnodes) == 1, f"expected exactly 1 GetNode '{SETNODE_NAME}'; got {len(getnodes)}"


# ---------- subgraph wiring ----------

class TestSubgraphWiring:
    def test_subgraph_has_full_audio_latent_input(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        wf = orjson.loads(output_path.read_bytes())
        sg = wf["definitions"]["subgraphs"][0]
        full_audio_inputs = [i for i in sg.get("inputs", []) if i.get("name") == SETNODE_NAME]
        assert len(full_audio_inputs) == 1
        assert full_audio_inputs[0].get("type") == "LATENT"

    def test_subgraph_has_audio_latent_slice(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        wf = orjson.loads(output_path.read_bytes())
        sg = wf["definitions"]["subgraphs"][0]
        slicers = [n for n in sg["nodes"] if n.get("type") == "AudioLatentSlice"]
        assert len(slicers) == 1
        # Default widgets: [source_seconds, start_seconds, duration_seconds]
        assert slicers[0].get("widgets_values") == [300.0, 0.0, 17.92]

    def test_per_iter_encode_chain_bypassed(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        wf = orjson.loads(output_path.read_bytes())
        sg = wf["definitions"]["subgraphs"][0]
        for nid in (SG_AUDIO_VAE_ENCODE_ID, SG_TRIM_AUDIO_ID):
            node = next((n for n in sg["nodes"] if n.get("id") == nid), None)
            assert node is not None, f"#{nid} should still exist (bypassed, not deleted)"
            assert node.get("mode") == 4, (
                f"#{nid} should be bypassed (mode=4); got mode={node.get('mode')}"
            )

    def test_audio_video_mask_audio_latent_input_rewired(self, staged_paths):
        """LTXVAudioVideoMask.audio_latent (slot 1) should now be wired
        from AudioLatentSlice, not from #598 LTXVAudioVAEEncode."""
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path))
        wf = orjson.loads(output_path.read_bytes())
        sg = wf["definitions"]["subgraphs"][0]
        mask_node = next(n for n in sg["nodes"] if n.get("id") == SG_AUDIO_VIDEO_MASK_ID)
        audio_latent_input = next(i for i in mask_node["inputs"] if i.get("name") == "audio_latent")
        link_id = audio_latent_input.get("link")
        assert link_id is not None, "audio_latent input must be wired"
        link = next(l for l in sg["links"] if l.get("id") == link_id)
        slicer = next(n for n in sg["nodes"] if n.get("type") == "AudioLatentSlice")
        assert link.get("origin_id") == slicer["id"], (
            f"audio_latent should be wired from AudioLatentSlice (#{slicer['id']}); "
            f"got origin_id={link.get('origin_id')}"
        )


# ---------- CLI flags ----------

class TestCLIFlags:
    def test_source_seconds_flag_bakes_into_slicer_widget(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path, "--source-seconds", "180"))
        wf = orjson.loads(output_path.read_bytes())
        sg = wf["definitions"]["subgraphs"][0]
        slicer = next(n for n in sg["nodes"] if n.get("type") == "AudioLatentSlice")
        assert slicer["widgets_values"][0] == 180.0

    def test_window_seconds_flag_bakes_into_slicer_widget(self, staged_paths):
        input_path, output_path = staged_paths
        _assert_ok(_apply(input_path, output_path, "--window-seconds", "10.0"))
        wf = orjson.loads(output_path.read_bytes())
        sg = wf["definitions"]["subgraphs"][0]
        slicer = next(n for n in sg["nodes"] if n.get("type") == "AudioLatentSlice")
        assert slicer["widgets_values"][2] == 10.0
