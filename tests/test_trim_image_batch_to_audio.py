"""Behavioral tests for TrimImageBatchToAudio.

Last updated: 2026-05-10

Pins the contract that the node clips an IMAGE batch down to
`floor(audio_duration * fps)` frames so the saved mp4 never has
more video than audio. Postmortem evidence + per-render ffprobe
data are in `internal/analysis/loop_audio_overshoot_analysis.md`.

The audio path is unchanged: `VHS_VideoCombine` continues to receive
the raw `orig_audio` waveform. This node only trims the IMAGE batch
that feeds VHS_VideoCombine.images, eliminating the few-seconds
silence-at-end the saved mp4 currently exhibits.
"""

from __future__ import annotations

import torch

from nodes import TrimImageBatchToAudio


def _make_audio(seconds: float, sample_rate: int = 44100) -> dict:
    samples = int(round(seconds * sample_rate))
    waveform = torch.zeros(1, 1, samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _make_images(n_frames: int, h: int = 4, w: int = 4) -> torch.Tensor:
    return torch.zeros(n_frames, h, w, 3)


def test_trims_when_video_longer_than_audio():
    """The canonical bug: 4277 video frames + 166.733s audio @ 25fps
    should clip to 4168 frames."""
    images = _make_images(4277)
    audio = _make_audio(166.733)
    out = TrimImageBatchToAudio.execute(images=images, audio=audio, fps=25)
    trimmed = out[0]
    assert trimmed.shape[0] == 4168, (
        f"expected floor(166.733 * 25) = 4168 frames, got {trimmed.shape[0]}"
    )


def test_passthrough_when_video_shorter_than_audio():
    """If the video batch is already shorter than audio, leave it alone."""
    images = _make_images(50)
    audio = _make_audio(10.0)  # 250 frames @ 25fps
    out = TrimImageBatchToAudio.execute(images=images, audio=audio, fps=25)
    assert out[0].shape[0] == 50


def test_passthrough_when_video_exactly_matches_audio():
    """No-op when shapes already align."""
    images = _make_images(250)
    audio = _make_audio(10.0)
    out = TrimImageBatchToAudio.execute(images=images, audio=audio, fps=25)
    assert out[0].shape[0] == 250


def test_zero_duration_audio_keeps_at_least_one_frame():
    """Defensive: degenerate-input branch shouldn't return an empty tensor
    (which would crash VHS_VideoCombine downstream)."""
    images = _make_images(100)
    audio = _make_audio(0.0)
    out = TrimImageBatchToAudio.execute(images=images, audio=audio, fps=25)
    assert out[0].shape[0] >= 1


def test_floor_semantics_not_round():
    """audio = 4.99s @ 25fps → 124 frames (floor), not 125 (round)."""
    images = _make_images(200)
    audio = _make_audio(4.99)
    out = TrimImageBatchToAudio.execute(images=images, audio=audio, fps=25)
    assert out[0].shape[0] == 124


def test_preserves_image_dtype_and_shape_other_dims():
    """Trim only along the frame dimension; H/W/C/dtype unchanged."""
    images = torch.rand(500, 480, 832, 3, dtype=torch.float16)
    audio = _make_audio(5.0)  # 125 frames @ 25fps
    out = TrimImageBatchToAudio.execute(images=images, audio=audio, fps=25)
    trimmed = out[0]
    assert trimmed.shape == (125, 480, 832, 3)
    assert trimmed.dtype == torch.float16


def test_three_observed_render_cases_from_ffprobe_data():
    """Replays the exact bug from the user's recent renders. Each case
    confirms the trim brings video frames to floor(audio * fps)."""
    cases = [
        # (audio_seconds, observed_video_frames, expected_after_trim)
        (166.733, 4277, 4168),
        (94.602, 2485, 2365),
        (54.036, 1589, 1350),
    ]
    for audio_s, vframes, expected in cases:
        images = _make_images(vframes)
        audio = _make_audio(audio_s)
        out = TrimImageBatchToAudio.execute(images=images, audio=audio, fps=25)
        assert out[0].shape[0] == expected, (
            f"audio={audio_s}s vframes={vframes} → expected {expected}, "
            f"got {out[0].shape[0]}"
        )


def test_node_is_registered_in_extension():
    """Smoke test: the node is in the ComfyExtension's node list so
    ComfyUI can discover it. AST scan because `get_node_list` uses
    relative imports that fail when nodes.py is loaded outside a
    package context (pytest default)."""
    import ast
    import pathlib

    src = pathlib.Path("nodes.py").read_text()
    tree = ast.parse(src)
    found_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AudioLoopHelperExtension":
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    found_names.add(child.id)
    assert "TrimImageBatchToAudio" in found_names, (
        "TrimImageBatchToAudio not referenced inside AudioLoopHelperExtension. "
        "Add it to the get_node_list() return value."
    )
