"""Behavioral tests for RunIdPrefix.

Last updated: 2026-05-10

Pins the per-render unique-prefix contract: one render → one timestamp
shared across every save node (VHS_VideoCombine, SaveImage, SaveLatent)
that wires through this node. fingerprint_inputs returns a fresh value
each call so ComfyUI re-executes the node on every queue submission
(otherwise the same cached timestamp would carry across renders).
"""

from __future__ import annotations

import math
import re

from nodes import RunIdPrefix


def test_video_prefix_shape():
    """`<workflow_name>/<timestamp>` with timestamp matching the format."""
    out = RunIdPrefix.execute(
        workflow_name="audio-loop-music-video_latent",
        timestamp_format="%Y%m%d_%H%M%S",
    )
    video_prefix = out[0]
    assert video_prefix.startswith("audio-loop-music-video_latent/")
    ts = video_prefix.split("/", 1)[1]
    assert re.fullmatch(r"\d{8}_\d{6}", ts), f"timestamp shape wrong: {ts!r}"


def test_latent_prefix_shape():
    """`<workflow_name>/<timestamp>/latents/segment` — clusters under the
    same per-render folder as video_prefix."""
    out = RunIdPrefix.execute(
        workflow_name="audio-loop-music-video_latent",
        timestamp_format="%Y%m%d_%H%M%S",
    )
    video_prefix, latent_prefix = out[0], out[1]
    assert latent_prefix == f"{video_prefix}/latents/segment", (
        f"latent_prefix must nest under video_prefix; got {latent_prefix!r}"
    )


def test_custom_timestamp_format_honored():
    out = RunIdPrefix.execute(
        workflow_name="x",
        timestamp_format="%Y-%m-%d",
    )
    video_prefix = out[0]
    ts = video_prefix.split("/", 1)[1]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts)


def test_workflow_name_with_slash_preserved():
    """Caller may want a deeper sub-folder; don't sanitize slashes."""
    out = RunIdPrefix.execute(
        workflow_name="renders/2026/may",
        timestamp_format="%Y%m%d",
    )
    assert out[0].startswith("renders/2026/may/")


def test_fingerprint_forces_re_evaluation_each_run():
    """fingerprint_inputs returns NaN so ComfyUI sees a fresh fingerprint
    every queue submission and re-executes the node. Otherwise the same
    cached timestamp would propagate across all renders, defeating
    runid uniqueness."""
    fp = RunIdPrefix.fingerprint_inputs(
        workflow_name="x", timestamp_format="%Y%m%d_%H%M%S",
    )
    assert isinstance(fp, float) and math.isnan(fp), (
        "fingerprint_inputs must return NaN (NaN != NaN forces re-eval)"
    )


def test_node_is_registered_in_extension():
    from _node_registry import assert_node_registered
    assert_node_registered("RunIdPrefix")
