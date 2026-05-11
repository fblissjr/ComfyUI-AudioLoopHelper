"""Behavioral tests for LatentFrameCount.

Last updated: 2026-05-10

Pins the LTX video VAE conversion: pixel_frames = (latent_frames - 1) * 8 + 1.
Used by the latent-load-based upscale workflow to size the empty audio
latent from the loaded video latent (no AUDIO source needed for that
sizing step).
"""

from __future__ import annotations

import torch

from nodes import LatentFrameCount


def _make_latent(latent_frames: int, c: int = 128, h: int = 60, w: int = 104) -> dict:
    return {"samples": torch.zeros(1, c, latent_frames, h, w)}


def test_pixel_and_latent_counts_match_ltx_vae_formula():
    out = LatentFrameCount.execute(latent=_make_latent(63))
    assert out[0] == 497, "63 latent → 497 pixel frames (LTX video VAE)"
    assert out[1] == 63


def test_single_latent_frame_yields_one_pixel_frame():
    out = LatentFrameCount.execute(latent=_make_latent(1))
    assert out[0] == 1
    assert out[1] == 1


def test_full_song_latent_matches_observed_render():
    """534 latent frames (observed in 4277-pixel-frame renders) → 4265 pixel.
    Off-by-12 from the saved mp4's 4277 is the +12 from initial-render
    LatentConcat offset, not from this conversion."""
    out = LatentFrameCount.execute(latent=_make_latent(534))
    assert out[0] == 4265
    assert out[1] == 534


def test_node_is_registered_in_extension():
    """AST scan — same pattern as test_trim_image_batch_to_audio."""
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
    assert "LatentFrameCount" in found_names
