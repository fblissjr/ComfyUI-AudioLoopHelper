"""Contract tests for the audio IC-LoRA guide node (the "pass ONLY audio" node).

The one parity-critical thing is the patchify: our node must produce the byte-identical
`ref_audio` token shape that ComfyUI-LTXVideo's LTXVSetAudioRefTokens produces, because
the model (av_model.py) applies the -ref_dur-0.04 offset to whatever tokens it finds — so
matching the token layout == train/inference parity. These tests lock that without
ComfyUI/GPU (pure tensor math).

Run: uv run --group dev --group analysis python -m pytest tests/test_audio_iclora_guide.py -v --rootdir=.
"""

from __future__ import annotations

import torch

import nodes_audio_iclora as G


def test_patchify_shape_b_t_cf():
    """(b,c,t,f) -> tokens (b, t, c*f). The documented ref_audio layout."""
    b, c, t, f = 1, 8, 51, 16
    latent = torch.arange(b * c * t * f, dtype=torch.float32).reshape(b, c, t, f)
    out = G.patchify_audio_latent(latent)
    assert set(out.keys()) == {"tokens"}
    assert out["tokens"].shape == (b, t, c * f)


def test_patchify_matches_permute_reshape_contract():
    """Byte-identical to permute(0,2,1,3).reshape(b,t,c*f) — the exact op the stock
    LTXVSetAudioRefTokens uses. This IS the parity contract."""
    b, c, t, f = 2, 8, 10, 16
    latent = torch.randn(b, c, t, f)
    expected = latent.permute(0, 2, 1, 3).reshape(b, t, c * f)
    out = G.patchify_audio_latent(latent)
    assert torch.equal(out["tokens"], expected)


def test_patchify_parity_with_installed_ltxvideo_node():
    """If ComfyUI-LTXVideo is importable, assert byte-identical to its private
    _patchify_audio_latent. Skips cleanly when not in a comfy env."""
    import importlib.util
    from pathlib import Path

    import pytest

    ltxv = Path(__file__).resolve().parents[2] / "ComfyUI-LTXVideo" / "iclora.py"
    if not ltxv.is_file():
        pytest.skip("ComfyUI-LTXVideo not present")
    # The sibling module imports comfy at top level, so it only loads inside a comfy
    # runtime — exec it directly and skip cleanly if that import chain isn't available.
    try:
        spec = importlib.util.spec_from_file_location("_ltxv_iclora", ltxv)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stock = mod._patchify_audio_latent
    except Exception:
        pytest.skip("ComfyUI-LTXVideo iclora.py needs comfy runtime to import")
    latent = torch.randn(1, 8, 51, 16)
    assert torch.equal(G.patchify_audio_latent(latent)["tokens"], stock(latent)["tokens"])


def test_ensure_stereo_widens_mono():
    """Mono [b,1,n] -> [b,2,n] (duplicate). Matches core audio_vae's mono->2ch widen
    (waveform.expand(-1,2,...)) and the trainer's repeat_interleave — bit-identical for
    a size-1 channel. Defensive: our node widens before encode so a mono tone can't crash
    even if a future VAE path doesn't auto-widen."""
    mono = torch.randn(1, 1, 16000)
    out = G.ensure_stereo(mono)
    assert out.shape == (1, 2, 16000)
    assert torch.equal(out[:, 0], out[:, 1])
    assert torch.equal(out[:, 0], mono[:, 0])


def test_ensure_stereo_passes_through_stereo():
    st = torch.randn(1, 2, 16000)
    assert torch.equal(G.ensure_stereo(st), st)


def test_node_registered():
    """The node is exported for ComfyUI registration."""
    from _node_registry import assert_node_registered

    assert_node_registered("LTXAddAudioICLoRAGuide")
