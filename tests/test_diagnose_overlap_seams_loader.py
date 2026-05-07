"""Tests for scripts/diagnose_overlap_seams.py::_load_latent.

Last updated: 2026-05-07

Phase A enabler for the seam-zone diagnostic. ComfyUI core's `SaveLatent`
node writes `.latent` files (safetensors-format with key `latent_tensor`).
The diagnose script must accept that shape directly so the diagnostic
can run on a SaveLatent capture without manual file munging.

Existing supported shapes (preserved):
  - .pt with dict containing 'samples'
  - .pt with bare Tensor
  - .safetensors with key in ('samples', 'latent', 'video_latent')

New shape under test:
  - .latent (safetensors-format) with 'latent_tensor' key
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from diagnose_overlap_seams import _load_latent


def _save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file(tensors, str(path))


def test_load_pt_dict_with_samples(tmp_path: Path):
    t = torch.randn(1, 4, 16, 8, 8)
    p = tmp_path / "x.pt"
    torch.save({"samples": t}, p)
    out = _load_latent(p)
    assert torch.equal(out, t)


def test_load_pt_bare_tensor(tmp_path: Path):
    t = torch.randn(1, 4, 16, 8, 8)
    p = tmp_path / "x.pt"
    torch.save(t, p)
    out = _load_latent(p)
    assert torch.equal(out, t)


def test_load_safetensors_samples_key(tmp_path: Path):
    t = torch.randn(1, 4, 16, 8, 8)
    p = tmp_path / "x.safetensors"
    _save_safetensors(p, {"samples": t})
    out = _load_latent(p)
    assert torch.equal(out, t)


def test_load_dot_latent_with_latent_tensor_key(tmp_path: Path):
    """ComfyUI SaveLatent output: .latent extension, latent_tensor key."""
    t = torch.randn(1, 4, 16, 8, 8)
    p = tmp_path / "loop_video_latent_00001_.latent"
    # save_torch_file writes safetensors with the same shape ComfyUI emits.
    _save_safetensors(p, {"latent_tensor": t})
    out = _load_latent(p)
    assert torch.equal(out, t)


def test_load_unknown_extension_raises(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"not a tensor")
    with pytest.raises(SystemExit):
        _load_latent(p)


def test_load_missing_file_raises(tmp_path: Path):
    p = tmp_path / "absent.pt"
    with pytest.raises(SystemExit):
        _load_latent(p)
