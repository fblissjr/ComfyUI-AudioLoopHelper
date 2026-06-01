"""Pure-helper tests for the advanced audio IC-LoRA knobs.

Two REAL (input/weight-changing, not no-op) knobs for a CFG=1 distilled base:
  - per-stream LoRA strength: split the LoRA patches into the cross-modal bridge
    (audio<->video attention, the voice->face path) vs the audio-only modules, so each
    can be applied at a different strength;
  - reference window trim: use only the first N seconds of the reference audio.

No ComfyUI runtime needed (nodes_audio_iclora.py has a stub fallback for comfy imports;
these exercise the pure helpers directly).
"""

import torch

from nodes_audio_iclora import (
    is_bridge_lora_key,
    partition_lora_patches,
    trim_reference_waveform,
)


def test_bridge_key_detection():
    assert is_bridge_lora_key("diffusion_model.transformer_blocks.0.audio_to_video_attn.to_q.lora_A.weight")
    assert is_bridge_lora_key("diffusion_model.transformer_blocks.5.video_to_audio_attn.to_out.0.lora_B.weight")
    assert not is_bridge_lora_key("diffusion_model.transformer_blocks.0.audio_attn1.to_q.lora_A.weight")
    assert not is_bridge_lora_key("diffusion_model.transformer_blocks.0.audio_attn2.to_v.lora_A.weight")
    assert not is_bridge_lora_key("diffusion_model.transformer_blocks.0.audio_ff.net.2.lora_B.weight")


def test_partition_splits_bridge_from_audio():
    loaded = {
        "blk.0.audio_attn1.to_q": 1,
        "blk.0.audio_ff.net.2": 2,
        "blk.0.audio_to_video_attn.to_k": 3,
        "blk.0.video_to_audio_attn.to_v": 4,
    }
    audio, bridge = partition_lora_patches(loaded)
    assert set(audio) == {"blk.0.audio_attn1.to_q", "blk.0.audio_ff.net.2"}
    assert set(bridge) == {"blk.0.audio_to_video_attn.to_k", "blk.0.video_to_audio_attn.to_v"}
    # the partition is exhaustive and disjoint
    assert set(audio) | set(bridge) == set(loaded)
    assert not (set(audio) & set(bridge))


def test_partition_audio_only_lora_has_empty_bridge():
    # an audio-only LoRA (arm A / the pitch LoRA) has no cross-modal keys -> bridge is empty,
    # so the per-stream loader degrades cleanly to a single-strength load.
    loaded = {"blk.0.audio_attn1.to_q": 1, "blk.0.audio_attn2.to_v": 2}
    audio, bridge = partition_lora_patches(loaded)
    assert set(audio) == set(loaded)
    assert bridge == {}


def test_trim_reference_waveform_keeps_first_seconds():
    wav = torch.zeros(1, 2, 44100 * 5)  # 5s stereo @ 44100
    out = trim_reference_waveform(wav, 44100, 3.5)
    assert out.shape[-1] == int(round(3.5 * 44100))
    assert out.shape[:-1] == wav.shape[:-1]  # leading dims untouched


def test_trim_zero_or_negative_returns_whole_clip():
    wav = torch.zeros(1, 2, 1000)
    assert trim_reference_waveform(wav, 44100, 0).shape[-1] == 1000
    assert trim_reference_waveform(wav, 44100, -1).shape[-1] == 1000


def test_trim_longer_than_clip_returns_unchanged():
    wav = torch.zeros(1, 2, 1000)
    assert trim_reference_waveform(wav, 44100, 999).shape[-1] == 1000  # window >= clip -> unchanged
