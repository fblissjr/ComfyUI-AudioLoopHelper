"""Tests for the `.block.` key rename that makes our trained LoRA loadable in
ComfyUI-LTXVideo. Discovered the hard way: without this rename ALL 2304 keys
fail to attach and the render is silently baseline (no LoRA effect)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ on path for the importable converter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from convert_lora_for_comfyui import convert_key, convert_state_dict


class TestConvertKey:
    def test_strips_block_from_wrapped_transformer_key(self):
        """The .block. segment is a serialization artifact of training-time
        block-swap (StreamingBlockWrapper.block). ComfyUI's unwrapped model
        expects it gone."""
        assert convert_key(
            "diffusion_model.transformer_blocks.12.block.attn1.to_k.lora_A.weight"
        ) == "diffusion_model.transformer_blocks.12.attn1.to_k.lora_A.weight"

    def test_covers_all_six_attn_module_names(self):
        """Broad target_modules hits 6 attention modules per block, all need
        the rename when wrapped. Locks the regex isn't module-specific."""
        for module in ("attn1", "attn2", "audio_attn1", "audio_attn2",
                       "audio_to_video_attn", "video_to_audio_attn"):
            src = f"diffusion_model.transformer_blocks.5.block.{module}.to_q.lora_B.weight"
            out = convert_key(src)
            assert ".block." not in out, f"missed .block. strip for {module}"
            assert module in out, f"strip removed the module name too: {out}"

    def test_idempotent_when_block_already_absent(self):
        """Kept-on-GPU blocks (0-11 in our training run) already serialize
        without .block. — running the converter on them must not corrupt."""
        already_clean = "diffusion_model.transformer_blocks.7.attn1.to_v.lora_A.weight"
        assert convert_key(already_clean) == already_clean

    def test_non_transformer_keys_unchanged(self):
        """Connector / embedding keys don't have the .block. wrapper — must
        not be rewritten."""
        for key in (
            "diffusion_model.embeddings_connector.video_connector.ff.lora_A.weight",
            "diffusion_model.patchify_proj.lora_A.weight",
            "diffusion_model.audio_patchify_proj.lora_A.weight",
        ):
            assert convert_key(key) == key, f"falsely rewrote {key}"

    def test_double_digit_block_indices_match(self):
        """Regex \\d+ must match multi-digit indices — LTX-2 has blocks 0-47
        and the bug surfaced from block 12 onward (the swapped ones)."""
        for n in (0, 9, 10, 12, 47):
            src = f"diffusion_model.transformer_blocks.{n}.block.attn1.to_k.lora_A.weight"
            out = convert_key(src)
            assert f"transformer_blocks.{n}.attn1" in out, f"failed for block {n}"
            assert ".block." not in out, f"failed to strip for block {n}: {out}"


class TestConvertStateDict:
    def test_preserves_total_count_and_values(self):
        """The state-dict transform is key-only — no tensors added/removed.
        Mirrors our real trained LoRA: mix of kept (no .block.) + swapped
        (with .block.) keys; output should be uniformly clean."""
        import torch
        sd = {
            # kept-block style (no .block., loaded fine)
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.zeros(2, 2),
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.ones(2, 2),
            # swapped-block style (.block., the broken ones)
            "diffusion_model.transformer_blocks.12.block.attn1.to_k.lora_A.weight": torch.zeros(2, 2),
            # non-transformer, untouched
            "diffusion_model.patchify_proj.lora_A.weight": torch.zeros(3),
        }
        out = convert_state_dict(sd)
        assert len(out) == 4
        # ALL keys should end without .block. now (uniform layout)
        assert all(".block." not in k for k in out), f"leaked .block. in {list(out)}"
        # transformer keys present in clean form
        assert "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight" in out
        assert "diffusion_model.transformer_blocks.12.attn1.to_k.lora_A.weight" in out
        # non-transformer key untouched
        assert "diffusion_model.patchify_proj.lora_A.weight" in out
