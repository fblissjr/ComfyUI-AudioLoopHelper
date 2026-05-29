"""Audio IC-LoRA guide node — the audio twin of ComfyUI-LTXVideo's LTXAddVideoICLoRAGuide.

Lets you pass in ONLY audio as an in-context reference for an audio-reference IC-LoRA
(e.g. the pitch gate / voice-clone product). Unlike the stock LTXVSetAudioRefTokens it:
  - takes RAW audio (AUDIO) + the audio VAE, encoding internally (no pre-encode step), and
  - drops the LipDub-specific `frozen_audio` output (irrelevant for fresh A+V generation).

THE VIDEO QUESTION (Fred): there is NO video input here. The reference is appended to the
AUDIO token stream only; the model (comfy/ldm/lightricks/av_model.py:686-709) prepends it
at negative RoPE positions (-ref_dur-0.04) and strips it from the output. The generated
video is the normal t2v/i2v target (noise->denoise / init+generate) and follows the
generated audio via the joint model. So nothing masked/padded/blacked-out — the audio path
has no video knob to set wrong. This matches the trainer's audio_reference strategy
(audio-ref only, video generated).

PARITY: the attach is byte-identical to LTXVSetAudioRefTokens — same `_patchify_audio_latent`
(permute(0,2,1,3).reshape(b,t,c*f)) + same `ref_audio` conditioning key — so the model
applies the exact offset training used. Design: internal/audio_iclora_training/
audio_iclora_guide_node_spec.md (private clone only).

Strength dial (the Advanced node) is NOT here yet: the model's ref_audio path is
always-clean (no strength param), so a working audio strength dial needs a model-side
answer from the LTX-2 fork. Basic (strength=1.0) is what the gate needs and what training
used (reference_strength=1.0).
"""

from __future__ import annotations

import torch

try:
    from comfy_api.latest import io
    import node_helpers
except ImportError:
    # Outside ComfyUI runtime (pytest). See nodes.py for the canonical stub pattern;
    # the pure helpers below (patchify_audio_latent, ensure_stereo) stay testable.
    class _Passthrough:
        def __getattr__(self, _name):
            return _Passthrough()

        def __call__(self, *args, **kwargs):
            return _Passthrough()

    class _IOStub(_Passthrough):
        class ComfyNode:
            pass

        @staticmethod
        def NodeOutput(*args):
            return args

    io = _IOStub()
    node_helpers = _Passthrough()


def patchify_audio_latent(latent: torch.Tensor) -> dict:
    """(b, c, t, f) -> {"tokens": (b, t, c*f)}.

    BYTE-IDENTICAL to ComfyUI-LTXVideo iclora.py::_patchify_audio_latent — this equality
    IS the train/inference parity contract (the model offsets whatever ref tokens it
    finds, so matching the layout is what makes the trained LoRA read the reference
    correctly). Do not "optimize" the permute/reshape.
    """
    b, c, t, f = latent.shape
    tokens = latent.permute(0, 2, 1, 3).reshape(b, t, c * f)
    return {"tokens": tokens}


def ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Widen mono [b, 1, n] -> [b, 2, n] (duplicate L=R). Defensive belt-and-suspenders:
    core comfy's audio VAE already widens mono (waveform.expand(-1, 2, ...)), and this is
    bit-identical to that and to the trainer's repeat_interleave for a size-1 channel. We
    widen before handing audio to the encoder so a mono reference tone can never hit the
    2-channel-conv crash regardless of the VAE path."""
    if waveform.dim() >= 2 and waveform.shape[1] == 1:
        repeats = [1] * waveform.dim()
        repeats[1] = 2
        return waveform.repeat(*repeats)
    return waveform


class LTXAddAudioICLoRAGuide(io.ComfyNode):
    """Attach a raw-audio in-context reference to the conditioning (audio twin of
    LTXAddVideoICLoRAGuide). Encodes the audio via the audio VAE and patchifies it into
    `ref_audio` tokens on both positive and negative conditioning. The model handles the
    negative-RoPE positioning + output stripping (parity with training)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAddAudioICLoRAGuide",
            display_name="Add Audio IC-LoRA Guide",
            category="Lightricks/IC-LoRA",
            description=(
                "Attaches a raw AUDIO clip as an in-context reference for an "
                "audio-reference IC-LoRA. The audio is encoded via the audio VAE and "
                "appended to the conditioning as reference tokens (the model places them "
                "at negative temporal positions as out-of-timeline context). No video "
                "input: the reference touches the audio stream only; the video is the "
                "normal generated target. Audio-only twin of Add Video IC-LoRA Guide."
            ),
            inputs=[
                io.Conditioning.Input(
                    "positive", tooltip="Positive conditioning to attach the reference to."
                ),
                io.Conditioning.Input(
                    "negative", tooltip="Negative conditioning to attach the reference to."
                ),
                io.Vae.Input(
                    "audio_vae", tooltip="The audio VAE (e.g. from LTXV Audio VAE Loader)."
                ),
                io.Audio.Input(
                    "reference_audio",
                    tooltip="Raw reference audio (e.g. a pitch tone, or a voice sample). "
                    "Mono is fine — it is widened to stereo for the VAE.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive", tooltip="Positive conditioning with the reference attached."
                ),
                io.Conditioning.Output(
                    display_name="negative", tooltip="Negative conditioning with the reference attached."
                ),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, audio_vae, reference_audio) -> io.NodeOutput:
        # Widen mono->stereo defensively, then encode via the audio VAE (same encoder the
        # trainer precompute used -> identical [8,T,16] latent format).
        waveform = ensure_stereo(reference_audio["waveform"])
        audio_in = {"waveform": waveform, "sample_rate": reference_audio["sample_rate"]}
        latent = audio_vae.encode(audio_in)
        samples = latent["samples"] if isinstance(latent, dict) else latent

        ref_audio = patchify_audio_latent(samples)
        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})
        return io.NodeOutput(positive, negative)
