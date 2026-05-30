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

import logging
import re

import torch

logger = logging.getLogger("AudioICLoRA")


def _log(msg: str) -> None:
    """Telemetry to the ComfyUI console. Uses both logging (respects comfy's handlers) and
    print (unmissable on stdout) since the whole point of these debug nodes is visibility."""
    logger.info(msg)
    print(f"[AudioICLoRA] {msg}", flush=True)


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


_LORA_KEY_RE = re.compile(r"^diffusion_model\..+\.lora_(A|B)\.weight$")


def lora_grammar_problems(keys) -> list[str]:
    """Necessary condition for comfy to map an LTX LoRA onto the model: every key must be
    'diffusion_model.<...>.lora_A/lora_B.weight'. Returns the reasons it is not (an empty list
    means the grammar is OK). Pure and offline, so it both unit-tests and gives the runtime
    loader a precise diagnostic when a LoRA binds nothing to the model."""
    keys = list(keys)
    if not keys:
        return ["empty state dict (no keys)"]
    bad = [k for k in keys if not _LORA_KEY_RE.match(k)]
    if bad:
        return [
            f"{len(bad)}/{len(keys)} keys are not 'diffusion_model....lora_A/lora_B.weight' "
            f"(e.g. {sorted(bad)[:3]})"
        ]
    return []


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
        import torchaudio

        raw = reference_audio["waveform"]
        sr = reference_audio["sample_rate"]
        dur = raw.shape[-1] / sr
        # Mirror the STOCK VAEEncodeAudio.execute invocation EXACTLY (the authoritative path):
        #   1) widen mono->stereo on [b,c,n] BEFORE movedim,
        #   2) resample to the VAE's own sample rate (default 44100), and
        #   3) hand vae.encode a channels-LAST TENSOR (movedim(1,-1)) — NOT a {"waveform":...} dict.
        # The earlier dict/no-movedim/no-resample form produced garbage latents (LTX-2 caught it).
        waveform = ensure_stereo(raw)  # [b,c,n]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 44100)
        if vae_sr != sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        _log(f"GUIDE: ref audio {tuple(raw.shape)} @ {sr}Hz ({dur:.2f}s) -> stereo {tuple(waveform.shape)}, "
             f"vae_sr={vae_sr} (resampled={vae_sr != sr}); encoding channels-last")
        latent = audio_vae.encode(waveform.movedim(1, -1))  # channels-last tensor
        samples = latent["samples"] if isinstance(latent, dict) else latent
        _log(f"GUIDE: encoded ref latent shape {tuple(samples.shape)} dtype {samples.dtype}")

        ref_audio = patchify_audio_latent(samples)
        _log(f"GUIDE: patchified ref tokens {tuple(ref_audio['tokens'].shape)} (b,t,c*f) -> attaching to pos+neg")
        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})
        return io.NodeOutput(positive, negative)


def _folder_paths():
    """Lazy folder_paths import (absent under pytest -> stub so schema defs don't crash)."""
    try:
        import folder_paths  # type: ignore
        return folder_paths
    except ImportError:
        class _FP:
            def get_filename_list(self, _kind):
                return []

            def get_full_path_or_raise(self, _kind, name):
                return name
        return _FP()


class LTXAudioICLoRALoader(io.ComfyNode):
    """Debug-instrumented IC-LoRA loader for the AUDIO IC-LoRA. Functionally the same patch
    path as comfy's LoraLoaderModelOnly / LTXICLoRALoaderModelOnly (key_map -> load_lora ->
    add_patches), but logs the KEY-MATCH TELEMETRY we need: how many LoRA tensors loaded, how
    many produced patches, how many actually applied to the model, and samples of matched +
    UNMATCHED keys. A LoRA that produces garbage even at low strength is almost always a
    key-mapping or quant mismatch — this node makes that visible instead of silent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAudioICLoRALoader",
            display_name="Audio IC-LoRA Loader (debug)",
            category="Lightricks/IC-LoRA",
            description=(
                "Loads an audio IC-LoRA onto the model and logs key-match telemetry to the "
                "console (matched / unmatched / applied counts + samples). Use to diagnose "
                "an audio LoRA that loads but produces garbage."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=_folder_paths().get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model) -> io.NodeOutput:
        import comfy.lora
        import comfy.lora_convert
        import comfy.utils

        fp = _folder_paths()
        path = fp.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        # Mirror the production load path EXACTLY (comfy.sd.load_lora_for_models runs
        # convert_lora before load_lora). For our diffusion_model.…lora_A/lora_B grammar
        # convert_lora is a verified no-op, but this diagnostic loader must report what the
        # REAL loader does — so run the same step (code-review H1/parity #4).
        lora = comfy.lora_convert.convert_lora(lora)
        lora_keys = list(lora.keys())
        n_tensors = len(lora_keys)
        audio_tensors = sum(1 for k in lora_keys if "audio" in k)

        _log("=" * 60)
        _log(f"LOADER: {lora_name}  strength={strength_model}")
        _log(f"LOADER: {n_tensors} LoRA tensors in file ({audio_tensors} contain 'audio'); metadata={dict(metadata) if metadata else None}")
        _log(f"LOADER: sample LoRA keys: {lora_keys[:2]}")

        # build the model's lora target key_map (what comfy knows how to patch on THIS model)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        n_targets = len(key_map)
        audio_targets = sum(1 for k in key_map if "audio" in k)
        _log(f"LOADER: model exposes {n_targets} LoRA targets ({audio_targets} contain 'audio')")

        # map LoRA keys -> model patches
        loaded = comfy.lora.load_lora(lora, key_map)  # keyed by MODEL module
        n_patches = len(loaded)

        m = model.clone()
        applied = set(m.add_patches(loaded, strength_model))
        n_applied = len(applied)
        # `loaded` is keyed by MODEL modules that already matched key_map, so add_patches
        # applies ~all of them — built-but-unapplied is the rare model-side miss. The
        # FILE-side signal (LoRA tensors that mapped to NO target) is the one that matters,
        # and comfy.lora.load_lora already logs each as "lora key not loaded"; the n_patches
        # vs n_tensors gap is its summary (960 tensors -> 480 paired patches == full match).
        model_side_unapplied = [x for x in loaded if x not in applied]

        _log(f"LOADER: {n_patches} patches built from {n_tensors} tensors; {n_applied} APPLIED to model")
        # AUTOMATED TRUST GATE: a LoRA that binds to nothing makes the eval silently meaningless
        # (it looks exactly like a null result). Refuse to return a no-op model. Fail loud so an
        # eval can never run with a dead adapter. The offline half of this check is
        # lora_grammar_problems() (unit-tested); this is the runtime half that needs the real
        # model + key map, so it can't be tested without a GPU but it CAN refuse to proceed.
        if n_applied == 0:
            grammar = lora_grammar_problems(lora_keys)
            why = "; ".join(grammar) if grammar else "grammar looks right, so most likely the wrong base model"
            _log("=" * 60)
            raise RuntimeError(
                f"LTXAudioICLoRALoader: '{lora_name}' bound ZERO patches to the model "
                f"({n_tensors} tensors in file, {n_patches} built, model exposes {n_targets} LoRA "
                f"targets). The LoRA is doing NOTHING, so the eval would be meaningless. "
                f"Cause: {why}. Sample LoRA keys: {lora_keys[:3]}."
            )
        if model_side_unapplied:
            _log(f"LOADER: *** {len(model_side_unapplied)} built patches NOT applied (model-side miss), "
                 f"sample: {model_side_unapplied[:3]} ***")
        else:
            _log(f"LOADER: all {n_applied} built patches applied. sample target keys: {list(applied)[:2]} "
                 f"(any file-side key misses are logged above by load_lora)")
        _log("=" * 60)
        return io.NodeOutput(m)


class LTXAudioSetRefTokens(io.ComfyNode):
    """Debug-instrumented audio reference-token attach (the audio twin of LipDub's
    LTXVSetAudioRefTokens). Takes a PRE-ENCODED audio latent, patchifies it to ref tokens,
    attaches to pos+neg conditioning, and logs the shapes. The model applies the negative-RoPE
    offset (matching training). Mirrors the stock node's attach exactly + adds telemetry."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAudioSetRefTokens",
            display_name="Audio Set Ref Tokens (debug)",
            category="Lightricks/IC-LoRA",
            description=(
                "Attaches a pre-encoded audio latent as reference tokens on pos+neg "
                "conditioning (the model places them at negative temporal positions). Logs "
                "the latent + token shapes. Debug twin of LTXVSetAudioRefTokens."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("audio_latent", tooltip="Encoded audio latent (from an audio VAE encode)."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, audio_latent) -> io.NodeOutput:
        samples = audio_latent["samples"] if isinstance(audio_latent, dict) else audio_latent
        _log(f"REFTOKENS: audio latent {tuple(samples.shape)} (b,c,t,f) dtype {samples.dtype}")
        ref_audio = patchify_audio_latent(samples)
        _log(f"REFTOKENS: tokens {tuple(ref_audio['tokens'].shape)} (b,t,c*f) -> attach to pos+neg")
        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})
        return io.NodeOutput(positive, negative)
