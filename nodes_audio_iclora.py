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


try:
    from . import audio_reference_shaping as _shaping
except ImportError:  # flat import under pytest (conftest puts the repo root on sys.path)
    import audio_reference_shaping as _shaping


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


# Cross-modal bridge modules: the ONLY path the audio reference reaches the video stream
# (audio_to_video_attn = video queries attend to audio = voice->face; video_to_audio_attn = the
# reverse). Splitting on these lets the per-stream loader steer the bridge independently of the
# audio-only modules (audio_attn1/2, audio_ff). Substring match is robust to comfy's model-keyed
# patch names, which embed the transformer module path.
_BRIDGE_SUBSTRINGS = ("audio_to_video_attn", "video_to_audio_attn")


def is_bridge_lora_key(key: str) -> bool:
    """True if a LoRA patch key targets a cross-modal bridge module (voice<->face), not an
    audio-only module. Pure + offline, so it unit-tests and drives the per-stream partition."""
    return any(s in key for s in _BRIDGE_SUBSTRINGS)


def partition_lora_patches(loaded: dict) -> tuple[dict, dict]:
    """Split a loaded LoRA patch dict into (audio_only, bridge) by module name. An audio-only
    LoRA (no cross-modal keys) yields an empty bridge dict, so the per-stream loader degrades
    cleanly to a single-strength load (the audio strength applied to everything)."""
    audio_only: dict = {}
    bridge: dict = {}
    for k, v in loaded.items():
        (bridge if is_bridge_lora_key(k) else audio_only)[k] = v
    return audio_only, bridge


def trim_reference_waveform(waveform: torch.Tensor, sample_rate: int, seconds: float) -> torch.Tensor:
    """Keep only the first ``seconds`` of a ``[..., n]`` waveform (last dim = samples). ``seconds``
    <= 0, or a window >= the clip, returns the whole clip. A REAL input change (fewer reference
    samples -> fewer ref tokens), not a no-op; training used short references (~3.5s for the
    audio-only-context model; the length is dataset-specific), so a few-second window is
    in-distribution. Head-trim is the ``head`` mode of the shared windowing primitive."""
    start, end = _shaping.select_window_bounds(waveform, sample_rate, window_sec=seconds, mode="head")
    return waveform[..., start:end]


def _encode_and_attach_reference(positive, negative, audio_vae, waveform, sr, reference_scale: float = 1.0, log_tag: str = "GUIDE"):
    """Shared body for the audio IC-LoRA guide nodes: widen mono->stereo, resample to the VAE's
    sample rate, encode channels-last, optionally scale the latent magnitude, patchify, and attach as
    `ref_audio` tokens on pos+neg. The stereo -> resample -> movedim(1,-1) sequence mirrors stock
    VAEEncodeAudio EXACTLY (the train/inference parity that makes the trained LoRA read the reference);
    it lives here once so the basic and advanced guides can't drift. The caller does any trimming."""
    import torchaudio

    stereo = ensure_stereo(waveform)
    vae_sr = getattr(audio_vae, "audio_sample_rate", 44100)
    if vae_sr != sr:
        stereo = torchaudio.functional.resample(stereo, sr, vae_sr)
    _log(f"{log_tag}: ref {tuple(waveform.shape)} @ {sr}Hz ({waveform.shape[-1] / sr:.2f}s) -> stereo "
         f"{tuple(stereo.shape)}, vae_sr={vae_sr}; encoding channels-last")
    latent = audio_vae.encode(stereo.movedim(1, -1))  # channels-last tensor
    samples = latent["samples"] if isinstance(latent, dict) else latent
    if reference_scale != 1.0:
        samples = samples * reference_scale
    _log(f"{log_tag}: encoded ref latent {tuple(samples.shape)} dtype {samples.dtype} (scale={reference_scale})")
    ref_audio = patchify_audio_latent(samples)
    _log(f"{log_tag}: patchified ref tokens {tuple(ref_audio['tokens'].shape)} (b,t,c*f) -> attach to pos+neg")
    positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
    negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})
    return io.NodeOutput(positive, negative)


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
        # Encode the raw reference + attach as ref_audio tokens (the model places them at negative
        # RoPE positions as out-of-timeline context). Shared encode path -> _encode_and_attach_reference.
        return _encode_and_attach_reference(
            positive, negative, audio_vae, reference_audio["waveform"], reference_audio["sample_rate"]
        )


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


class LTXAudioICLoRALoaderPerStream(io.ComfyNode):
    """Audio IC-LoRA loader with SEPARATE strength for the cross-modal bridge vs the audio-only
    modules. The reference's voice->face influence flows through the bridge (audio_to_video_attn /
    video_to_audio_attn); voice->voice through the audio-only modules (audio_attn1/2, audio_ff). On
    a guidance-distilled base (CFG fixed at 1) strength is the ONLY inference amplifier, and a single
    global strength has a narrow usable band before garbling -- splitting it lets you push the bridge
    (more identity transfer) while keeping the audio modules in-band. Same load path + zero-bind trust
    gate as the debug loader. An audio-only LoRA (no bridge keys) loads fine: the bridge split is empty
    and bridge_strength is a no-op."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAudioICLoRALoaderPerStream",
            display_name="Audio IC-LoRA Loader (per-stream)",
            category="Lightricks/IC-LoRA",
            description=(
                "Loads an audio IC-LoRA with separate strengths for the cross-modal bridge "
                "(audio_to_video_attn / video_to_audio_attn -- the voice->face path) and the "
                "audio-only modules (audio_attn / audio_ff). Amplify identity transfer via the "
                "bridge while keeping the audio modules in their usable strength band, on a CFG=1 "
                "distilled model where strength is the only inference knob. An audio-only LoRA has "
                "no bridge keys, so bridge_strength is then a no-op."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=_folder_paths().get_filename_list("loras")),
                io.Float.Input(
                    "audio_strength", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="Strength for the audio-only modules (audio_attn1/2, audio_ff).",
                ),
                io.Float.Input(
                    "bridge_strength", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="Strength for the cross-modal bridge (audio_to_video_attn / "
                    "video_to_audio_attn) -- the voice->face path. Push above audio_strength to "
                    "amplify identity transfer. No-op for an audio-only LoRA (no bridge keys).",
                ),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, lora_name, audio_strength, bridge_strength) -> io.NodeOutput:
        import comfy.lora
        import comfy.lora_convert
        import comfy.utils

        fp = _folder_paths()
        path = fp.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        lora = comfy.lora_convert.convert_lora(lora)
        lora_keys = list(lora.keys())
        n_tensors = len(lora_keys)

        _log("=" * 60)
        _log(f"PER-STREAM LOADER: {lora_name}  audio_strength={audio_strength}  bridge_strength={bridge_strength}")
        _log(f"PER-STREAM LOADER: {n_tensors} LoRA tensors in file; metadata={dict(metadata) if metadata else None}")

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(lora, key_map)  # keyed by MODEL module
        audio_patches, bridge_patches = partition_lora_patches(loaded)
        _log(
            f"PER-STREAM LOADER: {len(loaded)} patches -> {len(audio_patches)} audio-only @ {audio_strength}, "
            f"{len(bridge_patches)} bridge @ {bridge_strength}"
        )
        if not bridge_patches:
            _log("PER-STREAM LOADER: no cross-modal bridge keys (audio-only LoRA) -> bridge_strength is a no-op")

        # Two add_patches calls = two patch tuples per stream; comfy sums them at forward time, so
        # each stream gets its own strength. (strength=0 still 'applies' a tuple, so the zero-bind
        # gate below only fires on a true binding failure, not a deliberate 0 strength.)
        m = model.clone()
        applied = set(m.add_patches(audio_patches, audio_strength))
        applied |= set(m.add_patches(bridge_patches, bridge_strength))
        n_applied = len(applied)

        # Same zero-bind trust gate as the debug loader: a LoRA that binds nothing makes the eval
        # silently meaningless. Refuse to return a no-op model; fail loud.
        if n_applied == 0:
            grammar = lora_grammar_problems(lora_keys)
            why = "; ".join(grammar) if grammar else "grammar looks right, so most likely the wrong base model"
            _log("=" * 60)
            raise RuntimeError(
                f"LTXAudioICLoRALoaderPerStream: '{lora_name}' bound ZERO patches to the model "
                f"({n_tensors} tensors in file, {len(loaded)} built, model exposes {len(key_map)} LoRA "
                f"targets). The LoRA is doing NOTHING. Cause: {why}. Sample keys: {lora_keys[:3]}."
            )
        _log(f"PER-STREAM LOADER: {n_applied} patches applied (audio @ {audio_strength}, bridge @ {bridge_strength})")
        _log("=" * 60)
        return io.NodeOutput(m)


class LTXAddAudioICLoRAGuideAdvanced(io.ComfyNode):
    """Advanced audio IC-LoRA guide: the basic guide plus two REAL reference knobs that work on a
    CFG=1 distilled base. reference_window_sec trims the reference to its first N seconds (fewer ref
    tokens; ~3.5s matches training); reference_scale multiplies the encoded reference latent
    magnitude. Both default to a no-op (full clip, scale 1.0) == the basic guide. No CFG /
    attention-strength knob is exposed: the distilled base is CFG=1 and the model's ref_audio path is
    always-clean, so those would be SILENT no-ops without a model-side change (see module docstring).
    The parity-locked bits (patchify layout, the model's negative-RoPE offset) are untouched."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAddAudioICLoRAGuideAdvanced",
            display_name="Add Audio IC-LoRA Guide (Advanced)",
            category="Lightricks/IC-LoRA",
            description=(
                "Audio in-context reference with extra knobs over the basic guide: trim the "
                "reference to its first N seconds (reference_window_sec; ~3.5s matches training) "
                "and scale the encoded reference latent magnitude (reference_scale). Both default "
                "to the basic behavior (full clip, scale 1.0). Values far from the defaults push "
                "the reference off the trained distribution (like strength >1.0), so sweep gently. "
                "No CFG / attention-strength knob: the distilled base is CFG=1 and the ref_audio "
                "path is always-clean, so those would do nothing without a model change."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to attach the reference to."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to attach the reference to."),
                io.Vae.Input("audio_vae", tooltip="The audio VAE (e.g. from LTXV Audio VAE Loader)."),
                io.Audio.Input(
                    "reference_audio",
                    tooltip="Raw reference audio (a voice sample). Mono is widened to stereo for the VAE.",
                ),
                io.Float.Input(
                    "reference_window_sec", default=0.0, min=0.0, max=60.0, step=0.1,
                    tooltip="Use only the first N seconds of the reference (0 = whole clip). Training "
                    "used 3.5s, so ~3.5 is the in-distribution window.",
                ),
                io.Float.Input(
                    "reference_scale", default=1.0, min=0.0, max=4.0, step=0.05,
                    tooltip="Multiply the encoded reference latent magnitude (1.0 = unchanged). A coarse "
                    "'reference loudness' lever; values far from 1.0 go off-distribution, so sweep gently.",
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
    def execute(
        cls, positive, negative, audio_vae, reference_audio, reference_window_sec, reference_scale
    ) -> io.NodeOutput:
        # Trim (a real input change, or no-op at 0), then the shared encode+attach with the scale knob.
        sr = reference_audio["sample_rate"]
        waveform = trim_reference_waveform(reference_audio["waveform"], sr, reference_window_sec)
        return _encode_and_attach_reference(
            positive, negative, audio_vae, waveform, sr, reference_scale=reference_scale, log_tag="GUIDE-ADV"
        )


class LTXAudioReferenceShaper(io.ComfyNode):
    """Pick which slice of a reference clip to use, and shape what within it the model weighs
    most, BEFORE the guide node (AUDIO -> AUDIO). Drop between Load Audio and the guide's
    reference_audio input; the guide's encode/patchify and the model's negative-RoPE attach are
    untouched, so train/inference parity is preserved.

    input_type picks a domain preset (the two real inputs are dialogue and songs):
      - dialogue: whole short clip (or a sustained-energy window if long), gate breaths/silence
        to the floor, otherwise flat -- conservative, matches the trained reference distribution.
      - song: search the FULL clip for the most representative ~window_sec by SUSTAINED energy
        (the hook/chorus, skipping quiet intro/outro -- head-trim is the wrong default here),
        then gently energy-weight within it.
      - manual: take the window at window_offset_sec and apply a spotlight at focus_center.

    Honest limits (design doc): the gain is a waveform-domain PROXY for attention weight (louder
    content -> bigger latents -> more pull), not a true per-token attention scalar; RMS can't tell
    vocal from instrumental; high emphasis goes off-distribution. emphasis=0 is a no-op (the
    selected window, unshaped). The shaping math lives in audio_reference_shaping (unit-tested)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAudioReferenceShaper",
            display_name="Audio Reference Shaper (window + emphasis)",
            category="Lightricks/IC-LoRA",
            description=(
                "Pick which ~window_sec slice of a reference clip to use and shape what within it "
                "the model weighs most, before the guide node. Presets: dialogue (gate silence, "
                "stay near the trained distribution), song (find the loudest sustained window -- "
                "the hook -- across a full track), manual (spotlight at an offset). Waveform-domain "
                "gain is an in-distribution proxy for attention weight; emphasis=0 is a no-op. Drop "
                "between Load Audio and the guide's reference_audio."
            ),
            inputs=[
                io.Audio.Input("reference_audio", tooltip="Raw reference audio to window + shape."),
                io.Combo.Input(
                    "input_type", options=["dialogue", "song", "manual"], default="dialogue",
                    tooltip="dialogue: gate silence, else flat (matches training). song: find the "
                    "loudest sustained window across the whole clip, then energy-weight. manual: "
                    "spotlight at window_offset_sec / focus_center.",
                ),
                io.Float.Input(
                    "window_sec", default=3.5, min=0.0, max=60.0, step=0.1,
                    tooltip="Seconds to keep (token-count parity with training; ~3.5s for the audio-only-context "
                    "model -- the trained length is dataset-specific). 0 = whole clip.",
                ),
                io.Combo.Input(
                    "window_select", options=["auto", "manual", "whole"], default="auto",
                    tooltip="auto: sustained-energy search (song) / whole-if-short (dialogue). "
                    "manual: start at window_offset_sec. whole: ignore window_sec, use it all.",
                ),
                io.Float.Input(
                    "window_offset_sec", default=0.0, min=0.0, max=600.0, step=0.1,
                    tooltip="Window start (seconds) when window_select = manual.",
                ),
                io.Float.Input(
                    "emphasis", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How strongly to apply the within-window shaping. **Default 0 = window-only** "
                    "(the in-distribution behaviour: training never saw a weighted reference). Raising it "
                    "imposes a loudness contour the model didn't train on -- an off-distribution proxy, "
                    "experimental, sweep gently. 1 = full preset envelope.",
                ),
                io.Float.Input(
                    "silence_floor", default=0.1, min=0.0, max=1.0, step=0.01,
                    tooltip="Minimum weight for de-emphasized frames (gated silence / quiet). 0 fully "
                    "mutes them; ~0.1 keeps a little.",
                ),
                io.Float.Input(
                    "focus_center", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="manual spotlight peak position (0 = window start, 1 = window end).",
                ),
                io.Float.Input(
                    "focus_width", default=0.5, min=0.05, max=1.0, step=0.01,
                    tooltip="manual spotlight width (small = tight, large = broad).",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="Windowed + shaped reference audio.")],
        )

    @classmethod
    def execute(
        cls, reference_audio, input_type, window_sec, window_select, window_offset_sec,
        emphasis, silence_floor, focus_center, focus_width,
    ) -> io.NodeOutput:
        sr = int(reference_audio["sample_rate"])
        wav = reference_audio["waveform"]
        in_sec = wav.shape[-1] / sr
        shaped = _shaping.shape_reference_waveform(
            wav, sr, input_type=input_type, window_sec=window_sec, window_select=window_select,
            window_offset_sec=window_offset_sec, silence_floor=silence_floor, emphasis=emphasis,
            focus_center=focus_center, focus_width=focus_width,
        )
        _log(
            f"SHAPER: {input_type}/{window_select} {in_sec:.2f}s -> {shaped.shape[-1] / sr:.2f}s "
            f"(emphasis={emphasis}, floor={silence_floor})"
        )
        return io.NodeOutput({"waveform": shaped, "sample_rate": sr})


class LTXLoadComposeReferenceAudio(io.ComfyNode):
    """Compose the in-context reference from one or more (possibly non-contiguous) slices of an input
    reference clip, each with its own gain. Sits between Load Audio and the guide's reference_audio
    (AUDIO -> AUDIO): wire `Load Audio -> here -> guide.reference_audio`.

    A single slice == a simple window; `[]` segments == the whole clip. **Keep the TOTAL composed
    duration to ~a few seconds**: the reference becomes ~15 tokens/sec at negative RoPE positions and
    the model only saw short references in training, so longer is off-distribution (see
    internal/audio_iclora_training/reference_window_selection.png). Slice *selection* is in-distribution;
    per-slice *gain* is an off-distribution emphasis proxy (default unity 1.0 = no emphasis). Slices are
    given as `segments` JSON (hand-edit for now; a visual waveform editor that owns the upload + draws
    the waveform is forthcoming -- owning the upload is coupled to shipping that widget, so until then
    this takes an AUDIO socket)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLoadComposeReferenceAudio",
            display_name="Compose Reference Audio (slices)",
            category="Lightricks/IC-LoRA",
            description=(
                "Compose the IC-LoRA reference from one or more slices (each with a gain) of the input "
                "audio. Wire Load Audio -> here -> the guide's reference_audio. One slice = a window; no "
                "slices = whole clip. Keep total duration to ~a few seconds (longer is off-distribution). "
                "Slices are `segments` JSON (hand-edit for now; a waveform editor will manage it)."
            ),
            inputs=[
                io.Audio.Input("reference_audio", tooltip="The reference clip (e.g. from Load Audio)."),
                io.String.Input(
                    "segments", default="[]", multiline=False,
                    tooltip="JSON list of slices: [{\"start_sec\":..,\"end_sec\":..,\"gain\":1.0}, ...]. "
                    "[] = whole clip. Keep total duration small. (A waveform editor will manage this.)",
                ),
                io.Float.Input(
                    "fade_sec", default=0.01, min=0.0, max=0.5, step=0.005,
                    tooltip="Edge fade applied at each slice boundary so non-contiguous joins don't "
                    "encode as clicks.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The composed reference AUDIO.")],
        )

    @classmethod
    def execute(cls, reference_audio, segments, fade_sec) -> io.NodeOutput:
        import json

        sr = int(reference_audio["sample_rate"])
        waveform = reference_audio["waveform"]
        try:
            segs = json.loads(segments) if segments else []
        except Exception:
            segs = None
        if not isinstance(segs, list):
            _log(f"COMPOSE: segments not a JSON list, using whole clip: {str(segments)[:80]!r}")
            segs = []
        composed = _shaping.compose_reference(waveform, sr, segs, fade_sec=float(fade_sec))
        _log(f"COMPOSE: {len(segs)} slice(s) -> {composed.shape[-1] / sr:.2f}s @ {sr}Hz")
        return io.NodeOutput({"waveform": composed, "sample_rate": sr})
