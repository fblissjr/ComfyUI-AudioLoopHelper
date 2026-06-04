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


def test_av_model_negative_rope_offset_tripwire():
    """SOURCE TRIPWIRE on comfy core's negative-RoPE reference placement (av_model.py).

    Three copies of the ref-position convention exist: the trainer, upstream lipdub
    (cross-locked by the trainer's parity test), and ComfyUI's av_model.py — the leg that
    updates underneath us with every ComfyUI release. Our nodes hand the model bare
    `ref_audio` tokens and TRUST av_model to place them at negative temporal positions
    (ref ends just before t=0, offset by end_of_last_token + one time-per-latent) and to
    strip them from the output. If upstream changes that convention, every trained audio
    IC-LoRA silently degrades — no error, just a reference that stops biting.

    This pins the load-bearing source lines, not behavior (the placement math is inline in
    forward(); there is no importable function to assert against without instantiating the
    model). ON FAILURE: diff upstream av_model.py against the trainer's parity test
    (LTX-2 fork, cross-parity RoPE test) before re-pinning — the change may be a rename
    (re-pin) or a real convention change (escalate; do NOT just update the needles).
    """
    from pathlib import Path

    import pytest

    # parents[2] = custom_nodes, parents[3] = the ComfyUI root holding comfy/
    av_model = Path(__file__).resolve().parents[3] / "comfy" / "ldm" / "lightricks" / "av_model.py"
    if not av_model.is_file():
        pytest.skip("comfy core not present (not running inside a ComfyUI tree)")
    src = av_model.read_text()
    needles = {
        "conds wiring": 'ref_audio = kwargs.get("ref_audio", None)',
        "offset anchor (ref ends just before t=0)": "time_offset = ref_end[-1].item() + tpl",
        "offset applied to start positions": "ref_start = (ref_start - time_offset)",
        "output strip keyed on this call's seq len": "ax = ax[:, ref_audio_seq_len:]",
    }
    missing = [f"{label}: {needle!r}" for label, needle in needles.items() if needle not in src]
    assert not missing, (
        "comfy core av_model.py no longer matches the pinned negative-RoPE reference "
        "convention — upstream drift; re-verify train/inference parity before re-pinning:\n  "
        + "\n  ".join(missing)
    )


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


class _RecordingVAE:
    """Fake audio VAE that records exactly what .encode() is handed — to lock the
    stock-VAEEncodeAudio invocation contract (channels-last TENSOR at the VAE sample rate,
    NOT a dict). This is the execute-level test the earlier patchify-only tests lacked,
    which let the dict/no-movedim/no-resample garbage bug through."""

    audio_sample_rate = 44100

    def __init__(self):
        self.encode_arg = None

    def encode(self, x):
        self.encode_arg = x
        # mimic comfy's audio VAE: returns a [b, 8, t, 16] latent tensor
        b = x.shape[0]
        return torch.zeros(b, 8, 5, 16)


def test_guide_encode_is_channels_last_tensor_at_vae_rate():
    """The guide must hand audio_vae.encode a channels-LAST TENSOR resampled to the VAE's
    rate — mirroring stock VAEEncodeAudio. A dict, or [b,c,n] (channels-first), or the wrong
    sample rate all produce garbage latents (the LTX-2-diagnosed bug)."""
    vae = _RecordingVAE()
    # mono 16 kHz tone (our reference format); VAE wants 44100 stereo channels-last
    ref_audio = {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}
    G.LTXAddAudioICLoRAGuide.execute(
        positive=[], negative=[], audio_vae=vae, reference_audio=ref_audio
    )
    arg = vae.encode_arg
    assert arg is not None, "audio_vae.encode was never called"
    assert isinstance(arg, torch.Tensor), f"encode got a {type(arg)}, not a tensor (the dict bug)"
    # channels-last: after movedim(1,-1) on [b,c,n] -> [b,n,c], last dim == channels (2 after stereo)
    assert arg.shape[-1] == 2, f"not channels-last stereo: {tuple(arg.shape)}"
    # resampled to the VAE rate: 16000 -> 44100 lengthens the sample axis
    assert arg.shape[1] > 16000, f"not resampled to {vae.audio_sample_rate}: {tuple(arg.shape)}"


def test_set_ref_tokens_attaches_patchified_to_both(monkeypatch):
    """LTXAudioSetRefTokens is the node the EVAL WORKFLOW actually uses (the guide node is the
    all-in-one alternative). It takes a PRE-encoded audio latent, patchifies it, and attaches the
    SAME ref_audio tokens to BOTH positive and negative conditioning. This locks that contract,
    independent of comfy's real node_helpers, by recording what gets attached."""
    calls = []

    def rec(cond, values):
        calls.append((cond, values))
        return cond

    monkeypatch.setattr(G.node_helpers, "conditioning_set_values", rec)

    samples = torch.randn(1, 8, 51, 16)
    out = G.LTXAudioSetRefTokens.execute(positive="POS", negative="NEG", audio_latent={"samples": samples})

    # exactly two attaches: positive then negative, each carrying ref_audio
    assert [c[0] for c in calls] == ["POS", "NEG"], "must attach to positive then negative"
    expected = G.patchify_audio_latent(samples)["tokens"]
    for _cond, values in calls:
        assert "ref_audio" in values, "did not set ref_audio on conditioning"
        assert torch.equal(values["ref_audio"]["tokens"], expected), "attached tokens are not the patchified latent"
    assert len(out) == 2, "must return (positive, negative)"


def test_guide_attach_to_negative_false_leaves_negative_untouched(monkeypatch):
    """attach_to_negative=False must attach ref_audio to the POSITIVE only and return the
    negative unchanged. This is what makes the CFG-analog amplification trick expressible:
    (positive_with_ref, positive_without_ref) into a CFGGuider needs a ref-free arm
    (docs/reference/cfg_analog_amplification.md)."""
    calls = []

    def rec(cond, values):
        calls.append((cond, values))
        return cond

    monkeypatch.setattr(G.node_helpers, "conditioning_set_values", rec)
    vae = _RecordingVAE()
    ref = {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}
    out = G.LTXAddAudioICLoRAGuide.execute(
        positive="POS", negative="NEG", audio_vae=vae, reference_audio=ref,
        attach_to_negative=False,
    )
    assert [c[0] for c in calls] == ["POS"], "must attach to positive ONLY when toggle is off"
    assert out[1] == "NEG", "negative must be returned untouched (same object, no ref_audio)"


def test_guide_attach_to_negative_default_attaches_both(monkeypatch):
    """Calling execute WITHOUT the toggle (old call shape) must keep today's behavior:
    ref_audio attached to positive AND negative. Locks the non-breaking default."""
    calls = []

    def rec(cond, values):
        calls.append((cond, values))
        return cond

    monkeypatch.setattr(G.node_helpers, "conditioning_set_values", rec)
    vae = _RecordingVAE()
    ref = {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}
    G.LTXAddAudioICLoRAGuide.execute(
        positive="POS", negative="NEG", audio_vae=vae, reference_audio=ref
    )
    assert [c[0] for c in calls] == ["POS", "NEG"], "default must attach to both (current behavior)"
    for _cond, values in calls:
        assert "ref_audio" in values


def test_advanced_guide_attach_to_negative_false(monkeypatch):
    """The Advanced guide exposes the same toggle and threads it through the shared
    encode+attach path (the two guides must not drift)."""
    calls = []

    def rec(cond, values):
        calls.append((cond, values))
        return cond

    monkeypatch.setattr(G.node_helpers, "conditioning_set_values", rec)
    vae = _RecordingVAE()
    ref = {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}
    out = G.LTXAddAudioICLoRAGuideAdvanced.execute(
        positive="POS", negative="NEG", audio_vae=vae, reference_audio=ref,
        reference_window_sec=0.0, reference_scale=1.0, attach_to_negative=False,
    )
    assert [c[0] for c in calls] == ["POS"]
    assert out[1] == "NEG"


def test_lora_grammar_accepts_valid_keys():
    """The audio IC-LoRA grammar comfy can map: diffusion_model.<...>.lora_A/lora_B.weight."""
    keys = [f"diffusion_model.transformer_blocks.{i}.audio_attn1.to_q.lora_A.weight" for i in range(3)]
    keys += ["diffusion_model.transformer_blocks.0.audio_ff.net.2.lora_B.weight"]
    assert G.lora_grammar_problems(keys) == []


def test_lora_grammar_flags_wrong_grammar():
    """A checkpoint in the wrong key grammar (lora_down/up, no diffusion_model prefix) would bind
    nothing; the gate must catch it with a clear reason."""
    bad = ["transformer.blocks.0.attn.lora_down.weight", "lora_unet_x.lora_up.weight"]
    problems = G.lora_grammar_problems(bad)
    assert problems and "lora_A/lora_B" in problems[0]


def test_lora_grammar_flags_empty():
    assert G.lora_grammar_problems([]) == ["empty state dict (no keys)"]


def test_shipped_lora_passes_grammar_if_present():
    """If the trained checkpoint is discoverable here, its keys must pass the grammar gate (the
    necessary condition for the loader to bind it). Skips when not present, so it stays portable
    but becomes a real check inside the ComfyUI runtime."""
    from pathlib import Path

    import pytest

    path = None
    try:
        import folder_paths  # only importable inside a comfy runtime

        path = folder_paths.get_full_path("loras", "pitch_gate_audio_ref_step02000.safetensors")
    except Exception:
        path = None
    if not path or not Path(path).is_file():
        pytest.skip("trained checkpoint not discoverable in this env")

    from safetensors import safe_open

    with safe_open(path, framework="pt") as h:
        keys = list(h.keys())
    assert G.lora_grammar_problems(keys) == [], "shipped LoRA fails the comfy LoRA grammar gate"
