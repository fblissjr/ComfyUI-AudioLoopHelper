"""Pure-helper tests for the advanced audio IC-LoRA knobs.

REAL (input/weight-changing, not no-op) knobs for a CFG=1 distilled base:
  - per-stream LoRA strength: split the LoRA patches into the cross-modal bridge
    (audio<->video attention, the voice->face path) vs the audio-only modules, so each
    can be applied at a different strength;
  - reference window trim: use only the first N seconds of the reference audio;
  - timestep-range gate: apply the reference only over a noise band — per-entry split
    so exactly one conditioning entry is active per sampler step (comfy's
    get_area_and_mult skips entries whose timestep range excludes the step).

No ComfyUI runtime needed (nodes_audio_iclora.py has a stub fallback for comfy imports;
these exercise the pure helpers directly).
"""

import torch

import nodes_audio_iclora as G
from nodes_audio_iclora import (
    is_bridge_lora_key,
    partition_lora_patches,
    split_conditioning_for_reference_band,
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


# --- timestep-range gate: split_conditioning_for_reference_band ----------------------------------

_REF = {"tokens": torch.randn(1, 5, 128)}


def _entry(**extra):
    """One conditioning entry in comfy's [tensor, options_dict] shape."""
    return [torch.randn(1, 3, 8), {"frame_rate": 25.0, **extra}]


def test_band_split_interior_band_yields_three_entries():
    """Band strictly inside (0,1): band copy + pre-complement + post-complement. ref_audio rides
    ONLY the band entry; complements stay bare so exactly one entry is active per step."""
    cond = [_entry()]
    out = split_conditioning_for_reference_band(cond, _REF, 0.2, 0.7)
    assert len(out) == 3
    band, pre, post = out
    assert band[1].get("ref_audio") is _REF
    assert (band[1]["start_percent"], band[1]["end_percent"]) == (0.2, 0.7)
    assert "ref_audio" not in pre[1]
    assert (pre[1]["start_percent"], pre[1]["end_percent"]) == (0.0, 0.2)
    assert "ref_audio" not in post[1]
    assert (post[1]["start_percent"], post[1]["end_percent"]) == (0.7, 1.0)


def test_band_split_skips_empty_complements():
    """Band touching an edge emits no zero-width complement on that side."""
    out_head = split_conditioning_for_reference_band([_entry()], _REF, 0.0, 0.5)
    assert len(out_head) == 2  # band + post only
    assert out_head[0][1].get("ref_audio") is _REF
    assert (out_head[1][1]["start_percent"], out_head[1][1]["end_percent"]) == (0.5, 1.0)

    out_tail = split_conditioning_for_reference_band([_entry()], _REF, 0.5, 1.0)
    assert len(out_tail) == 2  # pre + band only
    assert (out_tail[1][1]["start_percent"], out_tail[1][1]["end_percent"]) == (0.0, 0.5)


def test_band_split_preserves_other_entry_keys():
    """All copies carry the original entry's keys (frame_rate=25.0 etc.) — complements that
    dropped them would lose text conditioning / temporal embed scaling on their steps."""
    cond = [_entry(some_key="kept")]
    for _tensor, opts in split_conditioning_for_reference_band(cond, _REF, 0.25, 0.75):
        assert opts["frame_rate"] == 25.0
        assert opts["some_key"] == "kept"


def test_band_split_shares_entry_tensor():
    """Copies reference the SAME cond tensor (comfy's conditioning_set_values copies the dict,
    not the tensor) — no memory blowup from splitting."""
    cond = [_entry()]
    for tensor, _opts in split_conditioning_for_reference_band(cond, _REF, 0.2, 0.7):
        assert tensor is cond[0][0]


def test_band_split_strips_stale_ref_from_complements():
    """If the incoming conditioning already carries ref_audio (chained guides), complements must
    not inherit it — a stale ref on a complement defeats the gate."""
    stale = {"tokens": torch.randn(1, 2, 128)}
    cond = [_entry(ref_audio=stale)]
    out = split_conditioning_for_reference_band(cond, _REF, 0.2, 0.7)
    band, pre, post = out
    assert band[1]["ref_audio"] is _REF  # band overwrites with the new ref
    assert "ref_audio" not in pre[1]
    assert "ref_audio" not in post[1]


def test_band_split_empty_band_emits_no_ref_entry():
    """start == end (and inverted ranges, normalized to empty) = ref never applies: complements
    tile [0,1] with no ref entry, instead of emitting a zero-width ref band."""
    for start, end in ((0.5, 0.5), (0.7, 0.3)):
        out = split_conditioning_for_reference_band([_entry()], _REF, start, end)
        assert all("ref_audio" not in opts for _t, opts in out), (start, end)
        spans = sorted((opts["start_percent"], opts["end_percent"]) for _t, opts in out)
        assert spans[0][0] == 0.0 and spans[-1][1] == 1.0  # complements still tile the range


def test_band_split_multi_entry_conditioning():
    """Each incoming entry splits independently (e.g. schedule-blended conditioning)."""
    cond = [_entry(), _entry()]
    out = split_conditioning_for_reference_band(cond, _REF, 0.2, 0.7)
    assert len(out) == 6
    assert sum(1 for _t, opts in out if "ref_audio" in opts) == 2


# --- timestep-range gate: execute-level (Advanced guide) -----------------------------------------


class _StubVAE:
    audio_sample_rate = 44100

    def encode(self, x):
        return torch.zeros(x.shape[0], 8, 5, 16)


def _faithful_csv(cond, values):
    """Faithful mini conditioning_set_values: copy each entry's dict, merge values."""
    return [[e[0], {**e[1], **values}] for e in cond]


def _ref_audio_input():
    return {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}


def test_advanced_guide_default_band_is_passthrough(monkeypatch):
    """(0,1) band = today's behavior: no splitting, ref attached to every entry of pos+neg.
    Locks the new inputs as no-op defaults for saved workflows."""
    monkeypatch.setattr(G.node_helpers, "conditioning_set_values", _faithful_csv)
    pos, neg = [_entry()], [_entry()]
    out = G.LTXAddAudioICLoRAGuideAdvanced.execute(
        positive=pos, negative=neg, audio_vae=_StubVAE(), reference_audio=_ref_audio_input(),
        reference_window_sec=0.0, reference_scale=1.0, attach_to_negative=True,
        reference_start_percent=0.0, reference_end_percent=1.0,
    )
    for cond in (out[0], out[1]):
        assert len(cond) == 1, "default band must not split entries"
        assert "ref_audio" in cond[0][1]
        assert "start_percent" not in cond[0][1], "default band must not stamp timestep range"


def test_advanced_guide_gated_band_splits_pos_and_neg(monkeypatch):
    """Strict subrange splits BOTH pos and neg (attach_to_negative=True): band entry carries
    ref + range, complements cover the rest."""
    monkeypatch.setattr(G.node_helpers, "conditioning_set_values", _faithful_csv)
    out = G.LTXAddAudioICLoRAGuideAdvanced.execute(
        positive=[_entry()], negative=[_entry()], audio_vae=_StubVAE(),
        reference_audio=_ref_audio_input(),
        reference_window_sec=0.0, reference_scale=1.0, attach_to_negative=True,
        reference_start_percent=0.0, reference_end_percent=0.5,
    )
    for cond in (out[0], out[1]):
        assert len(cond) == 2
        assert "ref_audio" in cond[0][1] and cond[0][1]["end_percent"] == 0.5
        assert "ref_audio" not in cond[1][1] and cond[1][1]["start_percent"] == 0.5


def test_advanced_guide_gated_band_composes_with_attach_to_negative_false(monkeypatch):
    """attach_to_negative=False + band: positive split + gated, negative untouched (no split,
    no ref) — the ref-free CFG arm stays whole."""
    monkeypatch.setattr(G.node_helpers, "conditioning_set_values", _faithful_csv)
    neg = [_entry()]
    out = G.LTXAddAudioICLoRAGuideAdvanced.execute(
        positive=[_entry()], negative=neg, audio_vae=_StubVAE(),
        reference_audio=_ref_audio_input(),
        reference_window_sec=0.0, reference_scale=1.0, attach_to_negative=False,
        reference_start_percent=0.25, reference_end_percent=0.75,
    )
    assert len(out[0]) == 3
    assert out[1] is neg, "negative must be returned untouched"
