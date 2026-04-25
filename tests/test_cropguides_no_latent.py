"""Tests for LTXVCropGuidesNoLatent — the CONDITIONING-only variant of
LTXVCropGuides. Replicates the upstream node's CONDITIONING-side behavior
(`keyframe_idxs` clearing) without taking or producing a LATENT, eliminating
the unnecessary `latent["samples"].clone()` per loop iteration.

These tests don't require a ComfyUI runtime — they exercise the static
behavior on the conditioning structure directly.
"""

from nodes import LTXVCropGuidesNoLatent


def _make_cond(keyframe_idxs=None):
    """A single-entry CONDITIONING list mirroring how upstream LTX-2 nodes
    pass conditioning around: [(tensor_or_None, dict_of_values)]."""
    metadata = {}
    if keyframe_idxs is not None:
        metadata["keyframe_idxs"] = keyframe_idxs
    return [(None, metadata)]


def _get_keyframe_idxs(cond):
    return cond[0][1].get("keyframe_idxs")


class TestLTXVCropGuidesNoLatent:
    def test_no_keyframes_passthrough(self):
        """When positive has no keyframe_idxs, output is unchanged."""
        positive = _make_cond(keyframe_idxs=None)
        negative = _make_cond(keyframe_idxs=None)
        new_pos, new_neg = LTXVCropGuidesNoLatent.execute(positive, negative)
        assert _get_keyframe_idxs(new_pos) is None
        assert _get_keyframe_idxs(new_neg) is None

    def test_empty_keyframes_passthrough(self):
        """When num_keyframes is zero (empty list/tensor), no-op."""
        positive = _make_cond(keyframe_idxs=[])
        negative = _make_cond(keyframe_idxs=[])
        new_pos, new_neg = LTXVCropGuidesNoLatent.execute(positive, negative)
        # Already-empty inputs come through unchanged
        assert _get_keyframe_idxs(new_pos) == [] or _get_keyframe_idxs(new_pos) is None
        assert _get_keyframe_idxs(new_neg) == [] or _get_keyframe_idxs(new_neg) is None

    def test_keyframes_cleared(self):
        """When num_keyframes > 0, keyframe_idxs is cleared on both."""
        positive = _make_cond(keyframe_idxs=[[1, 2, 3]])  # any non-empty marker
        negative = _make_cond(keyframe_idxs=[[1, 2, 3]])
        new_pos, new_neg = LTXVCropGuidesNoLatent.execute(positive, negative)
        assert _get_keyframe_idxs(new_pos) == []
        assert _get_keyframe_idxs(new_neg) == []

    def test_no_latent_input_signature(self):
        """API contract: execute takes only positive + negative, no latent.
        Calling with a latent kwarg should be a TypeError."""
        positive = _make_cond()
        negative = _make_cond()
        try:
            LTXVCropGuidesNoLatent.execute(positive, negative, latent={"samples": None})
            raised = False
        except TypeError:
            raised = True
        assert raised, "execute() must reject `latent` parameter (the whole point)"
