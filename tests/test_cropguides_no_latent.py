"""Tests for LTXVCropGuidesNoLatent — the CONDITIONING-only variant of
LTXVCropGuides. Replicates the upstream node's CONDITIONING-side behavior
(`keyframe_idxs` clearing) without taking or producing a LATENT, eliminating
the unnecessary `latent["samples"].clone()` per loop iteration.

Critical invariant: cleared `keyframe_idxs` must be `None`, not `[]`.
KJNodes' `OuterSampleCallbackWrapper` (`ltxv_nodes.py:867`) gates with
`if keyframe_idxs is not None:`, then indexes as a 4D tensor (`[0, 0, :, 0]`).
An empty list slips past the gate and crashes with `TypeError: list indices
must be integers or slices, not tuple`. Upstream `LTXVCropGuides`
(`comfy_extras/nodes_lt.py:404`) sets None — we must match.

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

    def test_keyframes_cleared_to_none_not_empty_list(self):
        """When num_keyframes > 0, keyframe_idxs is cleared to None on both.

        Must be None — KJNodes' OuterSampleCallbackWrapper gates
        `if keyframe_idxs is not None:` then indexes as a 4D tensor.
        An empty list slips through the gate and TypeErrors on tuple-indexing.
        """
        positive = _make_cond(keyframe_idxs=[[1, 2, 3]])
        negative = _make_cond(keyframe_idxs=[[1, 2, 3]])
        new_pos, new_neg = LTXVCropGuidesNoLatent.execute(positive, negative)
        assert _get_keyframe_idxs(new_pos) is None
        assert _get_keyframe_idxs(new_neg) is None

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
