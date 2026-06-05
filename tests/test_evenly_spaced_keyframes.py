"""Tests for EvenlySpacedKeyframes — auto-pick N frames spread evenly across an IMAGE batch.

Replaces hand-loading keyframe images: feed the loaded video frames, get `count` frames
sampled evenly across the clip (count=3 -> first/middle/last). Feeds the keyframe encode
chain. KJNodes has GetImagesFromBatchIndexed (explicit indices) but nothing that computes
N evenly-spaced indices from the batch length, hence this node.

ComfyUI IMAGE batch is [B, H, W, C]; selection is along the batch axis (dim 0).

Warn surfaces (advisory only, selection behavior unchanged):
- count clamped to batch size (count=15 on a 9-frame video silently became 9 — the
  user has no signal their keyframes are now closer together in the source).
- near-duplicate consecutive selections (a loop window anchored between two
  nearly identical stills has almost no motion freedom and renders frozen).
"""

import logging

import torch


def _frames(n: int) -> torch.Tensor:
    """IMAGE batch of n frames where frame i is filled with value i (so the selected
    indices are recoverable from the output values)."""
    return torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 2, 2, 3).contiguous()


def _selected_indices(out: torch.Tensor) -> list[int]:
    return [int(out[k, 0, 0, 0].item()) for k in range(out.shape[0])]


class TestEvenlySpacedKeyframes:
    def test_three_from_nine_first_mid_last(self):
        """T=9, count=3 -> linspace(0,8,3) = [0, 4, 8]."""
        from nodes import EvenlySpacedKeyframes

        out = EvenlySpacedKeyframes.execute(images=_frames(9), count=3)[0]
        assert out.shape[0] == 3
        assert _selected_indices(out) == [0, 4, 8]

    def test_five_from_nine(self):
        """T=9, count=5 -> [0, 2, 4, 6, 8]."""
        from nodes import EvenlySpacedKeyframes

        out = EvenlySpacedKeyframes.execute(images=_frames(9), count=5)[0]
        assert _selected_indices(out) == [0, 2, 4, 6, 8]

    def test_count_one_is_first_frame(self):
        from nodes import EvenlySpacedKeyframes

        out = EvenlySpacedKeyframes.execute(images=_frames(20), count=1)[0]
        assert out.shape[0] == 1
        assert _selected_indices(out) == [0]

    def test_count_exceeds_batch_clamps_to_all(self):
        """count > T returns all T frames (no duplicates/padding)."""
        from nodes import EvenlySpacedKeyframes

        out = EvenlySpacedKeyframes.execute(images=_frames(4), count=20)[0]
        assert out.shape[0] == 4
        assert _selected_indices(out) == [0, 1, 2, 3]

    def test_empty_batch_returns_empty_no_crash(self):
        """A 0-frame batch (bad video path / frame_load_cap miss) must not IndexError —
        return the empty batch so the failure surfaces downstream, not as a crash here."""
        from nodes import EvenlySpacedKeyframes

        empty = torch.zeros(0, 2, 2, 3)
        out = EvenlySpacedKeyframes.execute(images=empty, count=3)[0]
        assert out.shape[0] == 0

    def test_count_zero_or_negative_clamps_to_one(self):
        from nodes import EvenlySpacedKeyframes

        for c in (0, -3):
            out = EvenlySpacedKeyframes.execute(images=_frames(10), count=c)[0]
            assert out.shape[0] == 1
            assert _selected_indices(out) == [0]

    def test_endpoints_always_included(self):
        """First and last frame of the clip are always in the selection (count>=2)."""
        from nodes import EvenlySpacedKeyframes

        out = EvenlySpacedKeyframes.execute(images=_frames(101), count=4)[0]
        idx = _selected_indices(out)
        assert idx[0] == 0 and idx[-1] == 100

    def test_preserves_frame_content(self):
        """Selected frames are the exact input frames (not just indices) — full HWC slice."""
        from nodes import EvenlySpacedKeyframes

        imgs = torch.rand(12, 8, 6, 3)
        out = EvenlySpacedKeyframes.execute(images=imgs, count=3)[0]
        # linspace(0,11,3).round() = [0, 6, 11]
        assert torch.equal(out[0], imgs[0])
        assert torch.equal(out[1], imgs[6])
        assert torch.equal(out[2], imgs[11])


class TestEvenlySpacedKeyframesWarnings:
    def test_count_clamp_warns(self, caplog):
        """count=20 on a 4-frame batch clamps to 4 — must say so out loud."""
        from nodes import EvenlySpacedKeyframes

        with caplog.at_level(logging.WARNING):
            EvenlySpacedKeyframes.execute(images=_frames(4), count=20)
        assert any("clamp" in r.message.lower() for r in caplog.records)

    def test_no_clamp_warn_when_count_fits(self, caplog):
        from nodes import EvenlySpacedKeyframes

        with caplog.at_level(logging.WARNING):
            EvenlySpacedKeyframes.execute(images=_frames(9), count=3)
        assert not [r for r in caplog.records if "clamp" in r.message.lower()]

    def test_near_duplicate_keyframes_warn(self, caplog):
        """All-identical frames -> every consecutive pair is a near-duplicate
        anchor pair -> warn (the frozen-window footgun)."""
        from nodes import EvenlySpacedKeyframes

        with caplog.at_level(logging.WARNING):
            EvenlySpacedKeyframes.execute(images=torch.zeros(9, 2, 2, 3), count=5)
        assert any("identical" in r.message.lower() for r in caplog.records)

    def test_distinct_keyframes_no_similarity_warn(self, caplog):
        """_frames() gives consecutive frames a full-pixel value delta of 1.0 —
        far above any near-duplicate threshold."""
        from nodes import EvenlySpacedKeyframes

        with caplog.at_level(logging.WARNING):
            EvenlySpacedKeyframes.execute(images=_frames(9), count=5)
        assert not [r for r in caplog.records if "identical" in r.message.lower()]

    def test_single_keyframe_no_similarity_warn(self, caplog):
        """count=1 has no consecutive pairs — no similarity warn possible."""
        from nodes import EvenlySpacedKeyframes

        with caplog.at_level(logging.WARNING):
            EvenlySpacedKeyframes.execute(images=torch.zeros(5, 2, 2, 3), count=1)
        assert not [r for r in caplog.records if "identical" in r.message.lower()]


def test_evenly_spaced_keyframes_registered():
    from _node_registry import assert_node_registered

    assert_node_registered("EvenlySpacedKeyframes")
