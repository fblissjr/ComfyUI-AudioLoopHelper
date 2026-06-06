"""Tests for pre-loop batch encoding of keyframe LATENT schedules.

Covers `KeyframeLatentScheduleBatchEncode` (runs ONCE outside the loop,
VAE-encodes every unique keyframe image up front) and
`LatentSelectByIteration` (runs inside the loop, picks pre-encoded
LATENT by index — no VAE dependency).

This pair replaces the per-iteration `KeyframeImageSchedule + ImageBlend
+ VAEEncode` chain on the latent-keyframe workflow. Mirrors the
TimestampPromptScheduleBatchEncode + ConditioningSelectByIteration
pattern shipped 2026-04-22 for the conditioning side.

Why mirror byte-for-byte:
- Same gate (downstream `current_iteration` invalidates ComfyUI's
  framework cache; we need our own LRU).
- Same dedup invariant (encode each unique resource once regardless
  of how many iterations share it).
- Same selector clamp semantics (overshoot returns last entry).
"""

from __future__ import annotations

import logging

import pytest
import torch


@pytest.fixture(autouse=True)
def _clear_keyframe_latent_cache():
    """FakeVAE + FakeImages instances are tiny and get GC'd between
    tests, so Python id() recycles rapidly — a stale cache entry from
    a previous test would produce a ghost hit. Clear before every test."""
    import nodes
    nodes._KEYFRAME_LATENT_CACHE.clear()
    yield
    nodes._KEYFRAME_LATENT_CACHE.clear()


class FakeVAE:
    """Records encode calls and returns a content-tagged tensor per
    unique pixel input, so tests can assert dedup + per-iteration
    output identity."""

    def __init__(self) -> None:
        self.encode_calls: list[int] = []  # records the IMAGE index encoded

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        # `pixels` shape is `[1, H, W, 3]` (single image slice). We tag
        # the returned latent with the index extracted from the tensor's
        # first pixel value, so tests can identify which image got
        # encoded into which list slot.
        idx = int(pixels[0, 0, 0, 0].item())
        self.encode_calls.append(idx)
        # Shape-of-record for an LTX video latent: [B, C, F, H, W].
        # Content tag in [0,0,0,0,0] so tests can read it back.
        latent = torch.zeros(1, 128, 8, 8, 8, dtype=torch.float32)
        latent[0, 0, 0, 0, 0] = float(idx)
        return latent


def _make_images(n: int) -> torch.Tensor:
    """N images, distinguishable by FakeVAE via the [:, 0, 0, 0] value."""
    images = torch.zeros(n, 16, 16, 3, dtype=torch.float32)
    for i in range(n):
        images[i, 0, 0, 0] = float(i)
    return images


def _schedule_text_three_sections() -> str:
    return (
        "0:00-0:20: 0\n"
        "0:20-0:40: 1\n"
        "0:40+: 2\n"
    )


class TestKeyframeLatentScheduleBatchEncode:
    """Batch encoder invariants — runs ONCE per generation."""

    def _execute(
        self,
        *,
        vae: FakeVAE,
        images: torch.Tensor,
        stride_seconds: float = 20.0,
        audio_duration: float = 60.0,
        schedule: str | None = None,
        snap_boundaries: bool = True,
    ) -> tuple[list, int]:
        from nodes import KeyframeLatentScheduleBatchEncode
        return KeyframeLatentScheduleBatchEncode.execute(
            vae=vae,
            images=images,
            schedule=schedule if schedule is not None else _schedule_text_three_sections(),
            stride_seconds=stride_seconds,
            audio_duration=audio_duration,
            snap_boundaries=snap_boundaries,
        )

    def test_emits_list_with_one_entry_per_iteration(self):
        """Output length matches ceil(audio/stride)+1 for safe overshoot."""
        vae = FakeVAE()
        images = _make_images(3)
        latent_list, iteration_count = self._execute(
            vae=vae, images=images, stride_seconds=20.0, audio_duration=60.0,
        )
        # 60/20 = 3 iterations + 1 headroom = 4
        assert iteration_count == 4
        assert len(latent_list) == 4

    def test_each_unique_image_encoded_exactly_once(self):
        """Dedup invariant: same image index across iterations encodes once."""
        vae = FakeVAE()
        images = _make_images(3)
        # At stride=10s, audio=40s, schedule covers indices 0,1,2 across
        # 5 iterations — but only 3 unique image indices.
        self._execute(
            vae=vae, images=images, stride_seconds=10.0, audio_duration=40.0,
        )
        assert sorted(set(vae.encode_calls)) == [0, 1, 2]
        assert len(vae.encode_calls) == 3

    def test_iteration_to_image_index_matches_schedule(self):
        """For each iteration i, list[i] is the LATENT for the image the
        schedule selects at i*stride_seconds."""
        vae = FakeVAE()
        images = _make_images(3)
        latent_list, _ = self._execute(
            vae=vae, images=images, stride_seconds=20.0, audio_duration=60.0,
        )
        # With snap_boundaries=True at 20s stride: iter 0 → 0:00 → idx 0;
        # iter 1 → 0:20 → idx 1; iter 2 → 0:40 → idx 2; iter 3 (headroom)
        # → 1:00 → idx 2 (open-end entry).
        expected_indices = [0, 1, 2, 2]
        for i, expected in enumerate(expected_indices):
            tag = latent_list[i]["samples"][0, 0, 0, 0, 0].item()
            assert tag == float(expected), (
                f"iter {i}: expected image idx {expected}, got tag {tag}"
            )

    def test_dedup_returns_same_latent_object_for_repeated_indices(self):
        """Identity stability: repeated index yields the SAME LATENT object,
        not a copy. Lets the selector return identity-stable refs."""
        vae = FakeVAE()
        images = _make_images(3)
        latent_list, _ = self._execute(
            vae=vae, images=images, stride_seconds=10.0, audio_duration=40.0,
        )
        # Schedule "0:40+: 2" makes iter 4 also use image 2; iter 2 also
        # uses image 2 (0:20 falls in 0:20-0:40 → index 1, hmm).
        # Stride=10, snap on: iter 0→0:00→0; iter 1→0:10→0; iter 2→0:20→1;
        # iter 3→0:30→1; iter 4→0:40→2.
        # iter 0 and iter 1 share image 0; their LATENTs must be the same dict.
        assert latent_list[0] is latent_list[1]
        assert latent_list[2] is latent_list[3]

    def test_out_of_bounds_image_index_clamps_to_last(self):
        """Schedule references idx 5 but batch has 3 images → clamp to 2.

        Validator catches this pre-render with WARN; the runtime fallback
        is the existing KeyframeImageSchedule's clamp behavior. Mirror it
        here so swapping the node doesn't change semantics.
        """
        vae = FakeVAE()
        images = _make_images(3)
        schedule = "0:00-0:20: 0\n0:20+: 99\n"
        latent_list, _ = self._execute(
            vae=vae, images=images, stride_seconds=10.0, audio_duration=40.0,
            schedule=schedule,
        )
        # iter 2 is at 0:20 → schedule says 99, clamp to 2 (batch_size-1).
        tag = latent_list[2]["samples"][0, 0, 0, 0, 0].item()
        assert tag == 2.0

    def test_negative_index_clamps_to_zero(self):
        """Schedule with negative index clamps to 0 (batch start)."""
        vae = FakeVAE()
        images = _make_images(3)
        schedule = "0:00+: -5\n"
        latent_list, _ = self._execute(
            vae=vae, images=images, stride_seconds=10.0, audio_duration=20.0,
            schedule=schedule,
        )
        tag = latent_list[0]["samples"][0, 0, 0, 0, 0].item()
        assert tag == 0.0

    def test_auto_schedule_is_identity_mapping(self):
        """schedule="auto" -> window i anchors keyframe i, stride-aligned by
        construction (iteration i covers [i*stride, (i+1)*stride)). No schedule
        text to hand-author per song; pairs with the planner-driven
        EvenlySpacedKeyframes count (= iterations + 1)."""
        vae = FakeVAE()
        images = _make_images(5)
        latent_list, n = self._execute(
            vae=vae, images=images, stride_seconds=20.0, audio_duration=60.0,
            schedule="auto",
        )
        # 60/20 = 3 iterations + 1 headroom = 4 entries, indices 0..3.
        assert n == 4
        tags = [latent_list[i]["samples"][0, 0, 0, 0, 0].item() for i in range(n)]
        assert tags == [0.0, 1.0, 2.0, 3.0]

    def test_auto_schedule_clamps_when_batch_short(self, caplog):
        """auto with fewer keyframes than iterations clamps to the last image
        (and the existing clamp WARN fires) — same fallback as text schedules."""
        vae = FakeVAE()
        images = _make_images(2)
        with caplog.at_level(logging.WARNING):
            latent_list, n = self._execute(
                vae=vae, images=images, stride_seconds=10.0, audio_duration=40.0,
                schedule="auto",
            )
        tags = [latent_list[i]["samples"][0, 0, 0, 0, 0].item() for i in range(n)]
        assert tags == [0.0, 1.0, 1.0, 1.0, 1.0]
        assert any("clamp" in r.message.lower() for r in caplog.records)

    def test_auto_schedule_case_and_whitespace_tolerant(self):
        vae = FakeVAE()
        images = _make_images(3)
        latent_list, _ = self._execute(
            vae=vae, images=images, stride_seconds=20.0, audio_duration=40.0,
            schedule="  AUTO \n",
        )
        tags = [latent_list[i]["samples"][0, 0, 0, 0, 0].item() for i in range(3)]
        assert tags == [0.0, 1.0, 2.0]

    def test_empty_schedule_behavior_unchanged(self):
        """Empty schedule keeps its historical meaning (no entries -> default
        index 0 everywhere) — only the explicit "auto" sentinel opts in."""
        vae = FakeVAE()
        images = _make_images(3)
        latent_list, n = self._execute(
            vae=vae, images=images, stride_seconds=20.0, audio_duration=40.0,
            schedule="",
        )
        tags = [latent_list[i]["samples"][0, 0, 0, 0, 0].item() for i in range(n)]
        assert all(t == 0.0 for t in tags)

    def test_out_of_bounds_clamp_warns(self, caplog):
        """The clamp must not be silent: iterations that clamp to the same
        last keyframe anchor start==end to one still -> frozen-window risk."""
        vae = FakeVAE()
        images = _make_images(3)
        schedule = "0:00-0:20: 0\n0:20+: 99\n"
        with caplog.at_level(logging.WARNING):
            self._execute(
                vae=vae, images=images, stride_seconds=10.0, audio_duration=40.0,
                schedule=schedule,
            )
        assert any("clamp" in r.message.lower() for r in caplog.records)

    def test_in_range_schedule_no_clamp_warn(self, caplog):
        vae = FakeVAE()
        images = _make_images(3)
        with caplog.at_level(logging.WARNING):
            self._execute(
                vae=vae, images=images, stride_seconds=20.0, audio_duration=60.0,
            )
        assert not [r for r in caplog.records if "clamp" in r.message.lower()]

    def test_empty_schedule_uses_index_zero(self):
        """Missing schedule defaults to first image."""
        vae = FakeVAE()
        images = _make_images(3)
        latent_list, _ = self._execute(
            vae=vae, images=images, schedule="",
            stride_seconds=20.0, audio_duration=40.0,
        )
        # Encoder called exactly once (only index 0 used).
        assert len(vae.encode_calls) == 1
        assert vae.encode_calls[0] == 0

    def test_minimum_one_iteration(self):
        """Audio shorter than stride still yields ≥1 entry."""
        vae = FakeVAE()
        images = _make_images(2)
        latent_list, iteration_count = self._execute(
            vae=vae, images=images, stride_seconds=20.0, audio_duration=5.0,
        )
        assert iteration_count >= 1
        assert len(latent_list) >= 1

    def test_latent_shape_matches_5d_video_convention(self):
        """LATENT["samples"] is [B, C, F, H, W] per LTX video latent shape."""
        vae = FakeVAE()
        images = _make_images(2)
        latent_list, _ = self._execute(
            vae=vae, images=images, stride_seconds=20.0, audio_duration=20.0,
        )
        assert latent_list[0]["samples"].dim() == 5


class TestLatentSelectByIteration:
    """Per-iteration selector — runs inside the loop, no VAE ref."""

    def _execute(
        self, *, latent_list: list, current_iteration: int,
    ) -> dict:
        from nodes import LatentSelectByIteration
        return LatentSelectByIteration.execute(
            latent_list=latent_list,
            current_iteration=current_iteration,
        )[0]

    def _fake_latents(self, n: int) -> list[dict]:
        """Distinguishable LATENT dicts; tag in samples[0,0,0,0,0]."""
        out = []
        for i in range(n):
            samples = torch.zeros(1, 128, 8, 8, 8, dtype=torch.float32)
            samples[0, 0, 0, 0, 0] = float(i)
            out.append({"samples": samples})
        return out

    def test_selects_by_index(self):
        latent_list = self._fake_latents(3)
        for i in range(3):
            got = self._execute(latent_list=latent_list, current_iteration=i)
            assert got["samples"][0, 0, 0, 0, 0].item() == float(i)

    def test_clamps_beyond_list_to_last(self):
        """Overshoot returns last entry (absorbs batch encoder's +1 headroom)."""
        latent_list = self._fake_latents(2)
        got = self._execute(latent_list=latent_list, current_iteration=99)
        assert got["samples"][0, 0, 0, 0, 0].item() == 1.0

    def test_clamps_negative_index_to_zero(self):
        latent_list = self._fake_latents(2)
        got = self._execute(latent_list=latent_list, current_iteration=-5)
        assert got["samples"][0, 0, 0, 0, 0].item() == 0.0

    def test_empty_list_raises(self):
        """Empty batch is a wiring bug; fail loudly."""
        from nodes import LatentSelectByIteration
        with pytest.raises((ValueError, IndexError)):
            LatentSelectByIteration.execute(
                latent_list=[], current_iteration=0,
            )


class TestBatchEncoderCaching:
    """The batch encoder MUST cache its LATENT_LIST output across
    repeated execute() calls with equivalent inputs.

    Same gate as TimestampPromptScheduleBatchEncode: AudioLoopController
    re-execution per iteration cascades into this node, which would
    re-VAE-encode all N unique keyframes per loop tick if not cached.
    Symptom in the console log would be N "VAE encoding 1 frames" lines
    per iteration where N = unique image index count.
    """

    def test_repeated_execute_calls_with_same_inputs_skip_vae(self):
        from nodes import KeyframeLatentScheduleBatchEncode
        vae = FakeVAE()
        images = _make_images(3)
        kwargs = dict(
            vae=vae, images=images, schedule=_schedule_text_three_sections(),
            stride_seconds=20.0, audio_duration=80.0, snap_boundaries=True,
        )
        list_a, count_a = KeyframeLatentScheduleBatchEncode.execute(**kwargs)
        first_call_encode_count = len(vae.encode_calls)
        assert first_call_encode_count >= 1

        list_b, count_b = KeyframeLatentScheduleBatchEncode.execute(**kwargs)
        list_c, count_c = KeyframeLatentScheduleBatchEncode.execute(**kwargs)
        assert len(vae.encode_calls) == first_call_encode_count, (
            "Batch encoder re-encoded on cache-hit path. "
            f"Expected {first_call_encode_count} VAE calls, got {len(vae.encode_calls)}."
        )
        # Cached output identity is stable across calls.
        assert list_b is list_a
        assert list_c is list_a
        assert count_b == count_a == count_c

    def test_input_change_invalidates_cache(self):
        from nodes import KeyframeLatentScheduleBatchEncode
        vae = FakeVAE()
        images = _make_images(3)
        base = dict(
            vae=vae, images=images, schedule=_schedule_text_three_sections(),
            stride_seconds=20.0, audio_duration=80.0, snap_boundaries=True,
        )
        KeyframeLatentScheduleBatchEncode.execute(**base)
        encodes_after_first = len(vae.encode_calls)

        other_schedule = "0:00+: 1\n"
        KeyframeLatentScheduleBatchEncode.execute(
            **{**base, "schedule": other_schedule},
        )
        # Different schedule, image 1 not yet encoded → cache miss.
        assert len(vae.encode_calls) > encodes_after_first

    def test_stride_change_invalidates_cache(self):
        from nodes import KeyframeLatentScheduleBatchEncode
        vae = FakeVAE()
        images = _make_images(3)
        base = dict(
            vae=vae, images=images, schedule=_schedule_text_three_sections(),
            stride_seconds=20.0, audio_duration=80.0, snap_boundaries=True,
        )
        KeyframeLatentScheduleBatchEncode.execute(**base)
        encodes_after_first = len(vae.encode_calls)

        # Different stride → different iteration→image mapping → potentially
        # different encode set. Even if encode count happens to match, the
        # cache key must differ so output isn't a stale ref.
        list_a, _ = KeyframeLatentScheduleBatchEncode.execute(**base)
        list_b, _ = KeyframeLatentScheduleBatchEncode.execute(
            **{**base, "stride_seconds": 10.0},
        )
        # Outputs are DIFFERENT list objects (cache miss).
        assert list_a is not list_b


class TestIntegrationWithKeyframeImageSchedule:
    """End-to-end: batch encode then select, compare per-iteration output
    against what KeyframeImageSchedule would emit at the same indices.

    Confirms swapping `KeyframeImageSchedule + ImageBlend + VAEEncode`
    out for `KeyframeLatentScheduleBatchEncode + LatentSelectByIteration`
    is structurally equivalent — same image index per iteration, same
    final LATENT shape.
    """

    def test_per_iter_image_index_matches_legacy_schedule(self):
        from nodes import (
            KeyframeImageSchedule,
            KeyframeLatentScheduleBatchEncode,
            LatentSelectByIteration,
        )

        vae = FakeVAE()
        images = _make_images(3)
        schedule = _schedule_text_three_sections()
        stride = 20.0
        audio_duration = 80.0

        latent_list, iter_count = KeyframeLatentScheduleBatchEncode.execute(
            vae=vae, images=images, schedule=schedule,
            stride_seconds=stride, audio_duration=audio_duration,
            snap_boundaries=True,
        )

        for i in range(iter_count):
            # Legacy node returns (image, next_image, blend_factor, current_time, image_index)
            legacy_out = KeyframeImageSchedule.execute(
                images=images,
                current_iteration=i,
                stride_seconds=stride,
                schedule=schedule,
                blend_seconds=0.0,
            )
            legacy_image_index = legacy_out[4]

            # New pair: select latent at iteration i, read its content tag
            new_latent = LatentSelectByIteration.execute(
                latent_list=latent_list, current_iteration=i,
            )[0]
            new_tag = int(new_latent["samples"][0, 0, 0, 0, 0].item())

            assert new_tag == legacy_image_index, (
                f"iter {i}: legacy KeyframeImageSchedule selected idx "
                f"{legacy_image_index} but new pair returned idx {new_tag}"
            )
