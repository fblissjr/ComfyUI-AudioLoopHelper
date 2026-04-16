"""Tests for performance-oriented nodes: CachedTextEncode, IterationCleanup.

CachedTextEncode: verifies LRU cache behavior (hits, misses, eviction) via
a mock CLIP object that counts tokenize calls.

IterationCleanup: verifies LATENT passthrough and that the intended side
effects (torch.cuda.empty_cache, gc.collect) fire in the right modes.
"""

import gc
import torch


class _MockClip:
    """Stand-in for a ComfyUI CLIP object. Counts tokenize/encode calls."""

    def __init__(self):
        self.tokenize_calls = 0
        self.encode_calls = 0

    def tokenize(self, text: str):
        self.tokenize_calls += 1
        return {"text": text}  # opaque "tokens" -- passed to encode_from_tokens_scheduled

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        # Return a CONDITIONING-shaped structure: list of [tensor, metadata] pairs
        return [[torch.zeros(1, 4, 8), {"attention_mask": torch.ones(1, 4)}]]


# --- CachedTextEncode ---


class TestCachedTextEncodeHits:
    def setup_method(self):
        from nodes import _COND_CACHE
        _COND_CACHE.clear()

    def test_cache_hit_skips_encode(self):
        from nodes import CachedTextEncode

        clip = _MockClip()
        CachedTextEncode.execute(clip=clip, text="hello world")
        CachedTextEncode.execute(clip=clip, text="hello world")

        # Second call should hit the cache -- tokenize/encode called once total
        assert clip.tokenize_calls == 1
        assert clip.encode_calls == 1

    def test_different_prompts_both_miss(self):
        from nodes import CachedTextEncode

        clip = _MockClip()
        CachedTextEncode.execute(clip=clip, text="first prompt")
        CachedTextEncode.execute(clip=clip, text="second prompt")

        assert clip.tokenize_calls == 2
        assert clip.encode_calls == 2

    def test_different_clips_miss_even_with_same_prompt(self):
        from nodes import CachedTextEncode

        clip_a = _MockClip()
        clip_b = _MockClip()
        CachedTextEncode.execute(clip=clip_a, text="same prompt")
        CachedTextEncode.execute(clip=clip_b, text="same prompt")

        # Different id(clip) -> different cache keys -> both miss
        assert clip_a.tokenize_calls == 1
        assert clip_b.tokenize_calls == 1

    def test_cached_conditioning_is_returned_unchanged(self):
        from nodes import CachedTextEncode

        clip = _MockClip()
        first = CachedTextEncode.execute(clip=clip, text="prompt")
        second = CachedTextEncode.execute(clip=clip, text="prompt")

        # Same object identity -- we return the cached object, not a copy
        assert first[0] is second[0]


class TestCachedTextEncodeLRU:
    def setup_method(self):
        from nodes import _COND_CACHE
        _COND_CACHE.clear()

    def test_eviction_when_exceeding_max(self):
        from nodes import CachedTextEncode, _COND_CACHE, _COND_CACHE_MAX

        clip = _MockClip()
        # Fill the cache to exactly max capacity
        for i in range(_COND_CACHE_MAX):
            CachedTextEncode.execute(clip=clip, text=f"prompt_{i}")
        assert len(_COND_CACHE) == _COND_CACHE_MAX

        # Add one more -- should evict the oldest (prompt_0)
        CachedTextEncode.execute(clip=clip, text="prompt_new")
        assert len(_COND_CACHE) == _COND_CACHE_MAX
        assert (id(clip), "prompt_0") not in _COND_CACHE
        assert (id(clip), "prompt_new") in _COND_CACHE

    def test_lru_reorders_on_hit(self):
        from nodes import CachedTextEncode, _COND_CACHE, _COND_CACHE_MAX

        clip = _MockClip()
        for i in range(_COND_CACHE_MAX):
            CachedTextEncode.execute(clip=clip, text=f"prompt_{i}")

        # Hit prompt_0 -- now it's most recently used
        CachedTextEncode.execute(clip=clip, text="prompt_0")

        # Adding a new entry should evict prompt_1 (the new oldest), not prompt_0
        CachedTextEncode.execute(clip=clip, text="prompt_new")
        assert (id(clip), "prompt_0") in _COND_CACHE
        assert (id(clip), "prompt_1") not in _COND_CACHE


# --- IterationCleanup ---


class TestIterationCleanup:
    def test_passthrough_always_mode(self):
        from nodes import IterationCleanup

        latent = {"samples": torch.randn(1, 128, 4, 60, 104)}
        result = IterationCleanup.execute(latent=latent, mode="always")
        assert result[0] is latent

    def test_passthrough_gpu_only_mode(self):
        from nodes import IterationCleanup

        latent = {"samples": torch.randn(1, 128, 4, 60, 104)}
        result = IterationCleanup.execute(latent=latent, mode="gpu_only")
        assert result[0] is latent

    def test_passthrough_never_mode(self):
        from nodes import IterationCleanup

        latent = {"samples": torch.randn(1, 128, 4, 60, 104)}
        result = IterationCleanup.execute(latent=latent, mode="never")
        assert result[0] is latent

    def test_never_mode_skips_side_effects(self, monkeypatch):
        from nodes import IterationCleanup

        gc_calls = [0]
        empty_cache_calls = [0]

        def fake_gc_collect():
            gc_calls[0] += 1

        def fake_empty_cache():
            empty_cache_calls[0] += 1

        monkeypatch.setattr(gc, "collect", fake_gc_collect)
        monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

        latent = {"samples": torch.zeros(1)}
        IterationCleanup.execute(latent=latent, mode="never")

        assert gc_calls[0] == 0
        assert empty_cache_calls[0] == 0

    def test_always_mode_calls_gc_and_empty_cache(self, monkeypatch):
        from nodes import IterationCleanup

        gc_calls = [0]
        empty_cache_calls = [0]

        def fake_gc_collect():
            gc_calls[0] += 1

        def fake_empty_cache():
            empty_cache_calls[0] += 1

        monkeypatch.setattr(gc, "collect", fake_gc_collect)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

        latent = {"samples": torch.zeros(1)}
        IterationCleanup.execute(latent=latent, mode="always")

        assert gc_calls[0] == 1
        assert empty_cache_calls[0] == 1

    def test_gpu_only_mode_skips_gc(self, monkeypatch):
        from nodes import IterationCleanup

        gc_calls = [0]
        empty_cache_calls = [0]

        def fake_gc_collect():
            gc_calls[0] += 1

        def fake_empty_cache():
            empty_cache_calls[0] += 1

        monkeypatch.setattr(gc, "collect", fake_gc_collect)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

        latent = {"samples": torch.zeros(1)}
        IterationCleanup.execute(latent=latent, mode="gpu_only")

        assert gc_calls[0] == 0
        assert empty_cache_calls[0] == 1

    def test_no_cuda_available_is_a_noop_for_empty_cache(self, monkeypatch):
        from nodes import IterationCleanup

        empty_cache_calls = [0]

        def fake_empty_cache():
            empty_cache_calls[0] += 1

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

        latent = {"samples": torch.zeros(1)}
        IterationCleanup.execute(latent=latent, mode="always")

        # No CUDA -> empty_cache should not be called
        assert empty_cache_calls[0] == 0
