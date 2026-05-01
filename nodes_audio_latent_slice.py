"""AudioLatentSlice — slice an audio latent tensor by source-timeline seconds.

Companion to the audio-latent pre-encode pattern (see
`internal/design/audio_latent_pre_encode_design.md`). The current loop
subgraph re-encodes the windowed audio slice via `LTXVAudioVAEEncode`
each iteration (~1.7s × 5 loop iters = ~8.5s/render of pure encode +
~5-15s of AudioVAE re-stage overhead). Pre-encoding the full song's
audio latent ONCE outside the loop and slicing in latent space per
iter eliminates both costs.

Latent rate is inferred empirically from the encoded latent's temporal
dim vs the source seconds — works regardless of the audio VAE's
specific mel hop_length / autoencoder downscale factor (which would
otherwise need to come from VAE metadata).

Design risk acknowledged in the design doc: audio VAE may have causal
padding / mel-frame quantization effects that make
`encode_full[T1:T2] != encode_slice([T1:T2])` to non-trivial rtol.
Equivalence test before shipping into a workflow is mandatory; see
the design doc § Equivalence test.
"""

from __future__ import annotations

import torch

try:
    from comfy_api.latest import io
    from typing_extensions import override
except ImportError:
    # pytest-stub fallback. Same shape as nodes_sage / nodes_easycache /
    # nodes_regional_compile. 4 sites now — extraction tracked but
    # deferred per CLAUDE.md "promote at 3rd call site" + spike status
    # of consumer nodes argues for waiting until they validate.
    class _Passthrough:
        def __getattr__(self, _name): return _Passthrough()
        def __call__(self, *_args, **_kwargs): return _Passthrough()

    class _IOStub(_Passthrough):
        class ComfyNode: pass

        @staticmethod
        def NodeOutput(*args): return args

    io = _IOStub()  # type: ignore[assignment]

    def override(fn):  # type: ignore[no-redef]
        return fn


def _infer_latent_rate(latent_temporal_dim: int, source_seconds: float) -> float:
    """Latents-per-second of the encoded audio latent.

    Empirical: audio VAE produces a fixed-rate latent given a fixed-rate
    waveform, so the ratio is constant for any source clip. Inferred at
    slice time so the node doesn't need to know the VAE's mel hop_length
    / downscale factor.
    """
    if source_seconds <= 0:
        raise ValueError(
            f"source_seconds must be > 0; got {source_seconds}. The encoded "
            "latent must come from a non-zero-duration audio source."
        )
    if latent_temporal_dim <= 0:
        raise ValueError(
            f"latent_temporal_dim must be > 0; got {latent_temporal_dim}."
        )
    return latent_temporal_dim / source_seconds


def _compute_slice_indices(
    latent_temporal_dim: int,
    source_seconds: float,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[int, int]:
    """Return (start_idx, end_idx) into the latent's temporal dim for the
    requested window. Clamps to valid range; never returns an empty slice
    (start_idx is clamped to latent_temporal_dim - 1 max so callers
    always get at least 1 latent frame).
    """
    rate = _infer_latent_rate(latent_temporal_dim, source_seconds)
    start_idx = int(round(start_seconds * rate))
    end_idx = start_idx + max(1, int(round(duration_seconds * rate)))
    # Clamp into [0, latent_temporal_dim]; preserve at least 1 frame.
    start_idx = max(0, min(start_idx, latent_temporal_dim - 1))
    end_idx = max(start_idx + 1, min(end_idx, latent_temporal_dim))
    return start_idx, end_idx


def _slice_latent(
    samples: torch.Tensor,
    source_seconds: float,
    start_seconds: float,
    duration_seconds: float,
) -> torch.Tensor:
    """Slice the LATENT samples tensor along its temporal dimension.

    Convention follows ComfyUI's LATENT shape: typically [B, C, T, ...]
    where T is the temporal dim (audio: time; video: frames). For audio
    this is dim=2 (after batch + channels).
    """
    if samples.ndim < 3:
        raise ValueError(
            f"audio latent must have shape [B, C, T, ...] (>=3 dims); "
            f"got shape {tuple(samples.shape)}"
        )
    temporal_dim_size = samples.shape[2]
    start_idx, end_idx = _compute_slice_indices(
        temporal_dim_size, source_seconds, start_seconds, duration_seconds,
    )
    # Slice along dim 2 (temporal). Use index_select-equivalent slicing
    # to keep the gradient + autograd-graph behavior consistent with
    # the rest of the latent pipeline (we don't want to materialize a
    # contiguous copy here; downstream nodes will copy as needed).
    return samples[:, :, start_idx:end_idx, ...]


class AudioLatentSlice(io.ComfyNode):
    """Slice an audio LATENT by source-timeline seconds.

    Use case: pre-encode the full song's audio latent ONCE outside the
    loop subgraph (via `LTXVAudioVAEEncode` on the full waveform), then
    slice the relevant per-iter window in latent space rather than
    re-encoding the windowed waveform each iter.

    Inputs:
    - latent: full audio latent tensor (encoded once outside loop)
    - source_seconds: total seconds of audio that produced `latent`
    - start_seconds: window start time (per-iter from
      `AudioLoopController.video_start_time`)
    - duration_seconds: window length (typically `window_seconds` from
      the planner)

    Output: sliced latent matching the per-iter window.

    Latent rate is inferred empirically from `latent.shape[temporal_dim]
    / source_seconds` — works regardless of audio VAE's mel hop_length /
    autoencoder downscale factor.

    See `internal/design/audio_latent_pre_encode_design.md` for the full
    pre-encode topology + equivalence-test caveats.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioLatentSlice",
            display_name="Audio Latent Slice (by source seconds)",
            category="AudioLoopHelper/audio",
            description=(
                "Slice an audio LATENT by source-timeline seconds. Companion "
                "to the audio-latent pre-encode pattern: encode once outside "
                "the loop, slice per-iter in latent space (cheaper than re-"
                "encoding the windowed waveform each iter). Latent rate "
                "inferred empirically — no VAE-config awareness needed."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="Full audio latent (encoded once from full song).",
                ),
                io.Float.Input(
                    "source_seconds",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.001,
                    tooltip=(
                        "Total seconds of audio that produced `latent`. "
                        "Determines the latent rate (latents/sec)."
                    ),
                ),
                io.Float.Input(
                    "start_seconds",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.001,
                    tooltip="Window start time (typically video_start_time per iter).",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=10.0,
                    min=0.001,
                    max=86400.0,
                    step=0.001,
                    tooltip="Window length (typically window_seconds from planner).",
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    @override
    def execute(  # type: ignore[override]
        cls,
        latent,
        source_seconds: float,
        start_seconds: float,
        duration_seconds: float,
    ) -> io.NodeOutput:
        samples = latent["samples"]
        sliced = _slice_latent(samples, source_seconds, start_seconds, duration_seconds)
        if "noise_mask" in latent:
            # Canonical workflow strips noise_mask via StripLatentNoiseMask
            # before the audio chain. If a future workflow carries
            # noise_mask through here, the mask shape no longer matches the
            # sliced samples — silent passthrough would corrupt downstream
            # masking. Warn loudly rather than fail-silently.
            import warnings
            warnings.warn(
                "AudioLatentSlice received noise_mask; passing through "
                "unchanged but mask shape no longer matches sliced samples. "
                "Strip noise_mask before this node or extend AudioLatentSlice "
                "to slice the mask too.",
                stacklevel=2,
            )
        out: dict = {**latent, "samples": sliced}
        return io.NodeOutput(out)
