"""Audio-aware loop helper nodes for ComfyUI.

Provides nodes to automatically manage loop iteration timing against an
audio track, eliminating manual iteration count calculation and preventing
crashes from overshooting audio boundaries.

Built for use alongside ComfyUI-NativeLooping (TensorLoopOpen/Close),
ComfyUI-VideoHelperSuite, ComfyUI-KJNodes, and ComfyUI-MelBandRoFormer
for generating full-length music videos with LTX 2.3.
"""

import gc
import math
import re
from collections import OrderedDict

import torch
from typing_extensions import override

try:
    from comfy_api.latest import ComfyExtension, io
except ImportError:
    # Outside ComfyUI runtime (e.g., pytest). Provide minimal stubs so
    # helper functions and execute() methods remain testable.
    # __getattr__ handles io.Schema, io.Int.Input, etc. used in annotations
    # and define_schema() without enumerating every attribute.
    class _Passthrough:
        """Returns itself for any attribute access or call."""
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

    ComfyExtension = type("ComfyExtension", (), {})
    io = _IOStub()


LTX_TEMPORAL_SCALE = 8  # LTX 2.3 VAE temporal compression factor (pixel_frames // 8 = latent_frames)


def _compute_tile_count(audio_duration: float, stride: float) -> int:
    """Number of valid loop iterations. Matches AudioLoopController stop condition.

    Note: caps at 200 for display/planning purposes. AudioLoopController itself
    has no cap -- it runs until should_stop fires. For audio > 200 * stride seconds,
    the planner and ScheduleToMultiPrompt will show/generate 200 tiles but the
    loop will continue past that (last prompt repeats via fallback).
    """
    return max(1, min(math.ceil(audio_duration / stride) - 1, 200))


def _audio_duration(audio: dict) -> float:
    """Extract duration in seconds from a ComfyUI AUDIO dict."""
    return audio["waveform"].shape[-1] / audio["sample_rate"]


def _parse_timestamp(ts: str) -> float:
    """Parse a timestamp string into seconds.

    Supports formats:
      - "1:23"     -> 83.0
      - "1:23.5"   -> 83.5
      - "0:05"     -> 5.0
      - "83"       -> 83.0
      - "83.5"     -> 83.5
    """
    ts = ts.strip()
    if ":" in ts:
        parts = ts.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    return float(ts)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or M:SS.ss if fractional."""
    m = int(seconds) // 60
    s = seconds - m * 60
    if s == int(s):
        return f"{m}:{int(s):02d}"
    return f"{m}:{s:05.2f}"


_TS_PATTERN = r"\d+(?::\d{1,2})?(?:\.\d+)?"
_LINE_RE = re.compile(
    rf"^({_TS_PATTERN}(?:\s*-\s*{_TS_PATTERN})?\+?)\s*:\s*(.+)$"
)


from typing import Callable, TypeVar

_T = TypeVar("_T")


def _parse_schedule_generic(
    schedule: str,
    convert_value: Callable[[str], _T | None],
) -> list[tuple[float, float | None, _T]]:
    """Parse a timestamp-based schedule with a pluggable value converter.

    Each line: `timestamp_range: value`
    Range formats:
      - "0:00-0:38: value"   (start-end, inclusive)
      - "1:15+: value"       (from here onward)
      - "38-75: value"       (bare seconds)

    convert_value receives the raw string after the colon. Return None to
    skip the line (e.g. invalid integer).
    """
    entries: list[tuple[float, float | None, _T]] = []
    for line in schedule.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        range_part = match.group(1).strip()
        value = convert_value(match.group(2).strip())
        if value is None:
            continue

        if range_part.endswith("+"):
            start = _parse_timestamp(range_part[:-1])
            entries.append((start, None, value))
        elif "-" in range_part:
            parts = range_part.split("-", 1)
            start = _parse_timestamp(parts[0])
            end = _parse_timestamp(parts[1])
            entries.append((start, end, value))
        else:
            t = _parse_timestamp(range_part)
            entries.append((t, t, value))
    return entries


def _match_schedule_generic(
    entries: list[tuple[float, float | None, _T]],
    current_time: float,
    default: _T,
) -> _T:
    """Find the matching value for the given time. Last match wins."""
    result: _T | None = None
    for start, end, value in entries:
        if end is None:
            if current_time >= start:
                result = value
        else:
            if start <= current_time <= end:
                result = value
    if result is None and entries:
        result = entries[-1][2]
    return result if result is not None else default


def _match_schedule_with_next_generic(
    entries: list[tuple[float, float | None, _T]],
    current_time: float,
    blend_seconds: float,
    default: _T,
) -> tuple[_T, _T, float]:
    """Find current value, next value, and blend factor.

    Returns (current_value, next_value, blend_factor).
    blend_factor is 0.0 when not near a boundary, and ramps to 1.0
    at the boundary over blend_seconds.
    """
    current_value = _match_schedule_generic(entries, current_time, default)

    if blend_seconds <= 0:
        return current_value, current_value, 0.0

    next_boundary = None
    next_value = current_value
    for start, _end, value in entries:
        if start > current_time:
            if next_boundary is None or start < next_boundary:
                next_boundary = start
                next_value = value

    if next_boundary is None:
        return current_value, current_value, 0.0

    time_to_boundary = next_boundary - current_time
    if time_to_boundary < blend_seconds:
        blend_factor = 1.0 - (time_to_boundary / blend_seconds)
        return current_value, next_value, blend_factor

    return current_value, current_value, 0.0


# --- Prompt schedule (str values) ---


def _parse_schedule(schedule: str) -> list[tuple[float, float | None, str]]:
    """Parse a timestamp-based prompt schedule."""
    return _parse_schedule_generic(schedule, str.strip)


def _match_schedule(entries: list[tuple[float, float | None, str]], current_time: float) -> str:
    """Find the matching prompt for the given time. Last match wins."""
    return _match_schedule_generic(entries, current_time, "")


def _match_schedule_with_next(
    entries: list[tuple[float, float | None, str]],
    current_time: float,
    blend_seconds: float,
) -> tuple[str, str, float]:
    """Find current prompt, next prompt, and blend factor."""
    return _match_schedule_with_next_generic(entries, current_time, blend_seconds, "")


# --- Image schedule (int values) ---


def _safe_int(s: str) -> int | None:
    """Convert string to int, returning None on failure (skips the entry)."""
    try:
        return int(s)
    except ValueError:
        return None


def _parse_image_schedule(schedule: str) -> list[tuple[float, float | None, int]]:
    """Parse a timestamp-based image schedule (values are integer indices)."""
    return _parse_schedule_generic(schedule, _safe_int)


def _match_image_schedule(
    entries: list[tuple[float, float | None, int]], current_time: float
) -> int:
    """Find the matching image index for the given time. Last match wins."""
    return _match_schedule_generic(entries, current_time, 0)


def _match_image_schedule_with_next(
    entries: list[tuple[float, float | None, int]],
    current_time: float,
    blend_seconds: float,
) -> tuple[int, int, float]:
    """Find current image index, next image index, and blend factor."""
    return _match_schedule_with_next_generic(entries, current_time, blend_seconds, 0)


class AudioLoopController(io.ComfyNode):
    """Computes start_index, stop signal, and iteration seed for audio-conditioned
    video extension loops.

    Wire current_iteration from TensorLoopOpen, connect the audio track, and
    this node outputs the correct start_index for TrimAudioDuration, a
    should_stop boolean for TensorLoopClose, and a per-iteration seed.
    No manual constants needed -- audio duration is read directly from the tensor.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioLoopController",
            display_name="Audio Loop Controller",
            category="looping/audio",
            description=(
                "Computes start_index, auto-stop signal, and per-iteration seed "
                "for audio-conditioned video extension loops. Reads audio duration "
                "directly from the tensor so no manual constants are needed."
            ),
            inputs=[
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration (1-based) from TensorLoopOpen.",
                ),
                io.Float.Input(
                    "window_seconds",
                    default=19.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Duration of each video generation window in seconds.",
                ),
                io.Float.Input(
                    "overlap_seconds",
                    default=1.0,
                    min=0.0,
                    step=0.01,
                    tooltip=(
                        "Overlap between consecutive windows in seconds. "
                        "Stride is computed as window - overlap. "
                        "overlap_frames output auto-computes the frame count at 25fps."
                    ),
                ),
                io.Audio.Input("audio", tooltip="The audio track being used for generation."),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Base seed. Output iteration_seed = seed + current_iteration.",
                ),
                io.Int.Input(
                    "fps",
                    default=25,
                    min=1,
                    tooltip="Video frame rate. Used to compute overlap_frames output.",
                ),
            ],
            outputs=[
                io.Float.Output(
                    "start_index",
                    tooltip="Start time in seconds for this iteration's audio window.",
                ),
                io.Boolean.Output(
                    "should_stop",
                    tooltip=(
                        "True when the next iteration would overshoot the audio. "
                        "Wire to TensorLoopClose's stop input."
                    ),
                ),
                io.Float.Output(
                    "audio_duration",
                    tooltip="Total duration of the input audio in seconds.",
                ),
                io.Int.Output(
                    "iteration_seed",
                    tooltip="seed + current_iteration. Wire to extension's noise_seed.",
                ),
                io.Float.Output(
                    "stride_seconds",
                    tooltip="Computed stride (window - overlap). Wire to TimestampPromptSchedule and AudioLoopPlanner.",
                ),
                io.Int.Output(
                    "overlap_frames",
                    tooltip="overlap_seconds * fps. Wire to extension component's overlap_frames input.",
                ),
                io.Int.Output(
                    "overlap_latent_frames",
                    tooltip="overlap_frames in latent space ((pixel-1)//8+1). Wire to LatentContextExtract / LatentOverlapTrim.",
                ),
                io.Float.Output(
                    "overlap_seconds",
                    tooltip="Pass-through of overlap_seconds input. Wire to Extension subgraph's video_start_time on LTXVAudioVideoMask.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        current_iteration: int,
        window_seconds: float,
        overlap_seconds: float,
        audio: dict,
        seed: int,
        fps: int,
    ) -> io.NodeOutput:
        audio_duration = _audio_duration(audio)
        stride = window_seconds - overlap_seconds

        start_index = current_iteration * stride

        # Clamp start_index so TrimAudioDuration always has enough audio
        # for the mel spectrogram (needs >1024 samples). Without this,
        # the loop body crashes on the final iteration because
        # TensorLoopClose checks should_stop AFTER the body executes.
        min_audio_seconds = 0.5  # ~22050 samples at 44.1kHz, well above mel minimum
        max_start = max(0.0, audio_duration - min_audio_seconds)
        start_index = min(start_index, max_start)

        # Stop if the NEXT iteration would start past the audio.
        next_start = (current_iteration + 1) * stride
        should_stop = next_start >= audio_duration

        iteration_seed = seed + current_iteration
        overlap_frames = round(overlap_seconds * fps)

        # Video VAE: first pixel frame → 1 latent frame, then 1 latent per 8 pixels.
        # Formula: latent = (pixel - 1) // scale + 1. Matches vae.downscale_index_formula.
        overlap_latent_frames = (overlap_frames - 1) // LTX_TEMPORAL_SCALE + 1

        return io.NodeOutput(start_index, should_stop, float(audio_duration), iteration_seed, stride, overlap_frames, overlap_latent_frames, overlap_seconds)


class TimestampPromptSchedule(io.ComfyNode):
    """Selects a prompt based on the current audio position using a timestamp schedule.

    Write prompts for different sections of your song using timestamps you
    already know (verse, chorus, bridge). The node computes the current
    audio position from the iteration number and stride, then returns the
    matching prompt.

    When blend_seconds > 0, also outputs the next_prompt and a blend_factor
    for smooth transitions. Wire both prompts through text encoders into
    ConditioningBlend for gradual prompt transitions.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TimestampPromptSchedule",
            display_name="Timestamp Prompt Schedule",
            category="looping/audio",
            description=(
                "Selects a prompt based on the current audio position. "
                "Write timestamp-based schedules matching your song structure. "
                "Supports gradual blending between prompts at transitions."
            ),
            inputs=[
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration (1-based) from TensorLoopOpen.",
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per iteration (same as AudioLoopController).",
                ),
                io.String.Input(
                    "schedule",
                    default="0:00+: default prompt",
                    multiline=True,
                    tooltip=(
                        "Timestamp-based prompt schedule. One entry per line.\n"
                        "Formats:\n"
                        "  0:00-0:38: prompt for this range\n"
                        "  0:38-1:15: prompt for chorus\n"
                        "  1:15+: prompt from here onward\n"
                        "Timestamps: M:SS, M:SS.ss, or bare seconds."
                    ),
                ),
                io.Float.Input(
                    "blend_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.5,
                    tooltip=(
                        "Transition duration in seconds. 0 = hard switch (default). "
                        "Set to e.g. 5.0 to blend over ~5 seconds before each boundary. "
                        "Wire next_prompt and blend_factor to ConditioningBlend."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("prompt", tooltip="The prompt for this iteration's audio position."),
                io.String.Output("next_prompt", tooltip="The upcoming prompt at the next boundary. Same as prompt when not near a transition."),
                io.Float.Output("blend_factor", tooltip="0.0 = fully current prompt, ramps to 1.0 at the boundary. Wire to ConditioningBlend."),
                io.Float.Output("current_time", tooltip="Current position in seconds."),
            ],
        )

    @classmethod
    def execute(
        cls,
        current_iteration: int,
        stride_seconds: float,
        schedule: str,
        blend_seconds: float,
    ) -> io.NodeOutput:
        current_time = current_iteration * stride_seconds
        entries = _parse_schedule(schedule)
        prompt, next_prompt, blend_factor = _match_schedule_with_next(
            entries, current_time, blend_seconds
        )
        return io.NodeOutput(prompt, next_prompt, blend_factor, current_time)


class AudioLoopPlanner(io.ComfyNode):
    """Shows the iteration timeline for planning prompt schedules.

    Connect the same audio/stride/window as AudioLoopController and this
    node outputs a text summary of all iteration boundaries with timestamps.
    Leave it in the workflow -- it auto-updates when inputs change.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioLoopPlanner",
            display_name="Audio Loop Planner",
            category="looping/audio",
            description=(
                "Shows iteration timeline with timestamps. "
                "Helps you write prompt schedules by showing what time each iteration covers."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="The audio track."),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per iteration.",
                ),
                io.Float.Input(
                    "window_seconds",
                    default=19.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Video generation window per iteration.",
                ),
            ],
            outputs=[
                io.String.Output("summary", tooltip="Iteration timeline text."),
                io.Int.Output("total_iterations", tooltip="Estimated total iteration count."),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        stride_seconds: float,
        window_seconds: float,
    ) -> io.NodeOutput:
        audio_duration = _audio_duration(audio)

        iterations = _compute_tile_count(audio_duration, stride_seconds)

        lines = [
            f"Audio: {audio_duration:.1f}s ({_format_timestamp(audio_duration)})",
            f"Stride: {stride_seconds:.2f}s | Window: {window_seconds:.2f}s",
            f"Overlap: {window_seconds - stride_seconds:.2f}s",
            f"Estimated {iterations} iterations:",
            "",
            f"  Initial:  {_format_timestamp(0)} - {_format_timestamp(window_seconds)}"
            f"  (0.0s - {window_seconds:.1f}s)  [uses static prompt, not schedule]",
        ]
        for i in range(1, iterations + 1):
            start = i * stride_seconds
            end = start + window_seconds
            lines.append(
                f"  Iter {i:2d}:  {_format_timestamp(start)} - {_format_timestamp(end)}"
                f"  ({start:.1f}s - {end:.1f}s)"
            )

        return io.NodeOutput("\n".join(lines), iterations)


class ScheduleToMultiPrompt(io.ComfyNode):
    """Converts a timestamp-based schedule into a pipe-separated prompt list
    for LTXVLoopingSampler's MultiPromptProvider.

    Computes how many temporal tiles the audio needs, then maps each tile's
    midpoint to the matching schedule entry. Outputs a single string with
    prompts separated by | (one per tile).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ScheduleToMultiPrompt",
            display_name="Schedule to Multi-Prompt",
            category="looping/audio",
            description=(
                "Converts a timestamp schedule into pipe-separated prompts "
                "for LTXVLoopingSampler via MultiPromptProvider. One prompt per temporal tile."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="The audio track (for duration)."),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per tile. Wire from AudioLoopController.",
                ),
                io.Float.Input(
                    "window_seconds",
                    default=19.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Temporal tile size in seconds.",
                ),
                io.String.Input(
                    "schedule",
                    default="0:00+: default prompt",
                    multiline=True,
                    tooltip="Timestamp-based schedule (same format as TimestampPromptSchedule).",
                ),
            ],
            outputs=[
                io.String.Output("prompts", tooltip="Pipe-separated prompts, one per tile. Wire to MultiPromptProvider."),
                io.Int.Output("tile_count", tooltip="Number of temporal tiles."),
                io.String.Output("tile_map", tooltip="Debug: shows which prompt maps to which tile."),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        stride_seconds: float,
        window_seconds: float,
        schedule: str,
    ) -> io.NodeOutput:
        audio_duration = _audio_duration(audio)
        tile_count = _compute_tile_count(audio_duration, stride_seconds)
        entries = _parse_schedule(schedule)

        # Tiles are 1-based (matching AudioLoopPlanner/Controller)
        prompts = []
        tile_map_lines = []
        for i in range(1, tile_count + 1):
            tile_start = i * stride_seconds
            tile_mid = tile_start + window_seconds / 2
            prompt = _match_schedule(entries, tile_mid)
            prompts.append(prompt)
            label = (prompt[:60] + "...") if len(prompt) > 60 else prompt
            tile_map_lines.append(
                f"Tile {i}: {_format_timestamp(tile_start)}-{_format_timestamp(tile_start + window_seconds)} -> {label}"
            )

        prompt_string = " | ".join(prompts)
        tile_map = "\n".join(tile_map_lines)

        return io.NodeOutput(prompt_string, tile_count, tile_map)


class AudioDuration(io.ComfyNode):
    """Returns the duration of an audio tensor in seconds."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioDuration",
            display_name="Audio Duration",
            category="audio",
            description="Returns the duration of an audio tensor in seconds.",
            inputs=[
                io.Audio.Input("audio"),
            ],
            outputs=[
                io.Float.Output("duration_seconds"),
                io.Int.Output("sample_rate"),
                io.Int.Output("total_samples"),
            ],
        )

    @classmethod
    def execute(cls, audio: dict) -> io.NodeOutput:
        duration = _audio_duration(audio)
        return io.NodeOutput(float(duration), int(audio["sample_rate"]), int(audio["waveform"].shape[-1]))


class ConditioningBlend(io.ComfyNode):
    """Blends two conditionings with a factor. Works with any text encoder
    including LTX 2.3 Gemma 3 (no pooled_output required).

    When blend_factor = 0.0, passes conditioning_a through unchanged.
    When blend_factor = 1.0, passes conditioning_b through unchanged.
    Values between lerp the conditioning tensors.

    Wire TimestampPromptSchedule's blend_factor here for smooth transitions.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConditioningBlend",
            display_name="Conditioning Blend",
            category="looping/audio",
            description=(
                "Blends two conditionings with a factor. Works with LTX Gemma 3 "
                "Also compatible with CLIP conditioning. Use with TimestampPromptSchedule "
                "for smooth prompt transitions."
            ),
            inputs=[
                io.Conditioning.Input("conditioning_a", tooltip="Current prompt conditioning."),
                io.Conditioning.Input("conditioning_b", tooltip="Next prompt conditioning."),
                io.Float.Input(
                    "blend_factor",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0.0 = all A, 1.0 = all B. Wire from TimestampPromptSchedule.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning_a: list,
        conditioning_b: list,
        blend_factor: float,
    ) -> io.NodeOutput:
        # Passthrough when no blending needed
        if blend_factor <= 0.0:
            return io.NodeOutput(conditioning_a)
        if blend_factor >= 1.0:
            return io.NodeOutput(conditioning_b)

        out = []
        # Uses only conditioning_b[0] -- LTX Gemma 3 produces single-element conditioning.
        # For multi-element CLIP conditioning, this would need zip/min indexing.
        cond_b = conditioning_b[0][0]

        for i in range(len(conditioning_a)):
            t_a = conditioning_a[i][0]
            t_b = cond_b

            # Align sequence lengths by zero-padding the shorter one
            if t_b.shape[1] < t_a.shape[1]:
                t_b = torch.cat([t_b, torch.zeros((1, t_a.shape[1] - t_b.shape[1], t_b.shape[2]), device=t_b.device)], dim=1)
            elif t_a.shape[1] < t_b.shape[1]:
                t_a = torch.cat([t_a, torch.zeros((1, t_b.shape[1] - t_a.shape[1], t_a.shape[2]), device=t_a.device)], dim=1)

            # Lerp the conditioning tensors
            blended = t_a * (1.0 - blend_factor) + t_b * blend_factor

            # Copy metadata from conditioning_a, blend pooled_output if present
            opts = conditioning_a[i][1].copy()
            pooled_a = conditioning_a[i][1].get("pooled_output", None)
            pooled_b = conditioning_b[0][1].get("pooled_output", None)
            if pooled_a is not None and pooled_b is not None:
                opts["pooled_output"] = pooled_a * (1.0 - blend_factor) + pooled_b * blend_factor

            # Combine attention masks (OR -- valid if either is valid)
            mask_a = conditioning_a[i][1].get("attention_mask", None)
            mask_b = conditioning_b[0][1].get("attention_mask", None)
            if mask_a is not None and mask_b is not None:
                # Pad masks to same length
                max_len = max(mask_a.shape[-1], mask_b.shape[-1])
                if mask_a.shape[-1] < max_len:
                    mask_a = torch.cat([mask_a, torch.zeros((*mask_a.shape[:-1], max_len - mask_a.shape[-1]), device=mask_a.device)], dim=-1)
                if mask_b.shape[-1] < max_len:
                    mask_b = torch.cat([mask_b, torch.zeros((*mask_b.shape[:-1], max_len - mask_b.shape[-1]), device=mask_b.device)], dim=-1)
                opts["attention_mask"] = torch.clamp(mask_a + mask_b, 0, 1)

            out.append([blended, opts])

        return io.NodeOutput(out)


class LatentContextExtract(io.ComfyNode):
    """Extracts the last N latent frames as context for the next loop iteration.

    Replaces LTXVSelectLatents + StripLatentNoiseMask in the latent-space loop.
    Takes the tail frames and strips noise_mask so LTXVAudioVideoMask creates
    a fresh mask (matching VAEEncode behavior from the IMAGE workflow).

    Wire: TensorLoopOpen previous_value → this → LTXVAudioVideoMask video_latent
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentContextExtract",
            display_name="Latent Context Extract",
            category="looping/audio",
            description=(
                "Extracts last N latent frames as context for the next loop iteration. "
                "Strips noise_mask for clean sampler behavior."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Previous iteration's video latent."),
                io.Int.Input("overlap_latent_frames", default=4, min=1,
                             tooltip="Number of tail latent frames to extract. Wire from AudioLoopController."),
            ],
            outputs=[
                io.Latent.Output("context", tooltip="Clean context latent (no noise_mask). Wire to LTXVAudioVideoMask."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict, overlap_latent_frames: int) -> io.NodeOutput:
        with torch.profiler.record_function("LatentContextExtract"):
            s = latent.copy()
            video = s["samples"]
            frames = video.shape[2]

            start = max(0, frames - overlap_latent_frames)
            s["samples"] = video[:, :, start:]

            # Strip noise_mask so downstream creates fresh (matches VAEEncode behavior)
            s.pop("noise_mask", None)

        return io.NodeOutput(s)


class LatentOverlapTrim(io.ComfyNode):
    """Trims the first N latent frames (overlap region) from a sampler's output.

    Replaces LTXVSelectLatents for output trimming in the latent-space loop.
    Keeps everything after the overlap region, strips noise_mask.

    Wire: LTXVCropGuides latent → this → subgraph output
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentOverlapTrim",
            display_name="Latent Overlap Trim",
            category="looping/audio",
            description=(
                "Trims first N latent frames (overlap) from sampler output. "
                "Keeps new content only."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Sampler output video latent (after CropGuides)."),
                io.Int.Input("overlap_latent_frames", default=4, min=0,
                             tooltip="Number of leading latent frames to trim. Wire from AudioLoopController."),
            ],
            outputs=[
                io.Latent.Output("trimmed", tooltip="New content only (overlap removed)."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict, overlap_latent_frames: int) -> io.NodeOutput:
        with torch.profiler.record_function("LatentOverlapTrim"):
            s = latent.copy()
            video = s["samples"]

            # Clamp to avoid empty tensor if overlap >= total frames
            trim = min(overlap_latent_frames, video.shape[2] - 1)
            s["samples"] = video[:, :, trim:]

            # Strip noise_mask for clean accumulation
            s.pop("noise_mask", None)

        return io.NodeOutput(s)


class StripLatentNoiseMask(io.ComfyNode):
    """Removes noise_mask from a latent dict so downstream nodes create fresh masks.

    Low-level utility. Prefer LatentContextExtract or LatentOverlapTrim which
    handle this automatically.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="StripLatentNoiseMask",
            display_name="Strip Latent Noise Mask",
            category="latent",
            description="Removes noise_mask from latent so downstream nodes create fresh masks.",
            inputs=[io.Latent.Input("latent")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent: dict) -> io.NodeOutput:
        out = latent.copy()
        out.pop("noise_mask", None)
        return io.NodeOutput(out)


class KeyframeImageSchedule(io.ComfyNode):
    """Selects a keyframe image based on the current audio position using a
    timestamp schedule, analogous to how TimestampPromptSchedule selects prompts.

    Write a schedule mapping time ranges to image indices (0-based into the
    input IMAGE batch). The node picks the right keyframe each iteration so
    different song sections can use different reference images.

    When blend_seconds > 0, outputs next_image and blend_factor for smooth
    visual transitions via ImageBlend.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KeyframeImageSchedule",
            display_name="Keyframe Image Schedule",
            category="looping/audio",
            description=(
                "Selects a keyframe image based on the current audio position. "
                "Maps timestamp ranges to image indices for per-iteration visual grounding. "
                "Supports gradual blending between keyframes at transitions."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Batch of keyframe images. Index 0 = first image in batch.",
                ),
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration from TensorLoopOpen (0 = initial render).",
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per iteration (same as AudioLoopController).",
                ),
                io.String.Input(
                    "schedule",
                    default="0:00+: 0",
                    multiline=True,
                    tooltip=(
                        "Timestamp-to-image-index schedule. One entry per line.\n"
                        "Formats:\n"
                        "  0:00-0:38: 0\n"
                        "  0:38-1:15: 1\n"
                        "  1:15+: 2\n"
                        "Values are 0-based image indices into the batch."
                    ),
                ),
                io.Float.Input(
                    "blend_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.5,
                    tooltip=(
                        "Transition duration in seconds. 0 = hard switch (default). "
                        "Set to e.g. 5.0 to blend over ~5 seconds before each boundary. "
                        "Wire next_image and blend_factor to ImageBlend."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Keyframe image for this iteration."),
                io.Image.Output("next_image", tooltip="Upcoming keyframe at next boundary. Same as image when not near a transition."),
                io.Float.Output("blend_factor", tooltip="0.0 = fully current image, ramps to 1.0 at the boundary. Wire to ImageBlend."),
                io.Float.Output("current_time", tooltip="Current position in seconds."),
                io.Int.Output("image_index", tooltip="Which image index was selected."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        current_iteration: int,
        stride_seconds: float,
        schedule: str,
        blend_seconds: float,
    ) -> io.NodeOutput:
        current_time = current_iteration * stride_seconds
        entries = _parse_image_schedule(schedule)
        batch_size = images.shape[0]

        current_idx, next_idx, blend_factor = _match_image_schedule_with_next(
            entries, current_time, blend_seconds
        )

        # Clamp indices to valid range
        current_idx = max(0, min(current_idx, batch_size - 1))
        next_idx = max(0, min(next_idx, batch_size - 1))

        image = images[current_idx : current_idx + 1]
        next_image = images[next_idx : next_idx + 1]

        return io.NodeOutput(image, next_image, blend_factor, current_time, current_idx)


class VideoFrameExtract(io.ComfyNode):
    """Extracts the frame from a reference video/image batch at the current
    iteration's timestamp. Enables video-to-video style transfer by using
    reference video frames as per-iteration guides.

    Wire the output image to the subgraph's init_image input to ground
    each iteration in the corresponding reference frame.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VideoFrameExtract",
            display_name="Video Frame Extract",
            category="looping/audio",
            description=(
                "Extracts a frame from a reference video at the current iteration's timestamp. "
                "Enables video-to-video style transfer across full songs."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Reference video as an image batch.",
                ),
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration from TensorLoopOpen (0 = initial render).",
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per iteration (same as AudioLoopController).",
                ),
                io.Float.Input(
                    "source_fps",
                    default=25.0,
                    min=0.01,
                    step=0.01,
                    tooltip="Frame rate of the source video batch.",
                ),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Single frame at the matching timestamp."),
                io.Int.Output("frame_index", tooltip="Which frame index was extracted."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        current_iteration: int,
        stride_seconds: float,
        source_fps: float,
    ) -> io.NodeOutput:
        current_time = current_iteration * stride_seconds
        frame_index = round(current_time * source_fps)
        batch_size = images.shape[0]

        # Clamp to valid range
        frame_index = max(0, min(frame_index, batch_size - 1))

        image = images[frame_index : frame_index + 1]
        return io.NodeOutput(image, frame_index)


class ImageBlend(io.ComfyNode):
    """Blends two images with a factor. Pairs with KeyframeImageSchedule
    for smooth visual transitions between keyframes.

    When blend_factor = 0.0, passes image_a through unchanged.
    When blend_factor = 1.0, passes image_b through unchanged.
    Values between lerp the pixel values.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ImageBlend_AudioLoop",
            display_name="Image Blend",
            category="looping/audio",
            description=(
                "Blends two images with a factor. Use with KeyframeImageSchedule "
                "for smooth transitions between keyframes."
            ),
            inputs=[
                io.Image.Input("image_a", tooltip="Current keyframe image."),
                io.Image.Input("image_b", tooltip="Next keyframe image."),
                io.Float.Input(
                    "blend_factor",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0.0 = all A, 1.0 = all B. Wire from KeyframeImageSchedule.",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        blend_factor: float,
    ) -> io.NodeOutput:
        if blend_factor <= 0.0:
            return io.NodeOutput(image_a)
        if blend_factor >= 1.0:
            return io.NodeOutput(image_b)

        blended = image_a * (1.0 - blend_factor) + image_b * blend_factor
        return io.NodeOutput(blended)


# Module-level LRU cache for CachedTextEncode.
# Persists across loop iterations (our goal) and across workflow runs.
# Keyed on (id(clip), text). Bounded so long-running sessions don't grow
# unbounded VRAM from cached CONDITIONING tensors.
#
# Hazard: id(clip) can be recycled by CPython if the original CLIP is freed
# and a new object lands at the same memory address. In practice CLIP models
# are large (>10GB) and stay resident across iterations, so this is a latent
# risk rather than an observed bug. If ghost hits ever appear, switch to
# weakref-based keying.
_COND_CACHE: OrderedDict = OrderedDict()
_COND_CACHE_MAX = 20


class CachedTextEncode(io.ComfyNode):
    """Drop-in replacement for CLIPTextEncode that caches conditioning by
    (clip, text). On cache hit, skips tokenize + encode entirely.

    Speedup is significant for LTX 2.3 Gemma 3 12B: TimestampPromptSchedule
    emits the same prompt string across multiple iterations when a schedule
    range covers more than one iteration (e.g. "0:00-0:38: ..." at stride 19s
    covers iterations 0-2). Without caching, Gemma re-encodes the identical
    text each time.

    The cache is module-level and bounded (LRU, max 20 entries). Each entry
    holds a CONDITIONING tensor on GPU; 20 entries at ~16MB is ~320MB --
    negligible next to the 22B DiT.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CachedTextEncode_AudioLoop",
            display_name="Cached Text Encode",
            category="looping/audio",
            description=(
                "CLIPTextEncode with an LRU cache keyed on (clip, text). "
                "Skips re-encoding when the same prompt is used across "
                "multiple loop iterations. Drop-in replacement for CLIPTextEncode."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP model (Gemma 3 for LTX 2.3)."),
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Prompt text to encode. Identical text + same CLIP hits the cache.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
            ],
        )

    @classmethod
    def execute(cls, clip, text: str) -> io.NodeOutput:
        key = (id(clip), text)
        cached = _COND_CACHE.get(key)
        if cached is not None:
            _COND_CACHE.move_to_end(key)
            return io.NodeOutput(cached)

        # Only the miss path hits GPU (Gemma encode) -- that's the only
        # branch worth a named span in the profile trace.
        with torch.profiler.record_function("CachedTextEncode.miss"):
            tokens = clip.tokenize(text)
            cond = clip.encode_from_tokens_scheduled(tokens)
            _COND_CACHE[key] = cond
            if len(_COND_CACHE) > _COND_CACHE_MAX:
                _COND_CACHE.popitem(last=False)
        return io.NodeOutput(cond)


class IterationCleanup(io.ComfyNode):
    """LATENT passthrough that runs PyTorch allocator hygiene as a side
    effect. Place in the subgraph output path so every iteration ends with
    a clean allocator state.

    comfy-aimdo's README recommends flushing the caching allocator between
    model runs to prevent fragmentation. This node is the idiomatic way to
    do that inside a TensorLoop iteration.

    Modes:
      - always:   gc.collect() + torch.cuda.empty_cache() (default)
      - gpu_only: only torch.cuda.empty_cache() (skips Python gc pass)
      - never:    passthrough only, no side effects
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IterationCleanup",
            display_name="Iteration Cleanup",
            category="looping/audio",
            description=(
                "LATENT passthrough that flushes the PyTorch caching allocator "
                "and runs Python gc. Reduces fragmentation across loop iterations."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Latent to pass through unchanged."),
                io.Combo.Input(
                    "mode",
                    options=["always", "gpu_only", "never"],
                    default="always",
                    tooltip=(
                        "always: gc + empty_cache. "
                        "gpu_only: empty_cache only. "
                        "never: passthrough (disables the cleanup)."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, latent, mode: str) -> io.NodeOutput:
        with torch.profiler.record_function("IterationCleanup"):
            if mode == "always":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif mode == "gpu_only":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return io.NodeOutput(latent)


# --- Profiling nodes ---
#
# Three coordinated nodes capture end-to-end profile data for the audio loop:
#   ProfileBegin    -> placed before the loop, starts torch.profiler
#   ProfileIterStep -> placed inside the loop body, marks iteration boundaries
#   ProfileEnd      -> placed after the loop, finalizes and writes outputs
#
# All settings live on ProfileBegin. ProfileIterStep and ProfileEnd have zero
# widgets -- they read shared state from _PROFILER_STATE. Toggle off via the
# `enabled` widget on ProfileBegin (master switch), or via ComfyUI's native
# node bypass (mode=4 on any of the three).
# Profiler state MUST survive ComfyUI-HotReloadHack reimports of this module.
# Module-level globals here would be reset mid-workflow if any file in our
# package changes (file mtime, git pull, IDE autosave). Attaching to `torch`
# (which never hot-reloads) keeps the state reachable even after our module
# is reimported -- ProfileBegin / ProfileIterStep / ProfileEnd then coordinate
# through a single live dict instead of three stale module copies.
_STATE_ATTR = "_audioloophelper_profiler_state"
_WARNED_ATTR = "_audioloophelper_warned_keys"


def _get_profiler_state() -> dict:
    state = getattr(torch, _STATE_ATTR, None)
    if state is None:
        state = {}
        setattr(torch, _STATE_ATTR, state)
    return state


def _get_warned_keys() -> set:
    warned = getattr(torch, _WARNED_ATTR, None)
    if warned is None:
        warned = set()
        setattr(torch, _WARNED_ATTR, warned)
    return warned


# Backward-compat names for tests / imports. These point to the same live
# objects attached to torch, so `.clear()` in tests works correctly.
_PROFILER_STATE = _get_profiler_state()
_WARNED_KEYS = _get_warned_keys()


def _log_once(key: str, message: str) -> None:
    """Emit a warning message once per key per Python process."""
    warned = _get_warned_keys()
    if key in warned:
        return
    warned.add(key)
    print(f"[AudioLoopHelper] {message}")


class ProfileBegin(io.ComfyNode):
    """Start torch.profiler before the audio loop.

    Place this node between the audio/model loaders and TensorLoopOpen.
    The `trigger` input is any value you want to pass through -- it exists
    only to force this node into the execution order before the loop.

    All profile settings live on this node. ProfileIterStep and ProfileEnd
    read shared state, so you only change settings here.

    Toggle off in three ways:
      1. Set `enabled=False` (zero overhead)
      2. Right-click bypass this node (mode=4)
      3. Remove all three profile nodes from the workflow
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProfileBegin_AudioLoop",
            display_name="Profile Begin",
            category="looping/audio/profile",
            description=(
                "Starts torch.profiler for end-to-end audio-loop profiling. "
                "Use with ProfileIterStep (inside loop) and ProfileEnd (after loop)."
            ),
            inputs=[
                io.AnyType.Input("trigger", tooltip="Any value to sequence this node before the loop (passthrough)."),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="Master on/off. False = all three profile nodes are passthroughs with zero overhead.",
                ),
                io.String.Input(
                    "output_dir",
                    default="profile_output",
                    tooltip=(
                        "Root dir for profile outputs. Relative paths resolve "
                        "against the ComfyUI-AudioLoopHelper plugin folder "
                        "(gitignored). Use an absolute path to write elsewhere. "
                        "A timestamped subdir is created per run."
                    ),
                ),
                io.Int.Input(
                    "warmup_iterations",
                    default=1,
                    min=0,
                    max=10,
                    tooltip="Skip this many iterations before recording (iteration 1 has compilation noise).",
                ),
                io.Int.Input(
                    "active_iterations",
                    default=3,
                    min=1,
                    max=20,
                    tooltip="Record this many iterations after warmup. More = better variance data, larger files.",
                ),
                io.Boolean.Input(
                    "include_cpu",
                    default=True,
                    tooltip="Profile CPU activities too (Python overhead, dispatcher cost). Adds ~10% overhead.",
                ),
                io.Boolean.Input(
                    "include_memory",
                    default=True,
                    tooltip="Record VRAM allocation timeline. Adds ~3% overhead.",
                ),
                io.Boolean.Input(
                    "include_shapes",
                    default=True,
                    tooltip="Record tensor shapes per op. Helps identify which layer is slow. Adds ~5% overhead.",
                ),
                io.Boolean.Input(
                    "include_flops",
                    default=False,
                    tooltip="Count FLOPS per op. Expensive; enable only for deeper analysis.",
                ),
            ],
            outputs=[
                io.AnyType.Output("trigger", tooltip="Passthrough of input trigger."),
            ],
        )

    @classmethod
    def execute(
        cls,
        trigger,
        enabled: bool,
        output_dir: str,
        warmup_iterations: int,
        active_iterations: int,
        include_cpu: bool,
        include_memory: bool,
        include_shapes: bool,
        include_flops: bool,
    ) -> io.NodeOutput:
        state = _get_profiler_state()

        # Stop any prior profiler that was left running (user cancelled a run
        # before ProfileEnd fired, ComfyUI workflow re-queued, etc.) to avoid
        # orphaning an active torch.profiler that keeps collecting invisibly.
        prior = state.get("profiler")
        if prior is not None:
            try:
                prior.stop()
            except Exception:  # noqa: BLE001 -- torch.profiler errors are unhelpful
                pass
        state.clear()

        if not enabled:
            return io.NodeOutput(trigger)

        # Torch profiler only meaningful with CUDA; guard gracefully.
        if not torch.cuda.is_available():
            _log_once("no_cuda", "ProfileBegin: CUDA not available, profiling disabled.")
            return io.NodeOutput(trigger)

        import datetime
        from pathlib import Path

        # Resolve relative output_dir against the plugin folder so profile
        # data lands alongside our code (and is covered by our .gitignore)
        # rather than wherever ComfyUI happened to be launched from.
        out_root = Path(output_dir)
        if not out_root.is_absolute():
            plugin_dir = Path(__file__).resolve().parent
            out_root = plugin_dir / out_root

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CUDA]
        if include_cpu:
            activities.append(torch.profiler.ProfilerActivity.CPU)

        schedule = torch.profiler.schedule(
            wait=0,
            warmup=warmup_iterations,
            active=active_iterations,
            repeat=1,
        )

        profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=include_shapes,
            profile_memory=include_memory,
            with_flops=include_flops,
            with_stack=False,
            acc_events=True,  # retain events across cycle transitions
        )
        profiler.start()

        state["profiler"] = profiler
        state["run_dir"] = run_dir
        state["settings"] = {
            "warmup_iterations": warmup_iterations,
            "active_iterations": active_iterations,
            "include_cpu": include_cpu,
            "include_memory": include_memory,
            "include_shapes": include_shapes,
            "include_flops": include_flops,
        }
        print(f"[AudioLoopHelper] ProfileBegin: recording to {run_dir}")
        return io.NodeOutput(trigger)


class ProfileIterStep(io.ComfyNode):
    """Mark an iteration boundary for torch.profiler.

    Place inside the TensorLoop body (typically after LatentOverlapTrim or
    IterationCleanup). Calls profiler.step() to advance its schedule.

    No widgets -- settings are shared from ProfileBegin. Passthrough when
    ProfileBegin isn't active or was set to disabled.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProfileIterStep_AudioLoop",
            display_name="Profile Iter Step",
            category="looping/audio/profile",
            description="Calls torch.profiler.step() at iteration boundary. Passthrough LATENT.",
            inputs=[
                io.Latent.Input("latent", tooltip="Latent passed through unchanged."),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        profiler = _get_profiler_state().get("profiler")
        if profiler is None:
            _log_once(
                "step_uninit",
                "ProfileIterStep called without an active ProfileBegin -- passthrough. "
                "Wire a ProfileBegin node before the loop to enable profiling.",
            )
            return io.NodeOutput(latent)
        profiler.step()
        return io.NodeOutput(latent)


class ProfileEnd(io.ComfyNode):
    """Stop torch.profiler and write outputs.

    Place AFTER the TensorLoop completes. The `trigger` input exists only
    to sequence this node after the loop (pass any downstream value, e.g.,
    the TensorLoopClose output).

    Emits (in the timestamped dir from ProfileBegin):
      - trace.json        : chrome trace (open at perfetto.dev or chrome://tracing)
      - summary.txt       : top kernels by cumulative time, categorized
      - memory_timeline.html : VRAM timeline (if include_memory was True)

    Passthrough when ProfileBegin isn't active.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProfileEnd_AudioLoop",
            display_name="Profile End",
            category="looping/audio/profile",
            description="Stops torch.profiler and writes chrome trace + summary to disk.",
            inputs=[
                io.AnyType.Input("trigger", tooltip="Any value to sequence this node after the loop (passthrough)."),
            ],
            outputs=[
                io.AnyType.Output("trigger"),
            ],
        )

    @classmethod
    def execute(cls, trigger) -> io.NodeOutput:
        state = _get_profiler_state()
        profiler = state.get("profiler")
        if profiler is None:
            _log_once(
                "end_uninit",
                "ProfileEnd called without an active ProfileBegin -- passthrough.",
            )
            return io.NodeOutput(trigger)

        run_dir = state["run_dir"]
        settings = state["settings"]

        try:
            profiler.stop()

            # Write trace atomically: .tmp then rename, so a partial write
            # on disk-full / permission error doesn't leave a corrupt file.
            trace_path = run_dir / "trace.json"
            tmp_path = run_dir / "trace.json.tmp"
            try:
                profiler.export_chrome_trace(str(tmp_path))
                tmp_path.replace(trace_path)
            except (RuntimeError, OSError, ValueError) as e:
                _log_once("trace_export", f"ProfileEnd: trace export failed: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()

            try:
                summary = profiler.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=50,
                )
            except (RuntimeError, ValueError) as e:
                summary = f"Summary generation failed: {e}"
            (run_dir / "summary.txt").write_text(str(summary))

            if settings.get("include_memory"):
                try:
                    profiler.export_memory_timeline(
                        str(run_dir / "memory_timeline.html"),
                        device="cuda:0",
                    )
                except (RuntimeError, OSError, ValueError) as e:
                    _log_once("mem_timeline", f"ProfileEnd: memory_timeline export failed: {e}")

            print(f"[AudioLoopHelper] ProfileEnd: wrote profile to {run_dir}")
        finally:
            state.clear()
        return io.NodeOutput(trigger)


class AudioLoopHelperExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .nodes_analysis import AudioPitchDetect

        return [
            AudioLoopController,
            TimestampPromptSchedule,
            ConditioningBlend,
            AudioLoopPlanner,
            ScheduleToMultiPrompt,
            LatentContextExtract,
            LatentOverlapTrim,
            StripLatentNoiseMask,
            AudioDuration,
            AudioPitchDetect,
            KeyframeImageSchedule,
            VideoFrameExtract,
            ImageBlend,
            CachedTextEncode,
            IterationCleanup,
            ProfileBegin,
            ProfileIterStep,
            ProfileEnd,
        ]


def comfy_entrypoint() -> AudioLoopHelperExtension:
    return AudioLoopHelperExtension()
