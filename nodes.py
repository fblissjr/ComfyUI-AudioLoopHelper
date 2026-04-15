"""Audio-aware loop helper nodes for ComfyUI.

Provides nodes to automatically manage loop iteration timing against an
audio track, eliminating manual iteration count calculation and preventing
crashes from overshooting audio boundaries.

Built for use alongside ComfyUI-NativeLooping (TensorLoopOpen/Close),
ComfyUI-VideoHelperSuite, ComfyUI-KJNodes, and ComfyUI-MelBandRoFormer
for generating full-length music videos with LTX 2.3.
"""

import math
import re

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


def _parse_schedule(schedule: str) -> list[tuple[float, float | None, str]]:
    """Parse a timestamp-based prompt schedule.

    Each line: `timestamp_range: prompt text`
    Range formats:
      - "0:00-0:38: prompt"   (start-end, inclusive)
      - "1:15+: prompt"       (from here onward)
      - "38-75: prompt"       (bare seconds)

    Returns list of (start, end_or_None, prompt) tuples.
    """
    entries = []
    for line in schedule.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        range_part = match.group(1).strip()
        prompt = match.group(2).strip()

        if range_part.endswith("+"):
            start = _parse_timestamp(range_part[:-1])
            entries.append((start, None, prompt))
        elif "-" in range_part:
            parts = range_part.split("-", 1)
            start = _parse_timestamp(parts[0])
            end = _parse_timestamp(parts[1])
            entries.append((start, end, prompt))
        else:
            # Single timestamp -- treat as point match (this iteration only)
            t = _parse_timestamp(range_part)
            entries.append((t, t, prompt))
    return entries


def _match_schedule(entries: list[tuple[float, float | None, str]], current_time: float) -> str:
    """Find the matching prompt for the given time. Last match wins."""
    result = ""
    for start, end, prompt in entries:
        if end is None:
            if current_time >= start:
                result = prompt
        else:
            if start <= current_time <= end:
                result = prompt
    if not result and entries:
        result = entries[-1][2]
    return result


def _match_schedule_with_next(
    entries: list[tuple[float, float | None, str]],
    current_time: float,
    blend_seconds: float,
) -> tuple[str, str, float]:
    """Find current prompt, next prompt, and blend factor.

    Returns (current_prompt, next_prompt, blend_factor).
    blend_factor is 0.0 when not near a boundary, and ramps to 1.0
    at the boundary over blend_seconds.
    """
    current_prompt = _match_schedule(entries, current_time)

    if blend_seconds <= 0:
        return current_prompt, current_prompt, 0.0

    # Find the next boundary: the earliest range start that is after current_time
    next_boundary = None
    next_prompt = current_prompt
    for start, end, prompt in entries:
        if start > current_time:
            if next_boundary is None or start < next_boundary:
                next_boundary = start
                next_prompt = prompt

    if next_boundary is None:
        # No upcoming boundary -- we're in the last range
        return current_prompt, current_prompt, 0.0

    time_to_boundary = next_boundary - current_time
    if time_to_boundary < blend_seconds:
        blend_factor = 1.0 - (time_to_boundary / blend_seconds)
        return current_prompt, next_prompt, blend_factor

    return current_prompt, current_prompt, 0.0


def _parse_image_schedule(schedule: str) -> list[tuple[float, float | None, int]]:
    """Parse a timestamp-based image schedule.

    Same format as _parse_schedule but values are integer image indices
    instead of prompt strings.  Example:
        0:00-0:38: 0
        0:38-1:15: 1
        1:15+: 2
    """
    entries: list[tuple[float, float | None, int]] = []
    for line in schedule.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        range_part = match.group(1).strip()
        value = match.group(2).strip()

        try:
            idx = int(value)
        except ValueError:
            continue

        if range_part.endswith("+"):
            start = _parse_timestamp(range_part[:-1])
            entries.append((start, None, idx))
        elif "-" in range_part:
            parts = range_part.split("-", 1)
            start = _parse_timestamp(parts[0])
            end = _parse_timestamp(parts[1])
            entries.append((start, end, idx))
        else:
            t = _parse_timestamp(range_part)
            entries.append((t, t, idx))
    return entries


def _match_image_schedule(
    entries: list[tuple[float, float | None, int]], current_time: float
) -> int:
    """Find the matching image index for the given time. Last match wins."""
    result: int | None = None
    for start, end, idx in entries:
        if end is None:
            if current_time >= start:
                result = idx
        else:
            if start <= current_time <= end:
                result = idx
    if result is None and entries:
        result = entries[-1][2]
    return result if result is not None else 0


def _match_image_schedule_with_next(
    entries: list[tuple[float, float | None, int]],
    current_time: float,
    blend_seconds: float,
) -> tuple[int, int, float]:
    """Find current image index, next image index, and blend factor.

    Returns (current_idx, next_idx, blend_factor).
    blend_factor is 0.0 when not near a boundary, and ramps to 1.0
    at the boundary over blend_seconds.
    """
    current_idx = _match_image_schedule(entries, current_time)

    if blend_seconds <= 0:
        return current_idx, current_idx, 0.0

    next_boundary = None
    next_idx = current_idx
    for start, _end, idx in entries:
        if start > current_time:
            if next_boundary is None or start < next_boundary:
                next_boundary = start
                next_idx = idx

    if next_boundary is None:
        return current_idx, current_idx, 0.0

    time_to_boundary = next_boundary - current_time
    if time_to_boundary < blend_seconds:
        blend_factor = 1.0 - (time_to_boundary / blend_seconds)
        return current_idx, next_idx, blend_factor

    return current_idx, current_idx, 0.0


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
                    tooltip="Current loop iteration (1-based) from TensorLoopOpen.",
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
        ]


def comfy_entrypoint() -> AudioLoopHelperExtension:
    return AudioLoopHelperExtension()
