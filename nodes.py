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
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


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

        return io.NodeOutput(start_index, should_stop, float(audio_duration), iteration_seed, stride, overlap_frames)


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

        # Last valid iteration N satisfies: (N+1)*stride < duration
        # => N < duration/stride - 1 => N = floor(duration/stride) - 1
        # Matches AudioLoopController's stop condition (next start >= duration)
        iterations = max(1, min(math.ceil(audio_duration / stride_seconds) - 1, 200))

        lines = [
            f"Audio: {audio_duration:.1f}s ({_format_timestamp(audio_duration)})",
            f"Stride: {stride_seconds:.2f}s | Window: {window_seconds:.2f}s",
            f"Overlap: {window_seconds - stride_seconds:.2f}s",
            f"Estimated {iterations} iterations:",
            "",
        ]
        for i in range(1, iterations + 1):
            start = i * stride_seconds
            end = start + window_seconds
            lines.append(
                f"  Iter {i:2d}:  {_format_timestamp(start)} - {_format_timestamp(end)}"
                f"  ({start:.1f}s - {end:.1f}s)"
            )

        return io.NodeOutput("\n".join(lines), iterations)


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
    including LTX 2.3's Gemma 3 (no pooled_output required).

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
                "and standard CLIP conditioning. Use with TimestampPromptSchedule "
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


class AudioLoopHelperExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AudioLoopController,
            TimestampPromptSchedule,
            ConditioningBlend,
            AudioLoopPlanner,
            AudioDuration,
        ]


def comfy_entrypoint() -> AudioLoopHelperExtension:
    return AudioLoopHelperExtension()
