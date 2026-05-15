"""Config validator for AudioLoopController / LatentContextExtract setups.

Reports the integer-latent math (stride, overlap quantization, iteration
seams) and flags configuration footguns: invalid length values, widget
disagreement between length and window_seconds, resolution not div by 32/64,
seams landing on prompt-schedule boundaries, thin context on short windows.

Pure python. Designed to sit alongside AudioLoopPlanner on the widget side
of the graph; wire its `report` output to PreviewAny.
"""

try:
    from comfy_api.latest import io
except ImportError:
    # Match the stub pattern in nodes.py so tests can import without ComfyUI.
    class _Passthrough:
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

    io = _IOStub()


try:
    from .nodes import (
        LTX_TEMPORAL_SCALE,
        _audio_duration,
        _compute_loop_geometry,
        _compute_tile_count,
        _format_timestamp,
        _parse_image_schedule,
        _parse_schedule,
    )
except ImportError:
    from nodes import (  # type: ignore[no-redef]
        LTX_TEMPORAL_SCALE,
        _audio_duration,
        _compute_loop_geometry,
        _compute_tile_count,
        _format_timestamp,
        _parse_image_schedule,
        _parse_schedule,
    )


# Report line prefixes. Tests grep on these literals — don't rename without
# updating tests/test_config_validator.py.
_OK = "[OK]   "
_WARN = "[WARN] "
_ERR = "[ERROR]"

_RES_DIV32 = "div_by_32"
_RES_DIV64 = "div_by_64"


def _bracket(value: int, step: int, offset: int = 0) -> tuple[int, int]:
    """(floor, ceil) of the offset+step*k lattice bracketing `value`.

    step=8, offset=1 -> valid LTX lengths (1, 9, 17, ...).
    step=32, offset=0 -> resolution divisibility.
    """
    if value < offset:
        return (offset, offset + step)
    k = (value - offset) // step
    floor = offset + k * step
    if floor == value:
        return (value, value)
    return (floor, floor + step)


def _reachable_overlaps(window_px: int, fps: int) -> list[float]:
    """Effective overlaps are quantized to window_px - new_latents*8 seconds."""
    window_latents = (window_px - 1) // LTX_TEMPORAL_SCALE + 1
    values = []
    for n in range(1, window_latents):
        eff_px = window_px - n * LTX_TEMPORAL_SCALE
        if eff_px < 0:
            continue
        values.append(eff_px / fps)
    return sorted(values)


def _widget_range_for_overlap_px(overlap_px: int, fps: int) -> tuple[float, float]:
    """Widget band of `overlap_seconds` values yielding a given overlap_px."""
    # round(x * fps) = n means n - 0.5 <= x*fps < n + 0.5, so x is in
    # [(n-0.5)/fps, (n+0.5)/fps). Apply per 8-pixel latent band.
    if overlap_px <= 0:
        return (0.0, 0.5 / fps)
    k = (overlap_px - 1) // LTX_TEMPORAL_SCALE
    lo_px = k * LTX_TEMPORAL_SCALE + 1
    hi_px = (k + 1) * LTX_TEMPORAL_SCALE
    return ((lo_px - 0.5) / fps, (hi_px + 0.5) / fps)


class LoopConfigValidator(io.ComfyNode):
    """Cross-checks loop settings against LTX constraints and flags footguns.

    Shows the exact integer-latent math the AudioLoopController performs, so
    you can see where your widget values land before spending 45 minutes on
    a bad run. Optional inputs (length, width/height, schedule) light up
    additional checks when provided.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoopConfigValidator",
            display_name="Loop Config Validator",
            category="looping/audio",
            description=(
                "Validates AudioLoopController/loop settings against LTX 2.3 "
                "constraints. Shows integer-latent math + flags common footguns. "
                "Wire `report` to PreviewAny. `ok` is False if any ERROR fires."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="Same audio track the loop sees."),
                io.Float.Input(
                    "window_seconds",
                    default=19.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Same value you feed AudioLoopController.window_seconds.",
                ),
                io.Float.Input(
                    "overlap_seconds",
                    default=2.0,
                    min=0.0,
                    step=0.01,
                    tooltip="Same value you feed AudioLoopController.overlap_seconds (target).",
                ),
                io.Int.Input(
                    "fps",
                    default=24,
                    min=1,
                    tooltip="Video frame rate (LTX 2.3 training distribution).",
                ),
                io.Int.Input(
                    "length",
                    default=0,
                    min=0,
                    tooltip=(
                        "EmptyLTXVLatentVideo.length. 0 = skip check. "
                        "Enables: (length-1)%8 validity, length vs window*fps agreement."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    tooltip="Resolution width. 0 = skip. Enables div-by-32/64 check.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    tooltip="Resolution height. 0 = skip.",
                ),
                io.String.Input(
                    "schedule",
                    default="",
                    multiline=True,
                    tooltip=(
                        "TimestampPromptSchedule text. Empty = skip. "
                        "Enables: seam-on-boundary check (prompt cuts aligned to "
                        "iter seams is a documented failure mode)."
                    ),
                ),
                io.Combo.Input(
                    "resolution_rule",
                    options=[_RES_DIV32, _RES_DIV64],
                    default=_RES_DIV32,
                    tooltip=(
                        "LTX 2.3 single-stage / distilled-1.1 = div_by_32. "
                        "Two-stage distilled = div_by_64."
                    ),
                ),
                io.Float.Input(
                    "seam_tolerance_seconds",
                    default=0.2,
                    min=0.0,
                    step=0.01,
                    tooltip="Seams within this many seconds of a schedule boundary trigger a WARN.",
                ),
                io.String.Input(
                    "keyframe_schedule",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Schedule string from `KeyframeLatentScheduleBatchEncode` "
                        "(current) or legacy `KeyframeImageSchedule`. Empty = "
                        "skip keyframe checks.\n"
                        "Example:\n"
                        "  0:00-0:42: 0\n"
                        "  0:42-1:28: 1\n"
                        "  1:28+: 2"
                    ),
                ),
                io.Int.Input(
                    "keyframe_batch_size",
                    default=0,
                    min=0,
                    tooltip=(
                        "Number of images in the batch wired to "
                        "`KeyframeLatentScheduleBatchEncode.images` (current) or "
                        "`KeyframeImageSchedule.images` (legacy). 0 = skip "
                        "keyframe checks. Enables: index-out-of-bounds detection, "
                        "schedule-collapses-to-single-index detection, "
                        "empty-schedule-with-batch detection."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("report", tooltip="Full diagnostic text. Wire to PreviewAny."),
                io.Boolean.Output("ok", tooltip="False iff any ERROR check fires."),
                io.Int.Output("warnings", tooltip="Count of WARN lines."),
                io.Int.Output("errors", tooltip="Count of ERROR lines."),
                io.Float.Output(
                    "effective_stride_seconds",
                    tooltip="Stride the controller would actually produce. Sanity-check against planner.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        window_seconds: float,
        overlap_seconds: float,
        fps: int,
        length: int,
        width: int,
        height: int,
        schedule: str,
        resolution_rule: str,
        seam_tolerance_seconds: float,
        keyframe_schedule: str = "",
        keyframe_batch_size: int = 0,
    ) -> io.NodeOutput:
        report, ok, warnings, errors, stride = _build_report(
            audio_duration=_audio_duration(audio),
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            fps=fps,
            length=length,
            width=width,
            height=height,
            schedule=schedule,
            resolution_rule=resolution_rule,
            seam_tolerance_seconds=seam_tolerance_seconds,
            keyframe_schedule=keyframe_schedule,
            keyframe_batch_size=keyframe_batch_size,
        )
        return io.NodeOutput(report, ok, warnings, errors, stride)


def _build_report(
    *,
    audio_duration: float,
    window_seconds: float,
    overlap_seconds: float,
    fps: int,
    length: int,
    width: int,
    height: int,
    schedule: str,
    resolution_rule: str,
    seam_tolerance_seconds: float,
    keyframe_schedule: str = "",
    keyframe_batch_size: int = 0,
) -> tuple[str, bool, int, int, float]:
    """Pure function: separated from execute() for testability."""
    g = _compute_loop_geometry(window_seconds, overlap_seconds, fps)
    delta = g.effective_overlap_seconds - overlap_seconds

    if audio_duration > window_seconds:
        iterations = _compute_tile_count(audio_duration, g.stride_seconds) + 1
    else:
        iterations = 1
    seams = [i * g.stride_seconds for i in range(1, iterations)]

    lines: list[str] = _math_block(
        audio_duration=audio_duration,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        fps=fps,
        length=length,
        width=width,
        height=height,
        g=g,
        delta=delta,
        iterations=iterations,
    )
    lines += _seams_block(seams)

    check_lines: list[str] = ["Checks:"]
    warn_count = 0
    err_count = 0

    if length > 0:
        floor, ceil = _bracket(length, LTX_TEMPORAL_SCALE, offset=1)
        if floor == length:
            check_lines.append(f"  {_OK}length={length} valid: (length-1) % 8 == 0")
        else:
            err_count += 1
            check_lines.append(
                f"  {_ERR} length={length} invalid: (length-1) % 8 = {(length - 1) % 8}. "
                f"ComfyUI auto-rounds UP silently. "
                f"Fix: set length={floor} or {ceil}."
            )

        expected = round(window_seconds * fps)
        if length == expected:
            check_lines.append(
                f"  {_OK}length == window*fps: {length} == round({window_seconds}*{fps})"
            )
        else:
            warn_count += 1
            actual_window = length / fps
            check_lines.append(
                f"  {_WARN}length ({length}) != window*fps ({expected}). "
                f"Controller treats window as {window_seconds:.4f}s but generation "
                f"is actually {actual_window:.4f}s. "
                f"Fix: set length={expected}, or window_seconds={actual_window:.4f}."
            )

    if width > 0 and height > 0:
        divisor = 32 if resolution_rule == _RES_DIV32 else 64
        w_floor, w_ceil = _bracket(width, divisor)
        h_floor, h_ceil = _bracket(height, divisor)
        if w_floor == width and h_floor == height:
            check_lines.append(
                f"  {_OK}resolution {width}x{height} divisible by {divisor} ({resolution_rule})"
            )
        else:
            err_count += 1
            check_lines.append(
                f"  {_ERR} resolution {width}x{height} not divisible by {divisor}. "
                f"Fix: use {w_floor}x{h_floor} (round down) or {w_ceil}x{h_ceil} (round up)."
            )

    if abs(delta) < 1e-4:
        check_lines.append(
            f"  {_OK}effective overlap matches target ({g.effective_overlap_seconds:.4f}s)"
        )
    else:
        warn_count += 1
        reachable = _reachable_overlaps(g.window_pixel_frames, fps)
        nearby = sorted(reachable, key=lambda v: abs(v - overlap_seconds))[:3]
        nearby.sort()
        nearby_fmt = ", ".join(f"{v:.2f}s" for v in nearby)
        lo, hi = _widget_range_for_overlap_px(g.effective_overlap_pixel_frames, fps)
        check_lines.append(
            f"  {_WARN}effective overlap {g.effective_overlap_seconds:.3f}s != target "
            f"{overlap_seconds:.3f}s (delta {delta:+.3f}). "
            f"Reachable values for window={window_seconds:.2f}s near target: [{nearby_fmt}]. "
            f"Fix: any overlap_seconds in [{lo:.3f}, {hi:.3f}] yields current effective overlap."
        )

    ratio = g.overlap_latent_frames / g.window_latent_frames if g.window_latent_frames else 0.0
    if ratio < 0.15 and window_seconds < 12.0 and g.overlap_latent_frames > 0:
        warn_count += 1
        suggested = max(2.0, window_seconds * 0.25)
        check_lines.append(
            f"  {_WARN}thin context on short window: "
            f"{g.overlap_latent_frames}/{g.window_latent_frames} = {ratio*100:.0f}%. "
            f"Short windows have less runway to absorb seam artifacts. "
            f"Fix: set overlap_seconds >= {suggested:.1f}."
        )
    elif g.overlap_latent_frames > 0:
        check_lines.append(
            f"  {_OK}context/window ratio {g.overlap_latent_frames}/{g.window_latent_frames} = {ratio*100:.0f}%"
        )

    if audio_duration < window_seconds:
        err_count += 1
        check_lines.append(
            f"  {_ERR} audio ({audio_duration:.1f}s) shorter than window "
            f"({window_seconds:.1f}s). Loop will never iterate."
        )
    elif audio_duration < 2 * window_seconds:
        warn_count += 1
        check_lines.append(
            f"  {_WARN}audio ({audio_duration:.1f}s) barely longer than window "
            f"({window_seconds:.1f}s). Only ~1 iteration will run."
        )

    if schedule.strip():
        hits = _seam_boundary_hits(seams, schedule, seam_tolerance_seconds, check_lines)
        if hits is not None:
            if hits:
                warn_count += 1
            # _seam_boundary_hits appends the OK or WARN line itself.

    if g.overlap_clamped:
        warn_count += 1
        check_lines.append(
            f"  {_WARN}overlap_seconds too large for window; clamped to "
            f"{g.overlap_latent_frames} latents. Stride collapses, convergence extremely slow."
        )

    # Keyframe checks — gated on batch_size > 0. Catches three footguns
    # that turn the keyframe path into a no-op or an error:
    #   1. Batch wired but schedule empty (silent index-0 lock).
    #   2. Schedule references an index beyond batch_size (runtime clamp
    #      swallows the user's intent).
    #   3. Schedule collapses to a single index when batch_size > 1
    #      (unused keyframes — the pre-fix shipped shape of
    #      _latent_keyframe.json).
    if keyframe_batch_size > 0:
        kf_warn, kf_err = _keyframe_check_block(
            keyframe_schedule, keyframe_batch_size, check_lines,
        )
        warn_count += kf_warn
        err_count += kf_err

    lines.extend(check_lines)
    lines.append("")
    if err_count == 0 and warn_count == 0:
        lines.append("All checks pass.")
    else:
        lines.append(f"{err_count} error(s), {warn_count} warning(s)")
    lines.append("=" * 68)

    return "\n".join(lines), (err_count == 0), warn_count, err_count, g.stride_seconds


def _keyframe_check_block(
    keyframe_schedule: str,
    keyframe_batch_size: int,
    check_lines: list[str],
) -> tuple[int, int]:
    """Emit OK/WARN/ERROR lines for keyframe wiring. Returns (warn_delta, err_delta).

    Caller is responsible for the `keyframe_batch_size > 0` gate.
    Three failure modes caught (pre-run):
      - Empty schedule with batched keyframes → silent index-0 lock.
      - Index out of bounds → runtime clamp swallows intent.
      - Single-index schedule with batch > 1 → unused keyframes.
    """
    warn_delta = 0
    err_delta = 0

    if not keyframe_schedule.strip():
        warn_delta += 1
        check_lines.append(
            f"  {_WARN}keyframe batch has {keyframe_batch_size} image(s) but "
            f"schedule is empty. Every iteration uses index 0; other "
            f"keyframes go unused. "
            f"Fix: author a schedule (e.g. '0:00-0:42: 0\\n0:42+: 1') "
            f"or reduce the batch to 1 image."
        )
        return warn_delta, err_delta

    entries = _parse_image_schedule(keyframe_schedule)
    if not entries:
        warn_delta += 1
        check_lines.append(
            f"  {_WARN}keyframe schedule did not parse any entries. "
            f"Every iteration uses index 0. "
            f"Fix: use timestamp→index format, e.g. '0:00-0:42: 0\\n0:42+: 1'."
        )
        return warn_delta, err_delta

    # _parse_image_schedule returns list[tuple[start, end_or_None, index]];
    # unpack the index directly to avoid a magic positional index.
    indices = {idx for _, _, idx in entries}
    out_of_bounds = sorted(i for i in indices if i >= keyframe_batch_size or i < 0)
    if out_of_bounds:
        err_delta += 1
        shown = ", ".join(str(i) for i in out_of_bounds[:5])
        more = f" (+{len(out_of_bounds) - 5} more)" if len(out_of_bounds) > 5 else ""
        check_lines.append(
            f"  {_ERR} keyframe schedule references index {shown}{more} "
            f"but batch has {keyframe_batch_size} image(s) (valid: "
            f"0..{keyframe_batch_size - 1}). Runtime clamps silently so the "
            f"intended keyframe never shows. Fix: add more images to the "
            f"batch OR correct the indices."
        )
        return warn_delta, err_delta

    if len(indices) == 1 and keyframe_batch_size > 1:
        (only,) = indices
        warn_delta += 1
        check_lines.append(
            f"  {_WARN}keyframe schedule always selects index {only}; "
            f"{keyframe_batch_size - 1} keyframe(s) in the batch are unused. "
            f"Add timestamp ranges pointing at other indices, e.g. "
            f"'0:42-1:28: 1' to activate them."
        )
        return warn_delta, err_delta

    check_lines.append(
        f"  {_OK}keyframe schedule: {len(indices)} distinct index(es) "
        f"in a {keyframe_batch_size}-image batch"
    )
    return warn_delta, err_delta


def _math_block(
    *,
    audio_duration: float,
    window_seconds: float,
    overlap_seconds: float,
    fps: int,
    length: int,
    width: int,
    height: int,
    g,
    delta: float,
    iterations: int,
) -> list[str]:
    lines = [
        "=" * 68,
        "LoopConfigValidator",
        "=" * 68,
        "",
        f"Inputs: audio={audio_duration:.2f}s  window={window_seconds:.2f}s  "
        f"overlap_target={overlap_seconds:.2f}s  fps={fps}",
    ]
    extras = []
    if length:
        extras.append(f"length={length}")
    if width or height:
        extras.append(f"res={width}x{height}")
    if extras:
        lines.append("        " + "  ".join(extras))
    s = LTX_TEMPORAL_SCALE
    lines += [
        "",
        "Pixel math:",
        f"  window_pixel_frames  = round({window_seconds} * {fps}) = {g.window_pixel_frames}",
        f"  overlap_pixel_frames = round({overlap_seconds} * {fps}) = {g.overlap_pixel_frames}",
        "",
        f"Latent math (LTX video VAE: latent = (pixel-1)//{s} + 1):",
        f"  window_latents  = ({g.window_pixel_frames}-1)//{s} + 1 = {g.window_latent_frames}",
        f"  overlap_latents = ({g.overlap_pixel_frames}-1)//{s} + 1 = {g.overlap_latent_frames}"
        + ("  [CLAMPED to window_latents-1]" if g.overlap_clamped else ""),
        f"  new_latents     = {g.window_latent_frames} - {g.overlap_latent_frames} = {g.new_latent_frames}",
        "",
        "Effective output:",
        f"  stride_pixel_frames   = {g.new_latent_frames} * {s} = {g.stride_pixel_frames}",
        f"  stride_seconds        = {g.stride_pixel_frames}/{fps} = {g.stride_seconds:.4f}",
        f"  effective_overlap_sec = {g.effective_overlap_pixel_frames}/{fps} = {g.effective_overlap_seconds:.4f}"
        f"  (target {overlap_seconds:.4f}, delta {delta:+.4f})",
        f"  iterations (est)      = {iterations}",
        "",
    ]
    return lines


def _seams_block(seams: list[float]) -> list[str]:
    if not seams:
        return []
    lines = ["Iteration seams (global time):"]
    n = len(seams)
    head = min(n, 10)
    for i in range(head):
        t = seams[i]
        lines.append(f"  iter {i+1:2d}: {_format_timestamp(t)}  ({t:.2f}s)")
    if n > 12:
        lines.append("  ...")
        for i in (n - 2, n - 1):
            t = seams[i]
            lines.append(f"  iter {i+1:2d}: {_format_timestamp(t)}  ({t:.2f}s)")
    lines.append("")
    return lines


def _seam_boundary_hits(
    seams: list[float],
    schedule: str,
    tolerance: float,
    out_lines: list[str],
) -> list[tuple[int, float, float]] | None:
    """Returns list of (iter_index, seam_time, boundary_time) hits, or None on parse error.

    Appends an OK/WARN/parse-error line to `out_lines` as a side effect."""
    try:
        entries = _parse_schedule(schedule)
    except (ValueError, IndexError) as e:
        out_lines.append(f"  {_WARN}could not parse schedule: {e}")
        return None

    boundaries: set[float] = set()
    for start, end, _ in entries:
        if start > 0.0:
            boundaries.add(round(start, 3))
        if end is not None:
            boundaries.add(round(end, 3))

    hits: list[tuple[int, float, float]] = []
    for i, seam in enumerate(seams, 1):
        closest = None
        for b in boundaries:
            if closest is None or abs(seam - b) < abs(seam - closest):
                closest = b
        if closest is not None and abs(seam - closest) <= tolerance:
            hits.append((i, seam, closest))

    if hits:
        preview = "; ".join(
            f"iter {i}@{_format_timestamp(seam)}~={_format_timestamp(b)}"
            for i, seam, b in hits[:4]
        )
        more = f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""
        out_lines.append(
            f"  {_WARN}{len(hits)} seam(s) within {tolerance:.1f}s of schedule "
            f"boundaries: {preview}{more}. Prompt cut on iter seam is a documented "
            f"failure mode in rapid-cut configurations. "
            f"Fix: nudge overlap_seconds to shift stride, or move schedule "
            f"boundaries by ~1s."
        )
    else:
        out_lines.append(
            f"  {_OK}no iter seams within {tolerance:.1f}s of schedule boundaries"
        )
    return hits
