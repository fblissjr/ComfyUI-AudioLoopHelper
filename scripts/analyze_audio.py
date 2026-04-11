"""Analyze audio file for music video prompt scheduling.

Outputs energy timeline, detected song structure, and loudness stats.
Uses ffmpeg for analysis -- no heavy Python dependencies.

For music-aware analysis (BPM, key, chromagram, vocal F0, auto-generated
prompt templates), see analyze_audio_features.py (requires: uv sync --group analysis).

Usage:
    uv run scripts/analyze_audio.py path/to/audio.wav
    uv run scripts/analyze_audio.py path/to/audio.wav --output analysis.md
    uv run scripts/analyze_audio.py path/to/audio.wav --window 2 --trim 10
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile


def run_ffmpeg(args: list[str]) -> str:
    """Run ffmpeg command and return stderr (where ffmpeg puts info)."""
    result = subprocess.run(
        ["ffmpeg"] + args,
        capture_output=True,
        text=True,
    )
    return result.stderr + result.stdout


def get_audio_info(path: str) -> dict:
    """Get basic audio file info."""
    output = run_ffmpeg(["-i", path, "-hide_banner"])
    info = {"path": path}

    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", output)
    if duration_match:
        h, m, s = duration_match.groups()
        info["duration"] = int(h) * 3600 + int(m) * 60 + float(s)

    sr_match = re.search(r"(\d+) Hz", output)
    if sr_match:
        info["sample_rate"] = int(sr_match.group(1))

    ch_match = re.search(r"(\d+) channels", output)
    if ch_match:
        info["channels"] = int(ch_match.group(1))
    elif "stereo" in output:
        info["channels"] = 2
    elif "mono" in output:
        info["channels"] = 1

    return info


def get_loudness_stats(path: str) -> dict:
    """Get integrated loudness, LRA, and true peak via EBU R128."""
    output = run_ffmpeg(["-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"])
    stats = {}

    # Parse the Summary section at the end (not per-frame values)
    summary = output.split("Summary:")[-1] if "Summary:" in output else output

    i_match = re.search(r"I:\s+([-\d.]+) LUFS", summary)
    if i_match:
        stats["integrated_lufs"] = float(i_match.group(1))

    lra_match = re.search(r"LRA:\s+([-\d.]+) LU", summary)
    if lra_match:
        stats["lra_lu"] = float(lra_match.group(1))

    peak_match = re.search(r"Peak:\s+([-\d.]+) dBFS", summary)
    if peak_match:
        stats["true_peak_dbfs"] = float(peak_match.group(1))

    return stats


def get_rms_timeline(path: str, window: float = 2.0) -> list[tuple[float, float]]:
    """Get RMS energy per time window using ffmpeg astats."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name

    try:
        run_ffmpeg([
            "-i", path,
            "-af", f"astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file={tmp_path}",
            "-f", "null", "-",
        ])

        entries = []
        with open(tmp_path) as f:
            lines = f.readlines()

        for i in range(0, len(lines) - 1, 2):
            time_match = re.search(r"pts_time:([\d.]+)", lines[i])
            val_line = lines[i + 1]
            if "inf" in val_line:
                continue
            val_match = re.search(r"RMS_level=([-\d.]+)", val_line)
            if time_match and val_match:
                entries.append((float(time_match.group(1)), float(val_match.group(1))))

        # Aggregate to windows
        if not entries:
            return []

        max_t = entries[-1][0]
        timeline = []
        t = 0.0
        while t < max_t:
            chunk = [v for (tt, v) in entries if t <= tt < t + window and v > -80]
            if chunk:
                avg_rms = sum(chunk) / len(chunk)
                timeline.append((t, avg_rms))
            t += window

        return timeline
    finally:
        os.unlink(tmp_path)


def detect_structure(timeline: list[tuple[float, float]]) -> list[dict]:
    """Detect song sections based on energy transitions."""
    if not timeline:
        return []

    # Classify each window as quiet/medium/loud using percentiles
    rms_values = sorted([v for _, v in timeline if v > -60])
    if not rms_values:
        return []

    # Use percentiles for adaptive thresholds
    p25 = rms_values[len(rms_values) // 4]
    p75 = rms_values[3 * len(rms_values) // 4]
    quiet_threshold = p25
    loud_threshold = p75

    levels = []
    for t, v in timeline:
        if v < quiet_threshold:
            levels.append((t, "quiet"))
        elif v < loud_threshold:
            levels.append((t, "medium"))
        else:
            levels.append((t, "loud"))

    # Group consecutive same-level windows into sections
    raw_sections = []
    current_level = levels[0][1]
    section_start = levels[0][0]

    for i in range(1, len(levels)):
        if levels[i][1] != current_level:
            raw_sections.append({
                "start": section_start,
                "end": levels[i][0],
                "level": current_level,
            })
            current_level = levels[i][1]
            section_start = levels[i][0]

    raw_sections.append({
        "start": section_start,
        "end": timeline[-1][0] + 2.0,
        "level": current_level,
    })

    # Merge short sections (< 6s) into their neighbors
    sections = [raw_sections[0]]
    for s in raw_sections[1:]:
        prev = sections[-1]
        duration = s["end"] - s["start"]
        prev_duration = prev["end"] - prev["start"]

        # Absorb short sections into the louder neighbor
        if duration < 6:
            if prev_duration < 6:
                # Both short -- merge into one
                prev["end"] = s["end"]
                # Keep the louder level
                if _level_rank(s["level"]) > _level_rank(prev["level"]):
                    prev["level"] = s["level"]
            else:
                prev["end"] = s["end"]
        elif prev_duration < 6:
            # Previous was short, absorb it into current
            s["start"] = prev["start"]
            if _level_rank(prev["level"]) > _level_rank(s["level"]):
                s["level"] = prev["level"]
            sections[-1] = s
        else:
            sections.append(s)

    # Label sections based on position and level
    for i, s in enumerate(sections):
        duration = s["end"] - s["start"]
        if i == 0 and s["level"] == "quiet" and duration < 15:
            s["label"] = "INTRO"
        elif i == len(sections) - 1 and s["level"] == "quiet":
            s["label"] = "OUTRO"
        elif s["level"] == "loud":
            s["label"] = "CHORUS"
        elif s["level"] == "quiet" and duration < 6:
            s["label"] = "BREAK"
        elif s["level"] == "quiet":
            s["label"] = "BRIDGE"
        elif s["level"] == "medium":
            s["label"] = "VERSE"
        else:
            s["label"] = s["level"].upper()

    return sections


def _level_rank(level: str) -> int:
    return {"quiet": 0, "medium": 1, "loud": 2}.get(level, 0)


def format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def format_report(
    info: dict,
    stats: dict,
    timeline: list[tuple[float, float]],
    sections: list[dict],
    trim_offset: float = 0.0,
) -> str:
    """Format the full analysis as a readable report."""
    lines = []
    lines.append(f"# Audio Analysis: {os.path.basename(info['path'])}")
    lines.append("")

    # Basic info
    dur = info.get("duration", 0)
    lines.append(f"**Duration:** {dur:.1f}s ({format_time(dur)})")
    lines.append(f"**Sample rate:** {info.get('sample_rate', '?')} Hz")
    lines.append(f"**Channels:** {info.get('channels', '?')}")
    lines.append(f"**Integrated loudness:** {stats.get('integrated_lufs', '?')} LUFS")
    lines.append(f"**Loudness range:** {stats.get('lra_lu', '?')} LU")
    lines.append(f"**True peak:** {stats.get('true_peak_dbfs', '?')} dBFS")
    if trim_offset > 0:
        lines.append(f"**Trim offset:** {trim_offset}s (timestamps below are pre-trim)")
    lines.append("")

    # Song structure
    lines.append("## Song Structure")
    lines.append("")
    lines.append("| Section | Time (song) | Time (after trim) | Duration | Energy |")
    lines.append("|---------|-------------|-------------------|----------|--------|")
    for s in sections:
        song_range = f"{format_time(s['start'])}-{format_time(s['end'])}"
        if trim_offset > 0:
            trimmed_start = max(0, s["start"] - trim_offset)
            trimmed_end = max(0, s["end"] - trim_offset)
            trim_range = f"{format_time(trimmed_start)}-{format_time(trimmed_end)}"
        else:
            trim_range = "—"
        dur = s["end"] - s["start"]
        lines.append(f"| {s['label']:8s} | {song_range:11s} | {trim_range:17s} | {dur:5.0f}s   | {s['level']:6s} |")
    lines.append("")

    # Transition timestamps
    lines.append("## Key Transitions")
    lines.append("")
    for i in range(1, len(sections)):
        t = sections[i]["start"]
        prev = sections[i - 1]["label"]
        curr = sections[i]["label"]
        trimmed = f" (trimmed: {format_time(max(0, t - trim_offset))})" if trim_offset > 0 else ""
        lines.append(f"- **{format_time(t)}**{trimmed}: {prev} → {curr}")
    lines.append("")

    # Energy timeline
    lines.append("## Energy Timeline")
    lines.append("")
    lines.append("```")
    rms_values = [v for _, v in timeline]
    max_rms = max(rms_values) if rms_values else -10
    min_display = -40

    for t, v in timeline:
        bar_val = max(0, (v - min_display) / (max_rms - min_display))
        bar = "#" * int(50 * min(1.0, bar_val))
        lines.append(f"  {format_time(t)}  {bar:50s}  {v:.1f}dB")
    lines.append("```")
    lines.append("")

    # Prompt schedule template
    lines.append("## Prompt Schedule Template (after trim)")
    lines.append("")
    lines.append("Copy and adapt for TimestampPromptSchedule:")
    lines.append("")
    lines.append("```")
    for s in sections:
        trimmed_start = max(0, s["start"] - trim_offset)
        trimmed_end = max(0, s["end"] - trim_offset)
        if trimmed_end <= 0:
            continue
        if s == sections[-1] or s["label"] == "OUTRO":
            lines.append(f"{format_time(trimmed_start)}+: [OUTRO - {s['level']}] describe fadeout")
        else:
            lines.append(
                f"{format_time(trimmed_start)}-{format_time(trimmed_end)}: "
                f"[{s['label']} - {s['level']}] describe action and audio here"
            )
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze audio for music video prompt scheduling")
    parser.add_argument("audio_path", help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--output", "-o", help="Write markdown report to file")
    parser.add_argument("--window", "-w", type=float, default=2.0, help="Analysis window in seconds (default: 2)")
    parser.add_argument("--trim", "-t", type=float, default=0.0, help="Trim offset in seconds for schedule timestamps")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {args.audio_path}...", file=sys.stderr)

    info = get_audio_info(args.audio_path)
    stats = get_loudness_stats(args.audio_path)
    timeline = get_rms_timeline(args.audio_path, window=args.window)
    sections = detect_structure(timeline)

    report = format_report(info, stats, timeline, sections, trim_offset=args.trim)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
