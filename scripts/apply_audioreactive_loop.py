"""apply_audioreactive_loop.

Last updated: 2026-05-23

Stages an audio-reactive, FULL-LENGTH (loop-preserving) variant of the
canonical latent workflow into `example_workflows/` (promoted from
experimental after render validation; top-level shipped surface). This is
the production counterpart to `apply_audio_driven_single_shot.py`:

  - single-shot fork = fast preview/tuning rig (~14s, no loop, no drift) —
    use it to dial in the look, prompt, and audio coupling.
  - THIS variant = the real render. Keeps the loop, so it tracks the WHOLE
    track (3 / 5 / 10 / 20 min — `AudioLoopPlanner.total_iterations`
    auto-sizes the loop to the loaded audio) and the prompt schedule can
    evolve the visual across the set's sections.

Topology is UNCHANGED from the canonical (loop intact) — this only presets
widgets, so it passes every canonical audit invariant. What it sets:
  - #1523 LTX2AttentionTunerPatch.audio_to_video_scale = 2.5
        (how hard the audio modality drives video attention)
  - #508 LTX2_NAG.nag_scale = 5  (canonical/KJNodes 11 is the documented
        distilled freeze-risk knob)
  - #507 CLIPTextEncode (NAG negative) = motion + frame-quality terms for a
        non-person subject (drops the canonical's singer tokens)
  - #1269 first_frame_guide_strength = 0.7  (the per-iter init re-anchor
        strength = the DRIFT vs MOTION dial: higher holds a painterly init
        harder but suppresses motion; lower frees motion but lets style
        drift across iterations — A/B this for your image)
  - #1615 TimestampPromptScheduleBatchEncode = a single `0:00+:` prompt held
        for the whole render (add more `M:SS+:` entries to evolve per section)

All preset values are `--flag`-overridable. The audio path stays frozen;
this only tunes how hard audio drives video and how the loop re-anchors.

Usage:
    uv run --group dev python scripts/apply_audioreactive_loop.py
    uv run --group dev python scripts/apply_audioreactive_loop.py --dry-run
    uv run --group dev python scripts/apply_audioreactive_loop.py --revert
    uv run --group dev python scripts/apply_audioreactive_loop.py \
        --audio-to-video-scale 3.0 --first-frame-guide-strength 0.5 --force

A second run is a NO-OP by default (detected via the marker Note) so in-UI
edits — especially your authored prompt schedule — survive. Pass `--force`
to regenerate fresh from the canonical with new params. `--revert` deletes
the output but refuses if the target isn't this variant (no marker Note).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

# --- Node IDs (canonical latent workflow) ---
N_ATTN_TUNER = 1523      # LTX2AttentionTunerPatch -- audio_to_video_scale (widget 3)
N_NAG = 508              # LTX2_NAG -- nag_scale (widget 0); [scale, alpha, tau, inplace]
N_NEG_PROMPT = 507       # CLIPTextEncode -- NAG negative text (widget 0)
N_FFG_STRENGTH = 1269    # FloatConstant "first_frame_guide_strength" (widget 0)
N_PROMPT_SCHEDULE = 1615  # TimestampPromptScheduleBatchEncode (widget 0 = schedule)
N_LOOP_BODY_SUBGRAPH = 843  # loop body invoker -- MUST be present (this keeps the loop)

# Pre-flight: refuse unless the canonical loop layout is present.
REQUIRED_SOURCE_NODES = (
    N_ATTN_TUNER, N_NAG, N_NEG_PROMPT, N_FFG_STRENGTH, N_PROMPT_SCHEDULE,
    1539, 1540, N_LOOP_BODY_SUBGRAPH, 1560, 444, 565,
)

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "example_workflows/audio_reactive_loop.json"

DEFAULT_A2V_SCALE = 2.5
DEFAULT_NAG_SCALE = 5.0
DEFAULT_FFG_STRENGTH = 0.7
DEFAULT_NEGATIVE = (
    "still image with no motion, frozen frame, static, blurry, low quality, "
    "watermark, subtitles, text"
)
# Single "0:00+:" entry — one look held for the whole render. The loop CAN still
# evolve the visual by section (add more "M:SS+: ..." entries, "In a [shot]"
# continuation framing — NOT "Cut to"); a single steady prompt is the default.
DEFAULT_PROMPT_SCHEDULE = (
    "0:00+: In a tight macro close-up, an expressive oil-painted anatomical "
    "heart pulses and contracts rhythmically, beating steadily, vivid "
    "brushstrokes flexing with each beat under soft warm light. The camera "
    "holds steady."
)

NOTE_MARKER = "Audio-reactive loop — read me"
NOTE_TEXT = (
    "AUDIO-REACTIVE LOOP VARIANT (full-length render)\n\n"
    "Renders the WHOLE track — the loop auto-tracks audio length.\n"
    "Use the single-shot fork to preview the look fast, then render here.\n\n"
    "Knobs (all --flag-overridable in apply_audioreactive_loop.py):\n"
    "  - #1523 audio_to_video_scale=2.5: how hard audio drives video.\n"
    "  - #508 NAG nag_scale=5 (+#507 negative): softened from freeze-prone 11.\n"
    "  - #1269 first_frame_guide_strength=0.7: the DRIFT vs MOTION dial.\n"
    "    Higher = holds the (painterly) init harder, less motion;\n"
    "    lower = more motion, more cross-iter style drift. A/B this.\n"
    "  - #1615 prompt: a single 0:00+: entry by default. To evolve per\n"
    "    section, add more 'M:SS+: ...' entries ('In a [shot]' framing,\n"
    "    NOT 'Cut to'); keep the beat verb (pulses/beats).\n\n"
    "To run: #444 init image, #565 full track. Output trims to audio.\n"
    "Long renders: render per-track, use the SaveLatent -> upscale path."
)


def _already_migrated(ed: WorkflowEditor) -> bool:
    # Topology is unchanged from canonical, so detect via the marker Note.
    return any(
        n.get("type") == "Note" and NOTE_MARKER.lower() in (n.get("title", "").lower())
        for n in ed.wf["nodes"]
    )


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This variant keeps the loop — run it against the canonical latent "
            "workflow, not a loop-removed fork."
        )


def _apply_ops(ed: WorkflowEditor, *, a2v_scale: float, nag_scale: float,
               ffg_strength: float, negative: str, prompt_schedule: str) -> None:
    ed.find_node(N_ATTN_TUNER)["widgets_values"][3] = a2v_scale       # audio_to_video_scale
    ed.find_node(N_NAG)["widgets_values"][0] = nag_scale              # nag_scale
    ed.find_node(N_NEG_PROMPT)["widgets_values"][0] = negative        # NAG negative text
    ed.find_node(N_FFG_STRENGTH)["widgets_values"][0] = ffg_strength  # first_frame_guide_strength
    ed.find_node(N_PROMPT_SCHEDULE)["widgets_values"][0] = prompt_schedule

    ed.add_top_level_node(
        node_type="Note",
        pos=[1430, 520],
        size=[440, 360],
        inputs=[], outputs=[],
        widgets_values=[NOTE_TEXT],
        properties={},
        title=NOTE_MARKER,
    )


def _migrate(input_path: Path, output_path: Path, *, dry_run: bool, force: bool,
             a2v_scale: float, nag_scale: float, ffg_strength: float,
             negative: str, prompt_schedule: str) -> None:
    _assert_required_nodes_present(WorkflowEditor(input_path))

    if dry_run:
        print(f"would copy {input_path} -> {output_path}")
        print(f"would set #{N_ATTN_TUNER}.audio_to_video_scale = {a2v_scale}")
        print(f"would set #{N_NAG}.nag_scale = {nag_scale}")
        print(f"would set #{N_NEG_PROMPT} (NAG negative) = {negative!r}")
        print(f"would set #{N_FFG_STRENGTH}.first_frame_guide_strength = {ffg_strength}")
        entry_count = prompt_schedule.count("\n") + 1
        print(f"would set #{N_PROMPT_SCHEDULE}.schedule ({entry_count} entries)")
        print("would add marker Note (loop topology unchanged)")
        return

    if not force and output_path.exists() and input_path != output_path and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping (preserves any hand-edits, "
              "esp. your prompt schedule). Pass --force to regenerate, or --revert to remove.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    _apply_ops(ed, a2v_scale=a2v_scale, nag_scale=nag_scale, ffg_strength=ffg_strength,
               negative=negative, prompt_schedule=prompt_schedule)
    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit:    uv run --group dev python scripts/audit_workflows.py {output_path}")
    print(f"  3. EDIT the #1615 prompt to your subject (add M:SS+: entries to evolve per section).")
    print(f"  4. Load in ComfyUI: open {output_path}; set #444 init image + #565 full track.")


def _revert(output_path: Path) -> None:
    if not output_path.exists():
        print(f"{output_path} does not exist; nothing to revert.")
        return
    # Safety: only delete files that carry this variant's marker Note.
    try:
        is_ours = _already_migrated(WorkflowEditor(output_path))
    except Exception:
        is_ours = False
    if not is_ours:
        raise SystemExit(
            f"Refusing to delete {output_path}: no audio-reactive-loop marker Note, "
            "so it is not this variant. Check your --output path."
        )
    output_path.unlink()
    print(f"removed {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--audio-to-video-scale", type=float, default=DEFAULT_A2V_SCALE,
                    help="LTX2AttentionTunerPatch audio_to_video_scale (default 2.5; 1.0 = neutral).")
    ap.add_argument("--nag-scale", type=float, default=DEFAULT_NAG_SCALE,
                    help="LTX2_NAG nag_scale (default 5; canonical 11 is freeze-prone on distilled).")
    ap.add_argument("--first-frame-guide-strength", type=float, default=DEFAULT_FFG_STRENGTH,
                    help="Per-iter init re-anchor strength (default 0.7 = canonical; the drift-vs-motion dial, "
                         "1.0 = max identity stability / minimal motion, lower = more motion + drift).")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE,
                    help="NAG negative text (#507). Default is motion/quality terms for a non-person subject.")
    ap.add_argument("--prompt-schedule", default=DEFAULT_PROMPT_SCHEDULE,
                    help="Newline-separated 'M:SS+: ...' schedule entries for #1615.")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output file (does not touch --input).")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the output exists (overwrites in-UI edits incl. the schedule).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report planned ops without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return

    _migrate(
        Path(args.input), output_path, dry_run=args.dry_run, force=args.force,
        a2v_scale=args.audio_to_video_scale,
        nag_scale=args.nag_scale,
        ffg_strength=args.first_frame_guide_strength,
        negative=args.negative,
        prompt_schedule=args.prompt_schedule,
    )


if __name__ == "__main__":
    main()
