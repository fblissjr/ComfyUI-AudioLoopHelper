"""apply_audio_driven_single_shot.

Last updated: 2026-05-23

Stages a single-shot, audio-reactive DRAFT variant of the canonical
latent workflow into gitignored `internal/scratch/`. Does NOT mutate the
canonical, and does NOT write to the tracked `example_workflows/` surface
(the variant is unvalidated — it has not been render-tested. Promotion to
`example_workflows/experimental/` + a paired audit invariant follows the
"ships AND stabilizes" criterion in `internal/PLAN.md`).

Symptom / motivation: the canonical workflow is a full-length looped
music video. For a short audio-reactive demo (a subject that pulses/moves
in time with the music — "make a heart beat to music"), the loop spine
adds drift risk and render time and buys nothing — the output is one
window, trimmed to the audio.

What this variant is: the canonical workflow with the loop subsystem
removed so only the initial render runs, and three knobs preset for an
audio-driven demo:
  1. The decode reads the initial-render latent directly (no LatentConcat
     prepend, no TensorLoop).
  2. `LTX2AttentionTunerPatch.audio_to_video_scale` pushed above 1.0 so
     the audio modality drives video attention harder (the only direct
     "audio-drives-motion intensity" dial we have — node #1523).
  3. `LTXVPreprocess.img_compression` raised to 35 on the init image
     (a pristine i2v init reads as a static photo and freezes; LTX-2 was
     trained on compressed video, so compression unlocks motion).
  4. A single heart-pulse schedule entry as a starting prompt.

The audio path stays sacred and frozen. Audio still drives the scene via
LTX 2.3's native joint cross-attention; this variant only changes which
latent the decoder reads and how hard audio attention is scaled.

Topology change (loop removed):
  Initial render: 444 LoadImage -> 445 SmartResize -> 446 LTXVPreprocess
    -> 531 ImgToVideoInplace -> 350 ConcatAVLatent (+ audio 566->570)
    -> 161 SamplerCustomAdvanced -> 245 SeparateAVLatent -> 381 CropGuides
  Decode (rewired): 381 CropGuides.latent -> 2028 TrimVideoLatentToAudio
    -> 1604 LTXVTiledVAEDecode -> 2029 TrimImageBatchToAudio -> 617 VHS

Removed (loop-only / loop-body-reference-only). After the single
keep->remove edge `2028.latent <- 1605` is rewired to 381, the removal
set is fully self-contained (every other consumer of a removed node is
itself removed — verified by `_assert_no_dangling_kept_inputs`):
  1539 TensorLoopOpen, 843 subgraph invoker, 1540 TensorLoopClose,
  1605 LatentConcat, 1582 AudioLoopController, 1586 PreviewAny (controller),
  1616 ConditioningSelectByIteration (loop body), 1617 VAEEncode (loop guide
  latent), 1618 LoopIterationStamp, 2027 SaveLatent (bypassed, fed by 1605),
  1633 LTXVReferenceAudio (loop body), and the loop-only reference-video
  chain 1636 VHS_LoadVideo / 1637 ImageResizeKJv2 / 1638 LTXVPreprocess.
The orphaned loop-body subgraph DEFINITION is also pruned.

Kept (initial-render path — do NOT remove): the model chain
414->268->504->1523->508->503->2014->2015->1635->572->1632->153->161
(the bypassed LoRA / IC-LoRA loader / ref-audio nodes are passthrough on
this path), 1560 AudioLoopPlanner (feeds 1615 prompt schedule),
2021 ConditioningSelectByIteration (initial-render conditioning;
current_iteration unwired -> widget default 0 -> schedule[0]).

Not touched (verify against your clip): render length comes from
`LTXFramePlanner #1634` (single source of truth for dims, ~14s at the
canonical 353-frame / 25 fps default). Final output is trimmed to the
loaded audio length (Get_orig_audio). For a clean N-second demo, load an
N-second audio clip and, if you want the video to match exactly, adjust
the frame planner (not EmptyLTXVLatentVideo's widget — it is wired from
the planner). `TrimAudioDuration #601` is left at its canonical default.

Usage:
    uv run --group dev python scripts/apply_audio_driven_single_shot.py
    uv run --group dev python scripts/apply_audio_driven_single_shot.py --dry-run
    uv run --group dev python scripts/apply_audio_driven_single_shot.py --revert
    uv run --group dev python scripts/apply_audio_driven_single_shot.py \
        --audio-to-video-scale 3.0 --img-compression 35

Idempotent on the OUTPUT path (re-detects via the removed loop node).
`--dry-run` always reports planned ops (never mutates). `--revert` deletes
the output staging file, but refuses if the target still contains the loop
body (node 843) — a guard against a misdirected --output nuking a real
workflow.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

# --- Node IDs (canonical latent workflow) ---
N_TRIM_VIDEO_LATENT = 2028   # TrimVideoLatentToAudio -- decode feed
N_CROP_GUIDES = 381          # LTXVCropGuides -- initial-render latent source (out slot 2)
N_ATTN_TUNER = 1523          # LTX2AttentionTunerPatch -- audio_to_video_scale knob (widget 3)
N_INIT_PREPROCESS = 446      # LTXVPreprocess -- init-image img_compression (widget 0)
N_PROMPT_SCHEDULE = 1615     # TimestampPromptScheduleBatchEncode (widget 0 = schedule text)
N_LOOP_BODY_SUBGRAPH = 843   # subgraph invoker (its node "type" == the subgraph def id)

# Loop subsystem + loop-only reference nodes to remove. Self-contained after
# the 2028 rewire (asserted post-removal).
LOOP_NODES_TO_REMOVE = (
    1605,  # LatentConcat "Prepend Initial Render"
    1540,  # TensorLoopClose
    843,   # subgraph invoker (loop body)
    1539,  # TensorLoopOpen
    1582,  # AudioLoopController
    1586,  # PreviewAny (controller overlap_frames)
    1616,  # ConditioningSelectByIteration (loop body)
    1617,  # VAEEncode (loop guide latent)
    1618,  # LoopIterationStamp
    2027,  # SaveLatent (bypassed, fed only by 1605)
    1633,  # LTXVReferenceAudio (loop body ID-LoRA) -- fed 1618/843 only
    1638,  # LTXVPreprocess (ref-video, loop only) -- fed 843 only
    1637,  # ImageResizeKJv2 (ref-video, loop only) -- fed 1638 only
    1636,  # VHS_LoadVideo (ref-video, loop only) -- fed 1637 only
)

# Pre-flight: refuse unless the canonical layout is present.
REQUIRED_SOURCE_NODES = (
    N_TRIM_VIDEO_LATENT, N_CROP_GUIDES, N_ATTN_TUNER, N_INIT_PREPROCESS,
    N_PROMPT_SCHEDULE, 1539, 1540, 843, 1582, 1560, 444, 565, 2021,
)

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/scratch/audio_driven_single_shot.json"

DEFAULT_A2V_SCALE = 2.0
DEFAULT_IMG_COMPRESSION = 35
DEFAULT_PROMPT_SCHEDULE = (
    "0:00+: In a tight macro close-up, a glistening human heart pulses and "
    "contracts rhythmically, beating steadily, its surface flexing and "
    "relaxing with each beat under soft warm light. The camera holds steady "
    "with a slow, gentle push-in."
)

NOTE_TEXT = (
    "SINGLE-SHOT AUDIO-REACTIVE VARIANT (loop removed)\n\n"
    "Audio drives the scene natively via LTX 2.3 cross-attention.\n"
    "Knobs preset for an audio-driven demo:\n"
    "  - #1523 LTX2AttentionTunerPatch.audio_to_video_scale: how hard\n"
    "    audio drives video. Push higher (e.g. 3-5) for tighter coupling.\n"
    "  - #446 LTXVPreprocess.img_compression=35: unlocks motion from a\n"
    "    still init (pristine init -> frozen frame).\n"
    "  - #1615 schedule[0]: the heart-pulse prompt. Verb (pulses/beating)\n"
    "    is what binds the motion to the audio.\n\n"
    "To run: load your init image (#444 LoadImage) and a short music/audio\n"
    "clip (#565 LoadAudio). Output is trimmed to the audio length.\n"
    "Also try: boost the audio so it peaks; keep framing tight."
)


def _already_migrated(ed: WorkflowEditor) -> bool:
    # The loop body subgraph invoker is gone iff this variant was applied.
    return not ed.has_node(N_LOOP_BODY_SUBGRAPH)


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the canonical latent workflow layout."
        )


def _assert_no_dangling_kept_inputs(ed: WorkflowEditor) -> None:
    """Guard the docstring's self-containment claim: no surviving node may be
    left with an input whose link id no longer exists."""
    link_ids = {l[0] for l in ed.wf["links"] if isinstance(l, list)}
    dangling = [
        (n["id"], inp["name"], inp["link"])
        for n in ed.wf["nodes"]
        for inp in n.get("inputs", [])
        if inp.get("link") is not None and inp["link"] not in link_ids
    ]
    if dangling:
        raise SystemExit(f"BUG: removal left stale input link ids: {dangling}")


def _prune_orphan_subgraphs(ed: WorkflowEditor) -> int:
    """Drop subgraph definitions no longer instantiated by any node type."""
    defs = ed.wf.get("definitions") or {}
    sgs = defs.get("subgraphs")
    if not sgs:
        return 0
    used_types = {n.get("type") for n in ed.wf["nodes"]}
    kept = [sg for sg in sgs if sg.get("id") in used_types]
    removed = len(sgs) - len(kept)
    if removed:
        defs["subgraphs"] = kept
    return removed


def _apply_ops(ed: WorkflowEditor, a2v_scale: float, img_compression: int,
               prompt_schedule: str) -> None:
    # 1. Decode reads the initial-render latent directly (was LatentConcat #1605).
    slot = WorkflowEditor.find_input_slot(ed.find_node(N_TRIM_VIDEO_LATENT), "latent")
    ed.rewire_input(N_TRIM_VIDEO_LATENT, slot, N_CROP_GUIDES, 2, "LATENT")

    # 2. Remove the loop subsystem (the rewire above already moved the one
    #    keep->remove edge off of #1605), then assert nothing kept dangles.
    for nid in LOOP_NODES_TO_REMOVE:
        if ed.has_node(nid):
            ed.remove_node_and_links(nid)
    _assert_no_dangling_kept_inputs(ed)
    _prune_orphan_subgraphs(ed)

    # 3. Preset the audio-driven knobs.
    ed.find_node(N_ATTN_TUNER)["widgets_values"][3] = a2v_scale          # audio_to_video_scale
    ed.find_node(N_INIT_PREPROCESS)["widgets_values"][0] = img_compression
    ed.find_node(N_PROMPT_SCHEDULE)["widgets_values"][0] = prompt_schedule

    # 4. Drop a handoff Note on the canvas.
    ed.add_top_level_node(
        node_type="Note",
        pos=[1430, 520],
        size=[420, 320],
        inputs=[], outputs=[],
        widgets_values=[NOTE_TEXT],
        properties={},
        title="Single-shot audio-reactive — read me",
    )


def _migrate(input_path: Path, output_path: Path, *, dry_run: bool,
             a2v_scale: float, img_compression: int, prompt_schedule: str) -> None:
    # Pre-flight always runs (read-only) against the canonical source.
    _assert_required_nodes_present(WorkflowEditor(input_path))

    if dry_run:
        # Always report intent, regardless of whether the output already exists.
        print(f"would copy {input_path} -> {output_path}")
        print(f"would rewire #{N_TRIM_VIDEO_LATENT}.latent <- #{N_CROP_GUIDES}.out2 (LATENT)")
        print(f"would remove loop nodes: {list(LOOP_NODES_TO_REMOVE)}")
        print("would prune the orphaned loop-body subgraph definition")
        print(f"would set #{N_ATTN_TUNER}.audio_to_video_scale = {a2v_scale}")
        print(f"would set #{N_INIT_PREPROCESS}.img_compression = {img_compression}")
        print(f"would set #{N_PROMPT_SCHEDULE}.schedule[0] = {prompt_schedule!r}")
        print("would add handoff Note")
        return

    if output_path.exists() and input_path != output_path and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    _apply_ops(ed, a2v_scale, img_compression, prompt_schedule)
    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit:    uv run --group dev python scripts/audit_workflows.py {output_path}")
    print(f"  3. Load in ComfyUI: open {output_path}; set #444 init image + #565 audio.")


def _revert(output_path: Path) -> None:
    if not output_path.exists():
        print(f"{output_path} does not exist; nothing to revert.")
        return
    # Safety: single-shot variants have the loop body (node 843) removed. If the
    # target still has it, it is a canonical/loop workflow, not our variant.
    try:
        still_a_loop_workflow = WorkflowEditor(output_path).has_node(N_LOOP_BODY_SUBGRAPH)
    except Exception:
        still_a_loop_workflow = False
    if still_a_loop_workflow:
        raise SystemExit(
            f"Refusing to delete {output_path}: it still contains the loop body "
            f"(node {N_LOOP_BODY_SUBGRAPH}), so it is NOT a single-shot variant. "
            "Check your --output path."
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
                    help="LTX2AttentionTunerPatch audio_to_video_scale (default 2.0; 1.0 = neutral).")
    ap.add_argument("--img-compression", type=int, default=DEFAULT_IMG_COMPRESSION,
                    help="Init LTXVPreprocess img_compression (default 35; anti frozen-frame).")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT_SCHEDULE,
                    help="Schedule[0] prompt text (include the '0:00+:' prefix).")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report planned ops without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return

    _migrate(
        Path(args.input), output_path, dry_run=args.dry_run,
        a2v_scale=args.audio_to_video_scale,
        img_compression=args.img_compression,
        prompt_schedule=args.prompt,
    )


if __name__ == "__main__":
    main()
