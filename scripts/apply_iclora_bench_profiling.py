"""apply_iclora_bench_profiling.

Last updated: 2026-05-01

Wires ProfileBegin / ProfileIterStep / ProfileEnd around the audio loop
of `example_workflows/audio-loop-music-video_latent_iclora.json` to
produce a bench variant. The three profile nodes start torch.profiler
before the loop, advance schedule once per iteration, and stop +
write artifacts after the loop completes.

Splice plan (all three are AnyType / LATENT passthroughs):

  Top-level (1):
    LoadAudio(#565).AUDIO -> TrimAudioDuration(#567)
    becomes
    LoadAudio.AUDIO -> ProfileBegin.trigger -> TrimAudioDuration

  Subgraph (per-iter):
    IterationCleanup(#2007).LATENT -> output collector -20.0
    becomes
    IterationCleanup.LATENT -> ProfileIterStep.latent -> -20.0

  Top-level (1):
    TensorLoopClose(#1540).LATENT -> LatentConcat(#1605).slot1
    becomes
    TensorLoopClose -> ProfileEnd.trigger -> LatentConcat.slot1

No subgraph schema change (only internal node insertion), so users do
NOT need to delete-and-re-add the subgraph node when loading the bench
variant.

Output staged at:
  internal/scratch/audio-loop-music-video_latent_iclora_bench.json

Pre-flight refuses if the iclora source workflow doesn't have the
expected splice anchors (the 4 known node ids at known link slots).

When `start_experiment.sh` launched ComfyUI (RUN_ID set), profile
artifacts land at:
  data/runs/${RUN_ID}/profiler/
    - trace.json (chrome trace; open at perfetto.dev or chrome://tracing)
    - summary.txt (top kernels by cumulative time)
    - memory_timeline.html (VRAM timeline)

Without RUN_ID set, falls back to:
  internal/analysis/runs/profiler/<timestamp>/

Defaults: warmup_iterations=1, active_iterations=3 — captures iters 2-4
of the loop (skipping iter 1's compilation noise).

Compatibility:
  - Pure addition; doesn't modify any audit-visible topology
  - Coexists with all F2-F13 audit checks (none touched)
  - Does NOT modify the canonical or other shipped variants
  - Reverting deletes the staged file

Usage:
    uv run --group dev python scripts/apply_iclora_bench_profiling.py
    uv run --group dev python scripts/apply_iclora_bench_profiling.py --revert
    uv run --group dev python scripts/apply_iclora_bench_profiling.py --dry-run

Idempotent on the OUTPUT path. `--revert` deletes the staged file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent_iclora.json"
DEFAULT_OUTPUT = REPO_ROOT / "internal" / "scratch" / "audio-loop-music-video_latent_iclora_bench.json"

# Top-level anchor ids in the iclora workflow.
LOAD_AUDIO_ID = 565
TRIM_AUDIO_DURATION_ID = 567
TENSOR_LOOP_CLOSE_ID = 1540
LATENT_CONCAT_ID = 1605
LATENT_CONCAT_TARGET_SLOT = 1

# Subgraph-internal anchor ids.
ITERATION_CLEANUP_ID = 2007
SUBGRAPH_OUTPUT_COLLECTOR = -20
SUBGRAPH_OUTPUT_LATENT_SLOT = 0

# Profile widget defaults — match nodes.py ProfileBegin schema.
PROFILE_OUTPUT_DIR = "internal/analysis/runs/profiler"
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 3
PROFILE_INCLUDE_CPU = True
PROFILE_INCLUDE_MEMORY = True
PROFILE_INCLUDE_SHAPES = True
PROFILE_INCLUDE_FLOPS = False


def _already_applied(ed: WorkflowEditor) -> bool:
    return bool(
        ed.find_nodes_by_type("ProfileBegin_AudioLoop")
        or ed.find_nodes_by_type("ProfileEnd_AudioLoop")
    )


def _add_profile_begin(ed: WorkflowEditor) -> int:
    return ed.add_top_level_node(
        node_type="ProfileBegin_AudioLoop",
        pos=[-2400, 800],
        size=[300, 200],
        inputs=[
            {"name": "trigger", "type": "*", "link": None},
            {"name": "enabled", "type": "BOOLEAN",
             "widget": {"name": "enabled"}, "link": None},
            {"name": "output_dir", "type": "STRING",
             "widget": {"name": "output_dir"}, "link": None},
            {"name": "warmup_iterations", "type": "INT",
             "widget": {"name": "warmup_iterations"}, "link": None},
            {"name": "active_iterations", "type": "INT",
             "widget": {"name": "active_iterations"}, "link": None},
            {"name": "include_cpu", "type": "BOOLEAN",
             "widget": {"name": "include_cpu"}, "link": None},
            {"name": "include_memory", "type": "BOOLEAN",
             "widget": {"name": "include_memory"}, "link": None},
            {"name": "include_shapes", "type": "BOOLEAN",
             "widget": {"name": "include_shapes"}, "link": None},
            {"name": "include_flops", "type": "BOOLEAN",
             "widget": {"name": "include_flops"}, "link": None},
        ],
        outputs=[
            {"name": "trigger", "type": "*", "links": []},
        ],
        widgets_values=[
            True,                                  # enabled
            PROFILE_OUTPUT_DIR,
            PROFILE_WARMUP,
            PROFILE_ACTIVE,
            PROFILE_INCLUDE_CPU,
            PROFILE_INCLUDE_MEMORY,
            PROFILE_INCLUDE_SHAPES,
            PROFILE_INCLUDE_FLOPS,
        ],
        properties={
            "Node name for S&R": "ProfileBegin_AudioLoop",
            "cnr_id": "fblissjr/ComfyUI-AudioLoopHelper",
        },
        title="Profile Begin (bench)",
    )


def _add_profile_end(ed: WorkflowEditor) -> int:
    return ed.add_top_level_node(
        node_type="ProfileEnd_AudioLoop",
        pos=[400, 800],
        size=[260, 60],
        inputs=[
            {"name": "trigger", "type": "*", "link": None},
        ],
        outputs=[
            {"name": "trigger", "type": "*", "links": []},
        ],
        widgets_values=[],
        properties={
            "Node name for S&R": "ProfileEnd_AudioLoop",
            "cnr_id": "fblissjr/ComfyUI-AudioLoopHelper",
        },
        title="Profile End (bench)",
    )


def _add_profile_iter_step_subgraph(ed: WorkflowEditor) -> int:
    return ed.add_subgraph_node(
        node_type="ProfileIterStep_AudioLoop",
        pos=[2400, 1200],
        size=[260, 50],
        inputs=[
            {"name": "latent", "type": "LATENT", "link": None},
        ],
        outputs=[
            {"name": "latent", "type": "LATENT", "links": []},
        ],
        widgets_values=[],
        properties={
            "Node name for S&R": "ProfileIterStep_AudioLoop",
            "cnr_id": "fblissjr/ComfyUI-AudioLoopHelper",
        },
        title="Profile Iter Step (bench)",
    )


def _splice_profile_begin(ed: WorkflowEditor, profile_id: int) -> None:
    existing = ed.find_link_to_slot(TRIM_AUDIO_DURATION_ID, 0)
    if existing is None:
        raise SystemExit(
            f"TrimAudioDuration({TRIM_AUDIO_DURATION_ID}).audio has no inbound link; "
            "iclora workflow shape unexpected."
        )
    if existing[1] != LOAD_AUDIO_ID:
        raise SystemExit(
            f"Expected TrimAudioDuration.audio source = LoadAudio({LOAD_AUDIO_ID}), "
            f"got {existing[1]}."
        )
    ed.remove_link(existing[0])
    ed.add_link(LOAD_AUDIO_ID, 0, profile_id, 0, "AUDIO")
    ed.add_link(profile_id, 0, TRIM_AUDIO_DURATION_ID, 0, "AUDIO")


def _splice_profile_end(ed: WorkflowEditor, profile_id: int) -> None:
    existing = ed.find_link_to_slot(LATENT_CONCAT_ID, LATENT_CONCAT_TARGET_SLOT)
    if existing is None:
        raise SystemExit(
            f"LatentConcat({LATENT_CONCAT_ID}).slot{LATENT_CONCAT_TARGET_SLOT} "
            "has no inbound link; iclora workflow shape unexpected."
        )
    if existing[1] != TENSOR_LOOP_CLOSE_ID:
        raise SystemExit(
            f"Expected LatentConcat source = TensorLoopClose({TENSOR_LOOP_CLOSE_ID}), "
            f"got {existing[1]}."
        )
    ed.remove_link(existing[0])
    ed.add_link(TENSOR_LOOP_CLOSE_ID, 0, profile_id, 0, "LATENT")
    ed.add_link(profile_id, 0, LATENT_CONCAT_ID, LATENT_CONCAT_TARGET_SLOT, "LATENT")


def _splice_profile_iter_step(ed: WorkflowEditor, profile_id: int) -> None:
    existing = ed.find_subgraph_link_to_slot(
        SUBGRAPH_OUTPUT_COLLECTOR, SUBGRAPH_OUTPUT_LATENT_SLOT, 0,
    )
    if existing is None:
        raise SystemExit(
            f"Subgraph output -20.{SUBGRAPH_OUTPUT_LATENT_SLOT} (extended_latent) "
            "has no inbound link; iclora workflow shape unexpected."
        )
    if existing["origin_id"] != ITERATION_CLEANUP_ID:
        raise SystemExit(
            f"Expected subgraph output source = IterationCleanup({ITERATION_CLEANUP_ID}), "
            f"got {existing['origin_id']}."
        )
    ed.remove_subgraph_link(existing["id"], 0)
    ed.add_subgraph_link(
        ITERATION_CLEANUP_ID, 0, profile_id, 0, "LATENT", 0,
    )
    ed.add_subgraph_link(
        profile_id, 0,
        SUBGRAPH_OUTPUT_COLLECTOR, SUBGRAPH_OUTPUT_LATENT_SLOT,
        "LATENT", 0,
    )


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if input_path != output_path and output_path.exists():
        if _already_applied(WorkflowEditor(output_path)):
            print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
            return

    if not dry_run and input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed_target = output_path if (output_path.exists() and not dry_run) else input_path
    ed = WorkflowEditor(ed_target)

    if _already_applied(ed):
        print(f"{output_path.name}: already migrated, skipping.")
        return

    missing = ed.require_nodes((
        LOAD_AUDIO_ID, TRIM_AUDIO_DURATION_ID, TENSOR_LOOP_CLOSE_ID, LATENT_CONCAT_ID,
    ))
    if missing:
        raise SystemExit(f"Refusing to migrate: missing top-level nodes {missing}")
    sg = ed.get_subgraph(0)
    if sg is None:
        raise SystemExit("Refusing to migrate: no subgraph in workflow")
    sg_node_ids = {n["id"] for n in sg.get("nodes", [])}
    if ITERATION_CLEANUP_ID not in sg_node_ids:
        raise SystemExit(
            f"Refusing to migrate: subgraph missing IterationCleanup({ITERATION_CLEANUP_ID})"
        )

    if dry_run:
        print(f"would migrate {output_path.name} (3 profile nodes + 6 link edits)")
        return

    print(f"{output_path.name}: applying ProfileBegin/IterStep/End wiring...")

    pb_id = _add_profile_begin(ed)
    _splice_profile_begin(ed, pb_id)
    print(f"  ProfileBegin id={pb_id} (spliced LoadAudio -> ProfileBegin -> TrimAudioDuration)")

    pe_id = _add_profile_end(ed)
    _splice_profile_end(ed, pe_id)
    print(f"  ProfileEnd id={pe_id} (spliced TensorLoopClose -> ProfileEnd -> LatentConcat)")

    pis_id = _add_profile_iter_step_subgraph(ed)
    _splice_profile_iter_step(ed, pis_id)
    print(f"  ProfileIterStep id={pis_id} (spliced IterationCleanup -> ProfileIterStep -> output)")

    ed.save(output_path)
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print( "  2. Launch ComfyUI via ./start_experiment.sh (sets RUN_ID + telemetry envs)")
    print(f"  3. Load {output_path.name} in ComfyUI; queue prompt; wait for completion")
    print( "  4. Inspect artifacts:")
    print( "       data/runs/${RUN_ID}/profiler/trace.json   (perfetto.dev / chrome://tracing)")
    print( "       data/runs/${RUN_ID}/profiler/summary.txt  (top kernels)")
    print( "       data/runs/${RUN_ID}/profiler/memory_timeline.html  (VRAM)")
    print( "  5. Combine with exec_log_summary + sage_telemetry_summary for the full picture")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--revert", action="store_true",
                    help="Delete the staged bench variant.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return 0

    _migrate(Path(args.input), output_path, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
