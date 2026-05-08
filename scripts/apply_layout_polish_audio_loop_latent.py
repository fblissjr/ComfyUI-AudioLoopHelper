"""apply_layout_polish_audio_loop_latent.

Last updated: 2026-05-08

Stages a polished-layout variant of `audio-loop-music-video_latent.json`
into `internal/workflows/`. **Layout only** — no topology changes, no
new nodes, no rewires. Just `pos` / `size` / `groups[]` rewrites driven
by `scripts/_layout_grid.py`.

Symptom / motivation: shipped layout is hard to scan — Get/Set reroutes
look identical in weight to user inputs, notes float in arbitrary space,
no tier hierarchy ("must change every render" vs "set once"). User has
to hunt for what to tune. Screenshots: `internal/scratch/layout{1,2,3}.png`.

Root cause: `apply_intro_workflow._layout_workflow` uses one large
"1. Inputs" group for everything — input widgets, audio reroutes, seed
constants, image-strength constant. Visual hierarchy is flat.

Fix: split the inputs column into two sub-tier groups —

  1.1 REQUIRED — change every render (audio file, init image, seed,
                 frame planner, prompt schedule, initial-render prompt)
  1.2 COMMON   — change occasionally (audio trims, overlap target,
                 image strength)

Frozen-with-widgets nodes (loaders, sampler triplet) stay in their
existing functional columns (Models, Sampler) but those columns get a
clearer color signal so the user knows "these have widgets but are set
once / don't touch."

Layout deviates from `internal/PLAN.md` 4-tier proposal (REQUIRED/COMMON/
ADVANCED/FROZEN). Cramming all 30+ widget-bearing nodes into one column
overflows past 5000px — column 1 becomes a wall. Two-tier split keeps
inputs panel scannable and lets functional columns carry their own
"set once" signal via color.

Compatibility:
  - F-pair convention: staged variant (`internal/workflows/`) skips
    F-pair until promotion. Per `scripts/CLAUDE.md` "Carve-out for
    staged-variant scripts."
  - No new node types. No widget changes. No link changes. Existing
    audits pass unchanged.

Usage:
    uv run --group dev python scripts/apply_layout_polish_audio_loop_latent.py
    uv run --group dev python scripts/apply_layout_polish_audio_loop_latent.py --revert
    uv run --group dev python scripts/apply_layout_polish_audio_loop_latent.py --dry-run

Idempotent. `--revert` deletes the staging file. `--from-template
<golden.json>` reads node positions from a hand-laid-out workflow and
applies that grid to the target instead of the hardcoded spec.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _layout_classifications import compose  # noqa: E402
from _layout_grid import (  # noqa: E402
    GroupSpec,
    LayoutSpec,
    NoteAnchor,
    apply_layout,
    extract_template,
    summarize,
)
from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/workflows/audio_loop_latent_polished.json"

# --------------------------------------------------------------------------
# Group keys
# --------------------------------------------------------------------------

G_REQUIRED   = "1_1_required"
G_COMMON     = "1_2_common"
G_MODELS     = "2_models"
G_LORAS      = "3_loras"
G_COND       = "4_cond"
G_SAMPLER    = "5_sampler"
G_LOOP       = "6_loop"
G_OUTPUT     = "7_output"
G_PREENCODE  = "8_preencode"
G_ICLORA_REF = "9_iclora_ref"

# Tag keys (must match what apply_intro_workflow.py wrote into the
# canonical workflow's runtime-added LoRA loaders + Notes).
GROUP_TAG_KEY = "_alh_group"
NOTE_KEY_TAG = "_alh_note_key"

# --------------------------------------------------------------------------
# Color palette
# --------------------------------------------------------------------------

# Inputs sub-tiers carry the "what changes per render" signal.
COLOR_REQUIRED = "#29699c"   # bright blue — change every render
COLOR_COMMON   = "#3f789e"   # medium blue — change occasionally

# Functional columns. Loaders + sampler get a "set once" color hint.
COLOR_FROZEN_LOADERS = "#322"   # dark red — set once, do NOT touch widgets
COLOR_FROZEN_SAMPLER = "#322"
COLOR_LORAS    = "#1b4669"      # deep blue — bypassed by default
COLOR_COND     = "#485248"      # green — conditioning chain (topology)
COLOR_LOOP     = "#3f789e"      # medium blue — loop iteration spine
COLOR_OUTPUT   = "#b58b2a"      # gold — terminal sinks
COLOR_PREENCODE = "#3f789e"
COLOR_ICLORA   = "#a18c25"      # gold — IC-LoRA (bypassed)

# --------------------------------------------------------------------------
# Column origins
# --------------------------------------------------------------------------

# Row 0 — main pipeline. Wider gap between Inputs and Models because
# the inputs column houses tall multi-line CLIP encode + schedule nodes.
ROW0_Y_TOP = 200
ROW0_REQUIRED_Y = ROW0_Y_TOP
# COMMON sub-tier sits below REQUIRED. The y-offset is hand-tuned to
# accommodate the multiline CLIPTextEncode (Node 169) and the schedule
# batch encoder which together dominate REQUIRED's vertical extent.
# Iterate via Chrome MCP screenshots if this lands wrong.
ROW0_COMMON_Y = 2400

ROW0_COL_X = {
    "inputs":       0,
    "models":       900,
    "loras":        1500,
    "cond":         2050,
    "sampler":      2900,
    "loop":         3500,
    "output":       4500,
}

# Row 1 — preprocessing + IC-LoRA reference (below row 0).
ROW1_Y = 3300
ROW1_COL_X = {
    "preencode":  0,
    "iclora_ref": 1500,
}


# --------------------------------------------------------------------------
# Functional column → group-key mapping.
#
# Default tier for inputs is COMMON; specific input nodes that change
# every render are pinned to REQUIRED via _OVERRIDES below.
# --------------------------------------------------------------------------

_FUNCTION_TO_GROUP: dict[str, str] = {
    "inputs":     G_COMMON,
    "models":     G_MODELS,
    "loras":      G_LORAS,
    "cond":       G_COND,
    "sampler":    G_SAMPLER,
    "loop":       G_LOOP,
    "output":     G_OUTPUT,
    "preencode":  G_PREENCODE,
    "iclora_ref": G_ICLORA_REF,
}

# Overrides + additions on top of `SHARED_NODE_FUNCTIONS`. Two roles:
#   1. Pin inputs that change every render to REQUIRED (separating from
#      the COMMON default).
#   2. Move 1634/1615 from cond column to REQUIRED tier (their functional
#      column in shared is "cond" because intro lays them out there; the
#      polish surfaces them as user inputs).
# Tier rationale:
#   REQUIRED — change every render (audio, image, seed, prompts, dim plan)
#   COMMON   — tune occasionally (trims, overlap target, image strength)
_OVERRIDES: dict[int, str] = {
    565:  G_REQUIRED,        # LoadAudio
    444:  G_REQUIRED,        # LoadImage
    1527: G_REQUIRED,        # INTConstant start_seed
    1634: G_REQUIRED,        # LTXFramePlanner (functional: cond; surfaced as REQUIRED)
    1615: G_REQUIRED,        # TimestampPromptScheduleBatchEncode (functional: cond; surfaced as REQUIRED)
    1631: G_COMMON,          # TrimAudioDuration ID-LoRA Reference Slice (functional: loras; surfaced as COMMON)
}

# Additions for nodes not present in shared classifications (post-intro).
_ADDITIONS: dict[int, str] = {
    2013: G_COMMON,          # FloatConstant (overlap_seconds target)
    2021: G_COND,            # ConditioningSelectByIteration (initial render selector)
}

NODE_GROUPS: dict[int, str] = compose(_FUNCTION_TO_GROUP, overrides=_OVERRIDES) | _ADDITIONS


# --------------------------------------------------------------------------
# Note placement: anchored to the closest meaningful group.
# (note_key, NoteAnchor(group, dx, dy, w, h))
# --------------------------------------------------------------------------

# README is the workflow-level overview — anchor to REQUIRED but float
# above. Tighter dy than the previous -680 (was way out in space).
NOTE_PLACEMENTS: dict[str, NoteAnchor] = {
    "README":   NoteAnchor(group=G_REQUIRED, dx=0, dy=-360, w=660, h=320),
    "LORA":     NoteAnchor(group=G_LORAS,    dx=0, dy=-300, w=560, h=260),
    "NODE_169": NoteAnchor(group=G_REQUIRED, dx=720, dy=0, w=320, h=300),
    "SCHEDULE": NoteAnchor(group=G_REQUIRED, dx=720, dy=320, w=320, h=300),
    "ICLORA":   NoteAnchor(group=G_ICLORA_REF, dx=0, dy=-260, w=620, h=220),
}


def _build_spec() -> LayoutSpec:
    return LayoutSpec(
        groups={
            G_REQUIRED:   GroupSpec(origin=(ROW0_COL_X["inputs"], ROW0_REQUIRED_Y),
                                    color=COLOR_REQUIRED, title="1.1 REQUIRED — change every render"),
            G_COMMON:     GroupSpec(origin=(ROW0_COL_X["inputs"], ROW0_COMMON_Y),
                                    color=COLOR_COMMON, title="1.2 COMMON — tune occasionally"),
            G_MODELS:     GroupSpec(origin=(ROW0_COL_X["models"], ROW0_Y_TOP),
                                    color=COLOR_FROZEN_LOADERS,
                                    title="2. Models (set once — DiT, VAEs, CLIP, Sage)"),
            G_LORAS:      GroupSpec(origin=(ROW0_COL_X["loras"], ROW0_Y_TOP),
                                    color=COLOR_LORAS,
                                    title="3. LoRAs (bypassed)"),
            G_COND:       GroupSpec(origin=(ROW0_COL_X["cond"], ROW0_Y_TOP),
                                    color=COLOR_COND,
                                    title="4. Conditioning chain"),
            G_SAMPLER:    GroupSpec(origin=(ROW0_COL_X["sampler"], ROW0_Y_TOP),
                                    color=COLOR_FROZEN_SAMPLER,
                                    title="5. Sampler (set once — distilled 8-step path)"),
            G_LOOP:       GroupSpec(origin=(ROW0_COL_X["loop"], ROW0_Y_TOP),
                                    color=COLOR_LOOP,
                                    title="6. Loop"),
            G_OUTPUT:     GroupSpec(origin=(ROW0_COL_X["output"], ROW0_Y_TOP),
                                    color=COLOR_OUTPUT,
                                    title="7. Output"),
            G_PREENCODE:  GroupSpec(origin=(ROW1_COL_X["preencode"], ROW1_Y),
                                    color=COLOR_PREENCODE,
                                    title="8. Audio pre-encode + init render path"),
            G_ICLORA_REF: GroupSpec(origin=(ROW1_COL_X["iclora_ref"], ROW1_Y),
                                    color=COLOR_ICLORA,
                                    title="9. IC-LoRA reference (bypassed)"),
        },
        node_groups=NODE_GROUPS,
        note_anchors=NOTE_PLACEMENTS,
        group_tag_key=GROUP_TAG_KEY,
        note_key_tag=NOTE_KEY_TAG,
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _migrate(input_path: Path, output_path: Path, *, dry_run: bool, template_path: Path | None) -> None:
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")

    if template_path is not None:
        if not template_path.exists():
            raise SystemExit(f"template does not exist: {template_path}")
        template_ed = WorkflowEditor(template_path)
        spec = extract_template(template_ed.wf)
        # Preserve note-anchor convention from the hardcoded spec — the
        # extracted template doesn't know about notes (extracted spec
        # has empty note_anchors).
        spec.note_anchors = NOTE_PLACEMENTS
        spec.group_tag_key = GROUP_TAG_KEY
        spec.note_key_tag = NOTE_KEY_TAG
        print(f"  using template: {template_path}")
    else:
        spec = _build_spec()

    if dry_run:
        ed = WorkflowEditor(input_path)
        print(f"would copy   {input_path} -> {output_path}")
        print(summarize(spec, ed.wf))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path != output_path:
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    apply_layout(ed.wf, spec)
    ed.save()

    print(f"  laid out {len(ed.wf.get('groups', []))} groups")
    print(summarize(spec, ed.wf))
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit:         uv run --group dev python scripts/audit_workflows.py {output_path}")
    print(f"  3. Load in ComfyUI: open {output_path}")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"source workflow (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"output path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--from-template", default=None, metavar="GOLDEN_JSON",
                    help="extract layout grid from a hand-laid-out workflow "
                         "instead of using hardcoded spec; useful for "
                         "round-trip iteration in ComfyUI")
    ap.add_argument("--revert", action="store_true",
                    help="delete the output staged file")
    ap.add_argument("--dry-run", action="store_true",
                    help="report planned ops without writing")
    args = ap.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    template_path = None
    if args.from_template is not None:
        template_path = Path(args.from_template)
        if not template_path.is_absolute():
            template_path = REPO_ROOT / template_path

    if args.revert:
        _revert(output_path)
        return

    _migrate(input_path, output_path, dry_run=args.dry_run, template_path=template_path)


if __name__ == "__main__":
    main()
