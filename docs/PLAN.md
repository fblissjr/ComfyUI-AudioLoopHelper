Last updated: 2026-04-16

# Phase 1 Validation + Conditional Next Steps

## Context

We built `KeyframeImageSchedule`, `VideoFrameExtract`, `ImageBlend` and a new
example workflow `audio-loop-music-video_latent_keyframe.json`. The user has
not yet run this workflow end-to-end against any real audio + keyframe set.

Three uncertainties were investigated before deciding what to build next:
1. **DynamicCombo widget fragility** — only one real constraint exists
   (PrimitiveNode cannot feed sub-inputs). Existing workflow already uses
   DynamicCombo successfully via `LTXVImgToVideoInplaceKJ`. Not a blocker.
2. **End-of-window guide semantics** — confirmed via `comfy_extras/nodes_lt.py`
   `append_keyframe()` and LTX-2 native `KeyframeInterpolationPipeline`.
   Feeding two guides at different `frame_idx` values IS keyframe interpolation
   — the canonical pattern, not weird. Multi-guide approach is sound.
3. **Retake node feasibility** — confirmed simple. LTX-2 `TemporalRegionMask`
   is just `denoise_mask = 1` inside region, `0` outside. ComfyUI already
   has the `latent["noise_mask"]` field. A Retake node is ~50 lines:
   construct a 5D mask tensor, attach to latent dict.

Given Phase 1 is unvalidated, building Phase 2 (multi-guide subgraph) on top
risks debugging two layers at once. Better to test Phase 1 first.

## Plan

### Step 1: Validate Phase 1 (no code changes)

Run `audio-loop-music-video_latent_keyframe.json` end-to-end with a real
song and at least 3 distinct keyframe images. Use a `prompt3.md`-style
schedule with hard scene changes.

What to observe at iteration 5+ (where drift would normally show):
- **Does the keyframe image actually change visible scenes?** Not just
  lighting/color — does the spatial layout shift to match the new keyframe?
- **Is `blend_seconds=0.0` (hard cut) usable, or does it cause artifacts
  worse than a 1-2s blend?**
- **Does `blend_seconds=5.0` produce smooth visual transitions or just
  morph-mush?**
- **Lip sync at iteration boundaries** — does swapping the guide image
  break audio-video alignment?

### Step 2: Decision tree based on observations

Three possible outcomes drive different next steps:

#### Outcome A: Phase 1 works well — keyframe switching gives clear scene changes
**Action**: Don't build Phase 2. Document recommended settings in
`docs/keyframe_workflow_guide.md` (blend_seconds tuning, image batch
preparation, schedule patterns matching prompt3.md/prompt4.md). Move on
to Retake (Step 3).

#### Outcome B: Phase 1 partially works — transitions are mushy or model snaps back to first guide
**Action**: Build Phase 2 (multi-guide-per-window). Replace
`LTXVAddLatentGuide` (node 1519 in subgraph #843) with KJNodes'
`LTXVAddGuideMulti`. Wire `KeyframeImageSchedule.image` at frame_idx=0
and `KeyframeImageSchedule.next_image` at frame_idx=-1 (end of window).
This forces the model to interpolate within the window rather than just
using one anchor.

Implementation outline (when Outcome B is confirmed):
- Use `WorkflowEditor` from `scripts/workflow_utils.py` (per CLAUDE.md
  convention, never manual subgraph JSON edits)
- Add a new IMAGE input slot to subgraph #843 for `next_image`
- Replace node 1519's type from `LTXVAddLatentGuide` to `LTXVAddGuideMulti`
- DynamicCombo widget format: `[num_guides=2, strength_1, strength_2, frame_idx_1=0, frame_idx_2=-1]`
- `LTXVCropGuides` (node 655) already handles multiple guides correctly
  via `keyframe_idxs` metadata — no changes needed there
- Increment subgraph_node "size" — second IMAGE input adds a slot
- Bump versions in `pyproject.toml`, `CHANGELOG.md`

#### Outcome C: Phase 1 breaks lip sync or causes serious artifacts
**Action**: Diagnose root cause before building anything else. Likely
candidates: noise_mask not being stripped properly when init_image
swaps mid-loop, or guide encoding cost causing iteration timing drift.

### Step 3: Build Retake node (independent of Phase 1 outcome)

Useful regardless of which outcome above. Lets users fix one bad
iteration without re-rendering the whole song.

New node `LatentTemporalMask` in `nodes.py`:
- Inputs: LATENT, start_time (FLOAT), end_time (FLOAT), fps (FLOAT, default 25)
- Outputs: LATENT with noise_mask set to 1.0 inside [start_time, end_time],
  0.0 elsewhere
- Logic: build a 5D mask tensor matching the latent shape, where mask values
  along the temporal dim are determined by frame index ranges
- Reference implementation: `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py:110`
  (TemporalRegionMask.apply_to)
- ComfyUI already has noise_mask infrastructure: `comfy_extras/nodes_lt.py:175`
  `get_noise_mask()` shows the format

Workflow usage: after the loop completes, user feeds the accumulated
video latent + a time range into LatentTemporalMask, then runs another
SamplerCustomAdvanced pass with a fresh prompt for that region only.

This does NOT require subgraph changes. Standalone node + standalone
workflow snippet.

## Files

### Step 1 (no changes)
- `example_workflows/audio-loop-music-video_latent_keyframe.json` — test as-is

### Step 2B (only if needed)
- `example_workflows/audio-loop-music-video_latent_keyframe.json` — modify subgraph
- `scripts/build_keyframe_workflow.py` — extend to wire LTXVAddGuideMulti
- `CLAUDE.md` — document Phase 2 wiring

### Step 3
- `nodes.py` — add `LatentTemporalMask` node class
- `tests/test_keyframe_nodes.py` — add tests for mask construction
- `example_workflows/audio-loop-music-video_retake.json` — new minimal workflow
- `CLAUDE.md` — add LatentTemporalMask description

## Verification

### Step 1
Manual: load workflow, run with test audio + 3 keyframes + prompt3-style
schedule. Visual inspection of output video.

### Step 2B (if needed)
- All existing tests still pass: `uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.`
- Workflow integrity test passes for modified workflow
- Visual regression: compare output to Step 1 baseline

### Step 3
- Unit tests for mask construction (start/end time → mask values)
- Run a partial-region regeneration on a known-good video latent
