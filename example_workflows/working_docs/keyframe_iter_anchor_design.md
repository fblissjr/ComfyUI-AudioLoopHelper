# audio-loop keyframe variant — per-iter keyframe re-anchoring

Last updated: 2026-05-25

> **STATUS:** Generated + audit-clean (47 OK / 1 WARN pre-existing
> latent_volume / 0 ERR). Render-gate pending before relying on it.
> Ships as `example_workflows/audio-loop-music-video_latent_keyframe.json`;
> generator `scripts/apply_keyframe_iter_anchor.py`.

## Goal

Let the user pin different keyframe images to different loop iterations of
the canonical audio-loop workflow, to combat DiT drift on long renders and
to drive scene changes synced to song structure. "Image X anchors iters
[a, b, c]."

## Why on the canonical (not fml2v)

The fml2v flat-canvas build (210 nodes) reinvented the canonical's loop
body and fought it for a whole session. The canonical
(`audio-loop-music-video_latent.json`, 99+18 subgraph) is the proven spine:
single-pass full-res loop body with `LatentContextExtract`, `AudioLatentSlice`,
AdaIN, correct `LTXVAudioVideoMask` timing. The keyframe feature attaches to
it as a top-level add — no loop-body surgery.

## Mechanism

The canonical subgraph already exposes a `guide_latent` input (sg.input[8] →
`LTXVAddLatentGuide`), today fed a static init-image latent every iter. The
keyframe variant intercepts that feed at the **top level** with a selector:

```
LoadImage_kf → LTXSmartImageResize(FramePlanner dims) → LTXVPreprocess(18) → VAEEncode ─┐
                                                                                        ├→ LTXIterKeyframeSchedule ─→ #843.guide_latent
#1617 VAEEncode (init) ─────────────────────────────────────────────────────────────────┘ (fallback)
TLO #1539.current_iteration ──────────────────────────────────────────────────────────────┘
```

`LTXIterKeyframeSchedule` (in `nodes.py`) is a pure selector: per iter, returns
the keyframe latent whose `target_iters` contains `current_iteration` (lowest-
index row wins); else returns `fallback_latent` (the init latent). No VAE, no
tensor mutation — keyframes encode once outside the loop.

Each keyframe encoder chain mirrors the init guide chain
(`LoadImage→LTXSmartImageResize→LTXVPreprocess(18)→VAEEncode`) so the keyframe
latent is shape-compatible with what `guide_latent` expects. LoadImage defaults
to the init placeholder so unset keyframes don't crash the eager encode.

## Key decisions

| # | Decision | Why |
|---|---|---|
| 1 | Selector OUTSIDE the loop, feeds existing `guide_latent` | No subgraph schema change; `current_iteration` is available at top level (TLO out[3]); reuses the proven anchor machinery |
| 2 | `first_frame_guide_strength = 1.0` | At 1.0 the `LTXVAddLatentGuide` noise_mask = `max(0,1-strength)` = 0 → HARD lock (combats drift) AND the fast path (frozen guide frame skipped in attention/FFN; <1.0 denoises it as an active token → much slower, per sage/ffn testing) |
| 3 | `target_iters` is **1-based** | `TensorLoopOpen.current_iteration` emits 1,2,3,… (`ComfyUI-NativeLooping/nodes.py:113`). Iter 0 is the out-of-loop init render. `target_iters='0'` would be dead |
| 4 | Keyframe anchors at idx=-1 (window tail), via the existing `LTXVAddLatentGuide` | The overlap carries the tail into the next iter as frozen context, so a keyframe change drives a smooth one-iter transition (prev keyframe via overlap → new keyframe via anchor). `LatentOverlapTrim` trims the head, never the keyframe tail |
| 5 | Keyframe images replicate the init resize+preprocess chain | Otherwise a different-sized keyframe yields a wrong-shaped guide latent → crash at `LTXVAddLatentGuide` on matched iters (caught in code review) |

## Overlap × keyframe-change interaction

Verified sound. `LatentContextExtract` pulls prev iter's **tail** → current iter's
**head** (frozen context). `LatentOverlapTrim` trims the **head**. The keyframe
anchors the **tail** (idx=-1), which is never trimmed and becomes the next iter's
frozen head. So changing the keyframe at iter K → iter K ends on keyframe_K → iter
K+1 starts frozen on keyframe_K and denoises toward keyframe_{K+1} → smooth
transition, no duplication. Bigger `overlap_seconds` = more prior keyframe carried
= smoother/slower transition. Timing: "keyframe at iter K" lands at the END of iter
K's window.

## Usage

1. `bash start_experiment.sh nodynvram`
2. Load `audio-loop-music-video_latent_keyframe.json`
3. Set the 3 keyframe `LoadImage` files + the init `LoadImage #444`
4. On `LTXIterKeyframeSchedule`, set `target_iters` per row (1-based), e.g.
   `target_iters_1='1'`, `target_iters_2='3'`, `target_iters_3='5'`. Check
   `AudioLoopPlanner.summary` for the song's iter count.
5. Queue. Un-targeted iters use the init image (identical to the no-keyframe canonical).

If the DynamicCombo keyframe rows don't expand in the UI, delete + re-add
`LTXIterKeyframeSchedule` from the node menu and rewire (slot indices bake at
save time).

## Footgun: empty `target_iters` → silent fallback to init image

The shipped default leaves every `target_iters` row **EMPTY**. With no row
claiming any iteration, `LTXIterKeyframeSchedule` returns `fallback_latent`
(the init latent) on *every* iter — so the keyframes never fire and the render
is bit-identical to the no-keyframe canonical. No error, no warning: it just
looks like the keyframes did nothing.

If the keyframes "aren't working," check `target_iters` first:

- At least one row must name an iteration. `target_iters_1='1'` is the minimum
  to see any keyframe at all.
- **`target_iters` is 1-based** (see decision #3). `TensorLoopOpen.current_iteration`
  emits 1, 2, 3, …; iter 0 is the out-of-loop init render. `target_iters='0'` is
  dead — it matches no iteration and falls back silently.
- Rows accept comma-separated lists (`'1,2,3'`); lowest-index matching row wins
  when ranges overlap.

## Variant: `_keyframe_autoextract` (no hand-loading)

Staged under `example_workflows/experimental/audio-loop-music-video_latent_keyframe_autoextract.json`.
Same selector + anchor mechanism as the shipped keyframe workflow; only the
keyframe *source* differs. Instead of three hand-loaded `LoadImage` files, the
keyframes are sampled from a video clip:

```
VHS_LoadVideo (clip) → EvenlySpacedKeyframes(count=3) → frame batch ─┬→ GetImageRangeFromBatch(0,1) → keyframe-1 encode chain
                                                                     ├→ GetImageRangeFromBatch(1,1) → keyframe-2 encode chain
                                                                     └→ GetImageRangeFromBatch(2,1) → keyframe-3 encode chain
```

- `EvenlySpacedKeyframes` (in `nodes.py`) picks `count` frames spread evenly
  across the IMAGE batch — `count=3` = first / middle / last, endpoints always
  included. `count` is **fixed at 3** here to match the schedule's 3 keyframe
  slots; the three `GetImageRangeFromBatch` (KJNodes) nodes each split one frame
  (`start_index` 0/1/2, `num_frames=1`) into the existing keyframe encode chains.
- Each split frame then flows through the same `LTXSmartImageResize →
  LTXVPreprocess(18) → VAEEncode` chain as the hand-loaded variant, so the
  selector and anchor downstream are unchanged.
- The empty-`target_iters` footgun above applies identically — the auto-extract
  only replaces *which images* the keyframes are; you still have to assign them
  to iterations (1-based) or every iter falls back to the init.

## Open / next

- **Render gate**: structurally sound + audit-clean, not yet render-verified. Gate before promoting in docs as production.
- **Hard-lock escalation**: strength=1.0 via `LTXVAddLatentGuide` is a tail anchor, not a frame-0 `noise_mask=0` write. If keyframes still don't hold on very long renders, the escalation is an in-subgraph writer (needs the schema change avoided here).
- **Timestamp variant**: `KeyframeLatentScheduleBatchEncode + LatentSelectByIteration` (shipped) offers song-time scheduling instead of iter-index, with no DynamicCombo. Consider if iter-index proves unintuitive.
