# audio-loop endanchor variant — start/mid/end window anchoring

Last updated: 2026-06-05

> **STATUS:** Experimental; first full-song render surfaced the
> frozen-window footgun documented below (fixed at the node + workflow
> layer; render-gate with a varied keyframe source still pending).
> Ships as
> `example_workflows/experimental/audio-loop-music-video_latent_keyframe_endanchor.json`.
> Test predictions (P3.1-P3.3) pre-registered but ungraded.

## Goal

Anchor each loop window at its boundaries so window k's END matches window
k+1's START — "the start frame for this loop is the end of the prior loop."
Built on the canonical latent loop (same spine as the keyframe variant, see
`keyframe_iter_anchor_design.md`); keyframes auto-extracted from a source
video instead of hand-loaded stills.

## Mechanism

Per window k (inside the subgraph):

| anchor | node | latent_idx | strength | source |
|---|---|---|---|---|
| START | `#1519 LTXVAddLatentGuide` | -1 | 1.0 (`#1269`) | keyframe[k] (selector `#2057`) |
| MID (bypassed) | `#2044 LTXVAddLatentGuide` | 31 | 0.5 | keyframe via selector `#2059` |
| END | `#2043 LTXVAddLatentGuide` | 62 | 0.7 | keyframe[k+1] (selector `#2058`, iter+1 via `#2056 "a + 1"`) |

`latent_idx=62` assumes the 19.88s window / 25fps / overlap config (T=63
in-window latents); the in-workflow MarkdownNote carries the derivation —
recompute if the window config changes.

Keyframe supply chain (top level, encode-once-outside-loop):

```
VHS_LoadVideo #2047 (force_rate=1: 1 frame/sec, skip_first = seconds)
  → EvenlySpacedKeyframes #2048 (count auto-wired, see below)
  → LTXSmartImageResize #2053 (dims from LTXFramePlanner)
  → LTXVPreprocess #2054 (18)
  → KeyframeLatentScheduleBatchEncode #2055 (stride/duration from AudioLoopPlanner #1560)
  → LatentSelectByIteration #2057/#2058/#2059 (clamp-to-last = structural fallback)
```

`#2048.count` is wired from `AudioLoopPlanner.total_iterations + 1`
(`#2060 SimpleCalculatorKJ "a + 1"`) so the keyframe count tracks the song:
N windows consume keyframes 0..N (START of window k = keyframe k, END =
keyframe k+1). The widget value is a dead placeholder.

## The frozen-window footgun (found on first full-song render)

Symptom: initial render dynamic, every later window progressively less
dynamic, effectively frozen by the third window.

Cause chain — none of it image strength, all of it anchor *similarity*:

1. A short source clip sliced into many keyframes yields consecutive
   keyframes that are nearly identical (the shipped 9s test clip at
   `force_rate=1` gives 9 frames ~1s apart; `count=15` silently clamped
   to 9).
2. Window k is then pinned START (1.0) to still A and END (0.7) to still
   B ≈ A. The cheapest denoising path between two near-identical stills
   is a static morph. The initial render has no END anchor — which is why
   only iterations 2+ freeze.
3. Late in the song the selectors clamp to the LAST keyframe, so START
   and END become the *same* image — maximally frozen.

Keyframe COUNT does not add anchors per window (always exactly one START +
one END); count ÷ source-length sets the visual delta between consecutive
anchors, and that delta is each window's motion budget.

### Fixes shipped (2026-06-05)

- `EvenlySpacedKeyframes` WARNs on count clamp and on near-identical
  consecutive picks (mean abs pixel diff < 0.01 on a subsampled view), and
  emits a `placement_info` STRING output (node-family convention, mirrors
  `KeyframeGuidesTimeSpaced`).
- `KeyframeLatentScheduleBatchEncode` WARNs when schedule indices clamp to
  the batch range.
- Count autowire (above) replaces the magic widget value.

### Dial order when motion is still too damped

Pre-registered in the validation predictions doc; in order:

1. Fix the SOURCE: keyframes must span visually distinct moments roughly
   one window apart in narrative. A seconds-long clip cannot anchor a
   minutes-long song.
2. END strength `#2043`: 0.7 → 0.5 → 0.3.
3. Bypass `#2043` entirely (mode 4) → start-anchor-only behavior (the
   proven keyframe-variant shape).

## Alternatives considered

- **Dense per-window anchoring via `KeyframeGuidesTimeSpaced`** (multiple
  intra-window guides): parked — multiplies the anchor-similarity problem
  unless the source is genuinely varied.
- **Hand-wired per-keyframe node chains** (the pre-batch build): replaced
  by the batch encode + selector pattern — count is one widget (now one
  wire), no node duplication per keyframe.
- **MID anchor active by default**: shipped bypassed; a third pin further
  reduces motion freedom and the START/END pair already constrains the
  window.

## Open items

- Render-gate with a visually varied source video (grade P3.1-P3.3).
- If the variant proves out: promote out of experimental/ with an
  apply-script + audit F-pair (the live JSON has drifted from the original
  generator script; rebuild required at promotion).
- Possible follow-up: pre-render count-vs-source check in
  `LoopConfigValidator` (runtime WARN is the current backstop).
