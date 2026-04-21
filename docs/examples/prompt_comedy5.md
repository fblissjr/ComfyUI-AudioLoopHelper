Last updated: 2026-04-21

# Standup Comedy — Example Prompt Schedule (v5) — Unusual-Character Adaptation

Adapts the v4 schedule structure to an init image whose subject is
outside LTX's typical training distribution. v4 was authored around a
generic "male comedian in a striped sweater." When the init image
shows a very different character — a pale performer with an oversized
bald head, floral-patterned blazer, and tan mock-neck — every entry
needs the subject block rewritten to match what cross-attention will
see in the init image. The v4 cut-language and shot-size structure
still works; only identity strings and the 0:00 pose change.

## Why the rewrite

1. **Text/image conflict destroys identity anchoring.** If the prompt
   says "striped sweater" but the image shows a floral blazer,
   cross-attention gets contradictory signals and the character's
   clothing (and by extension body shape, since clothing is the
   loudest color cue) drifts across iterations.
2. **Node 169 must match 0:00 must match init image pose.** If the
   image is mid-bit (mouth wide open, free hand raised in an emphatic
   gesture) — not "pausing for the laugh, slight smile" — using the v4
   0:00 verbatim would create ~2s of morphing at the initial-render →
   first-iteration seam as the model tries to reconcile a pose that
   isn't in the frame.
3. **Head-shape normalization risk.** An enlarged dome cranium is
   outside LTX's typical distribution. At the 3-step commit phase of
   the distilled sampler (σ: 0.725 → 0.422 → 0), the model may try
   to regress proportions toward a conventional skull. Naming the
   feature ("oversized bald head") in every subject block gives
   cross-attention a loud, constant anchor to preserve it.

## Inputs

- **Audio**: same standup source as v4 (e.g., 184s preprocessed WAV).
- **Image**: an unusual-character standup init image (oversized bald
  head, floral-patterned blazer).
- **Subject (byte-exact, include in every schedule entry)**:
  `a pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck`

## Schedule

Grid-aligned for `overlap_seconds=2` (stride 17.92s). With
`snap_boundaries=True` (default) the same schedule also works at
`overlap_seconds=3` — the node rounds boundaries to the 16.96s grid
at runtime.

```
node_169_prompt: Style: cinematic. In a medium shot, static camera, a pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is mid-bit, mouth wide open, free hand raised in an emphatic gesture. Warm stage wash. The crowd on the right mid-laugh.

schedule:
0:00-0:17: Style: cinematic. In a medium shot, static camera, a pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is mid-bit, mouth wide open, free hand raised in an emphatic gesture. Warm stage wash. The crowd on the right mid-laugh.
0:17-0:35: Cut to a medium close-up, static camera. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is delivering the setup, raising an eyebrow, delivery in a dry deadpan. Warm stage wash. The crowd quiet, leaning in.
0:35-0:53: Cut to a close-up, slow dolly in. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is telling a joke, eyes wide with conviction, shaking his head slightly. Warm stage wash. Brisk rhythmic delivery.
0:53-1:11: Cut to a close-up, static camera. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is delivering the punchline, leaning into the mic, mouth open. Warm stage wash. One person in the crowd on the right slapping the table laughing.
1:11-1:29: Cut to a medium shot, slight handheld sway. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is mid-bit, gesturing with his free hand, shifting weight between feet. Warm stage wash. The crowd watching attentively.
1:29-1:47: Cut to a medium close-up, rack focus shifting to him. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is smiling wryly, looking out into the audience, head tilted. Warm stage wash. A couple of patrons on the right whispering.
1:47-2:05: Cut to a medium shot, static camera. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is telling a joke, gesturing sharply with his free hand to emphasize a point, mic slightly lowered. Warm stage wash. Crowd members shifting in their seats.
2:05-2:23: Cut to a close-up, slow dolly in. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is delivering the punchline, leaning back slightly, pointing at a crowd member, eyes narrowed playfully. Warm stage wash. Someone on the right wiping their eye from laughing.
2:23-2:41: Cut to a medium close-up, static camera. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is leaning into the mic, building the final premise, voice intimate and intense. Warm stage wash. The crowd leaning in, highly attentive.
2:41-2:58: Cut to a close-up, slow dolly in. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is delivering the final punchline, smiling wide, mic lowered. Warm stage wash. The crowd fully laughing, shoulders shaking.
2:58+: Cut to a wide shot, slow dolly out, camera pulling back. A pale man with an oversized bald head in a floral-patterned blazer and tan mock-neck is reacting to the crowd, waving his free hand, stepping back from the mic stand. Warm stage wash. The crowd animated, some standing, some wiping their eyes. Room tone settling.
```

## Recommended workflow settings (unchanged from v4-post-DR1)

| Widget | Node | Value |
|---|---|---|
| decoder type | 1597, 1604 | `LTXVTiledVAEDecode` |
| decoder widgets | 1597, 1604 | `[2, 2, 1, true, "auto", "auto"]` |
| `overlap_seconds` | AudioLoopController | `2.0` (start here; bump to `3.0` if iteration-boundary seams persist) |
| `blend_seconds` | 1558 TimestampPromptSchedule | `0.0` |
| `snap_boundaries` | 1558 TimestampPromptSchedule | `true` |
| `sampler_name` | 154 KSamplerSelect | `euler` |
| `scheduler` | 1421 BasicScheduler | `linear_quadratic, 8, 1` |
| `shift` | 1513 ModelSamplingSD3 | `13` |
| CFG | 153 CFGGuider | `1.0` |

## Negative prompt (start here)

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone
```

If head-shape drift shows up on the first run (subject starts looking
like a regular skull by iteration 5+), try adding:

```
, deformed head, misshapen skull, incorrect head shape, normal-sized head
```

## Watchouts specific to this character

- **Head-shape normalization**: the dome cranium is the single most
  at-risk feature. Most drift will show as the forehead slowly
  receding toward a conventional skull shape. Monitor iterations 3-6
  — that's usually where the commit-phase regression becomes visible.
- **1:29 rack focus**: the silhouette against blurred crowd may render
  oddly; if so, swap this entry's camera move to `static camera`.
- **Floral blazer color stability**: loud patterns sometimes get
  desaturated across iterations. If colors flatten, bump
  `overlap_seconds` to 3 (more conditioning runway per iteration).
- **Lip sync at extreme shot sizes**: close-ups help lip sync (more
  mouth pixels), but the cranium dominates the close-up frame in this
  character's case. If mouth resolution suffers, favor medium
  close-ups over true close-ups.

## Fallback path if the character drifts too much

1. First try: `overlap_seconds = 3` + rerun.
2. Second try: add the head-shape negative terms above.
3. Third try: remove the shot-size variety — use medium shot
   throughout (v3 conservative pattern). Sacrifices the
   comedy-special-edit feel but gives the most stable identity
   anchoring when the character is unusual.

## Patterns-that-transfer

- When the init image subject is outside LTX's training distribution
  (unusual proportions, loud patterns, non-photoreal styling), every
  schedule entry's subject block must describe the distinguishing
  feature in the same words so cross-attention has a constant anchor
  to preserve against the commit-phase regression pressure.
- Node 169's 0:00 prompt needs to describe the actual pose in the init
  image, not a generic entry point from a prior schedule.
- v4's cut-language and shot-size rotation still works for unusual
  characters; only identity strings change.
