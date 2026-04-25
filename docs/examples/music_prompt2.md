Last updated: 2026-04-25 (audio descriptors and init-committed redundancy stripped per `docs/guides/prompt_creation_guide.md`)

# Music Video — "the vocal track" / Warrior + Kitten (v2, simpler)

Companion to `internal/music_prompt1.md`. Same init image
(`your-illustrated-init-image.png`), same audio (`your-vocal-track.mp3`), same
grid (`overlap_seconds=2.0`, `stride=17.92s`), same subject
identity anchor. Differences:

1. **Simpler prompt text.** Each entry is one short sentence about
   shot + camera, one about action, done. No impact-frame /
   supersaturation / motion-blur-line art direction; no ambient
   audio description; no lighting shift clauses. The distilled LTX
   2.3 model's text conditioning has limited leverage at CFG 1.0
   anyway, so a crisp action verb and a canonical camera phrase
   carry most of the signal. Shorter prompts also cache better in
   `CachedTextEncode` when the same line recurs.
2. **Camera keywords are byte-exact to the canonical LTX 2.3 list**
   (`docs/guides/prompt_creation_guide.md` §7).
   Every camera clause is one of: `static camera, locked off shot`
   / `dolly in, camera pushing forward` / `dolly left, camera
   tracking left` / `dolly right, camera tracking right` /
   `jib up, camera rising up` / `jib down, camera lowering down` /
   `focus shift, rack focus`. Off-list phrasings from v1
   (`slight handheld sway`, `slow dolly in`, etc.) are removed.
3. **No `dolly out, camera pulling back` anywhere** — even on the
   outro. Pulling back over 18s of a sampler pass shrinks the face,
   loses lip-sync cross-attention signal, and can morph limbs. The
   outro is a held close-up instead; the audio fade closes the
   sequence. (Applied retroactively to v1 as well.)

## Why this might be the better default

- The v1 art-direction vocabulary ("impact frame", "supersaturation",
  "silhouetted accents", "motion blur lines") is the kind of layer
  that SHOULD help at CFG 4+ on a non-distilled model. At CFG 1.0
  on the distilled 8-step model, it's mostly paint that doesn't stick
  and may fight what the init image already encodes. Simpler prompts
  let the image and the audio do the driving.
- Fewer clauses per entry means fewer ways for Gemma 3's embedding
  to drift between iterations, which means less subject-identity
  drift across the 10-iteration loop. The same subject anchor phrase
  appears in every entry, surrounded by less noise.
- If this version produces better lip-sync and worse visual variety
  than v1, that tells you the art-direction vocabulary was
  contributing visual texture (not just noise). If it produces
  comparable lip-sync and comparable variety, v2 is strictly better
  (same output, less text to maintain). Worth A/B-ing both with the
  same seed.

## Inputs (unchanged from v1)

- **Audio**: `<your-vocal-track.mp3>`
- **Image**: `<your-illustrated-init-image.png>`
- **Subject string (byte-exact in every entry)**:

  `a warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=19.88`,
  `stride=17.92s`. Snap points:
  `0:00, 0:17, 0:35, 0:53, 1:11, 1:29, 1:47, 2:05, 2:23, 2:41`. With
  the 10s audio trim, the song is 171s in video-time (last iteration
  is `2:41+`, covering video 2:41-2:51).

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First schedule line is
byte-exact to Node 169.

```
node_169_prompt: Style: illustrated. In a medium shot, static camera, locked off shot, a warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together.

schedule:
0:00-0:17: Style: illustrated. In a medium shot, static camera, locked off shot, a warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together.
0:17-0:35: Cut to a medium close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, brow furrowing.
0:35-0:53: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, eyes narrowing.
0:53-1:11: Cut to a close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, eyes wide, mouth open.
1:11-1:29: Cut to a medium shot, jib up, camera rising up. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, shoulders squaring.
1:29-1:47: Cut to a medium close-up, focus shift, rack focus. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, chin lifting.
1:47-2:05: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, brow tightening.
2:05-2:23: Cut to a medium shot, dolly left, camera tracking left. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, head rocking a fraction.
2:23-2:41: Cut to a close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, eyes wide, mouth fully open.
2:41+: Cut to a medium close-up, jib down, camera lowering down. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, eyes lowering.
```

## Camera-move rotation (at a glance)

Ten iterations, seven distinct canonical moves, no dolly-out anywhere:

| Iter | Time | Shot | Camera | Section |
|---|---|---|---|---|
| init | 0:00-0:17 | medium | `static camera, locked off shot` | BRIDGE (quiet) |
| 1 | 0:17-0:35 | medium close-up | `static camera, locked off shot` | VERSE |
| 2 | 0:35-0:53 | close-up | `dolly in, camera pushing forward` | VERSE |
| 3 | 0:53-1:11 | close-up | `static camera, locked off shot` | **CHORUS 1** |
| 4 | 1:11-1:29 | medium | `jib up, camera rising up` | VERSE |
| 5 | 1:29-1:47 | medium close-up | `focus shift, rack focus` | VERSE |
| 6 | 1:47-2:05 | close-up | `dolly in, camera pushing forward` | VERSE |
| 7 | 2:05-2:23 | medium | `dolly left, camera tracking left` | VERSE |
| 8 | 2:23-2:41 | close-up | `static camera, locked off shot` | **CHORUS 2** |
| 9 | 2:41+ | medium close-up | `jib down, camera lowering down` | BRIDGE / OUTRO (held, no dolly-out) |

Pattern notes:
- Both CHORUS peaks are close-up + static. The *camera* holds still
  on the chorus, the *performance* does the work. Moving camera on
  the chorus would steal visual energy from the delivery.
- Dolly-ins appear on the approach to each chorus (iter 2 before
  chorus 1, iter 6 before chorus 2). They build tension toward the
  static chorus hit.
- Lateral and vertical moves (jib up, jib down, dolly left) sit
  mid-verse where less is at stake.
- `focus shift, rack focus` at iter 5 gives the kitten a featured
  moment without cutting away to a different subject (which i2v
  can't do safely).
- Outro is a held close-up — no dolly-out.

**Lip-sync note:** The integer-latent drift that used to desync at
`overlap_seconds=4` was fixed 2026-04-20. `AudioLoopController` now
derives stride from integer-latent counts so audio and video advance
by exactly the same amount per iteration. You can set overlap to
any value without accumulating lip-sync drift — widget values snap
to the nearest integer-latent effective overlap (e.g. target 2.0 →
effective 1.96s). See CLAUDE.md "Stride is derived from
integer-latent counts" for the full derivation.

## Workflow widget values

Identical to `music_prompt1.md`. Only the contents of Node 169 and
Node 1558 change when switching between v1 and v2; all other widgets
(resolution, sampler, scheduler, shift, CFG, window/overlap, decoder,
negative prompt) stay the same.

| Widget | Node | Value |
|---|---|---|
| `image` | 444 `LoadImage` | `your-illustrated-init-image.png` |
| `audio` filename | 565 `LoadAudio` | `your-vocal-track.mp3` |
| `start` (outer trim) | 567 `TrimAudioDuration` | **`10`** (skip the ~10s quiet intro) |
| `overlap_seconds` | 1582 `AudioLoopController` | `2.0` (committed default post-2026-04-20; effective 1.96s after integer-latent quantization) |
| `window_seconds` | 688 `FloatConstant` | `19.88` |
| `length` | 526 `PrimitiveNode` | `497` |
| resolution | 445 `ImageResizeKJv2` | `832 x 480` |
| Node 169 text | 169 `CLIPTextEncode` | paste `node_169_prompt` above |
| schedule text | 1558 `TimestampPromptSchedule` | paste `schedule:` block above |
| `blend_seconds` | 1558 | `0.0` |
| `snap_boundaries` | 1558 | `true` |
| `sampler_name` | 154 `KSamplerSelect` | `euler` |
| scheduler | 1421 `BasicScheduler` | `linear_quadratic, 8, 1` |
| `shift` | 1513 `ModelSamplingSD3` | `13` |
| `cfg` | 153 `CFGGuider` | `1.0` |
| decoder | 1604, 1597 `LTXVTiledVAEDecode` | unchanged |
| `start_seed` | 1527 `INTConstant` | your choice (same seed as v1 run if A/B-ing) |

## Negative prompt (unchanged)

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone
```

## Observations

- **`Style: illustrated.` not `Style: cinematic.`** `Style: cinematic`
  is one of Gemma 3's strongest photoreal anchors (film-look, skin
  texture, lens behavior). On a painterly fantasy init image it
  fights the image's native style, and across iterations the drift
  toward photoreal reads as the subject "aging" — skin detail, face
  geometry subtly realists over time. `docs/guides/prompt_creation_guide.md`
  already says to omit `Style:` when the init image sets style
  strongly; `your-illustrated-init-image.png` clearly does. `Style: illustrated.` keeps
  the structural prefix but pulls the text conditioning toward the
  image's style family instead of away from it. Alternatives if
  needed: `Style: painterly illustration.` / `Style: digital
  painting.` — more specific, slightly more tokens.
- **"Cut to ..." survives the simpler rewrite.** README rule 6
  warns against meta-language, but v4 standup confirmed that naming
  the iteration-boundary seam as a cut turns a technical discontinuity
  into a perceived edit. Dropping the rest of the art direction but
  keeping the cut language is the sweet spot for this simplified
  pass.
- **Why no `dolly out` even with the R7 outro exception.** The v2
  philosophy is "don't negotiate with the face". The only move that
  can mutate mouth geometry over a full iteration is one that shrinks
  it relative to the frame. On a distilled 8-step pass with already
  limited text leverage, that's a losing bet even if most of the
  time it works. Holding the close-up and letting the audio fade is
  strictly safer. Same rule applied retroactively to v1.
- **Tradeoff vs v1.** v1 tells Gemma 3 more about the *feel* of each
  iteration — the art-direction vocabulary nudges the latent toward
  painterly, high-contrast, peak-energy representations. v2 trusts
  the image to carry that and tells Gemma 3 only what's changing.
  For chorus iterations specifically, v1 may deliver a more
  "produced" look (the supersaturation + impact-frame language
  encodes "peak moment"); v2 relies entirely on the subject's
  expression. Run both with the same seed, see which reads better
  for this track.
- **Kitten visibility.** v1 gives the kitten an explicit beat in
  every entry ("the kitten alert", "goggles catching a stray
  highlight", etc.) specifically because small peripheral subjects
  morph in long loops. v2 drops those per-entry beats in the name
  of simplicity, but the plural `are singing together` verb still
  gives LTX a mouth-animation target for the kitten in every
  iteration via R1 multi-performer lip-sync cross-attention. That's
  probably enough. If the kitten still drifts, the first thing to
  restore is the per-entry kitten beats from v1 — not adding more
  descriptive language to the woman.
- **What I'd try next.** A third variation that keeps v2's canonical
  camera phrasing and terse structure but restores ONLY the kitten
  beats (not the art direction). That isolates whether the kitten
  drift is a real problem or a phantom one. Also: run v1 and v2 at
  the same seed and diff the chorus frames — that's the cleanest
  answer to whether the extra v1 vocabulary is earning its keep.
