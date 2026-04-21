Last updated: 2026-04-20

# Action Sequence — Gauntlet Climb (instrumental action)

First non-music-video prompt in this session. Maps the 7-act video plan
from `<your-assets>/a creative-direction doc` (Gemini-generated
from the cinematic-realism init image) onto our 9-iteration loop
architecture. Key architectural shift: **no lip-sync required**. The
audio (`your-instrumental-track.mp3`) is purely orchestral — no
vocals — so LTX 2.3's audio-video cross-attention isn't driving mouth
articulation. Instead it's binding visual impact frames to orchestral
hits (timpani, brass, anvil strikes), which is exactly what Gemini's
plan optimizes for via match cuts.

## What's different from the music-video prompts

1. **No `is singing` / `are singing together`.** The R1 rule in
   `scripts/analyze_audio_features.py` `_LLM_SYSTEM_PROMPT` is
   specifically for vocal-driven lip-sync. With instrumental audio
   we use concrete action verbs (`is driving a dagger`, `is lunging`,
   `is reaching`, `is sheathing`) which anchor the action pathway of
   LTX's cross-attention instead.

2. **Match cuts on musical beats become the primary technique.**
   Our iteration boundaries at `0:17, 0:35, 0:53, ...` are visual
   discontinuities anyway (independent sampler passes). Each entry's
   prompt names the musical beat it lands on (brass swell, timpani
   strike, anvil hit, string staccato) so the cut lands as intentional
   editing against audio, not as a seam.

3. **Cinematic-realism init + orchestral audio + action = LTX's
   training distribution.** No style-drift fight. Expect minimal
   subject drift across iterations; expect tight action-motion coupling.

4. **9 iterations instead of 10** — the song is 150s (2:30), shorter
   than the music track. Last iteration is `2:23+` covering
   `2:23-2:30`.

## Why 7 acts → 9 iterations works cleanly

Gemini's plan has specific beats at non-stride timestamps (0:08, 0:25,
0:42, 1:02, 1:22, 1:45, 1:52, 2:15, 2:28). We can't hit those directly
at 17.92s-stride granularity. What we CAN do: each iteration picks the
dominant Gemini beat within its window and leans into that one moment.
Cross-attention + audio timing handles the rest.

CHORUS peaks especially matter:
- CHORUS 1 (song 63.9-71.9) → iter 3 (`0:53-1:11`, video 53-71). Covers
  Gemini Act 3's lightning + lunge (0:55, 1:02). Brass swell meets the
  action payoff.
- CHORUS 2 (song 127.8-150.5) → iters 7-8 (`2:05-2:30`, video 125-151).
  Covers Gemini Act 6 SnorriCam ascent + wide silhouette + dragon
  shadow + Act 7 aftermath. The peak orchestral ensemble under the
  heroic summit.

## Inputs

- **Audio**: `<your-instrumental-track.mp3>`
  (150.47s, 123 BPM, C Minor, instrumental orchestral).
- **Image**: `<your-cinematic-init-image.png>` (cinematic-realism
  warrior mid-climb with daggers, stormy vertigo, photoreal).
- **Subject string (byte-exact in every entry)**:

  `a warrior woman with a brown-and-silver braid in black leather flight gear`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=19.88`,
  `stride=17.92s`. Snap points:
  `0:00, 0:17, 0:35, 0:53, 1:11, 1:29, 1:47, 2:05, 2:23`.
  No audio trim (`TrimAudioDuration.start=0`); song starts with useful
  content.

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First schedule line is
byte-exact to Node 169.

```
node_169_prompt: In a wide establishing shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear stands on the edge of the first wooden beam of a brutalist multi-tiered Gauntlet suspended thousands of feet above a mountain chasm, weathered iron and dark wood, volumetric fog below.

schedule:
0:00-0:17: In a wide establishing shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear stands on the edge of the first wooden beam of a brutalist multi-tiered Gauntlet suspended thousands of feet above a mountain chasm, weathered iron and dark wood, volumetric fog below.
0:17-0:35: Cut to an extreme close-up, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear is drawing a single dagger from her thigh sheath, blade sliding out with a sharp metallic rasp, her gaze hyper-focused.
0:35-0:53: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear is slamming her dagger deep into a narrow slot on a massive vertical beam, the blade biting into weathered wood with a shower of splinters, her other boot wedging onto the embedded hilt below.
0:53-1:11: Cut to a low-angle medium shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear is lunging mid-climb, driving her primary dagger into a narrow iron seam with a shower of sparks, a massive blue lightning bolt splitting the storm sky behind her.
1:11-1:29: Cut to a profile medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear is moving horizontally across a series of swinging logs, using her daggers like ice axes, biting each blade into weathered wood with mechanical tactical precision.
1:29-1:47: Cut to an extreme close-up, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear's gloved hand grips the hilt of a dagger, the blade beginning to slide in weathered wood with splinters falling, her expression locked in silent terror.
1:47-2:05: Cut to a tight shot, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear's hand snaps shut around the dagger hilt just as the slide stops, she begins a rapid vertical ascent driving dagger after dagger into the iron pillar.
2:05-2:23: Cut to a tracking medium shot, dolly left, camera tracking left. A warrior woman with a brown-and-silver braid in black leather flight gear is running upright along the final narrow iron beam, her ponytail and single white streak trailing behind like a banner, a massive dragon's shadow passing overhead and eclipsing the sun.
2:23+: Cut to a close-up, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear stands at the summit of the Gauntlet, shoulders heaving, breathing hard, slowly sheathing her primary dagger as rain and wind take over.
```

## Camera + action progression

Nine iterations, eight distinct canonical moves, no dolly-out anywhere:

| Iter | Time | Shot | Camera | Gemini Act | Musical beat |
|---|---|---|---|---|---|
| init | 0:00-0:17 | wide establishing | `static camera, locked off shot` | Act 1 (Verticality) | strings + timpani build |
| 1 | 0:17-0:35 | extreme close-up | `static camera, locked off shot` | Act 1 end + Act 2 start | staccato strings |
| 2 | 0:35-0:53 | close-up | `dolly in, camera pushing forward` | Act 2 (Tactical Geometry) | ticking snare |
| 3 | 0:53-1:11 | low-angle medium | `jib up, camera rising up` | **Act 3 (Leap of Logic)** | **brass swells / CHORUS 1** |
| 4 | 1:11-1:29 | profile medium | `dolly right, camera tracking right` | Act 4 (Mechanical Grind) | anvil strikes |
| 5 | 1:29-1:47 | extreme close-up | `static camera, locked off shot` | Act 5 (The Void) start | shivering violin |
| 6 | 1:47-2:05 | tight close-up | `dolly in, camera pushing forward` | Act 5 end + Act 6 start | full orchestral swell |
| 7 | 2:05-2:23 | tracking medium | `dolly left, camera tracking left` | **Act 6 (The Ascent)** | **peak chorus / CHORUS 2** |
| 8 | 2:23+ | close-up | `static camera, locked off shot` | Act 7 (Aftermath) | decaying outro |

Pattern notes:
- **Both chorus peaks** (iter 3 + iter 7) use CAMERA MOTION rather than
  static. For action scenes this is reversed from music-video prompts —
  there the chorus was "camera holds still, performance drives." Here
  the camera motion IS the action payoff (crane up through lightning;
  tracking run across the final beam). Both chorus entries lean into
  their specific Gemini-planned crane/tracking moves.
- **Verse-level iterations** (2, 4, 6) use dolly-in as tension-build
  before each chorus, dolly-right as horizontal-progression midpoint,
  dolly-in again as pre-ascent coil.
- **Void moment** (iter 5) is the ONLY mid-sequence static entry,
  enforcing "time stops" during the dagger-slip moment Gemini placed
  at 1:45-1:52.
- **Aftermath** (iter 8) is held close-up — same "audio fade closes,
  not the camera" rule from the music prompts.

## Workflow widget values

Applies to `example_workflows/audio-loop-music-video_latent.json`
(or `_latent_stg.json` if testing STG variant).

| Widget | Node | Value |
|---|---|---|
| `image` | 444 `LoadImage` | `your-cinematic-init-image.png` |
| `audio` filename | 565 `LoadAudio` | `your-instrumental-track.mp3` |
| `start` (outer trim) | 567 `TrimAudioDuration` | **`0`** (no cold-open to skip) |
| `overlap_seconds` | 1582 `AudioLoopController` | `2.0` |
| `window_seconds` | 688 `FloatConstant` | `19.88` |
| `length` | 526 `PrimitiveNode` | `497` |
| resolution | 445 `ImageResizeKJv2` | `832 x 448` |
| `img_compression` | 446 `LTXVPreprocess` | `18` |
| Node 169 text | 169 `CLIPTextEncode` | paste `node_169_prompt` above |
| schedule text | 1558 `TimestampPromptSchedule` | paste `schedule:` block above |
| `blend_seconds` | 1558 | `0.0` |
| `snap_boundaries` | 1558 | `true` |
| `sampler_name` | 154 `KSamplerSelect` | `euler` |
| scheduler | 1421 `BasicScheduler` | `linear_quadratic, 8, 1` |
| `shift` | 1513 `ModelSamplingSD3` | `13` |
| `cfg` | 153 `CFGGuider` | `1.0` |
| decoder | 1604, 1597 `LTXVTiledVAEDecode` | unchanged |

## Negative prompt

Action-sequence-specific additions vs the default. Paste into node
507 `CLIPTextEncode`:

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone, blurry motion, slow motion blur, cartoonish proportions, unrealistic physics, floating pose, mid-air suspended without anchor, dagger floating free of hand
```

Extras over the music-video base: `blurry motion`, `cartoonish
proportions`, `unrealistic physics`, `floating pose`, `dagger floating
free of hand`. These explicitly suppress the failure modes most likely
to appear in vertical-action content (weightless poses, dropped grip,
disconnected dagger).

## Observations

- **First non-singing prompt in the repo.** If this works well, it
  validates the broader "action verbs replace singing verb when no
  vocals" approach — LTX 2.3's audio-video cross-attention can drive
  impact frames against any acoustic events, not just vocal phonemes.
- **Expected: very low drift across iterations.** Cinematic-realism
  init + orchestral audio + action content all match LTX 2.3's training
  distribution. Unlike the illustrated runs, there's nothing for the
  model to drift TOWARD — it's already there.
- **Expected: the CHORUS iterations (3 and 7) should feel like the
  movie's payoff shots.** Iter 3 is "she uses the lightning"; iter 7
  is "she's the banner-ponytail victory run." These are the two
  frames viewers should remember.
- **Iter 5 risk**: static ECU with "silent terror" as the audio
  shivers is the most cinematically ambitious moment in the schedule.
  If it fails to read (e.g. face looks calm not terrified), the
  sequence's mid-point tension deflates. Fallback: add an explicit
  action beat: "a single bead of sweat tracing her temple" or "her
  grip white-knuckling visibly" — something LTX can animate.
- **Workflow choice**: `_latent.json` is the safer default for this
  first test. `_latent_stg.json` could be A/B'd if the baseline lacks
  visual detail on the chorus impact frames — STG's attention-block
  perturbation might sharpen the peak moments.
- **If the sequence is good, the Gauntlet sequence is a template for
  any non-vocal orchestral-action scene.** The "match cuts on musical
  beats + canonical camera + action verbs" pattern generalizes to
  combat, chase, montage — anywhere audio is instrumental and
  story is physical.
