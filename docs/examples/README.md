Last updated: 2026-04-25 (all schedules retroactively stripped of audio descriptors and init-committed redundancy per `docs/guides/prompt_creation_guide.md`; original discovery diffs preserved inline as historical case studies.)

# Prompt schedule examples

Real case studies showing how prompt schedules were designed for
specific use cases. Each file contains:
- Node 169 text (initial render prompt)
- Complete schedule block (paste into `TimestampPromptSchedule`)
- Workflow widget value table (which knobs to set)
- Negative prompt
- Observations on what the design is trying to achieve and what
  failed / worked during iteration

Asset filenames are scrubbed to placeholders (`<your-init-image.png>`,
`<your-instrumental-track.mp3>`). The patterns are what transfer, not
the specific assets.

## Music video / vocal-driven content

| File | Init image style | Notes |
|---|---|---|
| `music_prompt1.md` | Illustrated (painterly fantasy) | v1 rich vocabulary with illustrated beat pool |
| `music_prompt2.md` | Illustrated | v2 simpler structure — A/B against v1 to measure vocabulary-vs-identity tradeoff |
| `music_prompt3.md` | Cinematic realism | v3 with cinematic-realism init — validated that style drift fails on illustrated inits but works on cinematic |

## Action / instrumental content (no lip-sync)

| File | Approach | Notes |
|---|---|---|
| `action_prompt1.md` | Multi-shot narrative (Gemini-plan-driven) | First attempt; plan assumed multi-composition which single init can't produce |
| `action_prompt2.md` | Single-composition varied pacing | Refined — use the init's visual vocabulary, not Gemini's multi-location ambition |
| `action_prompt3.md` | Fast-paced, no contemplation | Punch-only verbs, no near-fall |
| `action_prompt4.md` | 9-iter with dragon-leap climax | Different audio track; out-of-distribution dragon challenge |
| `action_prompt5.md` | **20-iter rapid-cut architecture** | Halved `window_seconds` → 2.5× cut density |
| `action_prompt6.md` | Same 20-iter grid, no audio descriptors | A/B vs v5 — "frozen audio" insight: let the audio latent drive cross-attention alone |

## Standup comedy / dialogue

| File | Notes |
|---|---|
| `prompt_comedy1.md` | Early standup test |
| `prompt_comedy2.md` | Iteration on v1 |
| `prompt_comedy3.md` | Post-DR1 with widget trim variants |
| `prompt_comedy4.md` | Introduced the "Cut to ..." iteration-boundary technique that carries through to all action/music prompts |
| `prompt_comedy5.md` | **Unusual-character adaptation** — how to rewrite v4's subject blocks when the init image is outside LTX's typical training distribution (oversized cranium, loud patterned clothing) |

## Patterns that transfer

Reading these in order (comedy 1→4, music 1→3, action 1→6) documents
the discovery arc that produced the rules in
`../guides/prompt_creation_guide.md`. **All files have been
retroactively stripped (2026-04-25) to match those rules**; the
"What changed from <prior>" tables inside individual files document
the original discovery diffs. The arc:

1. **Subject byte-exact across entries** (rule stabilized in comedy
   series)
2. **"Cut to ..." language at iteration boundaries** (v4-standup
   finding — reframes seams as intentional edits; **now flagged as a
   hypothesis under review** in the guide §5.1, since loop architecture
   continues prior latents and the boundary marker may be undermining
   that continuity)
3. **Canonical LTX 2.3 camera phrasings only** (music series)
4. **No dolly-out anywhere for face-driven content** (music_prompt2
   onward; generalized to **no wide shots either** for music/comedy
   per guide §6.1, applied retroactively to comedy v4 / v5 outros)
5. **`Style:` prefix omitted when init commits style** (music_prompt3
   cinematic run)
6. **Action verbs replace singing verb for instrumental tracks**
   (action series)
7. **Audio descriptors removed when audio is frozen** (action_prompt6
   established the rule on 2026-04-20; on 2026-04-25 retroactively
   applied to all prior iterations)

See `../guides/prompt_creation_guide.md` and `../guides/debugging_guide.md`
for the distilled rules extracted from these case studies.
