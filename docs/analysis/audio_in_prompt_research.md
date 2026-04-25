Last updated: 2026-04-23 (merged audio_in_prompt_analysis.md + audio_in_prompt_guide_notebooklm.md; added project-workflow framing)

# Audio-in-prompt research: community practices for LTX-2 lip-sync

Research notes on lip-sync prompting for LTX-2 drawn from community
testing, NotebookLM synthesis, and Lightricks developer advice.

## When this applies vs when it doesn't

**This doc applies to:** LTX-2 workflows where the model is *generating*
audio, or T2V / I2V variants without a fixed audio track.

**This doc does NOT apply to our current music-video workflow.** Our
pipeline uses i2v + **frozen-audio conditioning** (`noise_mask=0` on
the audio latent via `SolidMask` + `SetLatentNoiseMask`). Because
audio and video share cross-attention in LTX 2.3 and the audio is
already carried by the frozen latent, the prompt is better when
**concise and less detailed** than what this research recommends:

- Strip music / instrumentation references from schedule prompts
  (the audio latent already carries them).
- Skip dialogue transcription for singing (audio already carries the
  vocal performance; the model's joint cross-attention reads the
  "is singing" verb and binds lip shape to the audio).
- Over-specifying audio descriptors double-signals and can over-crank
  visual intensity at beats.

See `CLAUDE.md` (Critical constraints → "Audio is FROZEN in our
workflow"). The strip rules above are derived from internal A/B runs
that validated the empirical mechanism.

**This doc is preserved** because future workflows (audio-generating
variants, pure-T2V without a frozen track, dialogue-heavy scenes
where audio is part of the generation target) will want the
transcribe-dialogue + detailed-delivery pattern.

---

## Is dialogue transcription required?

**Short answer: for audio-generating workflows, usually no — but
recommended as best practice.**

LTX-2 processes the mel spectrogram of the audio directly, so the
model often "knows what talking is" just from the audio file. Many
users get acceptable lip sync from a simple prompt ("a man is talking
to a woman") plus the audio track.

**When to transcribe anyway:**
- Precision / padding: exact text provides extra semantic context
  that tells the model "where to pad with precision" on mouth shapes.
- Heavy background music / complex pronunciations / fast singing:
  audio alone can confuse the model into visual "gibberish."
- Multi-speaker scenes: use `Person A says: "…" / Person B says: "…"`
  to bind lip sync to the correct character.

**Format:** `The man says: "Your exact audio transcript goes here."`
If extending a video or inpainting, include the transcript for the
entire video's audio — including unedited sections — so the model
sees the full context.

---

## Prompt rules for lip-sync quality

### Match vocal delivery and emotion
Prompt tone must match the audio's tone, pitch, and accent. If the
audio is high-pitched but the prompt says "deep voice," lip sync
fails. Use phrases like "in a sultry begging voice," "speaks with
great passion," or "speaking in a thick Australian accent."

**The "shouting" trick.** If generated lip movement is too subtle or
stiff, swap "speaking" for "shouting" in the prompt — it forces the
model to exaggerate mouth movements.

### Tight framing wins
Lip sync generations work significantly better in tight close-ups
or chest-up medium shots. If the face is too small or distant, the
model can't maintain identity or resolve lip movement.

### Highlight the mouth action
Phrases like "expressive mouth movement, clear lip sync" or "we
clearly see her lips moving in time with the speaking" reinforce
audio-to-video alignment.

### Action before dialogue
`The person speaks in a harsh low voice and says "…"` yields better
results than putting dialogue first.

### Avoid internal emotions
Don't say "sad" or "confused." Describe the physical cues: "furrowed
brow," "tremor of the chin," "tears welling."

### Structure
Long, descriptive, chronological single paragraphs up to ~200 words.
Present-tense verbs. Temporal connectors ("as," "then," "while").
Shot scale established first, then action, then audio layer.

---

## Fixing common lip-sync failure modes

| Symptom | Fix |
|---|---|
| Character looks like a documentary voiceover (mouth doesn't move) | Increase audio file volume — louder, peaking audio forces the model to drive lip sync harder |
| Frozen first frame (I2V) | Add subtle h.264 compression to the init image via `LTXVPreprocess` at strength 33-40. LTX-2 was trained on compressed video; a pristine image reads as a static photo |
| Over-emoting destroys character likeness | Add negatives: `exaggerated expressions, warped facial features, identity drift`. Optionally lower the audio-to-video attention scale |
| Multi-speaker confusion | Elaborate prompt with explicit Person A / Person B structure |

---

## Describing the input image (I2V)

If your prompt describes a character, outfit, or environment not in
the starting image, LTX-2 will often ignore the reference image and
switch to T2V mode, or freeze the frame.

**The VLM trick:** Pass the input image through a Vision-Language
Model (Qwen-VL, GPT-4o, Gemini) to generate a literal, accurate
description, then use that as the base of your prompt.

**Avoid overloading.** Prompting for actions that require a different
camera angle or setting than the init image fights the model. Start
with what is visible, then describe the transition.

---

## Concrete examples

These come from community testing and official guides. They illustrate
LTX 2.3 at training-distribution lengths and specificity levels — useful
for calibrating what "the model expects" even when our workflow writes
more concisely.

### 1. Intimate acoustic performance (slow, moody)

> *"A warm, intimate cinematic performance inside a cozy, wood-paneled
> bar, lit with soft amber practical lights and shallow depth of field
> that creates glowing bokeh in the background. The shot opens in a
> medium close-up on a young female singer in her 20s with short brown
> hair and bangs, singing into a microphone while strumming an acoustic
> guitar, her eyes closed and posture relaxed. The camera slowly arcs
> left around her, keeping her face and mic in sharp focus as two male
> band members playing guitars remain softly blurred behind her. Warm
> light wraps around her face and hair as framed photos and wooden
> walls drift past in the background. Ambient live music fills the
> space, led by her clear vocals over gentle acoustic strumming."*

### 2. Multi-character rap / dynamic exchanges

> *"Superman and Lois Lane perform together in a gritty rap music
> video. Their recognizable appearance and facial identity must remain
> consistent throughout the scene. At the beginning Lois Lane reacts to
> the beat with playful rhythmic hype sounds while looking at Superman,
> then briefly glancing at the camera with a teasing confident smile.
> The video alternates naturally between different music video shot
> types: wide shots showing both performers interacting with confident
> body language, medium performance shots capturing their rap delivery
> and movement, and occasional close-up reaction shots highlighting
> facial expressions and lip sync. Superman begins rapping with intense
> rhythmic delivery, strong mouth articulation and expressive lip
> movements while alternating his gaze between Lois Lane and the
> camera. He performs with sharp rap gestures and confident stage
> presence. When his line ends, Lois Lane steps forward and answers
> with her verse, rapping with energetic delivery and expressive lip
> movements while Superman reacts with amused approval."*

### 3. Stylized solo stage performance

> *"A single, completely solitary humanoid Shiba Inu performer sings
> passionately into a handheld microphone. No other people, animals,
> silhouettes, reflections, shadows, or background figures exist
> anywhere in the scene. The performer is the only living subject
> present at all times. He has orange-brown Shiba Inu fur, expressive
> canine features, a single black eyepatch over his right eye... His
> mouth opens and closes rhythmically in sync with the performance,
> with subtle head movement, gentle upper-body sway, and controlled
> side-to-side motion, creating a dynamic yet grounded stage presence.
> The microphone remains perfectly aligned with his mouth at all times,
> with no rotation. Lighting is moody and cinematic... The camera
> slowly pulls back and pans subtly to follow his movement while
> maintaining a full-body view, keeping both hands fully visible
> in-frame at all times."*

### 4. Musical theater / animated characters (dialogue included)

> *"A close-up of a cheerful girl puppet with curly auburn yarn hair
> and wide button eyes, holding a small red umbrella above her head.
> Rain falls gently around her. She looks upward and begins to sing
> with joy in English: 'It's raining, it's raining, I love it when
> its raining.' Her fabric mouth opening and closing to a melodic
> tune. Her hands grip the umbrella handle as she sways slightly from
> side to side in rhythm. The camera holds steady as the rain sparkles
> against the soft lighting. Her eyes blink occasionally as she sings."*

### 5. High-emotion singing with gestures

> *"A young woman sings with deep passion towards the camera, then
> slowly raises one hand to brush her hair back. She possesses long,
> voluminous dark brown wavy hair, deep blue eyes, and a sun-kissed
> complexion, wearing a rustic, textured dark burgundy off-shoulder
> top. She stands amidst a vast golden field, wildflowers swaying
> gently... The camera maintains a steady medium close-up, slightly
> low angle, focusing intently on her face. It subtly pushes in during
> her singing, then smoothly tracks her right hand as it rises,
> fingers lightly touching her temple, then fluidly sweeping the loose
> dark strands from her face, revealing her full expression. Intense
> golden backlighting creates a radiant halo around her hair..."*

### 6. Audio-reactive dancing (no lip-sync)

> *"A young woman with glowing eyes, crowned in black horns and
> adorned with intricate tattoos including wolves across her chest,
> slowly dances with passionate elegance under dim ambient light. Her
> dark hair flows as she sways, arms rising then falling rhythmically
> to unseen music; lips part slightly mid-motion. The camera glides
> smoothly around her from left to right, capturing her fluid grace
> against a misty, shadowed backdrop where faint snowflakes drift
> silently downward."*

### 7. Dialogue from a static image (I2V, static camera)

> *"A tight cinematic close-up of a male doctor speaking directly to
> the camera inside a modern health consultory. He wears a crisp white
> lab coat over a light blue shirt, subtle stubble, calm confident
> expression. Soft diffused daylight enters from a side window,
> creating gentle highlights on his face and clean shadows. The
> background is softly blurred with medical shelves and diagnostic
> equipment. The camera is locked in a shallow-depth close-up using a
> 50mm lens, with a very subtle push-in as he speaks, maintaining eye
> contact. Natural skin texture, realistic pores, professional medical
> atmosphere. Quiet room tone ambience, no music. He says: 'We need
> to run the tests again immediately, the results are inconclusive.'"*

### 8. Stylized non-verbal audio reactivity

> *"Live Action Mode futuristic fashion-dance tableau, neon sci-fi
> editorial: a dark-skinned dancer with a large textured afro is
> frozen in a dramatic off-balance tilt, wearing a reflective chrome
> set. Background is a luminous rectangular LED frame with
> blue/magenta rim lighting. At the start she holds the extreme lean
> pose, then slowly wakes into motion — micro tremor in shoulders,
> fingertips flex. Halfway through she transitions into a smooth hinge
> and recovery, moving rhythmically to the beat. Toward the end she
> rotates her head toward camera, eyes lock, and she breathes out one
> line: 'Watch me bend the light.' The camera makes a controlled slow
> push-in. Audio: low neon room hum, soft breath, faint fabric creak,
> subtle whoosh synced to arm sweep, minimal futuristic pulse very
> low in the mix."*

---

## Sources

- Community testing on r/LTXvideo, LTX Discord
- Lightricks developer advice (thread-level community posts)
- NotebookLM synthesis of the above (audio_in_prompt_guide_notebooklm.md
  merged into this file 2026-04-23)
