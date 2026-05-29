Last updated: 2026-05-29

# Audio→video IC-LoRA — experimental method notes (a process, not a result)

Experimental notes, written as a lab notebook. The goal here was **not** to ship a
working audio IC-LoRA. It was to build a *reproducible process* — data pipeline,
trainer that fits on one consumer GPU, eval harness — so that someone with more
time, better data, or a better task can fork it and try. Along the way we hoped to
show the trained LoRA *does something*, even if not useful. We have not shown that
yet, and this doc is honest about why.

Everything below is either (a) observed via inference-only conditioning (the same
way [`audio_reactive_workflows.md`](audio_reactive_workflows.md) and the ADR design
notes report), or (b) explicitly marked as a guess. Where we screwed up, we say so.
**If you have a better idea, fork it and change whatever you want.**

## The idea (and the thing that undercuts it)

LTX 2.3 is a joint audio-video model: it's trained on audio + video together, so
audio already influences the generated video through native cross-attention. You
can see this today with zero training — freeze an audio track, animate an init
image, and the motion tracks the beat ([`audio_reactive_workflows.md`](audio_reactive_workflows.md)).
We call this the **native coupling**, and it's loose but real.

The IC-LoRA idea: train a small adapter so audio drives video *on top of* that
native coupling — tighter, more predictable, or coupled to something the base model
doesn't already do. "IC-LoRA" = in-context LoRA: a reference is fed in-context
alongside the generation (the Lightricks Cameraman / LipDub LoRAs are the pattern).

**The thing that undercuts the whole demonstration, which we only fully internalized
late:** because the base model is *already* audio-reactive, any "it pulsed to the
beat!" result is confounded — you can't tell native coupling from the LoRA's
contribution without a controlled A/B, and even then a "pulse" is easy to fake. More
on this in [Where we screwed up](#where-we-screwed-up). If we did this again we would
pick a coupling the base model does **not** already do (see [What we'd do next](#what-wed-do-next-turn-leftturn-right)).

## What we built (the process)

Three pieces, two repos. The trainer lives in a fork of Lightricks' LTX-2 training
code; the data design, eval, and ComfyUI inference live here.

### 1. Data pipeline (this repo's design, runs in the trainer repo)

Synthetic, procedural, CPU-only (numpy + ffmpeg, no model). The substrate is a
**beat→pulse** coupling: a shape whose size/brightness pulses on an audio click
track at a chosen BPM. It's measurable — an FFT of per-frame brightness recovers the
BPM — so in principle a trained LoRA's output can be scored the same way.

- `synthetic_av.py` generates clips + a `captions.json` + a `manifest.jsonl`
  (ground-truth BPM per clip). Three dataset variants exist, and the progression
  between them is the interesting part — see [the dataset evolution](#the-dataset-evolution-v1--v2--v3).
- `process_dataset.py --with-audio` precomputes everything to latents once
  (Gemma text + video VAE + audio VAE), so training never runs a VAE in the loop.
- `verify_training_data.py` validates shapes/counts/pairing before you burn GPU.

### 2. Trainer (LTX-2 fork — see that repo's `docs/`)

The headline enabler: **the 22B distilled model, int8-quantized, fits on a single
24 GB GPU via block-swap** (stream most transformer blocks between CPU and GPU). We
observed ~17 GB peak and ~43 min for 300 steps on a 4090. Without this, training
this model is a datacenter task; with it, a solo person can run it overnight.
Details in the LTX-2 repo's trainer doc.

### 3. Inference / eval (this repo)

The trained LoRA loads into the canonical audio-loop ComfyUI workflow
(`example_workflows/audio-loop-music-video_latent.json`). A converter
(`scripts/convert_lora_for_comfyui.py`) re-keys the trainer's output to ComfyUI's
expected names. The eval idea: render with the LoRA on vs off, same everything else,
and look for a difference attributable to the LoRA. **This is the part we got
wrong twice** — see below.

## The dataset evolution (v1 → v2 → v3)

This sequence *is* the method's main lesson, so it's worth the space. Each step
fixed a real flaw and exposed the next one.

- **v1 — `reference = target`.** Each clip was its own IC-LoRA reference. Trained
  with a broad LoRA target-module set. **Observed result:** the LoRA made the
  ComfyUI render *less* audio-reactive than the no-LoRA baseline — it suppressed the
  native coupling. Best explanation (inference, not proven): with the reference
  identical to the target, the model can copy the pulse from the reference and never
  read the audio; the broad target set then spent adapter capacity in the wrong
  place and perturbed the base's working audio path. (An upstream trainer doc,
  musubi-tuner, describes a matching "broad preset degrades audio on an AV
  checkpoint" failure.)
- **v2 — paired references, different BPM.** Reference clip ≠ target clip, same
  visual identity, *different* BPM, so the audio is the only signal distinguishing
  them. We also restricted the LoRA to audio-side modules only. **Then we checked
  the data and found a leak:** in a narrow BPM range the resample logic made the
  reference's BPM *anti-correlated* with the target's (we measured corr ≈ −0.62). A
  model could infer the target rate from the reference's rate without reading the
  audio. Not shipped.
- **v3 — static reference.** The reference is a single frozen identity frame (no
  pulse, no audio). A still frame can't carry a rate, so the leak is gone *by
  construction* and the audio is the only time-varying input. This is what we
  trained: rank 16, audio-only target modules, LR 5e-4, 300 steps, final loss 0.0307
  (~15× v1's 0.002 — plausibly *because* the copy shortcut is gone and the audio now
  has to do work, but loss magnitude is not evidence of success either way).

Reasoning we leaned on throughout: an adapter learns to read whatever conditioning
channel *uniquely* carries the thing the loss needs. If two channels carry it,
gradient takes the cheaper one and the intended channel is never learned. So the
data's real job is to make the audio the *only* carrier of the controlled attribute
— caption rate-free, reference rate-free. We believe this is right; we have not
proven it isolates cleanly inside this model.

## Where we screwed up

Honestly, repeatedly, and mostly late at night:

1. **v1's `reference = target`** — a dataset that let the model ignore audio
   entirely. Caught only after a render came back *worse* than baseline.
2. **The v2 correlation leak** — we "fixed" v1 by making references different, then
   the fix introduced a measurable BPM correlation between reference and target. We
   only caught it by computing the correlation on the manifest *before* the next
   train. Lesson we'd bottle: check channel independence on the data, not after the
   render.
3. **The eval workflow conflated three signals** and we ran it anyway. The eval we
   first used fed a **pulsing reference video that had no audio** as the IC-LoRA
   input (we verified: zero audio streams, animated luminance), used the **same
   prompt as every training caption**, and drove the frozen-audio path with a music
   track. So three things could each produce a "pulse" in the output: (a) the base
   model's native reactivity, (b) the model copying the pulsing reference video, (c)
   the LoRA having memorized "this caption → a pulsing shape." A real test on an
   init image (a wolf/moon photo) produced a colored shape pulsing over the moon —
   which is consistent with the model having learned *a shape*, i.e. overfitting to
   the synthetic substrate, not learning a general audio→motion coupling. We can't
   attribute that render to the LoRA at all.

The common thread: we kept testing the *output* without isolating the *cause*.

## What the eval actually needs (designed, not yet clean-run)

To attribute anything to the LoRA you have to hold everything constant and vary one
thing, always against a no-LoRA baseline:

- **Static reference, matching training** (a frozen frame, not a pulsing video) —
  removes the copy-the-reference confound.
- **Silent-audio control** — feed silence; if the output still pulses, the LoRA
  learned "always pulse," not "pulse to the audio." This single control is the most
  decisive and the cheapest.
- **A non-training prompt and a real-photo init** — if the coupling only fires with
  the exact training caption on a shape-like init, it's memorization.
- **LoRA-on minus LoRA-off** is the only number that means anything. If the baseline
  already tracks the audio about as well, the LoRA earned nothing.

We built the static reference asset and the four-arm workflow set for this; we have
not produced a clean run of it yet (see [status](#status)). There is also a
**representability ceiling** to respect: the video VAE compresses ~8× in time, so a
per-beat pulse aliases above ~`fps × 3.75` ≈ 94 BPM at 25 fps. Test BPMs should stay
below that or the measurement is garbage regardless of the LoRA.

## A gap in the tooling worth knowing about

The IC-LoRA reference input in ComfyUI is a **video** node. There is no node that
takes an audio file (wav/mp3) as the in-context reference. So to test an
audio-reference IC-LoRA (the LipDub shape — reference is identity, audio is the
driver) you're forced to pass a video proxy, which carries both a video signal and
an audio signal and makes attribution impossible. If you want to build the
audio-reference variant properly, **a node that accepts audio as the IC-LoRA
reference is the missing piece** — mirroring how ComfyUI-LTXVideo's Advanced
Reference IC-LoRA guider nodes handle video, plus making sure the up/downstream
nodes handle an audio-only reference. We did not build this.

## What we'd do next: turn-left / turn-right

The strongest idea to come out of this, and the one we'd actually pursue: stop
trying to demonstrate a coupling the base model already does. Train on a coupling it
does **not** do natively — e.g. a voiceover saying "turn left" / "turn right" paired
with video where the camera pans that way.

Why it's a better experiment:
- LTX does not natively pan the camera from spoken words, so there's **no baseline
  confound** — any success is unambiguously the LoRA.
- Camera direction is **global motion you can't fake** by imposing a static shape.
- It's a **discrete, trivially measurable** signal (left vs right).
- The spoken word is the **only** place the direction can come from; the caption
  stays neutral, the init can be any scene.

This is speculative — we have not run it. But it's where we'd point a fork.

## Status

- Pipeline (gen → precompute → validate → train → convert → load) runs end-to-end.
- v3 LoRA trained and loads in ComfyUI. Its actual audio→video behavior is **not yet
  cleanly measured** — every render so far is confounded.
- Honest current read: what we've seen is consistent with native reactivity plus
  overfitting to a synthetic shape, **not** a demonstrated LoRA-specific coupling.
  We can't rule a small real effect in or out until the controlled eval runs.

## Cross-links

- Native audio→video coupling, inference-only (the baseline this builds on):
  [`audio_reactive_workflows.md`](audio_reactive_workflows.md)
- The two-pass audio-guides-video inference path (validated manually):
  `example_workflows/working_docs/combined_adr_workflow_design.md`
- Trainer side (block-swap, strategy, configs): the LTX-2 fork's `docs/` — see that
  repo. (Same author; the trainer is a fork of Lightricks' LTX-2 training code.)
