Last updated: 2026-06-04

# Audio-Only-Context IC-LoRA (voice -> identity / mannerism)

The successor to the pitch "helium" experiment ([audio_only_ic_lora.md](audio_only_ic_lora.md), which
covers the shared node mechanics and parity gates). Same idea, broader attribute: an in-context
reference **audio** clip steers the jointly-generated audio+video, this time trained on natural
talking-head clips to map a speaker's **voice to their identity** (no synthetic perturbation). Released
checkpoints + full model card:
[fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context).
Trained in the LTX-2 fork (`audio_reference` strategy).

Status: early. The reference visibly steers generation at default strength (hip-hop audio in -> hip-hop
output, mob-movie dialogue in -> mob mannerisms), but there is **no controlled quantitative eval yet**.
Proof of concept, not a benchmark.

## What it does, stated carefully

Hand it only a reference audio clip (plus a neutral text prompt); generate audio+video from noise. The
reference does **not** pick the scene (the prompt does that). It steers the **generated audio**, and the
**video follows that audio** through the joint model, so the speaker's mannerisms / energy / lip motion
track the reference under a fixed prompt. With out-of-distribution references (songs, movie audio) you
get broad emergent style/mannerism transfer; the narrow trained "this voice -> this person's face"
mapping is not cleanly separated from that yet.

## Two checkpoints

Same recipe, different cut (see the model card):
- **cross-modal** (`audio_attn` + the `audio_to_video_attn` / `video_to_audio_attn` bridges): the bridge
  is the audio reference's only path into the video stream, so its effect on the **video** is stronger.
  Default pick. ~290 MB.
- **audio-only** (`audio_attn` only): shapes the generated **audio**; the video follows via the frozen
  base coupling. Subtler video effect, perturbs the base video path least. ~164 MB.

Each has its own strengths/weaknesses; which is better for what is an open question. Both garble at high
strength, and the effect depends a lot on the reference audio (level / quality / content) in ways not
yet characterized.

## Nodes (in `nodes_audio_iclora.py`)

The basic loader + guide are documented in [audio_only_ic_lora.md](audio_only_ic_lora.md). For this
model two extra nodes give granular control (the base is guidance-distilled -> **CFG is fixed at 1**, so
strength, not CFG, is the only inference amplifier):

- **`LTXAudioICLoRALoaderPerStream`**: separate `audio_strength` and `bridge_strength`. Partitions the
  LoRA patches (bridge = `audio_to_video_attn` / `video_to_audio_attn`) and applies two passes. Push
  `bridge_strength` above `audio_strength` to amplify the audio->video coupling (voice->face) while
  keeping the audio modules in band. An audio-only LoRA has no bridge keys, so `bridge_strength` is then
  a no-op. Same zero-bind trust gate as the debug loader.
- **`LTXAddAudioICLoRAGuideAdvanced`**: the basic guide plus `reference_window_sec` (trim the reference;
  ~3.5 s matches training), `reference_scale` (scale the encoded reference latent magnitude),
  `attach_to_negative` (off = keep the negative conditioning ref-free, the arm the CFG-analog
  amplification trick needs on the full base; a no-op at CFG=1), and `reference_start_percent` /
  `reference_end_percent` (gate the reference to a band of the denoise schedule — outside the band the
  ref tokens vanish from the model call entirely, via a per-entry timestep split; on the 8-step
  distilled sampler the band resolves to ~12.5% increments). All default to the basic behavior. The
  parity-locked bits (negative-RoPE offset, patchify layout) are untouched. There is deliberately
  **no** reference-strength / attention-strength knob: the model reads the reference tokens with no
  scalar applied, so such a knob would be a silent no-op without a model-side change.
- **`LTXLoadComposeReferenceAudio`** ("Compose Reference Audio"): the reference loader (replaces
  Load Audio). A visual waveform editor for picking one or more (non-contiguous) slices of the
  reference clip, plus an **Auto-find hook** button that drops a slice on the loudest sustained
  window — the in-band, few-second window selection, run server-side through the same tested
  engine the head-trim uses (no graph queue). When you use it, set the Advanced guide's
  `reference_window_sec = 0` so the two don't double-trim. Keep the total composed duration to a
  few seconds (longer goes off-distribution).

## Usage

- Strength band ~0.3–0.75; both checkpoints garble toward 1.0. Start ~0.5.
- Keep the caption neutral so the audio drives the attribute.
- Generate from an empty latent (t2v); no image or video input. fp32 audio-VAE encode parity holds
  automatically (comfy forces float32 for the LTX audio VAE).
- A possible next knob (not built): a **timestep-range gate** to apply the reference only over the
  high-noise band where it matters. The stock `ConditioningSetTimestepRange` does gate the reference, but
  cleanly gating only the reference (not the text) needs it on its own conditioning branch, and the
  premise (reference bites at high sigma) should be confirmed first.

## The honest measurement note

We monitored a held-out reference-attribution gap during training; for an identity task it reads ~0,
**by construction**, because the target video already shows the face, so reconstruction never needs the
reference. That is not a verdict that the model fails. The real test is **generation from noise**: fix
the prompt and seed, swap only the reference, and see whether the audio + mannerisms change. The
quantitative version of that swap eval (including non-celebrity / unknown voices, so a face change can't
be the base model recalling a famous face) is the outstanding work.

## Next

Training-side: the target leaks the attribute, so the model has little pressure to use the reference.
Masking the target face region and/or biasing training toward high noise would force reliance, then
re-run the swap eval. Inference-side: sweep `bridge_strength`; build the timestep-range gate if the swap
eval reads weak.
