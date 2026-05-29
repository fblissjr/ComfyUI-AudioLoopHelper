Last updated: 2026-05-29

# Audio→video IC-LoRA training experiment — index

A self-contained record of an attempt to **train** a small audio-conditioned
IC-LoRA on LTX 2.3, on hardware a solo person owns (a single 24 GB GPU). The point
was to build a *reproducible process* others can fork — not to ship a working LoRA.
We have not shown the trained LoRA does anything useful; these docs are honest about
why, and where to go next.

This is distinct from the **inference-only** audio-reactive work (no training), which
already works and is documented separately — see
[`../experimental/audio_reactive_workflows.md`](../experimental/audio_reactive_workflows.md).

## Start here

- **[`method_notes.md`](./method_notes.md)** — the lab-notebook writeup. What we
  built, how it works, the dataset evolution (v1→v2→v3) and the two leaks we hit,
  the eval confounds we got wrong twice, the missing audio-reference node, and the
  turn-left/turn-right task we'd try next. Read this first.

## The work spans two repos

| Half | Where | What |
|---|---|---|
| **Data + eval + ComfyUI inference** | this repo | synthetic data design, the audio-swap eval idea, the LoRA→ComfyUI converter, the canonical audio-loop workflow the LoRA loads into |
| **Trainer** | the LTX-2 fork (a fork of Lightricks' LTX-2 training code) | block-swap fitting the 22B distilled model on one 24 GB GPU, the `condition`-mode strategy + audio-only target modules, the synthetic dataset builders, the pre-train validator |

The trainer-side overview lives in that fork's `docs/audio_iclora_trainer_notes.md`.
(Same author; not duplicated here to avoid drift from the code it describes.)

## The one-paragraph summary

LTX 2.3 is already audio-reactive natively (frozen audio drives video through joint
cross-attention — works with zero training). The IC-LoRA idea was to *tighten* that,
or couple to something the base doesn't already do, by training a small adapter. The
machinery works end-to-end (data → 4090-fittable trainer → ComfyUI load). What we
have **not** done is cleanly measure whether the LoRA adds anything beyond the base
model's native reactivity — every render so far is confounded, and the synthetic
beat→pulse task turned out to be a poor mechanism-prover precisely *because* the base
already does it. The reusable contributions are the block-swap training path, the
data-independence discipline, and the documented confounds. Fork and change whatever
you want.

## Related (in this repo)

- Inference-only audio-reactive workflows: [`../experimental/audio_reactive_workflows.md`](../experimental/audio_reactive_workflows.md)
- Two-pass audio-guides-video (ADR) design: `example_workflows/working_docs/combined_adr_workflow_design.md`
- Spectrogram-as-reference IC-LoRA experiment: [`../experimental/spectrogram_iclora_tutorial.md`](../experimental/spectrogram_iclora_tutorial.md)
