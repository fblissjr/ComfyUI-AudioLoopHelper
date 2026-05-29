Last updated: 2026-05-12

# Experimental features

Work-in-progress features. Each has a working implementation, but is **not production-validated**. Use at your own risk; gates for promotion to `docs/` + `example_workflows/` are listed per feature.

Corresponding workflow files live under `example_workflows/experimental/`. They are checked into git (unlike `internal/scratch/`) so users can download + run them. Experimental research-grade *nodes* (registered with ComfyUI but not wired into default workflows) live in `nodes.py` and are documented here.

## Current experiments

### Audio-reactive video (single-shot preview + full-length loop)

- **Guide:** [`audio_reactive_workflows.md`](./audio_reactive_workflows.md)
- **Workflows:** `example_workflows/experimental/audio_driven_single_shot.json` (preview rig, still experimental) + `example_workflows/audio_reactive_loop.json` (full render — **promoted to the top-level shipped surface** after render validation)
- **Build scripts:** `scripts/apply_audio_driven_single_shot.py`, `scripts/apply_audioreactive_loop.py`
- **Hypothesis:** an init image animated under a frozen audio track moves so its motion tracks the beat, via LTX 2.3 joint audio-video cross-attention; `audio_to_video_scale` controls the coupling strength.
- **Gate for promotion:** a render confirms visible audio-driven motion on the single-shot AND a full-length loop render holds style + tracks sections without unacceptable drift.
- **Status:** loop variant **promoted** (validated by a full-length render); single-shot preview rig stays experimental pending its own validation.

### Audio→video IC-LoRA — training method notes (process, not a result)

- **Notes:** [`audio_iclora_method_notes.md`](./audio_iclora_method_notes.md) — lab-notebook writeup of an attempt to *train* an audio-conditioned IC-LoRA (vs the inference-only audio-reactive workflows above).
- **Trainer side:** the 22B-on-one-4090 block-swap trainer is a fork of Lightricks' LTX-2 training code; its overview is in that fork's `docs/audio_iclora_trainer_notes.md`.
- **Goal:** build a reproducible *process* (data → 4090-fittable trainer → eval) others can fork — not to ship a working LoRA.
- **Status:** pipeline runs end-to-end; the trained LoRA's audio→video behavior is **not yet cleanly measured** (renders so far are confounded by the base model's native reactivity). The notes are candid about two dataset leaks and an eval we got wrong twice, and recommend a different task (turn-left/turn-right) next. Fork and change whatever you want.

### Spectrogram-as-reference IC-LoRA (Phase 2.0)

- **Tutorial:** [`spectrogram_iclora_tutorial.md`](./spectrogram_iclora_tutorial.md)
- **Workflow:** `example_workflows/experimental/spectrogram_iclora_minimal.json`
- **Build script:** `scripts/apply_spectrogram_iclora_minimal.py`
- **Internal design doc:** `internal/design/spectrogram_reference_design.md` (gitignored; architecture + kill switches + iteration ladder)
- **Hypothesis:** a Mel spectrogram fed to `LTXAddVideoICLoRAGuide` drives beat-locked visual rhythm.
- **Gate for promotion out of experimental:** qualitative A/B shows rhythm-alignment (Phase 2.0) AND `scripts/measure_beat_sync.py` scores > 0.5 (Phase 2.2, not yet built).
- **Status:** scaffolded + awaiting user A/B validation.

## Not-yet-experimental (still internal-only)

Things that haven't earned even experimental placement yet. They live in `internal/design/*.md` (gitignored). Once an initial PoC + tutorial exist, they graduate here.

- Upscale workflow (design: `internal/design/upscale_workflow_design.md`).
- IC-LoRA Phase 0b subgraph integration (design: `internal/ic_lora_assessment.md`).

## Promotion criteria

A feature graduates from `docs/experimental/` to the public-facing `docs/` tree when:

1. It ships with a validated `example_workflows/*.json` that audits clean (`scripts/audit_workflows.py`).
2. Its design doc moves from `internal/design/` to `docs/reference/` or `docs/guides/`.
3. There's at least one validated case study (internal log entry, in-repo experiment doc under `docs/experiments/`, or a real user validation note).
4. CLAUDE.md references it without qualifying as "experimental."

Until all four land, keep it here.
