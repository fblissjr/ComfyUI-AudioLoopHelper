Last updated: 2026-05-11

# Experimental features

Work-in-progress features. Each has a working implementation, but is **not production-validated**. Use at your own risk; gates for promotion to `docs/` + `example_workflows/` are listed per feature.

Corresponding workflow files live under `example_workflows/experimental/`. They are checked into git (unlike `internal/scratch/`) so users can download + run them. Experimental research-grade *nodes* (registered with ComfyUI but not wired into default workflows) live in `nodes.py` and are documented here.

## Current experiments

### Per-frame initial noise amplification

- **Doc:** [`noise_frame_amplifier.md`](./noise_frame_amplifier.md)
- **Node:** `LTXNoiseFrameAmplifier` (`nodes.py`, registered in `AudioLoopHelperExtension`)
- **Tests:** `tests/test_ltx_noise_frame_amplifier.py` (7 behavioral)
- **Hypothesis:** multiplying the first N temporal frames of the sampler's initial noise by `k > 1` is equivalent to a per-frame sigma boost, pushing the model out of its "ease into motion" temporal prior on i2v init-anchored clips so meaningful motion starts earlier.
- **Gate for promotion out of experimental:** 3+ seed A/B showing reduced filler with retained frame-0 init-image fidelity on `amplifier=1.5` vs. `amplifier=1.0`. Quantitative motion-start signal (e.g. optical-flow magnitude in early frames) preferred but not required.
- **Status:** working node, full unit-test coverage, no multi-seed visual A/B yet.

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
