# Experiments

Last updated: 2026-04-24

Per-experiment logs documenting hypothesis, setup, observations, and what we learned. Distinct from `docs/analysis/` (research/postmortems on shipped code) and `docs/experimental/` (scaffolded-but-not-validated features).

## Convention

- Filename: `exp_YYYY-MM-DD_<slug>.md`
- Lowercase, underscores, no spaces
- One experiment per file (chronological)
- Log evolves: open with hypothesis + setup, append findings as runs complete, close with status (`open` / `paused` / `concluded`) and links to follow-up experiments
- Neutral framing: "what we observed", not "success" / "failure" — observations have value regardless of whether the hypothesis held

## Template

```markdown
# Experiment: <slug>

Last updated: YYYY-MM-DD
Status: open | paused | concluded

## Hypothesis

What we're testing and why.

## Setup

Workflow, models, settings, inputs.

## Observations

What we saw — visual, audio, numeric, qualitative. Include image/spectrogram references.

## Inferences

What the observations imply about model behavior, architecture, or our assumptions.

## Next

Follow-up experiments, ablations to run, or pivot direction.
```

## Index

- [exp_2026-04-24_spectrogram_iclora_v2a.md](exp_2026-04-24_spectrogram_iclora_v2a.md) — Spectrogram-as-canny IC-LoRA + V2A round-trip. 9 runs across IC-LoRA families / strengths / colormaps / init-image presence. Key findings: B&W triggers vintage-broadcast audio prior; init images required for IC-LoRA to blend rather than copy; IC-LoRA + spectrogram not transmitting audio info. Pending single decisive run (R17) before close-out.
