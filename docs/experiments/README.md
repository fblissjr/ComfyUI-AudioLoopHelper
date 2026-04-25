# Experiments

Last updated: 2026-04-25

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

- [exp_2026-04-24_spectrogram_iclora_v2a.md](exp_2026-04-24_spectrogram_iclora_v2a.md) — Spectrogram-as-canny IC-LoRA + V2A round-trip. 9 runs across IC-LoRA families / strengths / colormaps / init-image presence. Key findings: B&W triggers vintage-broadcast audio prior; init images required for IC-LoRA to blend rather than copy; IC-LoRA + spectrogram not transmitting audio info. Pending single decisive run (R17) before close-out. **Spawned a generalized inference-time technique** (TTC1 CFG-analog amplification) that is independent of the spectrogram hypothesis — see §"Forward direction" in the log.

## Cross-experiment techniques

Some experiments produce reusable mechanisms that are independent of the specific hypothesis being tested. These outlive the experiment that birthed them:

- **TTC1 — CFG-analog amplification of any conditional contribution.** Generalized inference-time technique. Feed `(positive_with_X, positive_without_X)` to `CFGGuider` as `(positive, negative)`; the existing sampler computes `eps = eps_without + cfg * (eps_with - eps_without)` per denoising step, and `cfg` becomes the amplification knob for whatever conditional sits in the differential. Zero new sampler code; 2× inference cost; distinct from static control-vector / concept-slider techniques because the steering direction is recomputed per step from two full forward passes. Generalizes beyond IC-LoRA to any conditional (style LoRAs, identity LoRAs, per-reference ablation) — anything you can branch the prompt graph around so the two CONDITIONING streams differ only in `X`. Two POC wirings ship: `scripts/apply_ttc_iclora_amplification_poc.py` (IC-LoRA reference target, on the spectrogram experimental workflow) and `scripts/apply_ttc_init_guide_amplification_poc.py` (init-frame `LTXVAddLatentGuide` target, on the production audio-loop workflow — **demonstrates the mechanism without IC-LoRA in the graph at all**). Landscape and the other TTC methods (search, refine, sample, schedule) on this same conditional axis: `internal/analysis/iclora_landscape_analysis.md` §TTC.
