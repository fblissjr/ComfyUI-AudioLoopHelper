# CFG-analog amplification

Last updated: 2026-05-04

## Role

Inference-time pattern for amplifying any conditional contribution X by feeding `(positive_with_X, positive_without_X)` to `CFGGuider` as `(positive, negative)`. Existing sampler's CFG math runs the dual forward pass and blends — `cfg` widget becomes the amplification slider. Zero new sampler code; generalizes beyond IC-LoRA to any conditional that can be turned on/off (style LoRAs, identity LoRAs, attention guidance, per-reference ablation).

## Disambiguation

- ≠ **Control vectors / concept sliders** — those are static directions in latent space; this is a dynamic per-step differential between two model evaluations.
- ≠ **ControlNet** — different conditioning-insertion mechanism (additional input), not amplification of an existing one.
- ≠ **Standard CFG** — same math, but "negative" is the *without-X baseline*, not the unconditional null prompt. The role of the slot is overloaded for the duration of the amplification run.
- ≠ **NAG** — NAG is normalized attention guidance applied via model-level patches; this pattern operates at the sampler/conditioning level. Compatible in principle but conflicting on the negative slot in practice (see Failure modes).

## Key facts

- Math: sampler computes `eps_out = eps_neg + cfg * (eps_pos - eps_neg)` (standard CFG). Substituting `eps_pos = eps_with_X` and `eps_neg = eps_without_X` yields `eps_out = eps_without + cfg * (eps_with - eps_without)`.
- `cfg = 1` → standard X behavior (sanity check; should match unmodified workflow byte-close).
- `cfg > 1` → amplified X.
- `cfg = 0` → X contribution removed.
- `cfg < 0` → anti-X (push away from with-X).
- Distinct from control vectors: this is **dynamic per-step**, varying with the sampler trajectory; control vectors are fixed offsets.
- Canonical POC: `scripts/apply_ttc_iclora_amplification_poc.py` (forks experimental workflow; idempotent + reversible).

## When it applies / doesn't

| Applies | Doesn't apply |
|---|---|
| X has a separable conditioning source (LoRA can be loaded/unloaded; reference image can be swapped/dropped) | X fires per-step via model patches with no isolatable off-state |
| Sampler uses CFG math (any standard ComfyUI CFGGuider) | Sampler bypasses CFG (some distilled paths fix `cfg=1`) |
| You can construct a positive-without-X parallel to positive-with-X | You only have a single positive prompt with no isolatable component |
| Combined with one optional model-level effect (e.g. NAG bypassed during run) | Multiple negative-slot consumers (NAG + this) muddle the test |

## Failure modes

| Symptom | Likely cause |
|---|---|
| `cfg=1` doesn't match non-POC workflow byte-close | Negative-slot conflict: NAG still active, sharing the slot with without-X stream |
| Output drifts toward unrelated prior at high cfg | Distilled sigma chain accumulates amplification; test in [-1, 5] before pushing further |
| No visible amplification at high cfg | Without-X stream isn't actually different — verify the bypass / loader / image swap took effect |
| Reference leakage with no init image | IC-LoRA case: with no init, the model has no competing visual anchor; restore init before running the sweep |

Edge cases:
- POC protocol bypasses `LTX2_NAG` (Node 508) before running. With NAG active, the negative slot serves both NAG and the without-X stream — POC variable isolation fails.
- `cfg < 0` is exploratory territory for distilled LTX 2.3; not validated.
- Init image is required when X = IC-LoRA; without one, the IC-LoRA has no competing anchor and amplification produces reference leakage rather than amplified X.

## Audit + tests

None — this is an inference-time pattern, not a workflow invariant. POC behavior is reproducible via the apply script; outputs are evaluated visually per the cfg sweep.

## References

- `scripts/apply_ttc_iclora_amplification_poc.py` — canonical POC; cfg sweep protocol in module docstring
- `docs/reference/sampler_reference.md` — CFG math + sampler choice context
- `docs/reference/nag_technical_reference.md` — why NAG conflicts on the negative slot during amplification runs
- `internal/analysis/iclora_landscape_analysis.md` — TTC1 landscape (private clone only)
- `docs/reference/_atomic_note_template.md` — concept-note variant template
