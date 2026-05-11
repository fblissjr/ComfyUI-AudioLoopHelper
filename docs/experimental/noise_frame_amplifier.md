Last updated: 2026-05-11

# Per-frame initial noise amplification (experimental)

> **Experimental.** Working implementation, theoretical backing, but
> only n=1 visual A/B so far. Tune for your content; don't expect a
> universal recipe.

A drop-in `NOISE`-wrapping node — `LTXNoiseFrameAmplifier` — that
multiplies the first N temporal frames of the sampler's initial
noise tensor by a scalar amplifier. Equivalent in effect to giving
those frames a higher initial sigma, without touching the sigma
schedule itself.

Use case: LTX 2.3 i2v init-anchored renders frequently spend the
first 0.5–2 seconds essentially repainting the init image with
minor variations before any meaningful motion develops. This node
provides a sampler-level lever to push the model out of that
"filler" window earlier.

## The problem it addresses

LTX 2.3's distilled LoRA is trained on a video distribution where
shots tend to **ease into motion**: opening frames stay near the
init-image state, action develops over 1–2 seconds, motion settles
in mid-clip. On i2v with init-image conditioning, that temporal
prior compounds with the cross-attention pull from the init image
itself — producing the characteristic "frame 0 looks like the init,
frame 30 looks slightly different, frame 60 finally starts moving"
shape.

The standard knobs operate at the conditioning level:

- `LTXVImgToVideoInplaceKJ.strength` reduces how heavily init pixels
  overwrite frame 0 of the latent.
- `LTXVAddGuide.strength` reduces the conditioning-level guide weight.
- `LTXLatentAnchorAware.bypass=True` removes content-aware token pull.

These reduce *attachment* to the init image, but they don't touch
the model's **temporal prior** — it still wants to ease into motion
regardless of how strongly the init is anchored.

## The mechanism

The denoising process solves `x_{t-1} = x_t - σ_t · model(x_t)` at
each step. Initial state `x_T` is sampled noise scaled by `σ_T`.

If we multiply the first N temporal frames of `x_T` by a scalar
`k > 1`, those frames begin sampling at effective initial sigma
`k · σ_T`. The sampler then has more denoising work to do in that
region than the model's training distribution led it to expect for
those frames. To resolve that work, the model must exercise more
freedom relative to the init-image temporal prior — motion has to
manifest earlier in those frames because there's no longer a
match-to-init-state low-noise path through them.

It's **non-uniform initial sigma** achieved by perturbing the
input noise tensor rather than writing a custom non-uniform
sampler. Sigma chain, sage attention, anchor mechanisms, STG
guidance — all unchanged. The intervention is entirely upstream
of the sampler.

## The node

```
RandomNoise.NOISE  →  LTXNoiseFrameAmplifier.noise
                      LTXNoiseFrameAmplifier.noise  →  SamplerCustomAdvanced.noise
```

Schema:

| Input | Type | Default | Tooltip |
|-------|------|---------|---------|
| `noise` | `NOISE` | — | Upstream NOISE (typically from `RandomNoise`). |
| `n_frames` | `INT` | 8 | Latent-temporal frames to amplify starting at frame 0. 0 disables. 8 latent frames ≈ 57 pixel frames (LTX VAE temporal scale = 8) ≈ 0.32 s at 25 fps. |
| `amplifier` | `FLOAT` | 1.5 | Multiplier on early-frame noise. 1.0 is no-op. Typical range 1.3–2.0; >2.5 likely over-noisy. <1.0 *attenuates* early noise (inverse use case — anchor MORE strongly to init). |

Output: `NOISE` (a wrapped callable that produces amplified noise
when the sampler invokes it).

NestedTensor handling: when the sampler operates on an AV-concat
latent (LTX's video + audio NestedTensor shape), only the video
sub-tensor is amplified. Audio noise is left untouched.

## Tuning

Start with the defaults (`n_frames=8, amplifier=1.5`). Render. Watch
the full clip — the point of this node is to see *where motion
starts*, so post-trimming to skip the opening defeats the test.
Compare against the same workflow with `amplifier=1.0` (no-op) to
isolate the node's contribution.

Direction-of-effect for each knob:

| Symptom in render | Suggested change |
|-------------------|------------------|
| Filler still too long; opening seconds still match init | Increase `amplifier` (1.7, 2.0) and/or extend `n_frames` (12, 16) |
| Opening seconds are visibly noisy / grainy before resolving | Reduce `amplifier` (1.3) or shrink `n_frames` (4) |
| Init image abandoned entirely; frame 0 doesn't match the reference | Reduce `amplifier` toward 1.0, or pair with higher `LTXVImgToVideoInplaceKJ.strength` to re-anchor frame 0 |
| Mid-clip motion looks fine but earlier frames look mismatched in style | The amplifier is doing too much. Try `amplifier=1.2, n_frames=6` for a gentler push |

## When NOT to use it

- **t2v workflows.** No init-image temporal prior to break; the
  model is already free to start anywhere. The node would just add
  noise variance for no benefit.
- **Renders where opening filler is desired** — e.g. a slow dolly-in
  on a scene where you want the establishing shot, then action. The
  filler IS the shot composition; don't fight it.
- **Identity-critical i2v** where the init image must be reproduced
  exactly at frame 0 (face-stability tests, character-consistency
  benchmarks). Amplification breaks frame-0 fidelity by definition.
- **Loop workflows where each iteration's first frame is its own
  init image** — applying this would discontinuously perturb each
  iteration boundary. Stick to single-shot i2v.

## Status + promotion gates

Experimental. Has:

- Working implementation in `nodes.py`
- 7 behavioral tests in `tests/test_ltx_noise_frame_amplifier.py`
  (TDD red → green, covers amplification math, identity at
  amplifier=1.0, identity at n_frames=0, out-of-range clamping,
  seed pass-through, attenuation case, registration)
- Theoretical backing (non-uniform-initial-sigma equivalence)

Lacks:

- Multi-seed visual A/B
- Comparison against the alternatives in this doc's "standard knobs"
  list (which is cheaper / which generalizes better)
- A controlled benchmark across content types (face-anchored vs.
  scene-anchored i2v)

Promotion gate to `docs/reference/`: at minimum, a 3+ seed A/B
comparing renders with `amplifier=1.0` vs. `amplifier=1.5` on the
same init image + prompt, showing reduced filler with retained
init-image fidelity at frame 0. Add a measured-metric gate (e.g.
optical-flow magnitude in the first 30 frames as a quantitative
"motion start" signal) if a future contributor wants to harden the
evidence.

## Related

- The audio-loop pipeline already pre-encodes audio + freezes it via
  `noise_mask=0`, which is a different kind of non-uniform noise
  treatment (mask = 0 means "don't sample here" rather than "sample
  harder here"). Both manipulate the noise/sampling boundary; this
  node is the inverse direction — *more* sampling work on selected
  frames, not less.
- Conditioning-level i2v levers (`LTXVImgToVideoInplaceKJ.strength`,
  `LTXVAddGuide.strength`, `LTXLatentAnchorAware.bypass`) compose
  with this — they reduce init-image attachment; this node breaks
  the temporal prior. Different mechanisms, additive effects.
