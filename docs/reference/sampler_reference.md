Last updated: 2026-06-05

# Sampler reference — `euler` vs `euler_ancestral` vs `euler_ancestral_cfg_pp`

Grounded walkthrough of how the three Euler-family samplers behave in
ComfyUI, why they produce different results on LTX 2.3 distilled-1.1's
sigma schedule, and which to use for each of our workflows.

## TL;DR (2026-04-27 update)

**Sampler**: `euler` (plain). Confirmed against Lightricks's own
distilled inference in `coderef/LTX-Desktop/.../ltx_pipeline_common.py`
(uses `SimpleDenoiser` + `euler_denoising_loop`) and
`coderef/ID-LoRA/packages/ltx-core/src/ltx_core/components/diffusion_steps.py::EulerDiffusionStep`
(first-order Euler). NOT `euler_ancestral_cfg_pp` — that's a community
variant in some ComfyUI workflows; the 4-step plateau near σ≈0.99 in
the canonical sigma curve amplifies ancestral re-noise enough to bleed
across our TensorLoop iteration boundaries.

**Sigmas**: `ManualSigmas "1.0, 0.99375, 0.9875, 0.98125, 0.975,
0.909375, 0.725, 0.421875, 0.0"`. These are Lightricks's hand-tuned
`DISTILLED_SIGMA_VALUES` from
`coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:16` — what
their distilled checkpoint was trained to denoise. Pre-2026-04-27 we
used `BasicScheduler linear_quadratic 8 1` which approximated this
curve parametrically; the canonical hand-tuned values are the spec.
Migration: `scripts/apply_canonical_sigmas.py`.

**Other settings**: `CFGGuider cfg=1`, **no `ModelSamplingSD3` shift
node** (stripped from all distilled workflows 2026-05-01 via
`scripts/apply_strip_sd3_shift_node.py`; the canonical path feeds the
fixed sigmas directly — see CLAUDE.md "No flow-matching shift node"),
decoder `LTXVTiledVAEDecode [1,1,1,true,"cpu","float16"]` on 24GB+ (the device/dtype pair is load-bearing for full songs — `"auto","auto"` pre-allocates full-video fp32 buffers and kernel-OOMs the final decode; see `benchmarking_memory_pressure.md`)
(single-tile, ~3× faster cold-pass than [2,2,1]); fall back to
[2,2,1] on ≤16GB. Migration: `scripts/apply_no_tile_vae_decode.py`.

All code references are to `ComfyUI/comfy/k_diffusion/sampling.py`
(comfy-core) and `ComfyUI-LTXVideo/guiders/multimodal_guider.py`
(the KJ guider). All quotes are verbatim from those files.

## 1. The three samplers, end to end

### 1.1 `sample_euler` (line 190-212) — deterministic

```python
def sample_euler(model, x, sigmas, ...):
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]              # (s_churn=0 → gamma=0 path)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)   # d = (x - denoised) / sigma_hat
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt                     # Euler step — deterministic
    return x
```

Pure deterministic Euler integration. Same seed + same sigma schedule
→ bit-identical output. No re-injected noise mid-sampling.

Used by LTX-2's own pipeline code in `coderef/LTX-2/packages/
ltx-pipelines/src/ltx_pipelines/utils/samplers.py:34-74`
(`euler_denoising_loop`, called by `SimpleDenoiser` with no guidance).
Our committed `_latent.json` KSamplerSelect widget is `euler`
specifically to match this.

### 1.2 `sample_euler_ancestral` (line 216-237) — re-injects noise every step

```python
def sample_euler_ancestral(model, x, sigmas, ..., eta=1., s_noise=1., noise_sampler=None):
    # CONST model_sampling routes to sample_euler_ancestral_RF (rectified flow)
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_euler_ancestral_RF(...)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            # step to sigma_down (LOWER than target), then re-add noise up to sigma_up
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x
```

Ancestral step: deterministic-step down PAST the target, then re-add
`sigma_up` worth of fresh noise. `sigma_up` scales with remaining
sigma range (via `get_ancestral_step` + `eta`); at high sigma, it's
a substantial fraction of the step.

LTX 2.3 uses `ModelSamplingSD3` which is rectified flow → this call
routes to `sample_euler_ancestral_RF` (line 240+). Same principle,
RF-adapted math. The re-injection is still there.

### 1.3 `sample_euler_ancestral_cfg_pp` (line 1244-1284) — ancestral + CFG++

```python
def sample_euler_ancestral_cfg_pp(model, x, sigmas, ..., eta=1., s_noise=1., noise_sampler=None):
    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)

    uncond_denoised = None
    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True  # forces uncond branch
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            alpha_s = sigmas[i] * lambda_fn(sigmas[i]).exp()
            alpha_t = sigmas[i + 1] * lambda_fn(sigmas[i + 1]).exp()
            d = to_d(x, sigmas[i], alpha_s * uncond_denoised)   # DIRECTION FROM UNCOND

            sigma_down, sigma_up = get_ancestral_step(sigmas[i] / alpha_s, sigmas[i + 1] / alpha_t, eta=eta)
            sigma_down = alpha_t * sigma_down

            x = alpha_t * denoised + sigma_down * d             # magnitude still CFG'd
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x
```

Two things different from plain ancestral:

1. **CFG++**: the step direction `d` uses `uncond_denoised` (the negative
   branch's prediction), but the magnitude uses `denoised` (the CFG'd
   prediction). Separates "where to move" (uncond, unbiased) from
   "how far" (guided). Empirically better on rectified-flow models.
2. **Alpha-scaled coordinates**: `alpha_s` / `alpha_t` are log-SNR-
   derived; the ancestral step math operates in log-SNR space rather
   than raw sigma space. Another RF optimization.

**Critical caveat — CFG=1.0 collapses CFG++**: at `CFG=1.0`,
`uncond_denoised == denoised` (the model's uncond branch equals the
cond branch when guidance is unit-scale). Then `d = to_d(x, sigma, alpha_s * denoised)`,
which is effectively the same as plain ancestral plus the alpha-scaling.
You get the re-noise cost with no CFG++ benefit.

The `disable_cfg1_optimization=True` flag forces the uncond branch
to run even at CFG=1.0 (normally comfy-core short-circuits this for
performance). So the sampler works but doesn't *gain* anything from
CFG++ at CFG=1.

## 2. Interaction with LTX 2.3 distilled-1.1 sigma schedule

Our sigmas, verified bit-exact against `DISTILLED_SIGMAS` at
`coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:16`:

```
step:  0     1       2      3       4      5       6      7       8
sigma: 1.0  0.994  0.988  0.981  0.975  0.909  0.725  0.422  0.0
Δ:         0.006  0.006  0.006  0.006  0.066  0.184  0.303  0.422
```

Five plateau steps at σ ≈ 0.99 where very little denoising happens
per step. Then three collapse steps (0.909 → 0.725 → 0.422 → 0.0)
where most of the denoising is compressed. This shape is the
signature of the distilled schedule — it's what the distilled model
was trained to run on.

### What each sampler does on this specific schedule

**`euler`**: At each plateau step, denoises the tiny sigma delta,
no noise injection. The three collapse steps do the real work.
Reproducible; iteration N+1's context latent is deterministic given
iteration N's output.

**`euler_ancestral`**: Each plateau step injects `sigma_up` worth of
fresh noise. `sigma_up` at σ ≈ 0.99 is near-maximum (bounded by the
remaining sigma range + eta). **Five independent random re-injections
happen during a phase where the model is barely denoising.** Then
three collapse steps have to absorb all that accumulated noise. With
only three steps, the averaging is incomplete — some high-sigma
noise leaks through to the final sample.

In a loop workflow this leaked noise CONTAMINATES the context latent
fed to the next iteration. Over 10+ iterations this compounds. In
practice we see:
- Subject identity drift (face features change progressively)
- Style drift (illustrated → photoreal even stronger)
- Position drift (composition slides across iteration boundaries)

This is why CLAUDE.md documents "Don't use `euler_ancestral` —
plateau at σ ≈ 0.99 for 5 steps amplifies re-noise → iteration drift."

**`euler_ancestral_cfg_pp`**: Same re-injection pattern as
`euler_ancestral`, PLUS the CFG++ direction change. At CFG=1.0, the
CFG++ part is a no-op (uncond = cond), so you inherit all the
iteration-drift cost with no quality benefit. At cfg > 1, the CFG++
direction does help — but the ancestral re-noise still drifts across
iterations in our loop.

## 3. Interaction with `MultimodalGuider`

`ComfyUI-LTXVideo/guiders/multimodal_guider.py:161-179`:

```python
a_noise_pred_neg, v_noise_pred_neg = 0, 0
a_noise_pred_perturbed, v_noise_pred_perturbed = 0, 0
a_noise_pred_modality, v_noise_pred_modality = 0, 0

if any(params.do_uncond() for params in [audio_params, video_params]):
    # ... computes noise_pred_neg ...
    v_noise_pred_neg, a_noise_pred_neg = self.unpack_latents(noise_pred_neg)
```

`do_uncond()` is `not math.isclose(cfg_scale, 1.0)`. If both AUDIO
and VIDEO `GuiderParameters` have `cfg=1.0`, the uncond branch is
skipped and `noise_pred_neg` is never assigned. Line 269 then
references it unconditionally:

```python
"uncond_denoised": noise_pred_neg,
```

→ `UnboundLocalError: cannot access local variable 'noise_pred_neg'`.

**This is a bug in the guider, not in our design.** The workaround
for `_latent_stg.json` is `cfg=2.0` on both modalities (mild CFG,
forces the branch to run). A proper fix would patch the guider to
initialize `noise_pred_neg = None` before the conditional and handle
None downstream.

With `cfg=2.0`, the guider's math (`noise = pos + (cfg-1)*(pos-neg) + stg*(pos-perturbed)`)
becomes `pos + 1.0*(pos-neg) + 1.0*(pos-perturbed)`. CFG and STG
each contribute one unit. STG is no longer the sole quality signal,
but it's still half the directed modification.

## 4. Recommendations per workflow

| Workflow | Guider | CFG | KSamplerSelect | Why |
|---|---|---|---|---|
| `_latent.json` (baseline) | `CFGGuider` | 1.0 | **`euler`** | Matches `coderef/LTX-2/.../distilled.py` `SimpleDenoiser` pipeline — deterministic Euler on fixed sigmas. Zero iteration drift. |
| `_latent_stg.json` (STG hybrid; **archived** — `example_workflows/archive/`) | `MultimodalGuider` | 2.0 (AUDIO + VIDEO) | **`euler`** | STG adds quality via attention-block perturbation. Adding ancestral re-noise on top re-introduces the iter-drift mechanism STG was meant to help avoid. `euler_ancestral_cfg_pp` only makes sense single-shot. |
| `_latent_keyframe.json` | `CFGGuider` | 1.0 | **`euler`** | Same rationale as baseline; keyframes are orthogonal to sampling choice. |
| `_image_adain_perstep.json` (**archived** — `example_workflows/archive/`; the plain `_image.json` was retired) | `CFGGuider` | 1.0 | **`euler`** | Same baseline config; AdaIN is orthogonal. |
| Upstream `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json` (LoRA-on-full-22B, NOT our config) | `MultimodalGuider` | 3/7 (VIDEO/AUDIO) | `euler_ancestral_cfg_pp` | Single-shot, so iteration drift doesn't apply. CFG at 3-7 makes CFG++ direction meaningful. Ancestral adds dynamic range. Different model (LoRA not merged) + different use case. **Do not copy this stack onto our merged distilled-1.1.** |

## 5. When to reach for `euler_ancestral_cfg_pp`

Only defensible case in our architecture: you've run `euler` on
`_latent_stg.json` (with `MultimodalGuider cfg=2`) and the output
reads as too clean / lacks per-iteration variation. Ancestral adds
stochastic variety per step — at the cost of compounding iteration
drift across loops.

In that scenario, run 1-2 loop iterations (not 10+) with
`euler_ancestral_cfg_pp` to get dynamic-range sample variety without
accumulating drift across many iterations. Stitch shorter sequences
if you need that look.

For our default 10-iter audio-loop use case: **always `euler`**.

## 6. Common mistakes to avoid

- **Using `euler_ancestral`** because it's the comfy-core default
  for many non-LTX workflows. Wrong for our distilled schedule.
- **Using `euler_ancestral_cfg_pp`** because the ComfyUI-LTXVideo
  upstream workflows ship with it. Those workflows run the full-22B
  + distilled LoRA, not the merged distilled-1.1 checkpoint. Different
  training distribution.
- **Setting CFG=1.0 on `MultimodalGuider`**. Hits the
  `noise_pred_neg` unbound-variable bug in the guider. Use `cfg ≥ 1.01`
  on at least one modality, or stick with `CFGGuider CFG=1` if you
  don't want guidance.
- **Raising steps above 8 for the distilled model** ("more steps =
  better quality"). The distilled model was trained for exactly 8
  steps on these specific sigmas. More steps over-denoise.

## 7. How to verify your sampler choice is active

1. Load the workflow in ComfyUI.
2. Check KSamplerSelect (node 154) widget shows `euler`.
3. After a run, look at the ComfyUI console log. You'll see the
   traceback (on error) or the progress bar pointing to
   `sample_euler` (not `sample_euler_ancestral` or
   `sample_euler_ancestral_cfg_pp`).
4. `AudioLoopPlanner` summary should show the expected stride for
   your window/overlap values — confirms the full chain is live.

## 8. Source code references

- `ComfyUI/comfy/k_diffusion/sampling.py:190-212` — `sample_euler`
- `ComfyUI/comfy/k_diffusion/sampling.py:216-237` — `sample_euler_ancestral`
- `ComfyUI/comfy/k_diffusion/sampling.py:240-270` — `sample_euler_ancestral_RF` (rectified-flow path, where LTX 2.3 routes)
- `ComfyUI/comfy/k_diffusion/sampling.py:1244-1284` — `sample_euler_ancestral_cfg_pp`
- `ComfyUI/comfy/k_diffusion/sampling.py:1288-1290` — `sample_euler_cfg_pp` (delegates to ancestral_cfg_pp with eta=0, s_noise=0)
- `ComfyUI-LTXVideo/guiders/multimodal_guider.py:138-281` — `MultimodalGuider.predict_noise` (CFG=1.0 uncond-branch bug at :161-179 + :269)
- `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py:34-74` — LTX-2's own `euler_denoising_loop` (what our `euler` choice matches)
- `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:16` — `DISTILLED_SIGMAS` (the fixed schedule our `ManualSigmas` node feeds directly — no `linear_quadratic` / `shift=13` approximation)
