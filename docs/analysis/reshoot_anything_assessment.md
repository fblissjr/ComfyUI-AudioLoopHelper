Last updated: 2026-05-08

# Reshoot-Anything (Morphic, CVPR-W 2026): Viability Assessment for AudioLoopHelper

Read of `coderef/video-to-video/` and arXiv:2604.21776v2 against the LTX 2.3
audio-looped music-video pipeline. Verdict up front: **not a direct integration
target**, but two of its sub-techniques are worth filing for the photoreal-drift
and i2v-init problems already documented in root `CLAUDE.md`.

## What the paper does

Reshoot-Anything rewrites the camera trajectory of a *single* monocular input
video. Pipeline is four stages, each materialized as a separate script in
`coderef/video-to-video/`:

| Stage | Script | What it produces |
|---|---|---|
| 1. Per-frame metric depth | `estimate_depth.py` | `depths.npz` `[T,H,W]` float32, GeometryCrafter+MoGe v2 or DepthCrafter |
| 2. 4D camera keyframing | `visualizer/app.py` (Viser browser UI) | `cam_info.json` (camera trajectory) |
| 3. Anchor render | `render_from_cam_info.py` | `render.mp4` + `render_mask.mp4` + `reference.png` (forward-warped novel-view, hole mask, first frame) |
| 4. Reshoot inference | `inference_wan22_v2v_local.py` | reshot video |

Stage 4 is a **WAN 2.2 I2V-A14B** (14B MoE) fine-tune with rank-512 LoRA on
attention + FFN, fully trained patchify layer. Conditioning is **token
concatenation through self-attention** (not cross-attention) of `(z_a, z_n, M_a)`
and `(z_s, z_n, M_s)`, doubling sequence length. Both arms ride 3D RoPE; an
**Offset RoPE** of +50 is added to source temporal embeddings (well above max
train length of 20 latent frames) to decouple source perceived position from
target trajectory.

Self-supervision trick — the actual contribution: synthesize triplets
`(V_s, V_a, V_t)` from a single monocular clip by taking two distinct
random-walk crops of the same source as `(V_s, V_t)` and forward-warping the
first frame of `V_s` along a dense 2D tracking field (AllTracker) into `V_a`.
No paired multi-view data needed. Trained on 100k clips × 2k steps batch=24,
hybrid 85% monocular + 15% ReCamMaster synthetic (the synthetic mixture
is needed for extreme out-of-distribution camera rotations only).

## Why it doesn't slot into AudioLoopHelper

Four blockers, ordered by severity:

1. **Task mismatch.** Reshoot rewrites the camera path of an *existing* video.
   The audio-loop pipeline generates new music videos from `(audio, prompts)`;
   there is no source video to reshoot. The only conceivable integration is a
   post-process pass on a finished render — but that breaks the audio-frozen
   `noise_mask=0` invariant: the reshoot model has no concept of audio latents
   and would re-synthesize visuals without honoring `LTXVConcatAVLatent`. The
   reshoot output would not lip-sync the original audio.
2. **Base-model mismatch.** WAN 2.2 I2V-A14B (Mixture-of-Experts, 14B params,
   high-noise + low-noise experts) ≠ LTX 2.3. No sigma-chain compatibility, no
   `LTXVConcatAVLatent`, no audio-video cross-attention. Porting the *technique*
   to LTX 2.3 requires retraining a LoRA on LTX — out of solo-hobbyist range.
3. **Hardware envelope.** `run_wan22_inference.sh` defaults to `NUM_GPUS=8`
   with `--ulysses_size=8` (Ulysses sequence-parallel sharding across 8 GPUs).
   Inference at 832×480, 81 frames, source-token concat (2× sequence length),
   40 unipc steps, rank-512 LoRA on a 14B MoE backbone is not single-GPU 24GB
   territory without significant surgery (offload + sequence chunking + drop
   ulysses). Even *base* WAN 2.2 14B I2V single-GPU is tight.
4. **Weights not released.** README marks `[ ] Pretrained LoRA weights release`
   and `[ ] Self-supervised training pipeline release` as pending. Today, the
   install completes but `HIGH_NOISE_LORA_WEIGHTS` / `LOW_NOISE_LORA_WEIGHTS`
   point at nothing. Training the LoRA from scratch on 100k clips is not solo
   territory either.

Also worth noting — the install pulls **PyTorch3D** and **Flash-Attention 2**
from source (10–30 min), clones DepthCrafter and GeometryCrafter into `.deps/`,
and bakes a fixed WAN frame-count constraint (`(total // 4) * 4 + 1 - 4`,
encoded in `pipeline_spec.py:wan_consumed_frames`). All workable, just a lot
of moving parts for an integration that doesn't earn its way in.

## What's actually portable (the techniques worth keeping)

Three ideas in the paper are decoupled enough from WAN 2.2 to be worth filing
for problems we already have:

### 1. 3D-aware noise injection on the reference frame before warping

§3.2.3 + §4.1: before forward-warping the reference frame to build the anchor,
they inject Gaussian noise on RGB, sampled uniformly from `[0, 0.5]` per
channel on normalized images. Purpose: prevent the model from "simple texture
copying" of the anchor and force it to route textures from the source video
instead.

**Why it might apply here**: root `CLAUDE.md` notes "Illustrated inits drift
toward photoreal across iterations (cross-attention is photoreal-trained)."
The cross-attention bias toward photoreal is structurally adjacent to the
"model copies anchor texture too eagerly" problem the paper is correcting.

**Proposed probe**: noise-inject on the init image RGB (uniform `[0, 0.5]`
per channel on normalized values) immediately before `LTXVPreprocess`. Does
not require a pipeline re-architecture — single inline op on the init image,
A/B against current path. ~15 min experiment. If it nudges the
illustrated→photoreal drift, file as a knob on the init-image prep path.
If not, discard. Low cost, real signal either way.

Caveat: their noise is *training-time* on data that the model then learns to
denoise as part of the LoRA objective. Inference-time injection on a base LTX
2.3 i2v path is a different operation — the model wasn't trained to expect
that distribution. Treat as exploratory, not a principled fix.

### 2. Offset RoPE for decoupling temporal context from target position

§3.2.2: a constant +50 offset is added to source-token RoPE temporal
embeddings. 50 ≫ max train frames (20 latent), so the source pathway is
strictly outside the trajectory's temporal manifold. Empirically necessary
for the source tokens to act as content reference rather than positional
context.

**Why it might apply here**: conceptually adjacent to the "loop-body
CONDITIONING must carry `frame_rate`" rule in root `CLAUDE.md`. Both are
"make sure the model doesn't read context-token positional info as part of
the target trajectory." File for any future cross-window context conditioning
work (e.g. if a future iteration of `LatentSeamZoneMask` ever feeds prior-window
context tokens into the current window's denoising).

Not actionable today. LTX 2.3's loop body doesn't currently feed context
tokens through self-attention with the target — it freezes them via
`noise_mask=0` instead. The mechanism is different (mask-based vs RoPE-offset
based). But if the architecture ever changes shape, the offset-RoPE pattern
is the prior art.

### 3. Source token reconstruction loss (α=0.1)

§3.2 + §4.3: an auxiliary L1 loss between predicted source tokens and clean
source latents, weighted at α=0.1. Forces the source pathway to preserve
high-fidelity content rather than collapsing.

**Not actionable**: this is a training-time loss on a fine-tuning run we
won't be doing. Filed only as evidence that "concatenated context tokens
need an explicit content-preservation pressure or they collapse" — a useful
prior if anyone ever builds a content-token concat path on LTX.

## What's interesting but not portable

- **AllTracker dense 2D tracking** (Harley et al. ICCV 2025) for the
  forward-warping step that builds the anchor from the reference frame.
  Inference-only use is conceivable but the pretrained weights and integration
  cost don't match any current need.
- **GeometryCrafter + MoGe v2** depth pipeline. Strong stack, but we don't
  consume per-frame depth anywhere in the audio-loop path.
- **Hybrid 85/15 monocular/synthetic data mix.** Methodologically interesting
  (synthetic-only suffers texture degradation, monocular-only fails extreme
  rotations) but applies to training, not inference.

## Failure modes the paper documents (worth knowing)

Useful to know in case any of these surface later in unrelated contexts:

- **Anchor-only methods (EX-4D)** hallucinate when the anchor is imperfect
  (Fig. 10 rows 1–2). Geometric-only conditioning without source-content
  routing produces ghosting/inconsistency.
- **Synthetic-only methods (ReCamMaster)** fail real-world dynamics —
  tennis-ball deformation, smoke under colored lighting (Fig. 9, Fig. 10
  rows 3–4). Domain gap to real footage.
- **Cross-attention conditioning** is empirically *worse* than token-concat
  through self-attention for this task (Table 2 ablation: RotErr 3.53 vs 3.16,
  TransErr 4.31 vs 4.78, Mat. Pix. 1766 vs 2627, FVD-V 626.37 vs 571.06).
  Cross-attention "loses correct texture" — they argue self-attention's full
  pairwise routing is what enables high-fidelity content sourcing.

The cross-attn vs token-concat ablation is the most interesting takeaway from
a generic conditioning-architecture standpoint, but again, not actionable in
LTX 2.3 inference where the architecture is fixed.

## Experimental probes worth running

After the first pass above, looking at what's actually in the repo today
shifts the priority. Two probes, ordered by expected payoff and decreasing
confidence:

### Probe 1 (high-confidence, infra exists): TTC init-guide amplification sweep

The paper's *spirit* — "force the model to honor the source/anchor signal
harder than its prior is pulling it" — already has a tool here:
`scripts/apply_ttc_init_guide_amplification_poc.py` (staged workflow at
`example_workflows/experimental/init_guide_amplification_poc.json`).
Mechanism documented in `docs/reference/cfg_analog_amplification.md`:

```
eps_out = eps_without_init_guide
          + cfg * (eps_with_init_guide - eps_without_init_guide)
```

The existing POC's docstring already names the use case literally: *"Identity-drift
studies: does amplifying the init-frame guide reduce iter-over-iter drift,
or does it just over-anchor and freeze motion?"* That's exactly the failure
mode root `CLAUDE.md` flags as "illustrated inits drift toward photoreal across
iterations." The probe has been scaffolded since 2026-04-25; what's
missing is the actual sweep + writeup.

Concrete protocol (steal from the POC docstring):

1. Run `apply_ttc_init_guide_amplification_poc.py` (idempotent + reversible).
2. Open `example_workflows/experimental/init_guide_amplification_poc.json`.
3. Bypass `LTX2_NAG` in the loop subgraph (negative-slot conflict otherwise).
4. Wire an illustrated init image (the failure-mode trigger).
5. Fixed seed across runs. Sweep `CFGGuider(644).cfg`:
   - `1.0` — sanity (must match non-POC byte-close)
   - `2.0`, `3.0`, `5.0` — amplification stress
   - `0.0` — baseline without init-guide contribution
   - `-1.0` — anti-init-guide (deliberate repulsion; diagnostic)
6. Sample frames at each iteration boundary; diff against iter 1.
7. Score by visual identity preservation and by motion-vs-frozen tradeoff.

Decision criteria for the writeup (`internal/analysis/<topic>.md`):
- If `cfg=2-3` cleanly reduces drift without freezing motion → promote as
  a recommendation in `docs/reference/cfg_analog_amplification.md` use-case
  table.
- If amplification freezes motion before it fixes drift → file as evidence
  the drift mechanism is downstream of init guide and isn't fixable via
  amplification alone. That itself is useful: it would mean cross-attention
  bias is the load-bearing problem, not init-signal weakness.
- If the cfg=1 sanity arm doesn't reproduce non-POC byte-close → infrastructure
  bug, fix before continuing.

This is the first thing to do. Reshoot-Anything is the prompt; it isn't
contributing the mechanism — TTC already exists.

### Probe 2 (low-confidence, exploratory): RGB noise on init image pre-`LTXVPreprocess`

Direct paper port. Inject uniform noise per channel sampled from `[0, σ_max]`
on the init image RGB (normalized to `[0, 1]`) before `LTXVPreprocess`.

Honest mechanism caveat: the paper's noise is **3D-aware** (it gets
forward-warped along with the reference frame, so the noise pattern moves
coherently with geometry). Without depth + warping in our path, we have
flat i.i.d. noise on the init. The paper's mechanism doesn't fully transfer.

Speculative reason it might still do something:
- Init image is encoded once, frozen at frame 0 via `noise_mask=0`.
- RGB noise → noisy latent at frame 0.
- Subsequent frames denoise attending to frame 0; noisy frame-0 may break
  the "literal copy init" mode and let the model's prior contribute more.

Risk: if the model just preserves the noise (because `noise_mask=0` holds
the latent), the result is grainy throughout instead of helpful.

Cheapest implementation path — no new node:
- Open the canonical latent workflow.
- Insert ComfyUI core's image-noise node (or compose with `ImageBlend`
  against a noise IMAGE) between `LoadImage` and `LTXSmartImageResize`.
- Sweep `σ_max ∈ {0.0, 0.05, 0.1, 0.2, 0.5}` against a fixed seed.
- Score for grain artifacts vs identity-drift behavior.

If a sweep shows a useful operating point, *then* scaffold a dedicated
experimental node (`LTXInitImageRGBNoise`, default `noise_strength=0.0`
per the no-op-default rule in root `CLAUDE.md`) and an apply-script that
stages it into a workflow variant. Don't scaffold the node before the
sweep; the mechanism is too speculative to justify ahead of evidence.

### Not pursuing

- **Direct port of token-concat through self-attention** — architectural
  change to LTX 2.3's DiT. Out of scope for inference-only.
- **Offset RoPE for source tokens** — depends on a source-token pathway
  that doesn't exist in our loop (we use `noise_mask=0` instead of
  concatenated source tokens). File for later if the architecture ever
  changes shape.
- **Self-supervised pseudo-multi-view training** — requires LoRA training
  on LTX 2.3, not in scope.

## Bottom line

Skip Reshoot-Anything as a direct integration: wrong task, wrong base
model, wrong hardware envelope, weights unreleased. The paper's value
here is as motivation to *finally run* the TTC init-guide amplification
sweep that's been scaffolded since April — that POC is the closest tool
in the repo to the question "force the model to honor the init harder
across iterations" and it's a single afternoon's work to grade.

## Pointers

- Paper PDF: `internal/scratch/2604.21776v2.pdf` (private clone only)
- Code: `coderef/video-to-video/`
- Frame-count constraint reference: `coderef/video-to-video/pipeline_spec.py`
- Inference launcher (8-GPU assumption): `coderef/video-to-video/run_wan22_inference.sh`
- Related project context: root `CLAUDE.md` "Init image conditioning + IC-LoRA paths"
  section (illustrated→photoreal drift discussion)
- Related project context: `docs/reference/cfg_analog_amplification.md`
  (alternative mechanism for amplifying conditional signals without retraining)
