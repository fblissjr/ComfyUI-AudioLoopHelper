Last updated: 2026-04-22

# Why CLIP must not enter the loop body

## TL;DR

LTX2_NAG's negative conditioning lives inside a closure stored in
ComfyUI's `ModelPatcher.object_patches`. ComfyUI's model offload path
migrates tensors inside `transformer_options["patches"]` but does NOT
migrate `object_patches`. When per-iteration CLIP encoding evicts the
DiT to make VRAM room, NAG's captured `nag_cond_video` tensor is left
pointing at a device layout that no longer matches the reloaded
model — so NAG silently disengages. The symptom: every class NAG was
suppressing (microphones, `still image with no motion`, deformed
hands, duplicate characters, photoreal creep from illustrated inits)
returns simultaneously starting at iteration 2.

Fix: `TimestampPromptScheduleBatchEncode` runs CLIP ONCE outside the
loop, emits a pre-encoded conditioning list; `ConditioningSelectByIteration`
runs inside the loop with no CLIP dependency. DiT stays resident for
the full run.

## The asymmetry

### Where NAG stores its negative conditioning

`ComfyUI-KJNodes/nodes/ltxv_nodes.py:467-501` (LTX2_NAG.execute):

```python
context_video = nag_cond_video[0][0].to(device, dtype)    # tensor → GPU
...
patched = LTXVCrossAttentionPatch(context_video, ...)     # captures in closure
model_clone.add_object_patch(
    f"diffusion_model.transformer_blocks.{idx}.attn2.forward",
    patched,
)
```

`LTXVCrossAttentionPatch.__init__` stores `self.nag_context = context`
(line 423-428). `__get__` returns a bound method whose closure
captures `self.nag_context` by reference (line 430-440). That bound
method is what gets registered as an `object_patch`.

### What ComfyUI does on DiT offload

`comfy/model_patcher.py` (key lines):

- L339: `.clone()` shallow-copies `object_patches`. Closures are shared
  references across clones — migration of one is migration of all.
- L561-580: `model_patches_to(device)` walks
  `transformer_options["patches"]` and moves every tensor it finds to
  the target device. **It does NOT touch `object_patches`.** The
  captured `nag_context` tensor stays wherever it was.
- L1078-1085: `detach(unpatch_all=True)` offloads model weights but
  leaves `object_patches` closures untouched. Closures now point at
  tensors whose owning device-layout is gone.
- L900-915: `patch_model()` re-injects closures on reload. No
  device-validation of captured state.

### What triggers the offload

`nodes.py:1413-1418` (`CachedTextEncode.execute`, cache-miss path):

```python
tokens = clip.tokenize(text)
cond = clip.encode_from_tokens_scheduled(tokens)   # forces CLIP to GPU
```

`encode_from_tokens_scheduled` calls `load_models_gpu([CLIP])`, which
(if VRAM is tight) reaches `free_memory()` → `model.detach(unpatch_weights=True)`
at `comfy/model_management.py:664, 690, 764-766`. That evicts the
DiT. Next sampler step reloads the DiT but the closure's captured
tensor device state is already inconsistent.

## The failure mode in this workflow (pre-2026-04-22)

The legacy `_latent.json` had TWO `CachedTextEncode` nodes (1559 +
1607) running per iteration, feeding `ConditioningBlend` (1608). Each
schedule boundary crossing triggered a cache miss on at least one of
them. Iteration 2+ symptoms:

- Items explicitly negated (e.g. `microphones`) reappeared.
- Photoreal creep from illustrated/painterly inits.
- `still image with no motion` leaked back — frames went stiffer.
- Anatomy glitches (deformed hands, extra limbs, duplicate characters)
  returned.

All four mapped to NAG's single negative prompt
(`still image with no motion, subtitles, deformed facial features,
extra limbs, disfigured hands, duplicate character, twin, clone`)
going silent. They correlated exactly because they shared one
transformer-options cache entry that went stale together.

Schedule-bypassed runs stayed clean because CLIP loaded once at graph
build and never reloaded — DiT stayed resident and the closure's
captured tensor never fell out of sync.

## Fix: keep CLIP out of the loop body

`TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration`
(`nodes.py`, added 2026-04-22):

- Batch encoder: `clip`, `schedule`, `stride_seconds`,
  `audio_duration`, `snap_boundaries` → `conditioning_list` +
  `iteration_count`. Runs ONCE, outside the loop. Dedup means
  identical prompts across iterations are encoded once.
- Selector: `conditioning_list` + `current_iteration` → `conditioning`.
  Runs inside the loop. Indexes a Python list. No CLIP reference.

CLIP load count per run drops from `2 × num_iterations` to
`num_unique_prompts`. For a typical 6-entry schedule over 10
iterations: from 20 loads to 6. DiT stays resident. NAG's closure
never gets exposed to an offload/reload round-trip.

Migration: `scripts/apply_batch_encode_fix.py` (idempotent; validates
required source nodes up front so partial migration is impossible).

## Why we didn't "just fix `object_patches` migration"

Three reasons:

1. It's upstream ComfyUI behavior. We'd have to maintain a fork or
   ship a runtime monkey-patch; both fragile.
2. Even if `object_patches` migrated correctly, per-iteration CLIP
   loading would still evict the DiT and induce extra load churn.
   Keeping CLIP out of the loop avoids the problem entirely rather
   than patching around a symptom.
3. Pre-encoding is strictly better on other axes too: fewer Gemma
   forwards total (dedup), no LRU thrash, no cache-size tuning.

## Adjacent anti-patterns

Don't put these inside a loop subgraph either, for the same reason:

- Any new `CLIPTextEncode` variant that touches CLIP per-iteration.
- `clip_vision` models that force their own load on use.
- Any node that calls `load_models_gpu` internally on a per-iter path.

Rule of thumb: **if it has CLIP/VAE as an input AND depends on
`current_iteration` (directly or transitively), it belongs outside
the loop** — compute the per-iter discriminator elsewhere, index a
pre-materialized structure inside the loop.

## Follow-ups

- `IS_CHANGED` hardening on `TimestampPromptScheduleBatchEncode`:
  hash `(schedule, stride_seconds, audio_duration, snap_boundaries)`
  to guarantee ComfyUI skips re-execution when inputs are stable,
  independent of upstream AudioLoopController re-execution. Low
  priority because empirically the encoder only runs once per run
  today.
- Apply the same migration to `_image.json`, `_latent_stg.json`,
  `_latent_keyframe.json`. The apply script is parameterized — one
  invocation per variant.
- Upstream this doc (or a shorter version) to ComfyUI-KJNodes as a
  maintainer note, so other custom nodes using `add_object_patch`
  with captured tensors pick the same pattern.
