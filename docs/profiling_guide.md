Last updated: 2026-04-17

# Profiling Guide: End-to-End Audio Loop Profiling

Capture per-kernel and per-node timing across the audio loop so you can
make data-driven decisions about where to spend optimization effort.

## What you get

Three output files per profile run, in a timestamped subdirectory of
`./profile_output/<YYYYMMDD_HHMMSS>/`:

- **`trace.json`** — full chrome trace. Open at
  [perfetto.dev](https://ui.perfetto.dev/) (drag-drop the file) or
  `chrome://tracing`. Lets you see every kernel call in time order,
  zoom into individual iterations, and spot gaps / overlaps.
- **`summary.txt`** — top 50 GPU kernels by cumulative time, categorized.
  Good for answering "where is time going?" at first glance.
- **`memory_timeline.html`** — VRAM allocation over time (if
  `include_memory` was enabled). Shows peak usage per iteration and
  fragmentation patterns.

You can also re-run `scripts/profile_summary.py trace.json` afterwards to
produce an additional categorized summary from any saved trace.

## How to enable / disable

**Enable**: drop three nodes into your workflow (see wiring below).
**Disable**: three equivalent options, any one works:

1. Set `enabled=False` on `ProfileBegin` — zero overhead, nodes stay in
   the graph for easy re-enable.
2. Right-click → Bypass on any of the three profile nodes — standard
   ComfyUI bypass (`mode: 4`), nodes become passthroughs.
3. Delete the three nodes entirely.

Settings live only on `ProfileBegin`. `ProfileIterStep` and `ProfileEnd`
are widget-free — they read shared state from `ProfileBegin`.

## Where to place the three nodes

```
┌─ ProfileBegin ─┐
│   trigger ─────┼─→ (normal graph)
└────────────────┘
        │
        ▼ (the passthrough trigger output is not strictly required —
           placement forces execution order, but can be left unwired
           if ProfileBegin is wired somewhere early in the graph)

    ... rest of model / audio / VAE loaders ...

    TensorLoopOpen
        │
        ▼
    ┌── Subgraph #843 (extension) ───────────────────────────┐
    │                                                        │
    │    LatentContextExtract → LTXVAudioVideoMask → ...     │
    │                                                        │
    │                  ... sampler ...                       │
    │                                                        │
    │    LTXVCropGuides → LatentOverlapTrim → IterationCleanup
    │                                              │          │
    │                                              ▼          │
    │                                    ┌─ ProfileIterStep ─┐│
    │                                    │   latent ─────────┼┼──→ subgraph output
    │                                    └───────────────────┘│
    └────────────────────────────────────────────────────────┘
        │
        ▼
    TensorLoopClose
        │
        ▼
    ┌─ ProfileEnd ─┐
    │   trigger ───┼─→ (downstream consumer, e.g., VHS_VideoCombine)
    └──────────────┘
```

### `ProfileBegin` placement

Anywhere before `TensorLoopOpen`. Tap any existing connection point — the
`trigger` input is untyped passthrough. Easiest targets:

- Output of the `UNETLoader` (MODEL) — intercept before the first model
  patch node (`PathchSageAttentionKJ` or equivalent).
- Output of `AudioLoopController` (any scalar output).
- Output of an early `FloatConstant` that feeds the window_size.

The output of `ProfileBegin` goes to the consumer that was originally
connected. E.g., if you tap `UNETLoader → PathchSageAttentionKJ`, rewire
it as `UNETLoader → ProfileBegin → PathchSageAttentionKJ`.

### `ProfileIterStep` placement

Inside subgraph #843, **after `IterationCleanup`** (if present) or
`LatentOverlapTrim`. Before the subgraph output.

### `ProfileEnd` placement

After `TensorLoopClose`, intercepting the flow to `VHS_VideoCombine`.

```
TensorLoopClose → ProfileEnd → VHS_VideoCombine
```

Or tap any post-loop scalar flow.

## Recommended settings

For a first-run "where is time going?" look, use the defaults:

| Widget | Default | Why |
|---|---|---|
| `enabled` | `True` | — |
| `output_dir` | `./profile_output/` | Timestamped subdir created per run |
| `warmup_iterations` | `1` | Iteration 1 has first-time compilation noise |
| `active_iterations` | `3` | Captures variance across real iterations |
| `include_cpu` | `True` | Catches Python overhead and node dispatch cost |
| `include_memory` | `True` | VRAM timeline; diagnoses fragmentation |
| `include_shapes` | `True` | Shapes in trace — helps identify which layer is slow |
| `include_flops` | `False` | Expensive; enable only for deep analysis |

At defaults, overhead is ~15% (all the data options together). The
measured iteration time will be slower than normal — but relative
proportions (which kernel is the biggest share) are still accurate.

## Reading the output

### Quick answers from `summary.txt`

- **Is attention the bottleneck?** Look at the `attention` category
  percentage. If >50%, sparse-attention work (SpargeAttn) has real ROI.
- **Are norms or RoPE significant?** If `norm` or `rope` is >10% each, a
  triton kernel swap is worth considering.
- **Is Gemma still hot?** Check for any `aten::...` ops with large
  cumulative time. If Gemma shows up, the `CachedTextEncode` cache isn't
  hitting as expected.
- **Are VAE ops significant?** Look for VAE-related kernels. If >10%,
  tiled VAE or VAE residency might help.

### Reading the chrome trace

Drop `trace.json` into [perfetto.dev](https://ui.perfetto.dev/).

Key views:
- **Iteration boundaries** show as markers from `profiler.step()` calls.
- **Our named spans** appear with labels like `CachedTextEncode.hit`,
  `IterationCleanup.always`, `LatentContextExtract`, `LatentOverlapTrim`
  — look for them to see exactly when our code runs and for how long.
- **Gaps in GPU activity** reveal Python/dispatch overhead — if CPU is
  running but GPU is idle, that's a latency issue to investigate.

## Caveats

- **Profiler overhead is not zero.** Absolute per-iteration times will
  be slower with profiling on. Only trust the relative breakdown.
- **CUDA graphs / `torch.compile`** hide individual kernel names. LTX
  doesn't use compile in our default workflows, so this shouldn't matter.
- **First iteration has warmup noise**: CUDA kernel selection, model
  compilation, caching allocator initialization. That's why
  `warmup_iterations=1` is the default.
- **Trace files are large**: 3 iterations at 8 sampling steps with shapes
  = ~100-200 MB. Compressible. Default `output_dir` is outside the git
  repo.

## Re-running summaries on saved traces

```bash
uv run python scripts/profile_summary.py ./profile_output/20260417_120000/trace.json
```

Produces a categorized summary without re-running the workflow. Handy if
you want to explore different categorization rules or compare multiple
runs.

## Troubleshooting

### "ProfileIterStep called without an active ProfileBegin"

You placed `ProfileIterStep` (or `ProfileEnd`) but didn't wire
`ProfileBegin` — or `ProfileBegin` ran with `enabled=False`. Either
wire a `ProfileBegin` node before the loop, or disable the remaining
profile nodes via bypass.

**If this happens mid-workflow**: the profiler state is kept on the `torch`
module specifically to survive `ComfyUI-HotReloadHack` reimports of our
package. If you see the warning appear AFTER `ProfileBegin` logged its
"recording to ..." line, a hot reload may have orphaned the prior profiler.
The fix is already applied (state persists on `torch`), but if you edit
our source files during a profile run, results are undefined — finish the
run first, then edit.

### Harmless warnings in the console during a profile run

You may see these in the console:
- `Profiler clears events at the end of each cycle` — we pass
  `acc_events=True` so events are retained; this warning can be ignored.
- `External init callback must run in same thread as registerClient` —
  kineto (PyTorch profiler backend) is sensitive to cross-thread use.
  ComfyUI runs nodes via asyncio; the warning is benign.
- `Memory block of unknown size was allocated before the profiling
  started` — some tensors existed before `profiler.start()` fired, so
  their dealloc events aren't attributed. Does not affect summary data.

### No output files appeared

- Check `enabled=True` on `ProfileBegin`.
- Check for `[AudioLoopHelper] ProfileBegin: recording to ...` in the
  console — should appear when the workflow starts.
- Check for `[AudioLoopHelper] ProfileEnd: wrote profile to ...` — should
  appear at the end. If it doesn't, `ProfileEnd` isn't being reached
  (check your graph order).
- Confirm CUDA is available (`torch.cuda.is_available()`). On CPU-only
  systems, `ProfileBegin` logs a warning and disables itself.

### Trace is enormous (>500 MB)

Disable one or more expensive data options:
- `include_shapes=False` → ~30% smaller trace
- `include_memory=False` → ~15% smaller trace
- `include_flops=False` → already default off
- Reduce `active_iterations`

Or profile a shorter subset of the run by swapping in the profile nodes
on a non-looping variant of the workflow.
