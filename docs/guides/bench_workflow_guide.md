# Bench a workflow — find the bottlenecks

Last updated: 2026-05-01 (added sage A/B procedure + bench-variant apply scripts)

When you want to know "where did the time go" for a workflow render — which nodes dominate wall time, whether attention is the bottleneck, where the per-iteration cost lives — this guide walks the bench procedure end-to-end.

The pipeline composes four pieces that all key off a single `RUN_ID`:

1. **`start_experiment.sh`** — launcher that exports the three telemetry env vars
2. **`exec_logger.py`** (auto-loaded) — per-node wall-time JSONL via ComfyUI's executor monkey-patch
3. **`nodes_sage.py`** (auto-loaded) — per-attention-call routing telemetry
4. **`scripts/exec_log_summary.py`** + **`scripts/sage_telemetry_summary.py`** — aggregators that read the JSONLs and print bottleneck tables

Optional fifth piece for kernel-level granularity: wire `ProfileBegin` / `ProfileIterStep` / `ProfileEnd` into the workflow for `torch.profiler` chrome traces. Use this when the per-node breakdown points at one node and you need to see what's happening *inside* it.

---

## Quickstart (no workflow edits needed)

The shipped `audio-loop-music-video_latent.json` already has Sage attention active — just launching via `start_experiment.sh` instead of plain `start.sh` captures everything you need. To bench the IC-LoRA path specifically, un-bypass the IC-LoRA chain (loader, VHS_LoadVideo, in-loop guide) before queuing the prompt.

```bash
# 1. Launch ComfyUI with telemetry on
./start_experiment.sh

# (in the ComfyUI UI: load audio-loop-music-video_latent.json,
#  un-bypass IC-LoRA chain if benching that path, set your reference video
#  + IC-LoRA path, queue the prompt, wait for it to finish)

# 2. Note the RUN_ID printed at startup, e.g.:
#    [start_experiment.sh] RUN_ID=20260501T103045Z_a3f1
#    SAGE_TRACE=auto  EXEC_LOG=auto

# 3. Aggregate the per-node breakdown
uv run --group dev python scripts/exec_log_summary.py --latest

# 4. Aggregate the attention split + gate verdict
uv run --group dev python scripts/sage_telemetry_summary.py \
  --sage-log "data/runs/<RUN_ID>/sage.jsonl" \
  --exec-log "data/runs/<RUN_ID>/exec.jsonl"
```

Both summaries run on the same `data/runs/${RUN_ID}/` tree so they line up.

## What each summary shows

### `exec_log_summary.py` — per-node bottlenecks

Top-N node classes by total wall time. This is where you spot non-attention bottlenecks: VAE encode/decode, sampler step, text encoding, image preprocess, anything else.

Sample output:

```
=== exec.jsonl  total_wall=487.31s  (1834 node-runs) ===

CLASS_TYPE                                CALLS    TOTAL_S     PCT  MEDIAN_S      P90_S
------------------------------------------------------------------------------------------
SamplerCustomAdvanced                        25    298.420   61.2%   11.847    12.103
LTXVTiledVAEDecode                            1     47.020    9.7%   47.020    47.020
LTXVAddVideoICLoRAGuide                      25     31.150    6.4%    1.243    1.301
VAEEncode                                    25     22.500    4.6%    0.892    0.943
LTXVAudioVAEDecode                            1      8.140    1.7%    8.140    8.140
...
```

Use:
- **>50% in one row** → the bottleneck is that one node class. Optimization effort goes there.
- **VAEEncode high in IC-LoRA workflows** → per-iter VAE encode of ref-video frames; the deferred profiling concern from `internal/ic_lora_assessment.md`.
- **VAE decode is one-shot at the end** → expected; optimize via `apply_no_tile_vae_decode.py` if not on `[1,1,1]`.

Flags:
- `--per-prompt` — split the table by `prompt_id` if multiple prompts ran in one ComfyUI session
- `--top N` — show top N rows (default 15)
- `--json` — machine-readable output for downstream tools (autoresearch metrics, etc.)

### `sage_telemetry_summary.py` — attention split + gate verdict

Sage routing per call: (`effective_mode`, `has_mask`) cross-section + total share of gen wall time.

The two cross-sections that matter:
- `(fp16_triton, has_mask=True)` — masked cross-attention. If <5% of gen wall, mask-kernel work in sage-fork has no leverage.
- `(fp8_cuda++, has_mask=False)` — unmasked self-attention. The "where attention time actually goes" denominator.

Pair with `--exec-log` so the percentages compute against KSampler total wall (the right denominator — excludes VAE / load / other non-sampler work).

## When to wire `ProfileBegin` / `ProfileIterStep` / `ProfileEnd`

The two summaries above give per-node + per-attention-call resolution. That's enough for ~90% of "where's the bottleneck" questions.

Wire the Profile nodes when you need:
- **Per-iteration deep dive** — chrome trace shows kernel-level work inside one iteration. Open in `chrome://tracing` or `perfetto.dev`.
- **Memory timeline** — VRAM use over time, useful for OOM debugging.
- **Top-kernel categorization** — `summary.txt` lists top kernels by cumulative time, classified (matmul / attention / vae / other).

**The fastest way to wire them** is the apply script:
```bash
uv run --group dev python scripts/apply_iclora_bench_profiling.py
# → produces internal/scratch/audio-loop-music-video_latent_iclora_bench.json
```

This splices `ProfileBegin → LoadAudio chain`, `ProfileIterStep` inside the subgraph after `IterationCleanup`, and `ProfileEnd` between `TensorLoopClose` and `LatentConcat`. No subgraph schema change — load directly without delete-and-re-add.

If you'd rather wire by hand: the three nodes are at `nodes.py:2626/2810/2849`. Place `ProfileBegin` before `TensorLoopOpen` (any pre-loop trigger), `ProfileIterStep` inside the loop body on the LATENT path, `ProfileEnd` after `TensorLoopClose`. Defaults: `warmup_iterations=1, active_iterations=3` — captures iters 2-4 (skip iter 1 compilation noise).

Profiler outputs land at `data/runs/${RUN_ID}/profiler/{trace.json, summary.txt, memory_timeline.html}` when `RUN_ID` is set (i.e., when launched via `start_experiment.sh`); otherwise they fall back to `internal/analysis/runs/profiler/<ts>/`.

Toggle off in three ways: set `enabled=False`, bypass the node (`mode=4`), or remove the three nodes entirely. All three give zero overhead.

## Sage A/B comparison procedure

When you want to quantify what each layer of the sage stack contributes (our `AudioLoopHelperSageAttention` vs KJ's `LTX2MemoryEfficientSageAttentionPatch` vs no sage at all vs both stacked), use the controlled 4-arm procedure. Detailed comparison + the patch-level differences live in `internal/analysis/sage_attention_comparison.md`; this section is the operational recipe.

### The four arms

| Arm | `AudioLoopHelperSageAttention` | `LTX2MemoryEfficientSageAttentionPatch` | What it isolates |
|---|---|---|---|
| `ours` | active | absent | Today's default — baseline for our setup |
| `off` | bypassed (mode=4) | absent | Pure-pytorch attention floor — no sage at all |
| `kj` | bypassed (mode=4) | active | KJ's per-block + RoPE-fusion path only |
| `stacked` | active | active | KJ on DiT blocks + ours on leftover layers (Lightricks-cameraman shape) |

### Producing arm variants

```bash
# One-time: produce all four arm files in internal/scratch/
for arm in ours off kj stacked; do
    uv run --group dev python scripts/apply_iclora_bench_sage_arm.py --arm $arm
done
```

Each command stages a different `audio-loop-music-video_latent_iclora_bench_<arm>.json`. Pre-flight requires `apply_iclora_bench_profiling.py` to have run first (the bench variant with Profile* nodes is the input).

### Rendering each arm

For controlled comparison, every variable other than sage configuration must be held constant: same audio file, same init image, same ref-video, same prompts, **same seed** (the `Seed (rgthree)` widget), same window/overlap settings.

```bash
for arm in ours off kj stacked; do
    RUN_ID=arm_$arm ./start_experiment.sh
    # In the ComfyUI UI:
    #   1. Load internal/scratch/audio-loop-music-video_latent_iclora_bench_$arm.json
    #   2. Verify same inputs as previous arms (image / audio / ref / prompts / seed)
    #   3. Queue prompt; wait for completion
    # Then shut down ComfyUI (Ctrl-C the launcher) so the next iteration gets
    # a fresh process — avoids carry-over from object_patches sticky state.
done
```

The `RUN_ID=arm_$arm` prefix tags telemetry artifacts so you can later identify which arm produced which `data/runs/<id>/` tree.

### Comparing across arms

```bash
uv run --group dev python scripts/bench_compare_runs.py \
    --runs arm_ours arm_off arm_kj arm_stacked --baseline arm_ours
```

Output sections:
1. **Top-line metrics** — `total_wall_s` / `sampler_wall_s` / `sampler_pct` / `attention_calls` / `attention_wall_s` / `attention_pct_of_total` per arm, with `(+X.X%)` delta vs baseline
2. **Per-node-class wall-time table** — top N classes ranked by max-share-across-runs, with deltas vs baseline. Spots non-attention regressions (e.g., does `off` regress `LTXVTiledVAEDecode` for some reason).
3. **Sage attention by `(effective_mode, has_mask)`** — cross-section showing routing per arm. The `auto` vs `fp8_cuda++` split tells you which kernels each arm exercised.

`--json` for machine-readable output. `--latest N` to auto-pick the most recent N runs (useful for ad-hoc bench cycles).

### Two important caveats

**1. The `stacked` arm's telemetry is partial.** When KJ patches the LTX-2 DiT transformer blocks at the per-block level (`add_object_patch` on `attn1.forward`), those calls are invisible to our dispatcher-level tracer. The `sage.jsonl` for the `stacked` arm captures only the LEFTOVER attention layers (text-encoder cross-attn, etc.) — not the per-block-patched calls. Wall time via `exec.jsonl` is still accurate; just don't expect attention-share counts in `stacked` to be directly comparable to `ours`.

**2. Fork-vs-upstream `sageattention` is a separate axis.** All four arms above run with whatever `sageattention` Python package is currently installed. To attribute fork contribution vs upstream sage, run the same 4-arm matrix twice — once with the fork installed, once with `pip install sageattention` (upstream). Tag the two passes via `RUN_ID=arm_ours_fork` vs `RUN_ID=arm_ours_upstream` etc. The comparator treats every render identically; the suffix is purely for your bookkeeping.

### Realistic expectations

Based on the per-render bench from `data/runs/20260501T161401Z_8335` (attention = 8.2% of wall, sampler = 72.5%):

- The sage A/B will show **small deltas (~5% e2e at best)**. Amdahl: if attention is 8% and the best sage variant takes attention to zero, the workflow is ~8% faster. Realistic gains are smaller — sage doesn't take attention to zero, just makes it faster.
- **Look at the chrome trace BEFORE running the 4-arm A/B.** The `data/runs/<RUN_ID>/profiler/trace.json` from the bench-variant render is where actionable optimization information lives — kernel-level inside the 72.5% sampler slice. It might reveal the bottleneck is something sage-A/B can't touch (per-iter VAE encode of ref-video, NAG patching cost, autotune overhead on small kernels, etc.).
- **NAG-on vs NAG-off** is a complementary single-variable A/B in the same shape: produce a variant where `LTX2_NAG` is bypassed (`mode=4`), render with same seed + inputs, compare. CLAUDE.md L165 hints NAG adds +17 points beyond strict-attention Amdahl prediction; bench confirms locally.

## Telemetry env vars (all auto-set by `start_experiment.sh`)

| Variable | Purpose | Default | Disable |
|---|---|---|---|
| `RUN_ID` | Single correlation key per render. ISO8601 + 4-hex-char rand. | auto-generated | `RUN_ID=` (empty) |
| `AUDIOLOOPHELPER_SAGE_TRACE` | Per-attention-call sage telemetry. ~22k rows per 5-iter render. | `auto` | `AUDIOLOOPHELPER_SAGE_TRACE=` |
| `COMFYUI_EXEC_LOG` | Per-node-execution start/end events. | `auto` | `COMFYUI_EXEC_LOG=` |
| `AUDIOLOOPHELPER_PER_PROMPT` | Route artifacts under `data/runs/${RUN_ID}/${prompt_id}/`. | unset | unset |

Disable any individually for one launch:
```bash
AUDIOLOOPHELPER_SAGE_TRACE= ./start_experiment.sh
```
This runs ComfyUI with exec log + RUN_ID enabled but no sage tracing (useful when sage's per-call syscall overhead shows up as a confound on a perf-sensitive run).

## Artifact layout

After a render via `./start_experiment.sh`:

```
data/runs/${RUN_ID}/
├── exec.jsonl          # per-node start/end events (exec_logger)
├── sage.jsonl          # per-attention-call routing (nodes_sage)
└── profiler/<ts>/      # torch.profiler outputs (only if Profile* nodes wired)
    ├── trace.json
    ├── summary.txt
    └── memory_timeline.html
```

Plus the rendered output at ComfyUI's configured `--output-directory`.

## Privacy + retention

- **Sage trace**: shape/timing only. No prompt text, no tensor values, no model weights.
- **Exec log**: captures input/output shape snapshots and short string node-inputs up to 120 chars, **including prompt text**. Redact before sharing if your prompts are sensitive.
- **No auto-cleanup.** `data/runs/` accumulates until you `rm` manually. The launcher's docstring covers this.

## When the bench points at IC-LoRA specifically

Hypotheses to check on this workflow's video-reference IC-LoRA wiring:

- **Per-iter VAE encode of ref-video** — `LTXVAddVideoICLoRAGuide` does its own VAE encode each iter. If `VAEEncode` shows up high in `exec_log_summary` (>5% of total wall), the deferred wrapper-with-pre-encoded-LATENT optimization in `internal/ic_lora_assessment.md` becomes worth it.
- **NAG on / off** — re-run with `LTX2_NAG` bypassed (mode=4) to quantify NAG's overhead. CLAUDE.md notes NAG is +17 points beyond strict-attention Amdahl prediction; bench confirms locally.
- **Sage attention-share** — if attention is <30% of gen wall, sage's int8 amortization has limited headroom.

## Cross-reference

- Telemetry env-var registry: `docs/reference/environment.md`
- Telemetry deep-dive: `docs/reference/telemetry_and_tracing.md`
- Sage A/B comparison + patch-level differences: `internal/analysis/sage_attention_comparison.md`
- The launcher itself: `start_experiment.sh`
- The exec logger: `exec_logger.py` (auto-installed at module load when `COMFYUI_EXEC_LOG` is set)
- Sage telemetry source: `nodes_sage.py`
- Sage trace verifier (per-iter signature diff): `scripts/verify_sage_iteration_trace.sh`
- Bench-variant apply scripts:
  - `scripts/apply_iclora_bench_profiling.py` — wires Profile* nodes into the iclora workflow
  - `scripts/apply_iclora_bench_sage_arm.py --arm {ours,off,kj,stacked}` — produces sage A/B arm variants
- Bench summary scripts:
  - `scripts/exec_log_summary.py` — single-run per-node bottleneck breakdown
  - `scripts/sage_telemetry_summary.py` — single-run attention split + gate verdict
  - `scripts/bench_compare_runs.py` — multi-run comparator (the A/B roll-up)
- Autoresearch metric extractors using the same artifacts: `internal/autoresearch/metrics/{sage_summary,wall_time}.py`
