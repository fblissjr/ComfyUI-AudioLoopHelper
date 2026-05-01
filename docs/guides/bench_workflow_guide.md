# Bench a workflow — find the bottlenecks

Last updated: 2026-05-01

When you want to know "where did the time go" for a workflow render — which nodes dominate wall time, whether attention is the bottleneck, where the per-iteration cost lives — this guide walks the bench procedure end-to-end.

The pipeline composes four pieces that all key off a single `RUN_ID`:

1. **`start_experiment.sh`** — launcher that exports the three telemetry env vars
2. **`exec_logger.py`** (auto-loaded) — per-node wall-time JSONL via ComfyUI's executor monkey-patch
3. **`nodes_sage.py`** (auto-loaded) — per-attention-call routing telemetry
4. **`scripts/exec_log_summary.py`** + **`scripts/sage_telemetry_summary.py`** — aggregators that read the JSONLs and print bottleneck tables

Optional fifth piece for kernel-level granularity: wire `ProfileBegin` / `ProfileIterStep` / `ProfileEnd` into the workflow for `torch.profiler` chrome traces. Use this when the per-node breakdown points at one node and you need to see what's happening *inside* it.

---

## Quickstart (no workflow edits needed)

The shipped `audio-loop-music-video_latent_iclora.json` already has Sage attention active — just launching via `start_experiment.sh` instead of plain `start.sh` captures everything you need.

```bash
# 1. Launch ComfyUI with telemetry on
./start_experiment.sh

# (in the ComfyUI UI: load audio-loop-music-video_latent_iclora.json,
#  set your reference video + IC-LoRA path, queue the prompt, wait for it
#  to finish)

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

The three nodes are at `nodes.py:2626/2810/2849`. Wiring:
1. Place `ProfileBegin` *before* `TensorLoopOpen` on any `trigger` input
2. Place `ProfileIterStep` inside the loop body (typically after `LatentOverlapTrim` or `IterationCleanup`)
3. Place `ProfileEnd` *after* `TensorLoopClose` (`trigger` from any downstream value)
4. Defaults: `warmup_iterations=1, active_iterations=3` — captures iters 2-4 (skip iter 1 compilation noise).

Outputs land at `internal/analysis/runs/profiler/<ts>/{trace.json, summary.txt, memory_timeline.html}`.

Toggle off in three ways: set `enabled=False`, bypass the node (`mode=4`), or remove the three nodes entirely. All three give zero overhead.

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
- The launcher itself: `start_experiment.sh`
- The exec logger: `exec_logger.py` (auto-installed at module load when `COMFYUI_EXEC_LOG` is set)
- Sage telemetry source: `nodes_sage.py`
- Sage trace verifier (per-iter signature diff): `scripts/verify_sage_iteration_trace.sh`
- Autoresearch metric extractors using the same artifacts: `internal/autoresearch/metrics/{sage_summary,wall_time}.py`
