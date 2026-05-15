# Benchmarking memory pressure on ComfyUI + LTX 2.3

Last updated: 2026-05-15

## Role

How to measure GPU memory behavior under LTX 2.3 renders when the goal is to compare attention-kernel changes, mask-path changes, or anything that touches per-call working set. The naive "did it OOM?" gate misses the actual cost, which manifests as offload-driven slowdown rather than failure.

## Why "didn't OOM" is not "fit in budget"

ComfyUI on Ada/Hopper with `comfy-aimdo` enabled runs a **dynamic VRAM loader**: when GPU memory is tight, model weights shuffle to CPU and back transparently. The render finishes successfully, just slower. Two visible symptoms:

- Terminal: `Model X prepared for dynamic VRAM loading. NNN MB Staged. 0 patches attached.` per model load.
- Renders that "should" OOM at peak working-set actually complete because the loader provides a pressure-release valve.

Snapshot-based memory benchmarks (e.g. `torch.cuda.max_memory_allocated()` after a single call) miss this entirely. They report "peak allocation fit under cap" while the loader was actively offloading + reloading.

## What to actually measure

A complete memory-pressure benchmark on this stack captures three signals:

1. **Per-call kernel dispatch and timing** — sage trace. Tells you which kernel handled each attention call and how long it took.
2. **Per-model VRAM residency over time** — aimdo `/aimdo/vram` polling. Tells you how much of each model is actively in GPU memory vs offloaded, sampled at 1Hz.
3. **End-to-end wall time** — `Prompt executed in N seconds` from ComfyUI's terminal. The summary signal.

Comparing two runs across these three signals lets you attribute differences cleanly (kernel change vs offload pressure vs total-time).

## Tools (in this repo)

| Tool | Purpose | Input | Output |
|---|---|---|---|
| `start_experiment.sh` | Telemetry wrapper around `start.sh`; sets `RUN_ID` + `AUDIOLOOPHELPER_SAGE_TRACE=auto` + `COMFYUI_EXEC_LOG=auto` env vars before exec'ing the underlying start script | optional `[mode]` positional, any flags forwarded | `data/runs/${RUN_ID}/sage.jsonl` (per-attn-call), `data/runs/${RUN_ID}/exec.jsonl` (per-node) |
| `scripts/bench_aimdo_vram.py` | Polls `/aimdo/vram` HTTP endpoint at 1Hz, writes NDJSON | `--endpoint`, `--interval`, `--output` (required), optional `--max-duration` | NDJSON with `{ts, elapsed_s, data}` per poll; `data` is full endpoint response (per-model `loaded_size` vs `total_size`, driver-level free/total VRAM, pinned RAM, etc.) |
| `scripts/analyze_sage_traces.py` | Aggregates per-shape kernel timing across N sage.jsonl files | one or more sage.jsonl paths; defaults to scanning `data/runs/*/*/sage.jsonl` | stdout: per-run summary table + per-shape masked/unmasked p50+p95 |

The `/aimdo/vram` endpoint is provided by the third-party `ComfyUI-MemoryVisualization` custom node. Without that node installed + ComfyUI restarted to load it, the polling script returns 404. Verify with `curl http://localhost:8188/aimdo/vram | head -c 200` before running the bench.

## Run recipe — A/B comparison of two attention configs

Goal: compare sage routing modes (e.g. `auto` vs `auto_mask_aware`) under identical workflow + hardware, with all three signals captured.

1. **Verify endpoint is up:**
   ```bash
   curl -s http://localhost:8188/aimdo/vram | head -c 200
   ```
   Expect JSON with `"enabled": true`.

2. **Start polling for run A:**
   ```bash
   uv run python scripts/bench_aimdo_vram.py --output data/runs/aimdo_A.ndjson
   ```

3. **Run the workflow** in ComfyUI (the sage trace lands at `data/runs/${RUN_ID}/${prompt_id}/sage.jsonl` automatically because of `start_experiment.sh`).

4. **Stop polling** with Ctrl-C after the render completes.

5. **Flip the config** (sage node widget, or whichever variable you're comparing).

6. **Repeat steps 2-4 for run B**, with a different `--output` filename.

7. **Aggregate sage traces:**
   ```bash
   uv run python scripts/analyze_sage_traces.py data/runs/${RUN_ID}/${prompt_id_A}/sage.jsonl data/runs/${RUN_ID}/${prompt_id_B}/sage.jsonl
   ```

8. **Diff the VRAM residency curves** between `aimdo_A.ndjson` and `aimdo_B.ndjson` (no canned tool yet — load both as pandas or jq the `loaded_size / total_size` per model over `elapsed_s`).

## Removing the offload safety valve

When you want "would this OOM without the loader catching it" signal, disable dynamic VRAM via `--disable-dynamic-vram`:

```bash
bash start_experiment.sh default --disable-dynamic-vram
```

Caveat: this requires `start.sh` to actually forward trailing args to `main.py`. The vanilla shipped `start.sh` consumes only the first positional (the mode) and may drop the rest. Verify by adding `echo "args: $*"` before the python invocation in `start.sh`, or check the terminal for `prepared for dynamic VRAM loading` lines — they should NOT appear after a successful flag delivery. With the flag set, the comfy-aimdo init should print `No working comfy-aimdo install detected. DynamicVRAM support disabled` (line at `comfy/main.py:237`) instead of `DynamicVRAM support detected and enabled` (line 235).

## Reproducibility check

Sage traces from this stack are remarkably stable run-to-run when config is identical. A clean reproducibility signal: 6 consecutive FML2V renders with the same workflow + sage `auto` mode produced **identical** dispatch counts (672 masked calls × fp8_cuda++ each, zero fallbacks) and p95 timing within ~1% of p50. If your two A/B runs differ meaningfully in dispatch counts at the trace level (vs just timing), the config drifted between runs — investigate before claiming a perf delta.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Polling script returns 404 | `ComfyUI-MemoryVisualization` not loaded into running ComfyUI | Restart ComfyUI after installing the custom node |
| `--disable-dynamic-vram` ignored (still see `prepared for dynamic VRAM loading` lines) | `start.sh` not forwarding `"$@"` to `main.py` | Patch `start.sh` to `shift` after `MODE=${1:-default}` and append `"$@"` to the python invocation |
| Sage trace has zero `has_mask=True` entries despite workflow setting `guide_strength<1.0` | Workflow strips `guide_attention_entries` upstream of `_process_input` (e.g. audio-loop NestedTensor handling) — see `comfy/ldm/lightricks/model.py:1044` | Use a workflow that lets guide entries survive to the model (e.g. RuneXX FML2V at `example_workflows/benchmark_workflows/`); audio-loop family is structurally incompatible |
| Run-to-run variance > 5% at p95 across identical configs | Autotune cache cold for one run, or different mask sparsity per render | Run 3+ times; aggregate; quote p50 + p95 not single-run numbers |
| Trace file size suspiciously small (< 500 KB on LTX self-attn) | Partial render (user cancelled, OOM, error). | Check `exec.jsonl` for completion; treat partial trace as data-only-not-conclusion |

## FFN intermediate dominates at multi-guide scale

`LTXVChunkFeedForward` (KJNodes) is **load-bearing** for fitting LTX 2.3 multi-guide renders on 24 GiB cards. It chunks the FFN intermediate along the T dimension to reduce peak allocation; bypass it and the workflow OOMs deterministically at stage-2 step 0.

Failure mode (observed with `AudioLoopHelperSageAttention` in `auto` mode, fp8_cuda++ on masked attention, `--disable-dynamic-vram`, `LTXVChunkFeedForward` set to `mode=4`):

```
Stage-1: 8/8 steps completed cleanly
Stage-2: 0% [0/3] -- OOM immediately at:
  ldm/lightricks/av_model.py:356  ff_out = self.ff(vx_scaled)
  ldm/lightricks/model.py:322     return self.net(x)
  ldm/lightricks/model.py:308     gelu(self.proj(x), approximate="tanh")
```

The killer allocation is the FFN's first `Linear(dim, inner_dim)` output. At multi-guide-expanded T=44880 with `dim=4096`, `mult=4` → `inner_dim=16384`, the output tensor `(1, 44880, 16384)` bf16 = ~1.47 GiB. With `chunks=2` that becomes ~735 MiB per slice; sequential allocation/free keeps peak below the ceiling.

Stage-1 doesn't trigger this because stage-1 uses the smaller T=10780 shape — proj output there is ~360 MiB, comfortably fits.

Implication: at LTX 2.3 multi-guide scale on a 24 GiB card, attention-kernel choice is the **second-order** memory variable. FFN chunking is the first-order one. When benchmarking attention changes, always run with `LTXVChunkFeedForward` enabled (canonical config in `example_workflows/benchmark_workflows/fml2v_var_*.json` since 2026-05-14) — otherwise the FFN OOM dominates and attention-side deltas can't surface.

Upstream footgun worth knowing: `comfy/ldm/lightricks/model.py::FeedForward.__init__` accepts a `glu` kwarg but never references it. Both LTX 2.3 call sites pass `glu=True` expecting GEGLU; the class silently builds plain GELU. So `inner_dim = dim * 4`, not `dim * 8`. Doesn't affect the bench (the deployed behavior is what's measured), but anyone reading the call sites assuming GEGLU should know.

## Related

- `scripts/bench_aimdo_vram.py` — polling endpoint reader.
- `scripts/analyze_sage_traces.py` — sage trace aggregator.
- `start_experiment.sh` — telemetry wrapper enabling all three signal sources.
- `nodes_sage.py::_route_mask_aware` (audio-loop side) — what `auto_mask_aware` mode actually routes.
- `coderef/sage-fork/sageattention/core.py:225-248` — sage dispatcher routing logic on the fork side.
- ComfyUI-MemoryVisualization custom node — provides the `/aimdo/vram` endpoint; install separately.
