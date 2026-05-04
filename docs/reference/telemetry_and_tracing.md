Last updated: 2026-05-04

# Local logging and profiling

> **What this is.** Two opt-in local logging/profiling instruments that
> **this plugin** ships, plus an offline aggregator that reads the files
> they produce. All three default to off. **No code in this plugin makes
> any network call.** There is no telemetry endpoint, no analytics service,
> no "phone home." Everything is plain JSONL written to gitignored local
> files for debugging, profiling, and bench analysis on your own machine.

> **Scope.** This document covers only this plugin's instruments
> (`AUDIOLOOPHELPER_SAGE_TRACE`, `COMFYUI_EXEC_LOG`,
> `scripts/sage_telemetry_summary.py`). Anything ComfyUI itself does at
> runtime is upstream behavior we don't speak for here.

> **What "ours" means.** Both env-var-gated instruments are this plugin's
> own code. `AUDIOLOOPHELPER_SAGE_TRACE` writes from `nodes_sage.py`.
> `COMFYUI_EXEC_LOG` is our runtime monkey-patch on ComfyUI's `execute()`,
> installed by `exec_logger.py` only when the env var is set; otherwise it's
> a single attribute check at import time and that's it. The env-var name
> uses the `COMFYUI_` prefix because it controls our patch on a ComfyUI
> internal — the patch itself is plugin code.

This document is the single source of truth for what gets captured, where
it lands, how to control it, and what the privacy posture is.

## TLDR

| | sage logger | exec logger |
|---|---|---|
| Code home | `nodes_sage.py` (this plugin) | `exec_logger.py` (this plugin; runtime monkey-patch on ComfyUI's `execute()`) |
| Env var | `AUDIOLOOPHELPER_SAGE_TRACE` | `COMFYUI_EXEC_LOG` |
| Default | off (zero overhead) | off (zero overhead) |
| Activation | env var must be set **before launching ComfyUI** | env var must be set **before launching ComfyUI** |
| Granularity | one record per attention call | one record per ComfyUI node execution |
| Captures | tensor shapes, kernel mode, timing | node class, tensor shapes, **truncated input snapshot** (see below), timing |
| Captures user prompt text? | no | yes, truncated to 120 chars |
| Default sink (with `RUN_ID`) | `data/runs/${RUN_ID}/sage.jsonl` (gitignored) | `data/runs/${RUN_ID}/exec.jsonl` (gitignored) |
| Legacy fallback (no `RUN_ID`) | `internal/analysis/runs/sage/sage_<TS>.jsonl` (gitignored) | `internal/analysis/runs/exec_log/exec_<TS>.jsonl` (gitignored) |
| Auto-cleanup | none | none |
| Network access | **none** | **none** |
| Data leaves machine? | **no** | **no** |

## How it splits: in-ComfyUI capture vs. outside-ComfyUI analysis

There are three distinct steps. Two of them run **inside** ComfyUI (the
two loggers); the third runs **outside** ComfyUI as a standalone Python
script that reads the files the first two wrote.

```
  ┌──────────────────────────────────────┐
  │  inside ComfyUI                      │
  │                                      │
  │  AUDIOLOOPHELPER_SAGE_TRACE=auto ─┐  │
  │  COMFYUI_EXEC_LOG=auto ───────────┤  │      ┌──────────────────────────┐
  │                                   ▼  │      │  outside ComfyUI         │
  │  user runs workflow ─► loggers append│      │                          │
  │                        to JSONL files│  ──► │  scripts/                │
  │                                      │      │  sage_telemetry_summary  │
  │  internal/analysis/runs/             │      │  reads JSONL, prints     │
  │   ├── sage/sage_*.jsonl              │      │  per-mode stats and the  │
  │   └── exec_log/exec_*.jsonl          │      │  Phase 0 gate verdict    │
  └──────────────────────────────────────┘      └──────────────────────────┘
```

The two loggers and the aggregator never run together in the same
process. Loggers capture; aggregator analyzes. You can run the
aggregator any time after the logger files exist, even months later.

## End-to-end workflow

### 1. Set env vars BEFORE launching ComfyUI

The loggers install themselves at Python module import time
(`exec_logger.install()` runs at the bottom of the module and is a no-op
when the env var is unset; `SageTracer` opens its file handle when the
sage node first executes). Setting either env var after ComfyUI is already
running has no effect — restart ComfyUI to pick up the change.

```bash
# in the same shell that will launch ComfyUI:
export AUDIOLOOPHELPER_SAGE_TRACE=auto
export COMFYUI_EXEC_LOG=auto

# then launch:
cd <comfyui>
python main.py [your usual flags]
```

If you use a `start.sh` or systemd unit, set the env vars there. Both
tracers re-check the env var on every fresh ComfyUI process; there's no
persistent on-disk toggle.

**Recommended**: launch via `start_experiment.sh` at this plugin's
repo root rather than setting the env vars by hand. That wrapper
exports `RUN_ID` (a fresh `${ISO8601_UTC}_${rand4}` per launch) plus
the two logger vars defaulted to `auto`, then exec's the underlying
ComfyUI launcher. With `RUN_ID` set, every artifact for one render
correlates via a single directory key — see the path layout below.
Plain `<comfyui>/start.sh` (post-2026-04-26 minimization) is back to a
vanilla ComfyUI launcher with no plugin-specific instrumentation; nothing
is wired unless you set the env vars yourself or use `start_experiment.sh`.

**Path layout** (post-RUN_ID propagation):
- With `RUN_ID` set: `data/runs/${RUN_ID}/{exec.jsonl, sage.jsonl, profiler/}`.
- Without `RUN_ID` (legacy fallback): `internal/analysis/runs/<subdir>/<prefix>_<TS>.jsonl`.

The summarizer scripts (`scripts/sage_telemetry_summary.py`,
`scripts/verify_sage_iteration_trace.sh`) search both layouts and pick
the most recent by mtime. Full env-var registry at
`docs/reference/environment.md`.

### 2. Run your workflow as normal

Each attention call appends one line to the sage JSONL. Each node
execution appends two lines (`start` + `end`) to the exec JSONL. No
extra UI, no progress bars, no popups, no network calls. The files are
line-buffered, so even a mid-run crash leaves a useful partial log.

### 3. Stop ComfyUI cleanly

The sage logger writes a `{"event": "summary", ...}` line on model
cleanup (the `ON_CLEANUP` callback in `nodes_sage.py`). A clean
shutdown gives you the summary; a hard kill loses only the summary
line, not the per-call records.

### 4. Run the aggregator outside ComfyUI

This is a plain Python script. ComfyUI does not need to be running.

```bash
# from the plugin directory, with the dev group installed
uv run --group dev python scripts/sage_telemetry_summary.py \
    --sage-log internal/analysis/runs/sage/sage_<timestamp>.jsonl \
    --exec-log internal/analysis/runs/exec_log/exec_<timestamp>.jsonl
```

Output is a per-(kernel, mask) table plus the Phase 0 gate verdict
(see "Phase 0 aggregator" below for what the verdict means).

The aggregator never touches the live ComfyUI process, never modifies
the JSONL files, and never writes to disk. You can run it on the same
file as many times as you want.

## What data is used where, and why

| Data | Captured by | Used for | Used where |
|------|-------------|----------|------------|
| Per-attention-call tensor shape, kernel mode, timing | sage logger (in ComfyUI) | Kernel routing audit + per-mode timing breakdown | Aggregator (outside ComfyUI) |
| Per-node class type, input shape snapshot, duration | exec logger (in ComfyUI) | Pin total wall-clock time on the sampler nodes (the denominator for "% of gen time spent in masked-triton") | Aggregator (outside ComfyUI), `--exec-log` flag |
| Free text prompt strings ≤120 chars | exec logger (in ComfyUI) | Debugging which workflow path ran with which prompt | None automated; surfaces only if you read the JSONL by hand |
| Tensor values, model weights | NEITHER | n/a | n/a |
| Anything sent over the network | NEITHER | n/a | n/a |

The aggregator only consumes `class_type` and `duration_s` from the exec
log. It does not parse or display the `inputs` snapshot. So even with
the exec logger on, the **percent-of-gen-time number is computed without
ever reading prompt text** — the captured prompt strings are only
visible if you open the JSONL file directly.

## Why we have these

- Diagnose attention-kernel routing (is the mask-aware path actually
  routing masked calls to the triton kernel?)
- Build a perf budget (where does time actually go in an LTX gen?)
- Drive optimization decisions with numbers, not vibes (see
  `docs/reference/sage_attention.md` and the optimization-plan workflow)

## Sage logger: `AUDIOLOOPHELPER_SAGE_TRACE`

### Turning it on

```bash
# auto-generate a timestamped path under internal/analysis/runs/sage/
AUDIOLOOPHELPER_SAGE_TRACE=auto python <comfyui>/main.py

# or pin it to an explicit file path
AUDIOLOOPHELPER_SAGE_TRACE=/tmp/sage.jsonl python <comfyui>/main.py
```

The accepted "auto" tokens are `auto`, `1`, `true`, `yes` (case-insensitive).
Any other non-empty value is treated as an explicit file path. Empty or
unset means disabled, with a single attribute check per attention call as
the entire runtime cost.

### What gets recorded — every attention call

```json
{
  "ts": 1713912345.678,
  "iter": 0,
  "shape": [1, 31776, 2048],
  "has_mask": false,
  "mode": "auto_mask_aware",
  "effective_mode": "auto",
  "fell_back": false,
  "elapsed_us": 842.5
}
```

- `ts` — epoch float seconds.
- `iter` — loop-iteration counter pulled from `transformer_options`
  (falls back to sampler step when no `LoopIterationStamp` is present).
- `shape` — Q tensor dims as a list of integers. **No tensor values, no
  weights, no embeddings.**
- `has_mask` — boolean.
- `mode` / `effective_mode` — kernel name strings (e.g. `"auto"`,
  `"sageattn_qk_int8_pv_fp16_triton"`). Used to verify mask-aware
  routing is firing as expected.
- `fell_back` — boolean: did this call fall back to PyTorch attention
  because sage raised?
- `elapsed_us` — wall-clock microseconds for this call.

### What gets recorded — summary on cleanup

```json
{
  "ts": 1713912999.123,
  "event": "summary",
  "total_calls": 8960,
  "fallback_count": 0,
  "distinct_shapes": 3
}
```

### What DOES NOT get recorded

- Prompt text or any user input.
- Tensor values, weights, embeddings, or model state.
- File paths to your models or workflows.
- Anything network-related.

## Exec logger: `COMFYUI_EXEC_LOG`

### Turning it on

```bash
# auto-generate a timestamped path
COMFYUI_EXEC_LOG=auto python <comfyui>/main.py

# explicit file path
COMFYUI_EXEC_LOG=/tmp/exec.jsonl python <comfyui>/main.py

# stderr (useful for `tail -f` style debugging)
COMFYUI_EXEC_LOG=stderr python <comfyui>/main.py
```

### What gets recorded — every node execution

```json
{
  "ts": 1713912345.678,
  "event": "start",
  "prompt_id": "abc123",
  "node_id": "169",
  "class_type": "ConditioningCombine",
  "inputs": { ... shape-summary of the node's inputs ... }
}
{
  "ts": 1713912345.901,
  "event": "end",
  "prompt_id": "abc123",
  "node_id": "169",
  "class_type": "ConditioningCombine",
  "duration_s": 0.223,
  "outputs": [ ... shape-summary of outputs ... ]
}
```

On error: `event="error"` row with the exception string truncated to 500
characters.

### What gets recorded — privacy-relevant detail on `inputs`

The node-input snapshot uses `_shape_of()` (see `exec_logger.py`):

- **Tensors** → `{"shape": [...], "dtype": "...", "device": "..."}` only.
  No values.
- **Numbers and booleans** → recorded as-is.
- **Strings** → recorded as-is up to 120 characters; longer strings are
  truncated with `...`. **This includes prompt text, schedule strings,
  and any other string node-input.** If you don't want your prompt text
  ending up in the log, leave the exec logger off.
- **Lists / dicts** → recursively summarized to depth 2 with item
  truncation (default 8 items per container, configurable via
  `COMFYUI_EXEC_LOG_SHAPE_LIMIT`).

This shape summary is what makes the log useful for debugging (you can
see which nodes ran with what tensor shapes) but it does mean
prompt-shaped strings up to 120 chars survive. Treat the exec log as
having the same sensitivity as your prompt history.

### What DOES NOT get recorded

- Tensor values (only shapes/dtypes/devices).
- Model weights.
- Network calls (this logger is local-file-only).
- Any attempt to "phone home" or report telemetry off-machine.

## Retention and cleanup

**Neither logger auto-deletes its output.** Files accumulate at the active
sink until you manually delete them.

`start_experiment.sh` auto-cleans prior `trace.json` profiler files (~1.8GB
each) at startup; sage/exec JSONL are NOT touched.

### Manual cleanup

```bash
# wipe all per-run dirs (RUN_ID layout)
rm -rf data/runs/

# legacy fallback layout (when RUN_ID unset)
rm -rf internal/analysis/runs/sage/
rm -rf internal/analysis/runs/exec_log/

# selective
rm data/runs/2026-05-04_*/sage.jsonl
```

### Where the files actually live

Both default sinks are **gitignored**:
- `data/` is gitignored via the `data/*` rule with `.gitkeep` exception.
- `internal/` is gitignored entirely.

Either way, logger output is never accidentally pushed to a public repo.

If you specify an explicit path via the env var (e.g. `=/tmp/sage.jsonl`),
the file lands wherever you point it. Whether that path is gitignored is
on you.

## Phase 0 aggregator

`scripts/sage_telemetry_summary.py` reads sage-logger JSONL (and
optionally exec-logger JSONL) and prints a per-(kernel, mask) summary.
Used to gate kernel-side optimization decisions in the upstream sage
fork. Reads only; never writes; never makes network calls.

```bash
# basic: sage trace only, no % of total available
uv run --group dev python scripts/sage_telemetry_summary.py \
    --sage-log internal/analysis/runs/sage/sage_2026-04-25_*.jsonl

# with % of total via the exec log
uv run --group dev python scripts/sage_telemetry_summary.py \
    --sage-log internal/analysis/runs/sage/sage_2026-04-25_*.jsonl \
    --exec-log internal/analysis/runs/exec_log/exec_2026-04-25_*.jsonl

# explicit total wall time
uv run --group dev python scripts/sage_telemetry_summary.py \
    --sage-log internal/analysis/runs/sage/sage_2026-04-25_*.jsonl \
    --total-wall-ms 30000

# machine-readable JSON for scripting
uv run --group dev python scripts/sage_telemetry_summary.py \
    --sage-log <path> --exec-log <path> --json
```

The aggregator does not write any new files; it only reads existing
JSONL and prints to stdout.

## Privacy guarantees, in plain language

1. **Nothing is captured by default.** Both loggers require an explicit env var to activate. Unset = zero overhead, zero output, no install.
2. **Nothing leaves your machine.** **This plugin makes no HTTP calls, no analytics, no "anonymous usage data," no network I/O of any kind in either logger or in the aggregator.** Local file writes only. (For ComfyUI's own behavior, refer to ComfyUI documentation — that's outside this plugin's scope.)
3. **`internal/` is gitignored**, so anything landing under it cannot be accidentally committed.
4. **The sage logger never sees prompt text.** It only records tensor shapes, kernel names, and timing.
5. **The exec logger CAN see short prompt strings** (≤120 chars) as part of its node-input snapshot. If your prompts are sensitive, either keep the exec logger off or redact log files before sharing them.
6. **No automatic retention policy.** Log files persist until you delete them. There is no daily/weekly/monthly cleanup.
7. **Both loggers are part of this plugin's source.** `nodes_sage.py` (sage logger) and `exec_logger.py` (exec logger monkey-patch). Audit by reading the files; both are short and stdlib-only-plus-orjson.

## Quick on/off cheat sheet

```bash
# disable both (default)
unset AUDIOLOOPHELPER_SAGE_TRACE
unset COMFYUI_EXEC_LOG

# enable both with auto-pathed timestamped files
export AUDIOLOOPHELPER_SAGE_TRACE=auto
export COMFYUI_EXEC_LOG=auto

# enable just the sage logger to a specific file
export AUDIOLOOPHELPER_SAGE_TRACE=/tmp/sage.jsonl

# disable mid-session: just unset and restart ComfyUI
unset AUDIOLOOPHELPER_SAGE_TRACE
```

## Related

- `docs/reference/sage_attention.md` — sage kernel routing, modes, fallback
- `scripts/sage_telemetry_summary.py` — Phase 0 aggregator (offline, reads only)
- `nodes_sage.py` — the `SageTracer` class (sage logger) — this plugin's code
- `exec_logger.py` — the `COMFYUI_EXEC_LOG` monkey-patch (exec logger) — this plugin's code
