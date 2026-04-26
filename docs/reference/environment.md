Last updated: 2026-04-27

# Environment variables reference

Every env var the codebase reads, where it's read, default behavior, and
who sets it. **DRY rule**: each var is read at exactly one helper call
site. If you find a duplicate `os.environ.get(...)` for the same key in
two different files, that's a bug — route both through the existing
helper.

## Single-line summary

| Var | Default behavior | Set by | Read by |
|---|---|---|---|
| `RUN_ID` | unset → loggers use legacy timestamped paths | `start_experiment.sh` (this repo, root) | `scripts/workflow_utils.py::_current_run_id` |
| `COMFYUI_EXEC_LOG` | unset → exec logger inactive | `start_experiment.sh` | `exec_logger.py::_resolve_log_target` |
| `COMFYUI_EXEC_LOG_SHAPE_LIMIT` | unset → 8 items per list/dict shape snapshot | manual | `exec_logger.py::_shape_of` |
| `AUDIOLOOPHELPER_SAGE_TRACE` | unset → sage tracer inactive | `start_experiment.sh` | `nodes_sage.py::resolve_trace_path` |
| `COMFYUI_API_URL` | unset → `http://<local_comfyui_host>:<port>` (ComfyUI's loopback default, e.g. `localhost` on port `8188` for a stock install) | manual / Phase-2 harness | `internal/autoresearch/harness.py::_api_url` |
| `COMFYUI_OUTPUT_DIR` | unset → harness skips mp4 symlinking; video-content metrics report `*_status: video_missing` | manual (point at ComfyUI's VHS output dir) | `internal/autoresearch/harness.py::_locate_and_link_output_mp4` |

`<comfyui>` = your ComfyUI install root (the directory that contains
`main.py` and `custom_nodes/`).

## Detailed contracts

### `RUN_ID`

**Format**: `${ISO8601_UTC}_${rand4}` (e.g. `20260426T134522Z_a3f1`).
Lexicographically sortable, collision-resistant, readable in `ls`.

**Read by**: `scripts/workflow_utils.py:59 _current_run_id()` — the
single source of truth.

**Used by**:
- `scripts/workflow_utils.py::run_artifact_path(category, ext)` →
  `data/runs/${RUN_ID}/<category>.<ext>` when set; legacy timestamped
  path otherwise.
- `scripts/workflow_utils.py::run_artifact_dir(subdir="")` →
  `data/runs/${RUN_ID}/<subdir>/` when set; legacy
  `internal/analysis/runs/<subdir>/<TS>/` otherwise.
- `exec_logger.py::_resolve_log_target` calls `run_artifact_path` for
  the auto path.
- `nodes_sage.py::resolve_trace_path` same.
- `nodes.py::ProfileBegin.execute` calls `_current_run_id` to branch
  between RUN_ID-correlated path and legacy widget-driven path.

**Set by**: `start_experiment.sh` at this repo's root. Auto-generates
when unset; preserved when caller supplies a value (`RUN_ID=mytest
./start_experiment.sh`).

**To opt out for one launch**: `RUN_ID= ./start_experiment.sh`. Empty
string is treated identically to unset (`_current_run_id` returns
`None`).

**Why**: three telemetry systems (exec_log, sage tracer, profiler)
each stamped `time.time()` at startup, so files from one render drifted
apart by seconds and looked unrelated by name. Single shared key fixes
correlation. Diagnosed 2026-04-26 in
`internal/analysis/id_lora_ablation_and_seed_widget_audit.md`.

### `COMFYUI_EXEC_LOG`

**Values**: `auto` / `1` / `true` / `yes` (auto-generate path), `stderr`
(write to stderr), or any other string treated as an explicit file
path. Unset/empty → logger does not install.

**Read by**: `exec_logger.py:172 _resolve_log_target(value)` for the
value resolution. `exec_logger.py:280` reads it again at module-import
time as a presence-check guard so the import is a no-op when the var is
unset (zero overhead). Both reads are intentional; the import-time
check decides whether to install the patch, the install-time read
decides where to write.

**Set by**: `start_experiment.sh` (this repo, root) exports
`COMFYUI_EXEC_LOG=auto` when unset. As of 2026-04-26, plain
`<comfyui>/start.sh` does NOT export this — that script is back to
being a vanilla ComfyUI launcher with no plugin-specific telemetry.
Run via `start_experiment.sh` to get the auto-trace behavior.

**Auto path**: routes through `run_artifact_path("exec", "jsonl")` →
`data/runs/${RUN_ID}/exec.jsonl` when `RUN_ID` is set, else
`internal/analysis/runs/exec/exec_<TS>.jsonl`.

### `COMFYUI_EXEC_LOG_SHAPE_LIMIT`

**Values**: integer string. Default 8 if unset or unparseable.

**Read by**: `exec_logger.py::_shape_of` (lines 114, 120) — controls
the max list/dict items captured in input/output shape snapshots.

**Why two read sites in one function**: both inside `_shape_of` for
different recursion contexts (list vs dict). Same value, just convenient
to read at point-of-use rather than threading through. Negligible.

**Set by**: manually only. No default export.

### `AUDIOLOOPHELPER_SAGE_TRACE`

**Values**: `auto` / `1` / `true` / `yes` (auto-generate path), or any
other string treated as an explicit file path. Unset/empty → tracer
does not install.

**Read by**: `nodes_sage.py:213 resolve_trace_path()`.

**Set by**: `start_experiment.sh` (this repo, root) exports
`AUDIOLOOPHELPER_SAGE_TRACE=auto` when unset. Plain
`<comfyui>/start.sh` does NOT — same minimization rationale as
`COMFYUI_EXEC_LOG` above.

**Auto path**: routes through `run_artifact_path("sage", "jsonl")` →
`data/runs/${RUN_ID}/sage.jsonl` when `RUN_ID` is set, else
`internal/analysis/runs/sage/sage_<TS>.jsonl`.

### `COMFYUI_API_URL` (planned, Phase 2)

**Values**: full URL string. Default `http://<local_comfyui_host>:<port>` (ComfyUI's loopback default, e.g. `localhost` on port `8188` for a stock install) when
unset.

**Will be read by**: the experiment harness (`internal/autoresearch/harness.py`)
when it POSTs to ComfyUI's `/prompt` and polls `/history`.

**Why an env var instead of a hardcoded address**: easy to override per
invocation without editing files (`COMFYUI_API_URL=http://other-box:8188
./run.sh`); keeps any specific server address out of committed files.

**Sensitive endpoints / non-default ports**: keep the URL out of git.
For a single local box, just rely on the default. For a remote or
shared box, export the var in your shell profile (gitignored) or in a
gitignored `data/experiment.local.toml` that the harness reads.

### `COMFYUI_OUTPUT_DIR` (Phase 2.1)

**Values**: absolute path to ComfyUI's VHS_VideoCombine output
directory (where `LTX-2_${RUN_ID}_*.mp4` lands after a render).

**Read by**:
`internal/autoresearch/harness.py::_locate_and_link_output_mp4` once
per run, after `poll_until_done` returns. Globs the matching mp4 and
symlinks it into `data/runs/${RUN_ID}/output.mp4`.

**Soft failure**: when unset or the directory doesn't exist, the
harness skips the symlink step. Video-content metrics
(`subject_consistency`, eventually `style_consistency`,
`lip_sync`, `aesthetic`, `palette`) report
`*_status: video_missing` and the tracker row still records a
non-video baseline (wall-time + sage stats remain valid).

**Why a separate env var instead of inferring from ComfyUI config**:
ComfyUI's output dir is per-install + per-launch (`--output-directory`
override), not discoverable from outside. Asking the user to point at
it once is cheaper than spelunking into a running server.

**Privacy**: this var typically points outside the repo (ComfyUI
installs are usually elsewhere), so don't commit a default value into
the repo or shell scripts that ship in the repo. Set it once in your
gitignored shell profile or pass per-launch.

## Where exports live

| File | Exports | Notes |
|---|---|---|
| `start_experiment.sh` (this repo, root) | `RUN_ID`, `AUDIOLOOPHELPER_SAGE_TRACE`, `COMFYUI_EXEC_LOG` | Telemetry-enabled wrapper. Generates `RUN_ID`, sets the two tracer vars to `auto` when unset, then execs `<comfyui>/start.sh`. **Canonical entry point for experiment-mode launches.** |
| `<comfyui>/start.sh` | (nothing AudioLoopHelper-specific) | Vanilla ComfyUI launcher. As of 2026-04-26 carries no plugin-specific telemetry knowledge; use `start_experiment.sh` to layer that on top. |

## Adding a new env var

1. Pick one helper module that reads it (probably
   `scripts/workflow_utils.py` for cross-cutting state, or the module
   that owns the feature for narrow state).
2. Add a small `_current_<name>()` reader function with the type
   contract documented in its docstring (None for unset, or the parsed
   value).
3. **Do not** sprinkle `os.environ.get("MY_VAR", ...)` calls across the
   codebase. Every other call site goes through the helper.
4. Add a row to the table at the top of this doc with default behavior
   + who reads + who sets.
5. If it's an experiment-runner concern, document the override pattern
   in `internal/autoresearch/program.md` too.

## Audit command

```
grep -rn 'os\.environ\|os\.getenv' --include='*.py' . \
    | grep -v __pycache__ | grep -v coderef/ | grep -v '\.venv'
```

Each unique env-var name in the output should appear at exactly one
helper call site (the reads in this doc) plus optionally one
import-time presence-check guard (the `exec_logger.py:280` pattern).
Anything beyond that is a DRY violation — fix by routing through an
existing helper or factoring out a new one.
