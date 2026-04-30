---
name: diagnose-workflow
description: Canonical first-pass when a ComfyUI-AudioLoopHelper workflow won't run or fails validation. Use when the user mentions "workflow won't run", "dependency cycle detected", "validation failed", "Failed to convert ... to INT", "control_after_generate", or describes a recent ComfyUI render that didn't get past prompt validation. Runs the canonical diagnosis (ComfyUI log tail, audit, DAG inspection, exec log review) BEFORE any inline orjson scripting. Reference: docs/reference/debug_tools.md.
---

# Diagnose workflow failure

Canonical first-pass when a ComfyUI-AudioLoopHelper workflow fails to validate
or run. This skill exists because going inline-orjson-script before running the
existing tools wastes 80% of a session — every failure class has a tool that
identifies it within the first matching log line.

## Steps (run in order; each step takes < 30s)

### 1. Tail the ComfyUI log

Validation errors fire BEFORE any node executes. The log shows them.

```bash
# COMFYUI_ROOT is set by .claude/settings.local.json or your shell rc.
# Fallback to the typical install location if unset.  # path-privacy: ignore
COMFYUI_LOG_DIR="${COMFYUI_ROOT:-$HOME/ComfyUI}/user"  # path-privacy: ignore
ls -lt "$COMFYUI_LOG_DIR"/comfyui_*.log | head -3
tail -200 "$(ls -t "$COMFYUI_LOG_DIR"/comfyui_*.log | head -1)"
```

Look for: `got prompt`, `Failed to validate prompt`, `Dependency cycle detected`,
`Failed to convert an input value`, `Exception during processing`, `TypeError`.

The newest entries are at the bottom. Match the failure class:

| Log line | Likely fix |
|---|---|
| `Dependency cycle detected: A -> B -> C -> A` | A topology back-edge. Run audit step 2; check `graph_acyclic` ERR. |
| `Failed to convert an input value to a INT value: <name>, <stringval>` | Widget shift from a partial schema rename (control_after_generate leak). Run audit; check `widget_shape` ERR. |
| `TypeError: list indices must be integers or slices, not tuple` at `ltxv_nodes.py` | `keyframe_idxs` being `[]` instead of `None`. Run `pytest tests/test_node_schemas.py` — the AST test catches this. |
| `Exception during processing` mid-iteration | Real runtime bug. Use exec log (step 4) to find the last successful node. |
| Workflow runs to completion but output is wrong | Quality issue, not failure. Use `docs/guides/debugging_guide.md` instead. |

### 2. Run the audit

```bash
cd "$(git rev-parse --show-toplevel)"
uv run --group dev python scripts/audit_workflows.py
```

The audit covers F2–F10 topology checks plus quality-axis checks (sage,
decoder, guider, retake_*, prompt_*, etc.), plus 3 generic invariants
(`graph_acyclic`, `widget_shape`, `link_integrity`) plus 1 AST test
(`cond_metadata_types` in `tests/test_node_schemas.py`). Live inventory in
`docs/reference/debug_tools.md` — this skill intentionally doesn't enumerate
because the list grows with each fix.

Every ERR has a remediation pointer to a `scripts/apply_*.py` script.
Run the recommended migration with `--dry-run` first to see what changes:

```bash
uv run --group dev python scripts/apply_<thing>.py --dry-run
# verify
uv run --group dev python scripts/apply_<thing>.py
```

### 3. Inspect execution order

If the audit is clean but the workflow still fails, dump the topo-sorted DAG:

```bash
uv run --group dev python scripts/analyze_workflow_dag.py \
  example_workflows/audio-loop-music-video_latent.json \
  --format ascii --save-run
```

Output lands at `data/runs/dag/dag_<slug>_<ts>.txt` (or
`data/runs/${RUN_ID}/dag_<slug>.txt` if `RUN_ID` env var is set).

Compare against a known-good baseline diff. Look for: a new node in the loop
body, a wire that crosses the subgraph boundary in the wrong direction, an
unexpected execution order between two nodes.

### 4. Trace the suspect node

Once you've localized the failure to a specific node ID, get its source + wiring:

```bash
uv run --group dev python scripts/trace_node_source.py \
  example_workflows/audio-loop-music-video_latent.json <node-id> --include-inputs
```

Flags `object_patches`, captured tensors, bypasses, widget overrides. **Run
this before trusting any widget annotation in the JSON** — what's saved as
a widget value isn't always what runs (link supersedes widget).

### 5. Cross-reference the exec log

What actually ran in the most recent attempt:

```bash
ls -lt internal/analysis/runs/exec_log/ | head -5
# (or, if the run had RUN_ID:)
ls -lt data/runs/${RUN_ID}/exec.jsonl
tail -30 internal/analysis/runs/exec_log/<newest>.jsonl
```

Each line is a node start or end event. The last `start` without a matching
`end` is the node that crashed. The duration distribution tells you whether
the loop body completed at least one iteration before failing.

## When to escalate to inline scripting

Only after steps 1-5 don't localize the failure. Common reasons to drop to
inline:
- A subgraph internal cycle (the audit only checks top-level cycles by design
  — subgraph internals legitimately have feedback patterns).
- Cross-correlating sage trace + exec log + profiler (use
  `scripts/sage_telemetry_summary.py` first; only fall back to inline if it
  doesn't answer the question).
- An apply script's `--dry-run` shows unexpected changes — read the script
  source instead of inferring from output.

## Anti-patterns

- **Don't go straight to `uv run python -c "import orjson; ..."`.** Step 1
  alone catches Bug A (cycle), Bug B (widget shift), and most Bug C variants.
- **Don't trust widget values without running `trace_node_source.py`.** A
  saved widget value can be runtime-superseded by a link, or vice versa.
- **Don't manually fix workflows in the ComfyUI frontend without then re-saving
  to disk and re-running the audit.** In-frontend state can diverge from
  on-disk state silently.

## Reference

- `docs/reference/debug_tools.md` — full check inventory, tool usage, artifact
  paths.
- `docs/guides/debugging_guide.md` — symptom-first quality troubleshooting (when
  the workflow runs but the output looks wrong).
