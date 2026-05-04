Last updated: 2026-04-30

# Debug & Workflow Tooling Reference

Tooling reference for inspecting workflows, validating shipped JSONs, and
correlating runtime artifacts. **Symptom-first quality troubleshooting**
(why does the video look wrong) lives in `docs/guides/debugging_guide.md`.

---

## Inspection scripts

Read-only against any workflow JSON; none mutate state.

| Script | Purpose | Trigger |
|---|---|---|
| `scripts/audit_workflows.py [--verbose]` | Health audit across all `example_workflows/`: sage, batch-encode, sigma chain, resolution, `(L-1)%8`, preprocess, decoder, F2/F3 symmetry, plus 3 generic audit invariants (cycle / widget shape / link integrity) plus 1 AST test (cond-metadata types). Exits 1 on ERR. | After bulk edits, before commit, in CI. |
| `scripts/analyze_workflow_dag.py <wf> --format <ascii\|mermaid\|dot\|json> [--save-run]` | Topo-sorted execution order + graph rendering. `--save-run` lands the artifact under `data/runs/${RUN_ID}/dag_<slug>.<ext>` (correlates with sibling run logs) or `data/runs/dag/dag_<slug>_<ts>.<ext>` if `RUN_ID` unset. | Diagnosing execution order, post-validate cycle suspicion, comparing iteration-state across two workflows. |
| `scripts/trace_node_source.py <wf> <id> --include-inputs` | Resolve any node to AST-extracted source + wiring. Flags `object_patches`, captured tensors, bypasses, widget overrides. | Before trusting any widget annotation, before assuming bypass is inert. |
| `scripts/sage_telemetry_summary.py --sage-log <path> [--exec-log <path>]` | Outside-ComfyUI aggregator. Per-(kernel, mask) median/p90/count + Phase 0 gate verdict. Reads only; does not write. | After a traced render, when comparing kernel routing. |
| `scripts/verify_sage_iteration_trace.sh` | Diff per-iter sage kernel counts. | Suspecting per-iter kernel-routing drift. |

`audit_workflows.py` is **intentionally `WorkflowEditor`-independent** —
raw `orjson.loads` + inline link scans. Debug tools must stay usable when
the editor they audit has a bug; don't DRY against `WorkflowEditor`.

---

## Workflow validation

`audit_workflows.py` runs in CI on every push. New invariants land here
paired with their apply scripts. Two flavors:

**Named pattern checks** (one per known fix):

| Check | Pairs with apply script | What it ERRs on |
|---|---|---|
| `sage` / `sage_mode` / `sage_active` | `apply_sage_mode.py` | Missing or non-`auto_mask_aware` AudioLoopHelperSageAttention |
| `iteration_stamp` | `apply_loop_iteration_stamp.py` | Missing LoopIterationStamp |
| `preprocess_symmetry` (F2) | `apply_loop_guide_preprocess_symmetry.py` | Loop guide branch skips LTXVPreprocess |
| `loop_cropguides_symmetry` (F3) | `apply_loop_cropguides_symmetry.py` | Loop CFGGuider not via LTXVCropGuides |
| `alc_seed_legacy_name` (F4) | `apply_alc_seed_rename.py` | AudioLoopController has legacy `seed`/`noise_seed` input |
| `iterations_autowired` (F5) | `apply_iterations_autowire.py` | TensorLoopOpen.iterations_in not from AudioLoopPlanner.total_iterations |
| `alc_widget_drift` (F6) | `apply_strip_alc_control_after_generate.py` | AudioLoopController widgets_values has stale 6th `'randomize'` entry |
| `planner_no_stride_input` (F7) | `apply_planner_break_stride_cycle.py` | AudioLoopPlanner has legacy `stride_seconds` input (closes a cycle) |
| `dead_lora_loader_scaffolding_absent` (F11) | `apply_strip_dead_lora_loaders.py` | Bypassed `#1625/#1626/#1627` LoRA scaffolding nodes (inert UI clutter) still in canonical |
| `iclora_video_reference_guide_in_loop_with_cropguides` (F12a) | `apply_iclora_video_reference.py` | In-loop `LTXAddVideoICLoRAGuide` CONDITIONING outputs feed `CFGGuider` directly (must pass through `LTXVCropGuides[NoLatent]`) |
| `iclora_loader_present_when_guide_present` (F12b) | `apply_iclora_video_reference.py` | Subgraph has IC-LoRA guide but top-level has no `LTXICLoRALoaderModelOnly` |
| `iclora_ref_video_preprocess_symmetry` (F12c) | `apply_iclora_video_reference.py` | IC-LoRA guide present but no `LTXVPreprocess(val=18)` on the ref-video chain |
| `model_sampling_shift` (F13) | `apply_strip_sd3_shift_node.py` | `ModelSamplingSD3` present and active on a distilled workflow (Lightricks's distilled inference applies no shift; the SD3 node distorts the sigma-to-timestep mapping). WARN-level. |

**Generic structural invariants** (catch CLASSES of drift without per-bug rules):

| Check | Catches |
|---|---|
| `graph_acyclic` | Top-level dependency cycles. ComfyUI rejects with "Dependency cycle detected" before any node executes. |
| `widget_shape` | Stray `randomize`/`fixed`/`increment`/`decrement` strings in `widgets_values` of nodes that don't legitimately have a `control_after_generate` dropdown. Catches partial schema migrations. |
| `link_integrity` | Top-level link record vs node-level link references desync (slot out of range, source's `outputs[].links` doesn't list the link id, target's `inputs[].link != id`). Plus subgraph `linkIds` references to non-existent links. |
| (no audit; AST test) `tests/test_node_schemas.py::test_keyframe_idxs_cleared_to_none_not_empty_list` | `conditioning_set_values({"keyframe_idxs": []})` literal-list assignments. KJNodes' OuterSampleCallbackWrapper crashes on empty-list keyframe_idxs. |

**Bake new topology constraints into `audit_workflows.py`.** Every fix
that ships an apply script should ship a matching audit check (ERR with a
`Run scripts/apply_X.py` remediation pointer). Prevents silent regression
of fixes a sibling branch might revert.

---

## Apply scripts

Workflow migrations live in `scripts/apply_*.py`. Each script:

- **Default**: mutates `example_workflows/audio-loop-music-video_latent.json`
  in place (accepts an optional path arg).
- **Idempotent**: `md5sum` before + after re-run must match. Guard with
  `if _is_already_built(wf): return` to avoid burning `last_node_id` on
  strip-then-readd.
- **Has `--revert`** that restores the pre-fix shape.
- **Has `--dry-run`** that reports what WOULD change without writing.
  Pair with `audit_workflows.py` to verify a hypothetical state (HyDE).

**Three-tier staging:**
1. `internal/scratch/` — exploratory, gitignored.
2. `example_workflows/experimental/` — cross-machine reviewable; opt-in
   to audit via `EXPERIMENTAL_AUDITED_FILES` allowlist in
   `audit_workflows.py`.
3. `example_workflows/` — production, "ships AND stabilizes" per
   `internal/PLAN.md` (private clone only).

POCs that intentionally break a production invariant (e.g. F3 asymmetry)
ship a paired audit check that dispatches on a node-title prefix and ERRs
only if the rewire is damaged. Canonical TTC1 pair:
`apply_ttc_init_guide_amplification_poc.py` + `ttc1_init_guide_amplification`.

**Scratch-build apply scripts** use `WorkflowEditor.from_scratch(output_path)`
+ `add_top_level_node` + `add_link` — returns an empty-skeleton editor
with fresh uuid + reset `last_node_id` / `last_link_id`. Canonical:
`scripts/apply_spectrogram_iclora_minimal.py`.

**Shared apply-script helpers** live in `scripts/_apply_helpers.py`
(`add_link`, `find_node`, `remove_node_and_links`, `find_link_to_slot`,
`next_id`). Import with aliases to preserve call-site names; don't
re-define inline.

**Sweep orphan virtual GetNodes** after fork-and-strip. A GetNode whose
`widgets_values[0]` matches no live SetNode is orphaned; ComfyUI tolerates
it at runtime but it clutters the graph. Add the ID to `STRIP_IDS`.
Detect via:

```python
[n["id"] for n in wf["nodes"] if n["type"]=="GetNode"
 and not (n.get("outputs",[{}])[0].get("links") or [])]
```

Templates: `scripts/templates/apply_script_all_workflows.py` (in-place
edits) and `scripts/templates/apply_script_staged_variant.py`
(experimental staging). Both include the canonical `--revert`,
`--dry-run`, idempotence, and `require_nodes` guards.

---

## Runtime telemetry

| Source | Default path | RUN_ID path | What it captures |
|---|---|---|---|
| Exec log (`COMFYUI_EXEC_LOG=...` env var enables) | `internal/analysis/runs/exec_log/exec_<ts>.jsonl` | `data/runs/${RUN_ID}/exec.jsonl` | Per-node start/end with class_type, inputs, duration |
| Sage trace (`AUDIOLOOPHELPER_SAGE_TRACE=auto`) | `internal/analysis/runs/sage/sage_<ts>.jsonl` | `data/runs/${RUN_ID}/sage.jsonl` | Per-(kernel, mask) timing for sage attention |
| Profiler | `internal/analysis/runs/profiler/<ts>/` | `data/runs/${RUN_ID}/profiler/` | torch profiler trace.json + summary.txt + memory_timeline.html |
| ComfyUI stdout | `<comfyui>/user/comfyui_8188.log` (and `.prev.log`, `.prev2.log`) | (same) | Validation errors, prompt accepted/rejected, exception tracebacks |
| Output mp4 | (raw output dir) | `data/runs/${RUN_ID}/output.mp4` (symlink) | The rendered video |

`AUDIOLOOPHELPER_SAGE_TRACE=auto` is default in `start_experiment.sh`
at this repo's root. Plain `<comfyui>/start.sh` does NOT export it. Run
via `start_experiment.sh` for traced launches.

`docs/reference/telemetry_and_tracing.md` covers what each tracer captures
(and doesn't), retention, on/off semantics, why prompt text can leak via
the exec logger but not via the sage tracer.

`docs/reference/environment.md` — env-var registry, single-helper-call-site
DRY rule.

---

## Artifact paths

**Conventions** (post-2026-04-26 RUN_ID propagation):

- New per-render artifacts: `data/runs/${RUN_ID}/<category>.<ext>`
  via `scripts/workflow_utils.py::run_artifact_path(category, ext)`.
- Multi-file artifacts: `data/runs/${RUN_ID}/<subdir>/`
  via `run_artifact_dir(subdir)`.
- Without RUN_ID: legacy fallback to `internal/analysis/runs/` for most
  loggers. `analyze_workflow_dag.py --save-run` is the exception — always
  writes under `data/` (`data/runs/${RUN_ID}/dag_<slug>.<ext>` with
  RUN_ID, `data/runs/dag/dag_<slug>_<ts>.<ext>` without).

**The single env-var read site for `RUN_ID`** is
`scripts/workflow_utils.py::_current_run_id`. Route all reads through it.

**Rendered mp4 lands at `data/runs/${RUN_ID}/output.mp4`** (symlink) once
`internal/autoresearch/harness.py::_locate_and_link_output_mp4` runs
after `poll_until_done`. Source dir comes from `COMFYUI_OUTPUT_DIR` env
var (no hardcoded paths). Filename constant lives at
`internal/autoresearch/metrics/__init__.py::OUTPUT_MP4_FILENAME` — every
video-content metric (`subject_consistency`, `av_consistency`, future
style/aesthetic/palette) imports it; never inline `"output.mp4"` in a new
metric module.

---

## Canonical first-pass

When a workflow fails to run, work in this order before going inline:

1. **Tail the ComfyUI log** for the most recent prompt:
   ```
   tail -200 <comfyui>/user/comfyui_8188.log
   ```
   Validation errors ("Dependency cycle detected", "Failed to convert ...
   to a INT value") fail before any node executes — log shows them.
   Exception tracebacks fire mid-run and point you at the offending node.

2. **Run the audit**:
   ```
   uv run --group dev python scripts/audit_workflows.py
   ```
   ERRs map 1:1 to remediation apply scripts. The 4 generic checks
   (`graph_acyclic`, `widget_shape`, `link_integrity`,
   `cond_metadata_types` AST test) catch classes of drift.

3. **Inspect execution order** if a topology change is suspected:
   ```
   uv run --group dev python scripts/analyze_workflow_dag.py \
     example_workflows/audio-loop-music-video_latent.json \
     --format ascii --save-run
   ```
   Compare against a known-good baseline diff.

4. **Trace the suspect node** for source + wiring:
   ```
   uv run --group dev python scripts/trace_node_source.py <wf> <node-id> \
     --include-inputs
   ```

5. **Cross-reference the exec log** for what actually ran most recently:
   ```
   ls -lt internal/analysis/runs/exec_log/ | head
   ```
   (Or `data/runs/${RUN_ID}/exec.jsonl` if the run had RUN_ID.)

Skipping step 1 wastes the most time — ComfyUI's log usually identifies
the failure class within the first matching line.

**Iter-over-iter drift** specifically: trace CONDITIONING paths in
parallel (initial vs loop). Asymmetries (missing `LTXVConditioning`,
`frame_rate` mismatch, CLIP in subgraph) are load-bearing bugs.
