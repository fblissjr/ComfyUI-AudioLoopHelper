---
name: sage-trace-summary
description: Aggregate the most recent sage trace (and optional matching exec log) into a per-prompt + attribution + gate-verdict report. Use after a ComfyUI generation when you want to know "where did time actually go" and "is mask kernel work justified."
---

Wraps the post-gen sage telemetry workflow that emerged from the 2026-04-25 session: filename pairing between sage logs and exec logs is unreliable because writers rotate on different triggers. This skill defaults to the self-contained `--use-sage-span` denominator and only attempts exec-log pairing when ts ranges overlap.

## When to invoke

After a ComfyUI generation where `AUDIOLOOPHELPER_SAGE_TRACE=auto` was set. Typical flow:
1. ComfyUI starts (sage tracer creates a new `internal/analysis/runs/sage/sage_*.jsonl`).
2. User runs one or more workflows.
3. Invoke this skill — it picks up the latest sage log and reports.

## Steps

1. Find the most recent sage log:

```bash
cd "$(git rev-parse --show-toplevel)"
SAGE_LOG=$(ls -t internal/analysis/runs/sage/sage_*.jsonl 2>/dev/null | head -1)
echo "sage log: $SAGE_LOG"
```

2. **Run with `--use-sage-span` first** (self-contained denominator; honest per-trace fraction):

```bash
uv run --group dev python scripts/sage_telemetry_summary.py \
  --sage-log "$SAGE_LOG" --use-sage-span
```

3. Inspect the output's `attribution:` line:
   - `N sage_telemetry / 0 mirror_inferred / 0 unknown` — fresh trace post-sage-fork-upgrade. Trust the kernel names.
   - `0 sage_telemetry / N mirror_inferred / 0 unknown` — pre-upgrade trace. Mirror resolved auto → fp8_cuda++ via routing table; results trustworthy on sm89/CUDA12.8.
   - Mixed or `unknown > 0` — partial telemetry; check the trace's `arch` header.

4. **Optional: pair with an exec log only if ts ranges overlap.** Filename timestamps tell you when the writer opened a handle, not what window the file covers. Verify before pairing:

```bash
# First per-call ts in sage log:
SAGE_FIRST=$(uv run --group dev python -c "import orjson; print(next(orjson.loads(l)['ts'] for l in open('$SAGE_LOG') if orjson.loads(l).get('elapsed_us') is not None))")

# For each candidate exec log, print (ts_min, ts_max):
for f in internal/analysis/runs/exec_log/exec_*.jsonl; do
  uv run --group dev python -c "
import orjson, sys
ts = [orjson.loads(l)['ts'] for l in open('$f') if 'ts' in orjson.loads(l)]
print(f'$f: {min(ts):.1f} -> {max(ts):.1f}')
"
done
```

   Pick the exec log whose range contains `SAGE_FIRST`. If multiple do, pick the smallest range.

5. **Run with the verified pair** (per-prompt mode auto-engages when exec log spans >1 prompt_id):

```bash
uv run --group dev python scripts/sage_telemetry_summary.py \
  --sage-log "$SAGE_LOG" --exec-log "$VERIFIED_EXEC_LOG"
```

6. Read the gate verdict:
   - `<5%` masked-triton: mask kernel work in sage-fork closes permanently
   - `5-15%`: revisit in 6 months
   - `>15%`: justified
   - `skipped`: no masked calls in this trace (workload didn't hit cross-attn) OR no `pct_of_total` available

## Common output shapes

**Fresh trace, single prompt:**
```
attribution: 14523 sage_telemetry / 0 mirror_inferred / 0 unknown
fp16_triton    True    1247   ...    %_of_total=8.4%
fp8_cuda++     False   13276  ...    %_of_total=23.1%
gate verdict: 5-15% (8.4%) -- revisit in 6 months
```

**Stale trace, mirror-resolved:**
```
note: --arch not given; trace has no 'arch' field; using local autodetect: sm89_cuda12_8
attribution: 0 sage_telemetry / 26880 mirror_inferred / 0 unknown
fp8_cuda++     False   26880  ...
gate verdict: skipped (no pct_of_total available)
```

**Multi-prompt exec log:**
```
exec log has 5 prompt_ids; reporting per-prompt.

prompt_id=A1B2C3...  (5436 calls, 442.4s wall)
  ...
prompt_id=unknown  (87 calls, wall=unknown)   <- DATA QUALITY ALARM
  ...
```
The `unknown` bucket means rows whose ts fell outside every prompt window. Check if the exec log was started after the sage tracer.

## Reference

- Implementation: `scripts/sage_telemetry_summary.py`
- Cross-repo design: `internal/log/log_2026-04-25.md` Addendums 1-3
- Original gate criteria: `internal/design/sage_backlog.md`
