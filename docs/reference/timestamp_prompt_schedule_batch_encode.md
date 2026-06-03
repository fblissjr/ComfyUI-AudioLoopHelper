# TimestampPromptScheduleBatchEncode reference

Last updated: 2026-06-03

## Role

`nodes.py::TimestampPromptScheduleBatchEncode`. Pre-encodes the entire prompt schedule via CLIP **once**, **outside** the loop body. Emits a list of pre-encoded CONDITIONING (one per expected iteration) for `ConditioningSelectByIteration` to pluck inside the loop. Eliminates the CLIP-in-loop offload thrash that silently disengaged NAG iter 2+ in the legacy `TimestampPromptSchedule` workflow. Stamps `frame_rate` metadata on every emitted CONDITIONING — required to prevent identity drift iter-over-iter.

## Disambiguation

- ≠ **`TimestampPromptSchedule`** (legacy) — that one returns a string per iter; CLIP encodes inside the loop body. Causes the offload thrash this node fixes. **Don't wire in loop body.**
- ≠ **`KeyframeLatentScheduleBatchEncode`** — same architectural pattern (batch-encode-once-outside-loop) but for IMAGES via VAE rather than TEXT via CLIP.
- ≠ **`CachedTextEncode`** — earlier mitigation that cached text encodings but still kept CLIP in the loop. Replaced by this node.
- Output `conditioning_list` is a Python `list[CONDITIONING]`, not a single CONDITIONING. Pluck via `ConditioningSelectByIteration`.

## Key facts

- Source: `nodes.py::TimestampPromptScheduleBatchEncode`. Defined alongside the legacy `TimestampPromptSchedule`.
- Singleton per workflow; lives top-level (not in subgraph). Loop body has only `ConditioningSelectByIteration`.
- Pairs with: `nodes.py::ConditioningSelectByIteration`.
- Caching: module-level `_BATCH_ENCODE_CACHE` is an `OrderedDict` LRU of size 4 (covers A/B runs). Keyed on `(id(clip), schedule, stride_seconds, audio_duration, snap_boundaries, frame_rate)`. Dies on ComfyUI restart.
- Stamps `frame_rate` on every emitted CONDITIONING via `node_helpers.conditioning_set_values`. Canonical LTX 2.3 inference value is 25.0 (`docs/reference/ltx23_model_reference.md` § "`frame_rate`: canonical inference value is 25"); must match the `frame_rate` set on the initial-render's `LTXVConditioning`. A 2026-05-15 sweep flipped to 24.0; reverted 2026-05-16. `fps=25` is live in all shipped workflows.
- Iteration count output includes **+1 headroom** beyond expected loop length so the selector's clamp absorbs overshoot.
- Dedup: identical prompt strings encoded once regardless of how many iterations span them.

## Inputs / outputs

| Input | Type | Source |
|---|---|---|
| `clip` | CLIP | top-level CLIPLoader (Gemma 3 for LTX 2.3) |
| `schedule` | STRING (multiline) | widget |
| `stride_seconds` | FLOAT (seconds) | `AudioLoopController.stride_seconds` |
| `audio_duration` | FLOAT (seconds) | `AudioLoopController.audio_duration` |
| `snap_boundaries` | BOOLEAN | widget (default True) |
| `frame_rate` | FLOAT (Hz) | `LTXFramePlanner.fps_float` (canonical inference value: 25.0) |

| Output | Type | Wires to |
|---|---|---|
| `conditioning_list` | LIST[CONDITIONING] | `ConditioningSelectByIteration.conditioning_list` |
| `iteration_count` | INT | informational; loop count comes from `AudioLoopPlanner` |

## Wiring

```
LTXFramePlanner.fps_float ───────────┐
AudioLoopController.audio_duration ──┤
AudioLoopController.stride_seconds ──┼─→ TimestampPromptScheduleBatchEncode (top-level)
top-level CLIPLoader ────────────────┤             │
schedule (widget) ───────────────────┘             ▼
                                          conditioning_list
                                                   │
                                                   ▼
                                  ConditioningSelectByIteration (loop body)
                                                   │
                                                   ▼
                                              CFGGuider.positive
```

`clip` MUST come from a top-level `CLIPLoader`, never from inside the loop body. Wiring CLIP inside the loop reintroduces the offload thrash this node exists to prevent.

## Why outside the loop body

ComfyUI ModelPatcher's `object_patches` closures are never device-migrated on offload. NAG captures `nag_cond_video` tensor in such a closure. When CLIP loads inside the loop body, it triggers DiT eviction, which silently invalidates the captured tensor. Iter 2+: NAG calls execute against stale captures — looks correct but doesn't actually apply NAG. Symptom: microphones / anatomy regressions / style drift returning after iter 1.

Mechanism walkthrough: `docs/analysis/nag_object_patches_offload_asymmetry.md`.

## Failure modes

| Symptom | Likely cause |
|---|---|
| Microphones / hallucinated objects appear iter 2+ | CLIP back inside loop body (legacy `TimestampPromptSchedule` wired); NAG silently disengaged |
| Identity drift escalates iter-over-iter | `frame_rate` not stamped on emitted CONDITIONING — new loop-body conditioning-producer doesn't call `node_helpers.conditioning_set_values({"frame_rate": ...})` |
| Same prompt re-encoded per iter despite this node | `_BATCH_ENCODE_CACHE` invalidation — `id(clip)` changed (CLIP reloaded), or schedule/stride/duration changed |
| Test ghost-hits with FakeCLIP | `id()`-keyed cache without autouse clear-fixture; FakeCLIP GC + Python address recycling produces phantom hits |
| `frame_rate` mismatch between initial render and loop iters | Initial `LTXVConditioning.frame_rate` and this node's `frame_rate` widget diverged — must be identical |

Edge cases:
- Schedule format identical to `TimestampPromptSchedule` — same parser, same boundary semantics under `snap_boundaries`.
- `+1 iteration headroom` is deliberate — encodes one extra entry in case loop overshoots; selector clamps.
- Production cache keys include `type(clip).__name__` as cheap cross-class insurance against `id()` collisions.

## Audit + tests

| Audit / test | What it catches |
|---|---|
| `prompt_schedule` (predates F-numbering) | Workflow lacks a batch-encode prompt schedule entirely |
| `tests/test_batch_encode.py::TestBatchEncoderCaching` | REPEATED-call test confirms caching works across simulated iterations |
| `tests/test_node_schemas.py` | AST-walks `node_helpers.conditioning_set_values` calls; ensures `frame_rate` key is set |

The "any new CONDITIONING-producing loop-body node must stamp `frame_rate`" rule is enforced by code review + the test_node_schemas AST scan.

## References

- `nodes.py::TimestampPromptScheduleBatchEncode` — implementation
- `nodes.py::ConditioningSelectByIteration` — paired selector
- `nodes.py::TimestampPromptSchedule` — legacy variant; don't wire in loop body
- `docs/analysis/nag_object_patches_offload_asymmetry.md` — root-cause walkthrough for "CLIP can't enter the loop body"
- `tests/test_batch_encode.py::TestBatchEncoderCaching` — canonical repeated-call test
- `docs/reference/audio_loop_controller.md` — `stride_seconds` and `audio_duration` source
- `docs/reference/frame_planner_reference.md` — `fps_float` source for `frame_rate` stamping
- `docs/reference/nag_technical_reference.md` — NAG mechanism that silently disengages without this node
- `docs/reference/_atomic_note_template.md` — entity-note variant template
