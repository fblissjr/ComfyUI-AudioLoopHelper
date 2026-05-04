# AudioLoopController reference

Last updated: 2026-05-04

## Role

The pacing brain of the audio-conditioned video loop. Takes the iteration counter from `TensorLoopOpen`, the audio track, and dimension config (window/overlap/fps), and emits the timing signals every other loop-body node depends on: where this iteration's audio window starts, whether the loop should stop, what seed to use, and the *effective* overlap after integer-latent quantization. One node per workflow. The "stride from integer latents" pattern documented here is what eliminates lip-sync drift across overlap-widget changes.

## Key facts

- **Node ID**: `AudioLoopController`. Defined in `nodes.py` (~line 430).
- **One per workflow.** Receives `current_iteration` from `TensorLoopOpen`, dimension inputs from `LTXFramePlanner`, and the user's `AUDIO`. Emits eight outputs consumed across the rest of the loop body.
- **Stride is derived from integer latent counts**, not widget seconds. The `overlap_seconds` widget is a *target*; the node returns the *effective* quantized overlap as a separate output.
- **`base_seed`, not `seed`.** The widget was renamed 2026-04-26 to suppress ComfyUI's auto-attached `control_after_generate` dropdown — `seed` and `noise_seed` literal names trigger silent widget mutation across runs.

## Inputs / outputs

| Input | Type | Source |
|---|---|---|
| `current_iteration` | INT | `TensorLoopOpen.current_iteration` |
| `window_seconds` | FLOAT | `LTXFramePlanner.actual_seconds` |
| `overlap_seconds` | FLOAT | widget (target value) |
| `audio` | AUDIO | top-level audio track |
| `base_seed` | INT | widget |
| `fps` | INT | `LTXFramePlanner.fps_int` |

| Output | Type | Wires to |
|---|---|---|
| `start_index` | FLOAT | `TrimAudioDuration.start_time` |
| `should_stop` | BOOLEAN | `TensorLoopClose.stop` |
| `audio_duration` | FLOAT | downstream telemetry |
| `iteration_seed` | INT | extension's `noise_seed` (= `base_seed + current_iteration`) |
| `stride_seconds` | FLOAT | `TimestampPromptScheduleBatchEncode`, `AudioLoopPlanner` |
| `overlap_frames` | INT | extension component's `overlap_frames` |
| `overlap_latent_frames` | INT | `LatentContextExtract`, `LatentOverlapTrim` |
| `overlap_seconds` (effective) | FLOAT | subgraph `LTXVAudioVideoMask.video_start_time` |

## Stride math

The core insight: the LTX video VAE compresses every 8 pixel frames into 1 latent frame. If audio stride doesn't match what the decoder will actually emit per iteration, lip-sync drifts iteration-over-iteration. So stride is computed from integer latents, not user-typed seconds:

```
window_latents  = (window_pixel_frames - 1) // 8 + 1
overlap_latents = (overlap_pixel_frames - 1) // 8 + 1     # rounded
new_latents     = window_latents - overlap_latents
stride_pixel    = new_latents * 8
stride_seconds  = stride_pixel / fps
```

The user-typed `overlap_seconds = 2.0` at fps=25 doesn't quantize cleanly; the *effective* value emitted by the node is whatever (`new_latents * 8 / fps`) actually produces. Both values are present as outputs — the widget value never propagates downstream; the effective value does.

This is also why **`overlap_latents >= window_latents` is auto-clamped to `window_latents - 1`**: at least one new latent must be generated per iteration or the loop stops making progress.

## start_index clamping

```python
start_index = current_iteration * stride_seconds
max_start   = max(0.0, audio_duration - 0.5)
start_index = min(start_index, max_start)
```

The `0.5` floor exists because `TrimAudioDuration` feeds the mel spectrogram, which needs >1024 samples. Without the clamp, the final iteration crashes when there's <0.5s of audio left — and `TensorLoopClose` checks `should_stop` AFTER the body executes, so the body has already crashed by the time the loop would have stopped.

## Wiring

```
TensorLoopOpen.current_iteration ─┐
LTXFramePlanner.actual_seconds ───┤
                                  ├─→ AudioLoopController ─→ start_index ───→ TrimAudioDuration
overlap_seconds (widget) ─────────┤                       ─→ should_stop ───→ TensorLoopClose
top-level AUDIO ──────────────────┤                       ─→ stride_seconds → TimestampPromptScheduleBatchEncode
LTXFramePlanner.fps_int ──────────┘                                            AudioLoopPlanner
                                                          ─→ overlap_seconds → subgraph LTXVAudioVideoMask
                                                          ─→ overlap_latent_frames → LatentContextExtract / LatentOverlapTrim
                                                          ─→ iteration_seed → extension noise_seed
```

`window_seconds` and `fps` MUST come from `LTXFramePlanner`, not from independent widgets — the planner snaps to LTX-valid neighborhoods, the controller derives stride from those values; if the two diverge, the loop drifts. The auto-wiring is enforced by audit `frame_planner_present` (F8).

## Gotchas

- **Don't rename `base_seed` back to `seed` or `noise_seed`.** ComfyUI's frontend auto-attaches `control_after_generate` to those literal names, silently mutating the widget value across runs even when a wire supersedes it at execute time. Audit: `alc_seed_legacy_name` (F4). AST guard: `tests/test_node_schemas.py::test_no_seed_or_noise_seed_named_inputs`.
- **The schema rename to `base_seed` (2026-04-26) was bug-for-bug for one cycle.** Renaming `inputs[].name` without also stripping the leftover `'randomize'` widget value at `widgets_values[4]` shifted that string into the `fps` slot, causing INT-parse failure. Companion: `apply_strip_alc_control_after_generate.py`. Audit: `alc_widget_drift` (F6).
- **`stride_seconds` is an OUTPUT, not an input.** A short-lived 2026-04-26 wiring made `AudioLoopPlanner` consume `AudioLoopController.stride_seconds` as input — this closed a control-loop cycle (`TensorLoopOpen → ALC → ALPlanner → TensorLoopOpen.iterations_in`) that ComfyUI's prompt validator rejects with "Dependency cycle detected." `AudioLoopPlanner` now derives stride internally via the same `_compute_loop_geometry` helper. Audit: `planner_no_stride_input` (F7).
- **`should_stop` is checked AFTER the body, not before.** Loop body must handle the edge case where `start_index` has been clamped — the iteration runs once more even though the next iteration would overshoot.
- **Iteration count is auto-tracked.** `AudioLoopPlanner.total_iterations → TensorLoopOpen.iterations_in` is wired in every shipped workflow (audit: `iterations_autowired`, F5). For short benches, drag in an `INTConstant` and rewire — recipe in `docs/guides/debugging_guide.md`.
- **`current_iteration` is 1-based** in the controller's interface, but the wiring from `TensorLoopOpen` matches that convention. If you write a custom loop driver, mind the off-by-one.

## Audit + tests

| Check | What it catches |
|---|---|
| `alc_seed_legacy_name` (F4) | Schema reverted `seed` / `noise_seed` |
| `alc_widget_drift` (F6) | Stale `'randomize'` widget value after rename |
| `planner_no_stride_input` (F7) | Re-introduced `controller→planner` stride edge (cycle) |
| `iterations_autowired` (F5) | Iteration count not wired from planner |
| `frame_planner_present` (F8) | Dimension widgets scattered, ALC not fed from planner |

Unit tests: `tests/test_audio_loop_controller.py` — covers stride math, start_index clamping, should_stop boundary, seed pass-through, integer-latent quantization. Companion test: `tests/test_audit_frame_planner.py`.

## References

- `nodes.py` — class `AudioLoopController` (~line 430), `_compute_loop_geometry` helper (~line 65)
- `tests/test_audio_loop_controller.py` — stride + clamping + boundary tests
- `docs/reference/frame_planner_reference.md` — dimension feed (window_seconds, fps)
- `docs/reference/pipeline_flow_latent.md` — where ALC sits in the full loop
- `docs/reference/debug_tools.md` — F-pair convention and audit inventory
- `docs/reference/noise_mask_semantics.md` — `overlap_latent_frames` is the count `LatentContextExtract` strips at the boundary
- `docs/reference/_atomic_note_template.md` — this doc follows the entity-note variant
