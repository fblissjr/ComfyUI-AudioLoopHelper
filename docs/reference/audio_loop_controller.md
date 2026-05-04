# AudioLoopController reference

Last updated: 2026-05-04

## Role

`AudioLoopController` (`nodes.py::AudioLoopController`). Singleton per workflow. Inputs: iteration counter (from `TensorLoopOpen`), audio track, dimension config (from `LTXFramePlanner`). Outputs: 8 timing/control signals consumed across the loop body. Implements integer-latent stride quantization to eliminate lip-sync drift across overlap-widget changes.

## Disambiguation

- `stride_seconds` is an **OUTPUT**, not an input. The short-lived 2026-04-26 controller→planner edge created a dependency cycle. Audit `planner_no_stride_input` (F7) prevents reintroduction.
- `overlap_seconds` widget value (target) ≠ `overlap_seconds` output (effective, after integer-latent quantization). Both exist; only the effective output propagates downstream.
- `current_iteration` is **1-based** in this node's interface; matches `TensorLoopOpen` convention.
- `base_seed` (widget name) ≠ `seed` / `noise_seed` (which trigger ComfyUI's `control_after_generate` mutation trap; never use those literal names).

## Inputs / outputs

| Input | Type | Source |
|---|---|---|
| `current_iteration` | INT (iteration count, 1-based) | `TensorLoopOpen.current_iteration` |
| `window_seconds` | FLOAT (seconds) | `LTXFramePlanner.actual_seconds` |
| `overlap_seconds` | FLOAT (seconds, target) | widget |
| `audio` | AUDIO | top-level audio track |
| `base_seed` | INT | widget |
| `fps` | INT (frames/sec) | `LTXFramePlanner.fps_int` |

| Output | Type | Wires to |
|---|---|---|
| `start_index` | FLOAT (seconds) | `TrimAudioDuration.start_time` |
| `should_stop` | BOOLEAN | `TensorLoopClose.stop` |
| `audio_duration` | FLOAT (seconds) | downstream telemetry |
| `iteration_seed` | INT (= `base_seed + current_iteration`) | extension's `noise_seed` |
| `stride_seconds` | FLOAT (seconds, effective) | `TimestampPromptScheduleBatchEncode`, `AudioLoopPlanner` |
| `overlap_frames` | INT (pixel frames, effective) | extension's `overlap_frames` |
| `overlap_latent_frames` | INT (latent frames, 8:1 of pixel) | `LatentContextExtract`, `LatentOverlapTrim` |
| `overlap_seconds` | FLOAT (seconds, effective post-quantization) | subgraph `LTXVAudioVideoMask.video_start_time` |

## Stride math

```
window_latents  = (window_pixel_frames - 1) // 8 + 1
overlap_latents = (overlap_pixel_frames - 1) // 8 + 1     # rounded down
new_latents     = window_latents - overlap_latents
stride_pixel    = new_latents * 8
stride_seconds  = stride_pixel / fps
```

Why: LTX video VAE compresses 8 pixel frames → 1 latent frame. If audio stride doesn't match what the decoder emits per iteration, lip-sync drifts iter-over-iter. Integer-latent quantization eliminates the drift. Implementation: `nodes.py::_compute_loop_geometry`.

`overlap_latents >= window_latents` is auto-clamped to `window_latents - 1` — at least one new latent must be generated per iteration.

## Wiring

```
TensorLoopOpen.current_iteration ─┐
LTXFramePlanner.actual_seconds ───┤
overlap_seconds (widget) ─────────┤─→ AudioLoopController ─→ start_index ───→ TrimAudioDuration
top-level AUDIO ──────────────────┤                       ─→ should_stop ───→ TensorLoopClose
LTXFramePlanner.fps_int ──────────┘                       ─→ stride_seconds → TimestampPromptScheduleBatchEncode, AudioLoopPlanner
                                                          ─→ overlap_seconds → subgraph LTXVAudioVideoMask
                                                          ─→ overlap_latent_frames → LatentContextExtract, LatentOverlapTrim
                                                          ─→ iteration_seed → extension noise_seed
```

`window_seconds` and `fps` MUST come from `LTXFramePlanner`, not independent widgets — planner snaps to LTX-valid neighborhoods, controller derives stride from those values. Divergence = drift. Enforced by audit `frame_planner_present` (F8).

## Widget-order spec

`widgets_values[]` array layout (positional; ComfyUI backend pops by index):

| Index | Field | Type |
|---|---|---|
| 0 | `current_iteration` | INT |
| 1 | `window_seconds` | FLOAT |
| 2 | `overlap_seconds` | FLOAT |
| 3 | `base_seed` | INT |
| 4 | `fps` | INT |

Length = 5. **Any extra value at index 4 (e.g. leftover `'randomize'` from pre-rename `seed` widget's `control_after_generate` dropdown) shifts into the `fps` slot → INT-parse failure.** Companion: `scripts/apply_strip_alc_control_after_generate.py`. Audit: `alc_widget_drift` (F6).

## Bypass behavior

`mode=4` (bypassed) passes inputs to outputs of same TYPE only. With 8 typed outputs (FLOAT, BOOLEAN, FLOAT, INT, FLOAT, INT, INT, FLOAT) and 6 inputs (INT, FLOAT, FLOAT, AUDIO, INT, INT), most bypass paths dead-end silently. **Do not bypass `AudioLoopController` in production workflows** — the loop body has no recovery path.

## Failure modes

| Symptom | Likely cause |
|---|---|
| `Failed to convert <value> to INT` at "got prompt" | Stale `'randomize'` in `widgets_values[4]` (F6 — `alc_widget_drift`) |
| Workflow validator rejects with "Dependency cycle detected" | Reintroduced `controller→planner stride_seconds` edge (F7) |
| `seed` widget mutates across runs despite wire | Schema reverted to `seed`/`noise_seed` literal name (F4 — `alc_seed_legacy_name`) |
| Loop runs wrong iteration count | `AudioLoopPlanner.total_iterations` not wired to `TensorLoopOpen.iterations_in` (F5 — `iterations_autowired`) |
| Lip-sync drifts across overlap-widget changes | Dimension widgets scattered, not fed from `LTXFramePlanner` (F8) |
| Mel spectrogram crash on final iteration | `start_index` clamping disabled or audio < 0.5s remaining; loop body runs once after `should_stop` would fire (`TensorLoopClose` checks AFTER body) |

Edge case: `start_index` clamps to `max(0.0, audio_duration - 0.5)` to keep ≥0.5s of audio for the mel spectrogram (>1024 sample requirement). Implementation: `nodes.py::AudioLoopController.execute`.

## Audit + tests

| Audit ID | Catches |
|---|---|
| `alc_seed_legacy_name` (F4) | Schema reverted `seed` / `noise_seed` |
| `alc_widget_drift` (F6) | Stale `'randomize'` widget value after rename |
| `iterations_autowired` (F5) | Iteration count not wired from planner |
| `planner_no_stride_input` (F7) | Reintroduced controller→planner stride edge |
| `frame_planner_present` (F8) | Dimension widgets scattered |

Tests: `tests/test_audio_loop_controller.py` (stride math, clamping, boundary), `tests/test_audit_frame_planner.py` (audit-check behavior).

## References

- `nodes.py::AudioLoopController`, `nodes.py::_compute_loop_geometry` — implementation
- `tests/test_audio_loop_controller.py` — stride + clamping + boundary tests
- `docs/reference/frame_planner_reference.md` — dimension feed (window_seconds, fps source)
- `docs/reference/pipeline_flow_latent.md` — where ALC sits in the full loop
- `docs/reference/debug_tools.md` — F-pair convention and audit inventory
- `docs/reference/noise_mask_semantics.md` — `overlap_latent_frames` is the count consumed at boundaries
- `docs/reference/_atomic_note_template.md` — entity-note variant template
