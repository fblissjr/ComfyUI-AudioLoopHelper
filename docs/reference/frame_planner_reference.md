# LTXFramePlanner reference

Last updated: 2026-05-16

## Role

Single source of truth for LTX 2.3 dimension config in shipped workflows. The user types human-readable target values (832, 448, 19.88, 25); the node snaps them to LTX-architectural neighborhoods and emits matched values to every downstream consumer. Replaces the pre-2026-04-27 pattern of scattering width / height / length / window_seconds / fps across five separate widgets that drifted apart.

## Key facts

- **Node ID**: `LTXFramePlanner`. Defined in `nodes.py` (~line 1505).
- **Inputs**: `target_width` (default 832), `target_height` (default 448), `target_seconds` (default 19.88), `fps` (default 25; LTX 2.3 canonical inference value — see fps gotcha below).
- **Outputs**: `width`, `height`, `frames`, `actual_seconds`, `fps_int`, `fps_float`, `latent_volume`, `status`, `summary`.
- **One node per workflow.** All shipped workflows wire its outputs into 6 consumer inputs (see Wiring map).
- **`fps_int` vs `fps_float` is intentional.** `LTXVConditioning.frame_rate` is FLOAT-typed; `AudioLoopController.fps` and `AudioLoopPlanner.fps` are INT-typed. Same value, two output slots, no coercion at the wire.

## Snap rules (enforced)

| Rule | Source | Why |
|---|---|---|
| `width % 32 == 0`, `height % 32 == 0` (snap DOWN) | LTX 2.3 single-stage architecture | Patch-size-32 spatial transformer; off-grid dimensions break the conv stem |
| `(frames - 1) % 8 == 0` (snap DOWN) | Video VAE temporal compression | Encoder formula `latent = (pixel - 1) // 8 + 1`; non-conforming `length` is silently floored by `EmptyLTXVLatentVideo` (verified at `comfy_extras/nodes_lt.py:36`) |
| `actual_seconds = frames / fps` (always self-consistent) | Internal | Eliminates `length / fps != window_seconds` drift between consumer widgets |

## Latent-volume classification

`latent_volume = (width // 32) * (height // 32) * ((frames - 1) // 8 + 1)`

| Status | Range | Meaning |
|---|---|---|
| `OK` | ≤ 20,000 | Inside the artifact-free regime |
| `NEAR_EDGE` | 20,001 – 24,570 | At/near the artifact ceiling; render quality degrades |
| `OVER_EDGE` | > 24,570 | Past the ceiling; banding / artifacts likely |

Ceiling reference: `docs/reference/ltx23_model_reference.md`.

## Wiring map

`LTXFramePlanner` outputs flow into:

```
width          -> EmptyLTXVLatentVideo.width, ImageResizeKJv2.width
height         -> EmptyLTXVLatentVideo.height, ImageResizeKJv2.height
frames         -> EmptyLTXVLatentVideo.length
actual_seconds -> AudioLoopController.window_seconds,
                  AudioLoopPlanner.window_seconds,
                  subgraph video_end_time slot
fps_int        -> AudioLoopController.fps, AudioLoopPlanner.fps
fps_float      -> LTXVConditioning.frame_rate
status, summary -> PreviewAny (visibility for the operator)
```

The `apply_frame_planner_consolidation.py` migration removes four pre-existing helper nodes that the planner replaces (`FloatConstant "window_size_seconds"`, `SetNode "Set_window_size_seconds"`, `GetNode "Get_window_size_seconds"`, `PrimitiveNode "length"`). Net node count: -3 per workflow.

## Gotchas

- **`target_seconds` is per-iteration window duration, NOT total video length.** Total length is determined by the audio. Default 19.88s @ 25fps = 497 frames (`(497-1)%8==0`) ≈ 9 iterations on a 3-min song.
- **Lower `target_seconds` = more iterations = more re-anchoring.** Tradeoffs: better identity preservation, higher resolution headroom, more boundary seams. There's no universally-right value; tune per render.
- **`fps=25` is the LTX 2.3 canonical inference value.** Lightricks's shipped ComfyUI-LTXVideo example workflows set `LTXVConditioning.frame_rate=25` across T2V/I2V distilled + full; V2V uses 24 (preserves source-video fps). 8n+1 latent boundary aligns cleanly at 25. Full evidence + mechanism + symptom of mismatch: `docs/reference/ltx23_model_reference.md` § "`frame_rate`: canonical inference value is 25". A 2026-05-15 sweep flipped widgets to 24 on a misread of a library placeholder default; reverted 2026-05-16 (production render validation pending).
- **The wire supersedes the widget value at execution time** (verified at `apply_frame_planner_consolidation.py:44-48` source-level audit). Widget defaults exist only for orphan-node smoke testing; in shipped workflows they're never consulted.
- **`EmptyLTXVLatentVideo` silently floors invalid `length`** with `((L-1)//8)+1`. Without the planner upstream, the user-typed length and the actual rendered length can disagree by up to 7 frames. With the planner, they always agree.

## Migration

```bash
uv run --group dev python scripts/apply_frame_planner_consolidation.py
uv run --group dev python scripts/apply_frame_planner_consolidation.py --revert
uv run --group dev python scripts/apply_frame_planner_consolidation.py --dry-run
```

Idempotent + reversible. Operates on `example_workflows/_latent.json` by default; pass another path to migrate a different workflow. Independent of all other apply scripts (F2, F3, F4, F5, F6, F7, canonical sigmas, LoRA chain).

## Audit + tests

- **Audit check**: `frame_planner_present` in `scripts/audit_workflows.py` (~line 305). ERR on production workflows missing the node; WARN on experimental forks.
- **Unit tests**: `tests/test_frame_planner.py` covers snap math, latent-volume classification, output type contracts.
- **Audit tests**: `tests/test_audit_frame_planner.py` covers the audit-check behavior on present/missing/bypassed configurations.

## References

- `nodes.py` — class `LTXFramePlanner` (~line 1505)
- `scripts/apply_frame_planner_consolidation.py` — migration
- `scripts/audit_workflows.py` — `frame_planner_present` check (~line 299)
- `tests/test_frame_planner.py`
- `tests/test_audit_frame_planner.py`
- `docs/reference/ltx23_model_reference.md` — artifact-ceiling source
- `docs/reference/debug_tools.md` — F-pair convention (frame_planner_present is part of the F-series)
- `docs/reference/audio_loop_controller.md` — `actual_seconds` and `fps_int` are the dimension feed AudioLoopController consumes
- `docs/reference/timestamp_prompt_schedule_batch_encode.md` — consumes `fps_float` for `frame_rate` stamping
- `docs/reference/f_pair_convention.md` — F8 (`frame_planner_present`) follows this convention
- `docs/reference/_atomic_note_template.md` — this doc follows the entity-note variant
- `comfy_extras/nodes_lt.py:36` — silent-floor behavior in `EmptyLTXVLatentVideo`
