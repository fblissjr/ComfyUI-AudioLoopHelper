# ComfyUI-AudioLoopHelper

## TLDR

4 ComfyUI nodes that automate loop timing for full-length music video
generation with LTX 2.3. The main node (AudioLoopController) reads audio
duration from the tensor, computes stride from window + overlap, outputs
start_index / stop signal / seed / stride / overlap_frames. No manual
constants to keep in sync.

## Architecture

Single file: `nodes.py`. Uses ComfyUI's extension API (`ComfyExtension`,
`io.ComfyNode`). Entry point: `comfy_entrypoint()`.

4 nodes:
- `AudioLoopController` -- core: start_index, stop signal, seed, stride, overlap_frames
- `TimestampPromptSchedule` -- per-iteration prompt from timestamp ranges
- `AudioLoopPlanner` -- displays iteration timeline for planning
- `AudioDuration` -- extracts duration/sample_rate from audio tensor

Helper functions:
- `_audio_duration(audio)` -- shared duration extraction (used by 3 nodes)
- `_parse_timestamp(ts)` -- "M:SS" or bare seconds to float
- `_format_timestamp(seconds)` -- float to "M:SS" (preserves sub-second)
- `_parse_schedule(schedule)` -- multiline schedule text to entries list
- `_match_schedule(entries, time)` -- find matching prompt for a timestamp

## Key patterns

- AUDIO type: `{"waveform": Tensor, "sample_rate": int}`.
  Duration = `waveform.shape[-1] / sample_rate`.
  Note: the comfy_api TypedDict says `sampler_rate` but actual code uses `sample_rate`.
- Stride is computed internally: `window_seconds - overlap_seconds`.
  overlap_frames is computed as `round(overlap_seconds * fps)`.
  User sets overlap once; stride and overlap_frames propagate via outputs.
- Timestamp parsing regex `_LINE_RE` handles colons in M:SS timestamps
  vs the colon separator between range and prompt.
- Nodes that need per-iteration evaluation (TimestampPromptSchedule,
  CLIPTextEncode) must be inside the loop body (between TensorLoopOpen
  and TensorLoopClose in the dependency graph) to be cloned each iteration.
- AudioLoopPlanner runs once (outside the loop). It uses a closed-form
  formula matching AudioLoopController's stop condition.

## ComfyUI gotchas learned the hard way

- **Workflow JSON has two link representations**: node body `"link"` fields AND the `"links"` array. Both must be updated when editing JSON or wires break on reload.
- Link array format: `[link_id, source_node, source_output, target_node, target_input, type]`
- Node `"mode": 0` = active, `"mode": 4` = bypassed.
- **PrimitiveNode cannot feed DynamicCombo sub-inputs** (e.g., `mode.iterations` on TensorLoopOpen). Set values directly on the widget instead.
- **ComfyMathExpression rejects boolean results** (raises ValueError). Use `int(expr)` wrapper, or use KJNodes `SimpleCalculatorKJ` which outputs BOOLEAN natively.
- **Graph expansion preview limitation**: preview/display nodes connected to TensorLoopOpen outputs only show first-pass values. Cloned iterations produce correct values but those go to cloned preview nodes you can't see.
- Pyright `reportIncompatibleMethodOverride` on `execute()` methods is a false positive -- standard ComfyUI node API pattern.

## Dependencies

Companion custom nodes (not imported, just used alongside in workflows):
- ComfyUI-NativeLooping_testing -- TensorLoopOpen/Close. Don't fork; graph expansion is deeply coupled to ComfyUI execution engine. Likely headed for core inclusion.
- ComfyUI-VideoHelperSuite -- VHS_VideoCombine
- ComfyUI-KJNodes -- Set/Get nodes, FloatConstant, LTX helpers
- ComfyUI-MelBandRoFormer -- vocal separation

## Testing

No comfy_api outside ComfyUI runtime. Test parsing logic inline:
```bash
uv run python -c "
import re
# paste _parse_timestamp, _parse_schedule, _match_schedule
# run assertions
print('pass')
"
```

## Workflow docs

- Docs go in `workflows/internal/` in the parent ComfyUI directory.
- Step-by-step wiring guides: v4_upgrade_steps.md, v6_upgrade_steps.md.
