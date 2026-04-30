---
name: workflow-validator
description: Validate ComfyUI workflow JSON for the audio-looped music video pipeline. Supports both IMAGE and LATENT workflow variants. Checks AudioLoopController schema (post-rename: base_seed not seed; widgets_values length 5), AudioLoopPlanner schema (post-cycle-break: window_seconds + overlap_seconds + fps, NO stride_seconds input), subgraph wiring, conditioning path, noise_mask handling, link integrity. Begins by running scripts/audit_workflows.py — currently F2–F10 named topology checks, plus quality-axis checks (sage, decoder, guider, retake_*, prompt_*, etc.), plus 3 generic invariants (graph_acyclic / widget_shape / link_integrity) and the cond_metadata_types AST test in tests/test_node_schemas.py. Full inventory: docs/reference/debug_tools.md.
tools: Read, Grep, Glob, Bash
---

# Workflow Validator

Validate a ComfyUI workflow JSON for the LTX 2.3 audio-looped music video pipeline.
Supports both workflow types:
- **Image workflow** (`audio-loop-music-video_image.json`): per-iteration VAE decode/encode
- **Latent workflow** (`audio-loop-music-video_latent.json`): latent-space loop, one final decode

## Step 0: Run the automated audit FIRST

```
uv run --group dev python scripts/audit_workflows.py
```

The audit covers all named topology checks F2–F10 (`preprocess_symmetry`,
`loop_cropguides_symmetry`, `alc_seed_legacy_name`, `iterations_autowired`,
`alc_widget_drift`, `planner_no_stride_input`, `frame_planner_present`,
`ltx2_nag_reaches_loop`, `vae_decode_no_tile`) plus quality-axis checks
(`sage`, `sage_active`, `sage_mode`, `sage_node`, `decoder`, `guider`,
`cfg_value`, `iteration_stamp`, `latent_volume`, `preprocess_compression`,
`prompt_relay_wiring`, `prompt_schedule`, `id_lora_runtime_consistent`,
`retake_audio_passthrough`, `retake_no_loop_nodes`, `retake_temporal_mask_present`)
plus 3 generic invariants (`graph_acyclic`, `widget_shape`, `link_integrity`)
plus the `cond_metadata_types` AST test in `tests/test_node_schemas.py`. Any
ERR has a remediation pointer to a `scripts/apply_*.py` script. If audit is
clean and the workflow still fails, proceed to manual checks below.

Reference: `docs/reference/debug_tools.md` — full check inventory, what each
audit ID catches, paired apply scripts. Treat that doc + the live `record(...)`
calls in `scripts/audit_workflows.py` as the source of truth — this list rots
the moment a new check lands.

## Step 1: Detect workflow type

Read the workflow JSON. Determine type by checking:
- If subgraph contains `GetImageRangeFromBatch` nodes (615, 1509) → IMAGE workflow
- If subgraph contains `LatentContextExtract` (2004) and `LatentOverlapTrim` (2005) → LATENT workflow

## Step 2: Run ALL checks. Report PASS or FAIL for each.

### AudioLoopController (both workflows)
- [ ] Node exists (type "AudioLoopController")
- [ ] 6 inputs: current_iteration, window_seconds, overlap_seconds, audio, **base_seed** (NOT `seed` — renamed 2026-04-26 to defuse ComfyUI's control_after_generate auto-attach), fps
- [ ] `widgets_values` has exactly 5 entries `[current_iteration, window_seconds, overlap_seconds, base_seed, fps]`. A 6-element list with `'randomize'` / `'fixed'` at index 4 is the F6 drift — fix via `scripts/apply_strip_alc_control_after_generate.py`
- [ ] **8 outputs**: start_index, should_stop, audio_duration, iteration_seed, stride_seconds, overlap_frames, overlap_latent_frames, **overlap_seconds** (the post-quantization effective value, distinct from the input widget)
- [ ] overlap_seconds is a widget (~1.0-3.0, no link)
- [ ] fps is a widget (25, no link)
- [ ] start_index output → Extension #843 start_index
- [ ] should_stop output → TensorLoopClose stop
- [ ] iteration_seed output → Extension #843 noise_seed
- [ ] stride_seconds output → TimestampPromptSchedule (NOT into AudioLoopPlanner — that wire is the F7 cycle-closer, removed 2026-04-27. AudioLoopPlanner now derives stride internally.)
- [ ] overlap_frames output wired (IMAGE: → Extension #843 overlap input; LATENT: may be unused)
- [ ] **overlap_latent_frames output** (LATENT only): → Extension #843 overlap input (subgraph slot 14)

### Conditioning Path (both workflows)
- [ ] Text encode output goes through LTXVConditioning (frame_rate=25) BEFORE Extension #843 positive
- [ ] If blending: two text encode nodes, both from same DualCLIPLoader, into ConditioningBlend
- [ ] Extension #843 positive NOT from static Get_base_cond_pos (unless prompt scheduling is disabled)

### Extension Subgraph #843 -- IMAGE workflow
- [ ] Node 598 (LTXVAudioVAEEncode) "Audio Latent" → Node 606 (LTXVAudioVideoMask) "audio_latent"
- [ ] Node 615 (GetImageRangeFromBatch): start_index = -1, num_frames linked from overlap input
- [ ] Node 1509 (GetImageRangeFromBatch): start_index linked from overlap input, num_frames = 4096
- [ ] No stale references to deleted nodes
- [ ] Internal link origin_slot numbers match actual component input positions
- [ ] Subgraph input 14 (num_frames/overlap) linkIds non-empty, distributes to nodes 615 and 1509

### Extension Subgraph #843 -- LATENT workflow
- [ ] Node 2004 (LatentContextExtract): latent input wired from subgraph input (previous_latent)
- [ ] Node 2004: overlap_latent_frames input wired from subgraph input 14 (NOT hardcoded/null link)
- [ ] Node 2004: output wired to LTXVAudioVideoMask (#606) video_latent input
- [ ] Node 2005 (LatentOverlapTrim): latent input wired from CropGuides (#655) output
- [ ] Node 2005: overlap_latent_frames input wired from subgraph input 14 (NOT hardcoded/null link)
- [ ] Node 2005: output wired to subgraph output (-20)
- [ ] Subgraph input 14 linkIds non-empty, distributes to BOTH nodes 2004 and 2005
- [ ] Subgraph input 14 label is "overlap_latent_frames" (not "overlap")
- [ ] External link to subgraph input 14 comes from AudioLoopController **slot 6** (overlap_latent_frames), NOT slot 5 (overlap_frames)
- [ ] **noise_mask flow**: LatentContextExtract and LatentOverlapTrim strip noise_mask (verify from nodes.py source: `s.pop("noise_mask", None)`)

### noise_mask validation (LATENT workflow only)
- [ ] No LTXVSelectLatents nodes in the loop body (they preserve stale noise_mask)
- [ ] LatentContextExtract output has no noise_mask → LTXVAudioVideoMask creates fresh zeros
- [ ] LatentOverlapTrim output has no noise_mask → clean accumulation in TensorLoopClose

### Initial Render Path
- [ ] IMAGE: ImageBatch prepends decoded initial render to loop output
- [ ] LATENT: LatentConcat (#1605) prepends CropGuides output to TensorLoopClose output, dim="t"
- [ ] LATENT: VAEDecodeTiled (#1604) decodes concatenated latent → VHS_VideoCombine

### Loop Setup (both workflows)
- [ ] TensorLoopOpen: mode="iterations", iterations=50
- [ ] TensorLoopClose: accumulate=true, overlap=disabled, stop wired
- [ ] TensorLoopOpen initial_value type matches workflow: IMAGE for image, LATENT for latent

### Audio Setup (both workflows)
- [ ] MelBand #568/#569 both mode 0 (enabled)
- [ ] #569 vocals output → Set_actual_audio
- [ ] #567 does NOT wire directly to Set_actual_audio

### Output (both workflows)
- [ ] VHS_VideoCombine #617: trim_to_audio = true

### Links (both workflows)
- [ ] Every "link" field in node bodies has matching entry in links array
- [ ] No orphan link IDs
- [ ] Subgraph internal links: every linkId in subgraph inputs has a matching link entry

### Schema consistency (both workflows)
Cross-reference node inputs/outputs against `nodes.py` define_schema():
- [ ] AudioLoopController: 8 outputs match schema (the 8th is `overlap_seconds`, the effective post-quantization value)
- [ ] TimestampPromptSchedule: 4 inputs, 4 outputs match schema
- [ ] AudioLoopPlanner: 4 inputs (audio, window_seconds, overlap_seconds, fps — NO stride_seconds, removed 2026-04-27 to break controller→planner→tensorloop cycle), 2 outputs match schema
- [ ] ConditioningBlend (if present): 3 inputs, 1 output match schema
- [ ] LatentContextExtract (if present): 2 inputs, 1 output match schema
- [ ] LatentOverlapTrim (if present): 2 inputs, 1 output match schema
- [ ] StripLatentNoiseMask (if present): 1 input, 1 output match schema

## Report Format

Output a summary table:

| Check | Result | Detail |
|-------|--------|--------|
| ... | PASS/FAIL | ... |

Group by section. List all FAILs first with explanation, then PASSes.
