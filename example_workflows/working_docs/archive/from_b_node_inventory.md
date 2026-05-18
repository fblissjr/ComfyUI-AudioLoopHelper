Last updated: 2026-05-15

# From-B node inventory — porting `fml2v_var_b_first_keyframe_only.json` to audio-loop

> **STALE WARNING (added 2026-05-16)**: this doc was written 2026-05-15 when fps=24 was framed as canonical. **The fps 25→24 sweep was reverted on 2026-05-16; canonical inference fps is now 25** (matches Lightricks shipped workflows + 8n+1 latent boundary). Canonical `first_frame_guide_strength=0.7`, canonical `target_seconds=19.88`. Full postmortem: `internal/analysis/fps_24_partial_reading_postmortem.md` (private clone only). Re-read body claims through that lens.

Source: `example_workflows/benchmark_workflows/fml2v_var_b_first_keyframe_only.json`
Canonical target reference: `example_workflows/audio-loop-music-video_latent.json`

Scope: every top-level node in B classified as **reusable**, **reusable-with-tweak**, **strip**, **replace**, or **must-add** (the last is empty for B — it covers loop machinery added from the audio-loop side). Two subgraphs exist in `definitions.subgraphs` (`PROMPT ENHANCER`, `Frames split view`) but **no top-level node references them** (no invokers in the live graph), so they're inert dead defs — strip.

---

## §1 Summary

Top-level nodes: **156** (counted: B has 156, well above the rough 99 guess in the brief — most of the bulk is `GetNode`/`SetNode` plumbing and quality-pass duplicates).

| Category | Count | Approx % |
|---|---:|---:|
| reusable | 41 | 26% |
| reusable-with-tweak | 12 | 8% |
| strip | 58 | 37% |
| replace | 45 | 29% |
| must-add | 0 (handled by companion subagent inventory) | — |

Category totals by node-type bucket:

| Bucket | Count | Disposition |
|---|---:|---|
| Loaders (UNET/CLIP/VAE/Lora/upscale) | 11 | mostly reusable; 3 strip (alt GGUF loaders), 1 strip (`tiny_vae`), 1 reusable-with-tweak (`LoraLoaderModelOnly` bypassed) |
| Two-pass refine sampler region | 13 | **reusable** core (this is B's quality story) |
| Sage attention patch cluster | 5 | 1 reusable, 3 strip (bypassed alternates), 1 reusable-with-tweak (`LTXVChunkFeedForward`) |
| Init-image path (3-frame FML) | 15 | collapse to 1 — strip middle/last branches + their resize/preprocess/Set/Get duplicates |
| Conditioning (CLIPTextEncode + LTXVConditioning) | 4 | replace with `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` |
| Audio path | 2 | **replace** — B has only `LTXVEmptyLatentAudio` (silence); needs `LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode → AudioLoopController` chain |
| Resolution / frame plumbing | 7 | strip — `LTXFramePlanner` becomes the single source of truth |
| Set/Get named-variable plumbing | 76 | majority strip (3-frame plumbing); ~12 reusable on rename; ~20 conflict with canonical and need renaming/removing |
| MarkdownNotes | 9 | strip (benchmark / RuneXX-credit notes); replace with audio-loop-appropriate notes |
| Output (VHS_VideoCombine) | 1 | reusable-with-tweak (must add `TrimImageBatchToAudio` + `RunIdPrefix` via F14/F15 apply scripts post-graft) |

**Biggest gotcha**: B has no audio source. The "audio path" you'd graft onto isn't a chain to keep — it's `LTXVEmptyLatentAudio` feeding silence. The audio chain has to come wholly from the canonical (`LoadAudio` → `TrimAudioDuration` → `LTXVAudioVAEEncode` → `AudioLoopController` → `AudioLoopPlanner`), and `LTXVEmptyLatentAudio #9` should be stripped. **Second-biggest**: B's two-pass refine (pass-1 = `#13` SamplerCustomAdvanced with 9-sigma + `#25` `LTXVLatentUpsampler` between passes; pass-2 = `#21` SamplerCustomAdvanced with 4-sigma at upscaled res) is the quality story and is fundamentally incompatible with the canonical audio-loop's single-pass `TensorLoopOpen/Close` body — the two new sibling workflows will need to choose: (a) keep both passes but only loop the pass-1 sampler (pass-2 runs once on the assembled latent post-loop, which means `LTXVLatentUpsampler` operates on the concatenated multi-iteration latent), or (b) drop pass-2 and ship single-pass audio-loop with B's loader/preprocess wins only. This is the open question for the user.

**Third gotcha**: B uses `euler_ancestral_cfg_pp` (#1) and `euler_cfg_pp` (#4) — both `*_cfg_pp` variants. Root CLAUDE.md says canonical distilled path uses **plain `euler` only** and explicitly forbids `euler_ancestral*`. Both samplers need replacing with `KSamplerSelect("euler")` for the audio-loop variant.

---

## §2 Per-node detail (ordered by category)

### reusable (keep as-is)

| Node | Type | WHY |
|---|---|---|
| #187 | `UNETLoader` (`ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors`) | Matches canonical distilled model. |
| #190 | `DualCLIPLoader` (gemma_3_12B_it_fpmixed + ltx-2.3_text_projection_bf16, `ltxv`, `default`) | Canonical text-encoder loader. |
| #181 | `VAELoader` (`LTX2_video_vae_bf16.safetensors`) | Video VAE used in canonical. |
| #175 | `VAELoaderKJ` (`LTX2_audio_vae_bf16.safetensors`, `main_device`, `bf16`) | Canonical audio VAE loader. Root CLAUDE.md notes the emergency fallback `apply_audio_vae_fix.py` if KJ breaks; otherwise keep. |
| #182 | `LatentUpscaleModelLoader` (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`) | Required by pass-2 (`#25 LTXVLatentUpsampler`). Reusable IF two-pass refine is kept. |
| #13 | `SamplerCustomAdvanced` (pass 1) | Core of B's quality story — pass-1 at planner res. |
| #21 | `SamplerCustomAdvanced` (pass 2) | Pass-2 at 2× upscaled res. Reusable only if two-pass is kept. |
| #215 | `ManualSigmas` `"1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"` (9-step pass 1) | **Canonical 8-step distilled chain** (matches `apply_canonical_sigmas.py`'s reference). |
| #216 | `ManualSigmas` `"0.85, 0.7250, 0.4219, 0.0"` (4-step pass 2) | Refine-stage sigmas — keep with two-pass. |
| #25 | `LTXVLatentUpsampler` (`spatial`) | Spatial upsample between pass 1 and pass 2 — central to two-pass refine. |
| #14, #15 | `RandomNoise` (seeds 42, 43, both `fixed`) | One per pass. Reusable. (Will need an `AudioLoopController.base_seed` source post-graft.) |
| #8, #36 | `CFGGuider` (cfg=1) | Standard distilled-path guiders. |
| #18, #146 | `LTXVSeparateAVLatent` | A/V splitting — also used by canonical audio-loop. |
| #24, #34 | `LTXVConcatAVLatent` | A/V repacking — also used by canonical. |
| #2222 | `LTXVCropGuides` (between passes) | F3 chain element. |
| #2156 | `LTXVCropGuides` (post pass 2, pre decode) | F3 chain element. |
| #2182 | `LTXVAddGuideMulti` (pass-2 anchors, `strength=1.0`) | Re-anchors first frame at upscaled res. Reusable. |
| #2221 | `LTXVAddGuideMulti` (pass-1 anchors, `strength=0.7`) | Soft init anchor. Reusable; widget tied to `Get_firstframe_strength` (`#2110 PrimitiveFloat = 0.7`). |
| #2107 | `Power Lora Loader (rgthree)` | Currently empty (no LoRAs configured). Reusable. |
| #149 | `LTXVTiledVAEDecode` `[1,1,1,True,"auto","auto"]` | Canonical 24GB-class config (per `apply_no_tile_vae_decode.py`). |
| #150 | `LTXVAudioVAEDecode` | Standard A decode. |
| #43 | `VHS_VideoCombine` | Standard output node. (Tweak below.) |
| #228 | `LTXVChunkFeedForward` `[2, 4096]` | Root CLAUDE.md notes FFN chunking is on in canonical bench workflows; keep. |
| #2296 | `AudioLoopHelperSageAttention` `["auto", True, 1024]` (`auto_mask_aware`, `skip_under_seq_len=1024`) | Canonical attention node — exactly what the audio-loop variant wants. |
| #198 | `LTX2SamplingPreviewOverride` `[8]` | Preview override; canonical keeps it. Confirm wiring. |
| #197 | `LTX2_NAG` (see tweak — flagged for value review) | NAG node itself reusable; widget values flagged below. |
| #122, #124, #117, #120, #133, #147, #148, #196, #2155, #2255 | `GetNode` for `model`, `clip`, `vae_audio`, `vae`, `upscale_model`, `vae`, `vae_audio`, `negative`, `vae`, `vae` | Canonical-variable getters that map cleanly to the audio-loop's named map (see §3). |
| #2154, #2163, #2166, #2167, #2259, #2260 | `GetNode` for `negative_guider`, `positive_guider`, `negative_guider2`, `positive_guider2` | Guider plumbing — reusable as-is. |
| #2164, #2165, #2223, #2224, #2233, #2215, #2214 | `SetNode` for `positive_guider2`, `negative_guider2`, `positive_guider`, `negative_guider`, `negative`, `latent_audio`, `Get_latent_audio` | Guider/conditioning routing — reusable. |
| #199, #200, #201 | `Set_model_nag` / two `Get_model_nag` | NAG-wrapped model routing — reusable. |
| #230 | `Set_model_with_lora` | Routes patched model out. Reusable. |
| #188, #173, #172 | `Set_clip`, `Set_vae`, `Set_vae_audio` | Loader-output routing — reusable. |
| #153, #154, #203, #204 | `Set_final_video`, `Set_final_audio`, `Get_final_video`, `Get_final_audio` | Output plumbing — reusable. |

### reusable-with-tweak

| Node | Type | Tweak |
|---|---|---|
| #1 | `KSamplerSelect("euler_ancestral_cfg_pp")` | **Replace widget value with `"euler"`** — root CLAUDE.md forbids `euler_ancestral*` on distilled. (Border case: classifying as tweak not replace since the node type stays.) |
| #4 | `KSamplerSelect("euler_cfg_pp")` | **Replace widget value with `"euler"`** — same rule; CFG=1 makes `_cfg_pp` variants redundant anyway. |
| #186 | `LoraLoaderModelOnly` (`...lora-dynamic_fro09_avg_rank_111_bf16`, strength 0.6, `mode=4` bypassed) | **Leave bypassed** for canonical audio-loop production. The "dynamic" suffix and rank-111 size suggests an experimental motion-dynamics LoRA; no docs reference it. User can toggle on for experimentation but not the default. Flag in §4. |
| #197 | `LTX2_NAG` `[scale=11, alpha=0.25, tau=2.5, inplace=True]` | **Dial `scale` down to 3-7** per root CLAUDE.md ("KJNodes default `scale=11` is aggressive for distilled — dial to 3-7 if initial render freezes"). Recommend **5** as starting value for production audio-loop. |
| #43 | `VHS_VideoCombine` | **Post-graft must run `apply_trim_image_batch_to_audio.py` (F14) + `apply_run_id_layout.py` (F15)** to splice `TrimImageBatchToAudio` into `.images` and `RunIdPrefix` into `.filename_prefix`. Also flip `trim_to_audio` widget true (currently false) is moot since `-c:v copy` doesn't truncate. Update `filename_prefix` from `"LTX-2"` to something audio-loop-appropriate. |
| #149 | `LTXVTiledVAEDecode` | Reusable; just confirm `audit_workflows.py::vae_decode_no_tile` passes (it will with `[1,1,1]`). |
| #228 | `LTXVChunkFeedForward` `[2, 4096]` | Reusable; root CLAUDE.md confirms FFN chunking on by default. No tweak strictly required, but document the choice in the new workflow's MarkdownNote. |
| #2296 | `AudioLoopHelperSageAttention` `["auto", True, 1024]` | Reusable; the chain repositioning is the tweak — sage node must sit AFTER any model-mutating node (root CLAUDE.md: "Canonical order for compile-style patches: `UNETLoader → ... → LTXICLoRALoaderModelOnly → <module-mutating node> → SetNode "model"`"). |
| #9 | `LTXVEmptyLatentAudio` `[97, 25, 1]` | **Strip if real audio is loaded** (canonical path), OR keep with `frames_number` autowired from `LatentFrameCount` if the new workflow is silent. Brief says "produces two new sibling workflows" — both should have real audio, so this is strip. (Listed here only because the question of which sibling has audio is open.) |
| #2076 | `PrimitiveFloat` (`FPS = 24`) | **Keep at 24.0** — canonical for LTX 2.3 (training-distribution framerate; `comfy/ldm/lightricks/av_model.py:866`). The canonical audio-loop `LTXVConditioning #10` widget snapshot still shows `frame_rate=25` (pre-2026-05-15 default); the workflow-JSON sweep migrates it to 24 to match. |
| #2103 | `PrimitiveStringMultiline` `"video of a man dancing and singing"` | Replace prompt content with the audio-loop timestamped-schedule input (or strip if `TimestampPromptScheduleBatchEncode` takes a separate widget — it does). Disposition: strip after replace. |
| #91, #93, #137 | `GetNode "fps"` (3 copies) | If the canonical audio-loop pipeline supplies fps via `AudioLoopController.fps`, the named-var `fps` Get/Set still works — reusable, but their **upstream Set** (`#2074 Set_fps` driven by `#2076 PrimitiveFloat FPS`) needs source rewired to `AudioLoopController.fps`. See §3. |

### strip

| Node | Type | WHY |
|---|---|---|
| #2 | `LTXVScheduler` `[8, 2.05, 0.95, True, 0.1]` | **Orphaned** — `output[0] SIGMAS` has no consumers (`-> []`). Dead-code. Title says "for more steps" but unwired. Strip. |
| #5 | `ManualSigmas` `"0.909375, 0.725, 0.421875, 0.0"` | **Orphaned** — `out[0] SIGMAS -> []`. Dead. Strip. |
| #11 | `CLIPTextEncode` (negative prompt) | Replaced by canonical audio-loop's negative-prompt pre-encode path; B's hardcoded negative string goes into the schedule-encode widget or batch-encode equivalent. |
| #16 | `CLIPTextEncode` (positive, single prompt) | **Replace** by `TimestampPromptScheduleBatchEncode` (handled under §1 replace bucket). Strip the node. |
| #10 | `LTXVConditioning` | Replaced by `TimestampPromptScheduleBatchEncode`'s output, which already stamps `frame_rate`. Strip. |
| #9 | `LTXVEmptyLatentAudio` | Strip (B feeds silence; new workflows source from `LoadAudio` → `LTXVAudioVAEEncode`). |
| #32 | `EmptyLTXVLatentVideo` `[768, 512, 97, 1]` | Strip — canonical audio-loop sizes the empty video latent via `LTXFramePlanner` outputs, not hardcoded widgets. |
| #47, #2172 | `LoadImage` ("MIDDLE FRAME", "LAST FRAME") | `_first_keyframe_only` variant uses only frame 0. **Confirmed**: middle + last frame are still loaded structurally in B (the variant suffix is about which guides are SOFT not which are loaded), but for audio-loop production we want a single init image — strip both. |
| #48, #2171 | `ImageResizeKJv2` (middle / last) | Strip with their corresponding LoadImage. |
| #49, #2168 | `ResizeImagesByLongerEdge` (middle / last) | Strip. |
| #2174 | `LTXVPreprocess` (middle path) | Strip — no live consumer (`#50` also has `-> []`; both middle/last preprocess outputs go nowhere in `_first_keyframe_only` variant). |
| #50 | `LTXVPreprocess` (last path) | Strip — `out -> []` (already dead in B). |
| #78, #2169 | `SetNode "middleframe"`, `SetNode "lastframe"` | Strip — only `firstframe` survives. |
| #2173, #2106, #2220, #2225, #2224 | `GetNode "middleframe"`, `Get "lastframe"` (multiple), `Get "firstframe"` (multiple) | Strip middle/last gets; consolidate firstframe gets to one. |
| #2169 | `Set_lastframe` | Strip. |
| #2107 + #2108 + #2109 + #2110 + #2278 | `PrimitiveFloat LAST FRAME STRENGTH`, `FIRST FRAME STRENGTH`, `MIDDLE FRAME STRENGTH` + MarkdownNote about strengths | Strip last+middle strength primitives + their note; keep `FIRST FRAME STRENGTH=0.7` as the canonical `first_frame_guide_strength` value (root CLAUDE.md). |
| #2112, #2113, #2277, #2218, #2219, #2217 | `Set_firstframe_strength`, `Set_lastframe_strength`, `Set_middleframe_strength`, `Set_middleframe_resized`, `Set_lastframe_resized`, `Set_firstframe_resized` | Strip middle/last variants; reroute firstframe_strength = single canonical value. |
| #2185 | `Set_middleframe_count` | Strip. |
| #2187, #2188, #2189, #2276, #2279, #2280, #2281, #2226 | `Get_firstframe_strength` (×3), `Get_lastframe_strength` (×3), `Get_middleframe_strength` (×2) | Consolidate firstframe; strip rest. |
| #2191, #2192, #2216, #92, #2077 | `ComfyMathExpression`, `SimpleCalculatorKJ` (compute width/2, height/2, audio length) | Strip — math becomes implicit in `LTXFramePlanner` + `AudioLoopController` outputs. |
| #2072, #2073, #2074, #2075 | `Set_width`, `Set_height`, `Set_fps`, `Set_frames` | Strip — these are sourced from B's `INTConstant` widgets; the audio-loop replacement gets these from `LTXFramePlanner`. |
| #2078, #2079, #2080 | `INTConstant LENGTH=15`, `HEIGHT=720`, `WIDTH=1280` | **Strip** — these are the hardcoded resolution/length widgets that conflict with `LTXFramePlanner`. The audio length comes from audio source, not user input; resolution from planner. |
| #2076 | `PrimitiveFloat FPS=24` | Strip after retargeting (canonical uses 25). |
| #70, #71, #128, #129, #219, #220 | `Get_width` (×3), `Get_height` (×3) | Most rewire to `LTXFramePlanner` outputs; redundancy strip — keep one of each at most, or remove the named-var indirection entirely and direct-wire. |
| #127, #2175 | `Get_frames` (×2) | Strip — replaced by `AudioLoopController.frames_per_iter` / `LTXFramePlanner.actual_frames`. |
| #133, #193 | `Get_upscale_model`, `Get_vae_tiny` | `Get_upscale_model` reusable if two-pass kept (consumer of `#182`); `Get_vae_tiny` strip (preview-VAE bypassed). |
| #177 | `Set_vae_tiny` | Strip (preview-VAE source is bypassed loader #180). |
| #180 | `VAELoader` `[taeltx2_3.safetensors]` `mode=4` | Strip — bypassed, file not present locally (per B's note). |
| #189 | `DualCLIPLoaderGGUF` `mode=4` | Strip — alternate loader, bypassed. |
| #191 | `UnetLoaderGGUF` `mode=4` | Strip — alternate loader, bypassed. |
| #226 | `PathchSageAttentionKJ` `mode=4` | **Strip** — bypassed; replaced by `AudioLoopHelperSageAttention #2296` (which is the canonical audio-loop attention node). |
| #227 | `LTX2MemoryEfficientSageAttentionPatch` `mode=4` | Strip — bypassed alternate sage patch. |
| #229 | `LTX2AttentionTunerPatch` `mode=4` | Strip — bypassed tuner. |
| #170, #174, #176, #179, #183, #184, #2109, #9000, #9001 | All `MarkdownNote` | Strip — benchmark-specific (RuneXX credit, ada-sage caveats, frame-strength tips). Replace with audio-loop-appropriate notes in the new workflows (about prompt scheduling, audio path, F-pair compliance). |

### replace (functionally needed but canonical uses different node)

| B node | Replace with | WHY |
|---|---|---|
| #16 `CLIPTextEncode` (positive single prompt) + #2103 `PrimitiveStringMultiline` + #10 `LTXVConditioning` | `TimestampPromptScheduleBatchEncode` (pre-encodes outside loop, frame_rate stamped) + `ConditioningSelectByIteration` (per-iter pluck inside loop body) | Root CLAUDE.md: "CLIP must not enter the loop body. Pre-encode via `TimestampPromptScheduleBatchEncode` outside; `ConditioningSelectByIteration` plucks per-iter inside." |
| #11 `CLIPTextEncode` (negative prompt) | The negative side of `TimestampPromptScheduleBatchEncode` or a single static negative cond pre-encoded outside the loop (canonical pattern from `audio-loop-music-video_latent.json` — checks Node 169 + 420 zero-out + 164 + 153 wiring). | Negative is static for audio-loop; pre-encode once. |
| #32 `EmptyLTXVLatentVideo` `[768, 512, 97, 1]` | `LTXFramePlanner → EmptyLTXVLatentVideo` (widgets driven from planner outputs, not hardcoded) | Resolution/frames from planner SSoT. |
| #9 `LTXVEmptyLatentAudio` | `LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode` chain (canonical audio source) | B has no real audio source. Whole audio chain must come in from canonical. |
| #44 `ImageResizeKJv2 [960, 544, 'lanczos', 'crop', ...]` (firstframe) + #2083 `ResizeImagesByLongerEdge [1536]` | `LTXSmartImageResize` (adaptive multi-stage with bicubic intermediates + lanczos final) | Root CLAUDE.md: "Single-pass lanczos at >2× linear reduction aliases (model reads as motion cues → spurious zoom/dolly); naive multi-stage PIL-backed lanczos stacks float32→uint8 quantization rounds. `LTXSmartImageResize` solves both." |
| #2078, #2079, #2080 INTConstants | `LTXFramePlanner` widgets | SSoT for dimensions per F8 audit (`frame_planner_present`). |
| #92, #2077, #2216, #2191, #2192 math nodes | `LTXFramePlanner` outputs (implicit) + `AudioLoopController` outputs | Math becomes implicit in the planner/controller schemas. |
| #14, #15 `RandomNoise` (two passes) | Keep node type, but rewire `noise_seed` to derive from `AudioLoopController.iter_seed` (canonical pattern) | Per-iteration seed must be deterministic from controller. |
| #1, #4 `KSamplerSelect("euler_*_cfg_pp")` | `KSamplerSelect("euler")` (plain) for both passes | Distilled-path rule. Strictly a widget-value change; logged here as replace because the sampler chosen is wrong. |
| #43 `VHS_VideoCombine` | Same node + post-graft `apply_trim_image_batch_to_audio.py` (F14) + `apply_run_id_layout.py` (F15) | F14/F15 invariants. |
| #186 `LoraLoaderModelOnly` (bypassed) | Leave bypassed OR replace with `Power Lora Loader (rgthree) #2107` (already present, empty config) | Two LoRA paths in B; consolidate on one (rgthree power loader handles multi-LoRA more cleanly). |
| Entire `Set_*`/`Get_*` named-variable cluster for `firstframe`/`middleframe`/`lastframe`/`firstframe_strength`/etc. | Audio-loop named vars (`model`, `audio_vae`, `video_vae`, `actual_audio`, `orig_audio`, `full_audio_latent`, `sigmas`, `start_seed`) | See §3 for full conflict + rename plan. |

### must-add (from companion subagent's loop-machinery inventory — listed here just to mark slot locations)

Brief says these come from the companion. Slot annotations:

- `AudioLoopController` — slots **upstream** of the first-pass `RandomNoise #15` (drives `base_seed`), and feeds `frame_rate` + iteration metadata.
- `AudioLoopPlanner` — slots upstream of `TensorLoopOpen`'s `iterations_in`.
- `LTXFramePlanner` — slots **before** `EmptyLTXVLatentVideo` (replacement for #32) + provides `actual_seconds` to `TrimAudioDuration.duration` (per `apply_initial_render_audio_duration_autowire.py`).
- `LoadAudio` + `TrimAudioDuration` + `LTXVAudioVAEEncode` — replace #9's silent latent.
- `TensorLoopOpen` / `TensorLoopClose` — wrap the pass-1 sampler region (the per-iter denoise). Decision needed: does the loop wrap **just pass 1** (and pass 2 runs once on the concatenated assembled latent) or **both passes** (per-iter refine)? See §4.
- `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` — replace conditioning region.
- `KeyframeLatentScheduleBatchEncode` + `LatentSelectByIteration` — if keyframe timeline desired (sibling-workflow variant).
- `LatentContextExtract` / `LatentOverlapTrim` / `LatentConcat` — assembly path.
- `ProfileBegin/IterStep/End` — optional, for bench variants.
- `RunIdPrefix`, `TrimImageBatchToAudio`, `TrimVideoLatentToAudio` — F14/F15 post-graft.

---

## §3 Named-variable map: B vs canonical

### B's Set/Get vocabulary

Sets (with originating value):

| Var | B's source | Disposition |
|---|---|---|
| `model` | `#192 Set_model` ← `#2107` Power Lora Loader output | **Keep** (canonical uses same `model` name) |
| `clip` | `#188 Set_clip` ← `#190` DualCLIPLoader | **Keep** |
| `vae` | `#173 Set_vae` ← `#181` VAELoader | **RENAME to `video_vae`** to match canonical (canonical uses both `video_vae` and `audio_vae`; B's bare `vae` is ambiguous) |
| `vae_audio` | `#172 Set_vae_audio` ← `#175` VAELoaderKJ | **RENAME to `audio_vae`** (canonical's name) |
| `vae_tiny` | `#177 Set_vae_tiny` ← bypassed `#180` | **STRIP** (preview-VAE bypassed) |
| `upscale_model` | `#171 Set_upscale_model` ← `#182` | **Keep** if two-pass refine kept |
| `model_nag` | `#199 Set_model_nag` ← `#197 LTX2_NAG` | **Keep** |
| `model_with_lora` | `#230 Set_model_with_lora` ← `#2107` rgthree LoRA loader | **Keep** |
| `negative` | `#2233 Set_negative` ← `#11` neg-CLIPTextEncode | **STRIP** (replaced by pre-encoded static negative) |
| `positive_guider`, `negative_guider`, `positive_guider2`, `negative_guider2` | `#2223, #2224, #2164, #2165` | **Keep** (two-pass guider plumbing) |
| `latent_audio` | `#2215 Set_latent_audio` ← `#9 LTXVEmptyLatentAudio` | **REWIRE** source to `LTXVAudioVAEEncode` output. Keep var name. |
| `firstframe`, `middleframe`, `lastframe` | `#75, #78, #2169` | **Keep `firstframe`, STRIP middle/last** |
| `firstframe_resized`, `middleframe_resized`, `lastframe_resized` | `#2217, #2218, #2219` | **Keep `firstframe_resized`, STRIP middle/last** |
| `firstframe_strength`, `middleframe_strength`, `lastframe_strength` | `#2112, #2277, #2113` | **Keep `firstframe_strength`, STRIP middle/last** |
| `middleframe_count` | `#2185` | **STRIP** |
| `width`, `height`, `fps`, `frames` | `#2073, #2072, #2074, #2075` | **STRIP B's setters; canonical sources from `LTXFramePlanner`**. The named-var Gets stay; their setter changes to LTXFramePlanner output. |
| `final_video`, `final_audio` | `#153, #154` | **Keep** |

### Canonical audio-loop variables (from `audio-loop-music-video_latent.json`)

- `actual_audio` — trimmed audio (`LoadAudio → TrimAudioDuration`)
- `audio_vae` — canonical name (B uses `vae_audio`)
- `full_audio_latent` — full-song audio latent (`LTXVAudioVAEEncode` on the full song)
- `model` — same as B
- `orig_audio` — original `LoadAudio` output (pre-trim)
- `sigmas` — canonical exposes sigmas as a named var (B inlines them)
- `start_seed` — canonical base-seed driver (replaces B's hardcoded `#15 RandomNoise 42`)
- `video_vae` — canonical name (B uses bare `vae`)

### Conflicts + rename plan

| Conflict | Resolution |
|---|---|
| B `vae` ↔ canonical `video_vae` | Rename B's `Set_vae` / `Get_vae` (×5) to `Set_video_vae` / `Get_video_vae` |
| B `vae_audio` ↔ canonical `audio_vae` | Rename `Set_vae_audio` / `Get_vae_audio` (×2) to `Set_audio_vae` / `Get_audio_vae` |
| B has no `sigmas` named-var (sigmas direct-wired from `#215`/`#216`) | Add `Set_sigmas`/`Get_sigmas` if two-pass kept; OR direct-wire (low cost). |
| B has no `start_seed` named-var | Add — wire `AudioLoopController.start_seed → Set_start_seed`; both `RandomNoise` nodes' `noise_seed` widgets fed via `Get_start_seed` (+ iter-derivation). |
| B has no `actual_audio`/`orig_audio`/`full_audio_latent` | Must-add from canonical audio chain. |
| B's `width`/`height`/`fps`/`frames` setters from `INTConstant` | Rewire setters to `LTXFramePlanner` outputs; Gets stay. |

---

## §4 Open questions (need user's call)

1. **Two-pass refine: keep or drop?**
   - Option A: keep pass-1 + pass-2 + `LTXVLatentUpsampler`, but **only loop pass-1**. Pass-2 runs once on the post-`LatentConcat` assembled latent. This means the upsample + refine sees the full multi-iteration latent at once → big VRAM event. May not fit on 24GB at production length.
   - Option B: drop pass-2 entirely. New workflow is single-pass like canonical `audio-loop-music-video_latent.json`. Trades quality for tractability. This is the "B's loader wins only" variant.
   - Option C: ship two siblings — one Option A, one Option B. (Possible interpretation of "produces two new sibling workflows.")
2. **Pass-2's `LTXVLatentUpsampler` operating on assembled latent**: if Option A, the upsampler operates on a `LatentConcat`-assembled video of duration `total_iterations × stride_seconds`. Confirm this is the intent.
3. **`LoraLoaderModelOnly #186` (`...dynamic_fro09_avg_rank_111_bf16`)**: leave bypassed, or include as default-on? No public docs reference this LoRA name. Defensible default: bypassed (matches B). Flag for user awareness.
4. **`LTX2_NAG.scale` value**: B has `11`; root CLAUDE.md recommends 3-7 for distilled. Default to **5** unless user prefers a different value.
5. **fps**: B uses 24, canonical's live JSON still shows 25 (pre-2026-05-15 default). LTX 2.3 was trained at 24 (`comfy/ldm/lightricks/av_model.py:866`). Recommend **24** for the new audio-loop variants — matches training distribution.
6. **Init image**: `benchmark_test_frame.png` in B is a placeholder. The two new workflows should ship with a generic placeholder name (e.g. `example_init.png`) and a MarkdownNote pointing the user to replace it.
7. **Negative prompt**: B's `#11` has the full benchmark negative string ("blurry, oversaturated, pixelated, ...). Canonical audio-loop workflows tend to use a static `ConditioningZeroOut`-wired pseudo-negative at CFG=1 (since negative is inert at cfg=1). Decision: pre-encode B's negative once and keep it (preserves whatever marginal contribution it has) OR replace with zero-out. Recommend zero-out for cleanliness; preserve in a code comment for revert.
8. **Subgraphs in `definitions.subgraphs`** (`PROMPT ENHANCER`, `Frames split view`) — no invokers, dead defs. Strip from `definitions.subgraphs` entirely or leave (harmless). Recommend strip for hygiene.

---

## §5 Build-script implications

High-level steps your build script must perform:

1. **Load B with `WorkflowEditor`**; deep-copy to a working state.
2. **Strip dead defs**: clear `definitions.subgraphs` (or leave; cosmetic).
3. **Strip middle/last frame branches** (15 nodes: #47, #2172, #48, #2171, #49, #2168, #2174, #50, #78, #2169, #2173, #2106, #2185, plus their corresponding Set/Get of `middleframe_*` and `lastframe_*` vars and the matching strength primitives #2108, #2278).
4. **Strip alternate loaders**: #180 (taeltx2 VAELoader bypassed), #189 (DualCLIPLoaderGGUF bypassed), #191 (UnetLoaderGGUF bypassed), #177 (Set_vae_tiny), #193 (Get_vae_tiny).
5. **Strip bypassed sage patches**: #226, #227, #229. Keep #228 (`LTXVChunkFeedForward`) and #2296 (`AudioLoopHelperSageAttention`).
6. **Strip dead sigma/scheduler nodes**: #2 (LTXVScheduler orphaned), #5 (ManualSigmas orphaned).
7. **Strip benchmark MarkdownNotes**: all 9.
8. **Strip conditioning region**: #11, #16, #10, #2103.
9. **Strip empty-latent + INTConstant resolution plumbing**: #9, #32, #2078, #2079, #2080, #2076 (after confirming fps=24 source flows through `LTXFramePlanner.fps_float`), #2072-#2075 (width/height/fps/frames setters), #92, #2077, #2191, #2192, #2216.
10. **Rename Set/Get vars**: `vae` → `video_vae`, `vae_audio` → `audio_vae` (use `WorkflowEditor` widget rewrites; ~10 Get + 2 Set nodes).
11. **Replace sampler widgets**: #1 `"euler_ancestral_cfg_pp"` → `"euler"`; #4 `"euler_cfg_pp"` → `"euler"`.
12. **Tweak NAG widget**: #197 `scale=11` → `scale=5`.
13. **Add audio chain** (from canonical): `LoadAudio` + `TrimAudioDuration` (`[0, 600]` per `apply_fix_source_audio_trim_defaults.py`) + `LTXVAudioVAEEncode` + `Set_orig_audio` + `Set_actual_audio` + `Set_full_audio_latent`. Rewire `Set_latent_audio` (kept) source to `LTXVAudioVAEEncode` output.
14. **Add `LTXFramePlanner`** + wire its outputs to: `EmptyLTXVLatentVideo` (re-added via canonical pattern), `Set_width`/`Set_height`/`Set_fps`/`Set_frames` (or direct-wire, depending on naming policy), `TrimAudioDuration.duration` (autowire per F8).
15. **Add audio-loop spine**: `AudioLoopController` + `AudioLoopPlanner` + `TensorLoopOpen`/`TensorLoopClose` + `LatentContextExtract` + `LatentOverlapTrim` + `LatentConcat`. Wrap pass-1 sampler (`#13`) in the loop body. Decision-gated: optionally wrap pass-2 (`#21`) or leave it outside.
16. **Add conditioning region**: `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration`. Wire to both `CFGGuider` positive/negative (replaces direct `#10` LTXVConditioning wiring).
17. **Add `RandomNoise` seed wiring**: rewire `#14`/`#15` `noise_seed` widgets to `AudioLoopController.iter_seed` (canonical pattern).
18. **Post-graft apply-script chain** (must run after build, in order):
    1. `scripts/apply_loop_guide_preprocess_symmetry.py` (F2) — match init/loop preprocess.
    2. `scripts/apply_loop_cropguides_symmetry.py` (F3) — loop CFGGuider through CropGuides.
    3. `scripts/apply_trim_video_latent_to_audio.py` (F14 latent half).
    4. `scripts/apply_trim_image_batch_to_audio.py` (F14 image half).
    5. `scripts/apply_run_id_layout.py` (F15) — `RunIdPrefix` + bypassed `SaveLatent` for upscale path.
    6. `scripts/audit_workflows.py` — must exit 0 on both new workflows.
19. **Layout pass**: run `scripts/apply_layout_polish_audio_loop_latent.py --from-template` or its newer equivalent so the two new workflows match the canonical visual layout.
20. **Save to**: `example_workflows/<new_name_pass1_only>.json` and `example_workflows/<new_name_two_pass>.json` (or whatever the sibling naming is). Both must pass `audit_workflows.py`.

Read-only confirmation: nothing was mutated in B or any other JSON during this inventory pass.
