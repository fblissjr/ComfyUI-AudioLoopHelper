Last updated: 2026-05-15

# var_d node inventory — porting `fml2v_var_d_audio_input.json` to audio-loop

Source: `example_workflows/benchmark_workflows/fml2v_var_d_audio_input.json`
Canonical target reference: `example_workflows/audio-loop-music-video_latent.json`
Companion: `example_workflows/working_docs/from_b_node_inventory.md` (same shape for variant B)

Scope: every top-level node in var_d classified as **reusable**, **reusable-with-tweak**, **strip**, **replace**, or **must-add**. var_d is structurally the "B variant + real audio input + dynamic-dimension Set/Get plumbing"; the multi-keyframe topology is unchanged from B's `fml2v_var_b_first_keyframe_only.json` *baseline* (i.e. var_d still loads three frames and feeds them to `LTXVAddGuideMulti`). Two subgraphs live in `definitions.subgraphs` (`PROMPT ENHANCER`, `Frames split view`) — no top-level invokers — dead defs, strip.

---

## §1 Summary

Top-level nodes: **161** (vs B's 156; +5 = the audio chain `LoadAudio` / `TrimAudioDuration` / `LTXVAudioVAEEncode` / `SolidMask` / `SetLatentNoiseMask`).

| Category | Count | Approx % |
|---|---:|---:|
| reusable | 43 | 27% |
| reusable-with-tweak | 13 | 8% |
| strip | 62 | 39% |
| replace | 43 | 27% |
| must-add | 0 (handled by companion loop-machinery inventory) | — |

Category totals by node-type bucket:

| Bucket | Count | Disposition |
|---|---:|---|
| Loaders (UNET/CLIP/VAE/Lora/upscale) | 11 | mostly reusable; 3 strip (alt GGUF loaders), 1 strip (`tiny_vae`), 1 reusable-with-tweak (`LoraLoaderModelOnly` bypassed) |
| Two-pass refine sampler region | 13 | reusable core — same as B (this is var_d's quality story too) |
| Sage attention patch cluster | 5 | 1 reusable (`AudioLoopHelperSageAttention #2296`), 3 strip (bypassed alternates), 1 reusable-with-tweak (`LTXVChunkFeedForward`) |
| Init-image path (3-frame FML) | 15 | collapse to 1 — strip middle/last branches + their resize/preprocess/Set/Get duplicates |
| Conditioning (CLIPTextEncode + LTXVConditioning) | 4 | replace with `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` |
| Audio path | 5 | **partially reusable** — `LoadAudio` + `TrimAudioDuration` + `LTXVAudioVAEEncode` are the canonical audio chain; `SolidMask` + `SetLatentNoiseMask` are var_d's hand-rolled audio-freeze (canonical uses `LTXVAudioVideoMask` instead). |
| Resolution / frame plumbing | 9 | strip — `LTXFramePlanner` becomes the single source of truth (var_d's `INTConstant` + `SimpleCalculatorKJ` is structurally closer to canonical's planner than B's hardcoded `EmptyLTXVLatentVideo` widgets — less rename work). |
| Set/Get named-variable plumbing | ~80 | majority strip (3-frame plumbing); ~12 reusable on rename; ~22 conflict with canonical and need renaming/removing |
| MarkdownNotes | 9 | strip (benchmark / RuneXX-credit notes); replace with audio-loop-appropriate notes |
| Output (VHS_VideoCombine) | 1 | reusable-with-tweak (must add `TrimImageBatchToAudio` + `RunIdPrefix` via F14/F15 apply scripts post-graft) |

**Biggest discoveries vs B**:

1. **Real audio chain present.** Unlike B (which has only `LTXVEmptyLatentAudio #9` feeding silence), var_d carries a working `LoadAudio #2297 → TrimAudioDuration #2298 → LTXVAudioVAEEncode #2299` chain that feeds `Set_latent_audio #2215`. The `LTXVEmptyLatentAudio #9` node is still present BUT **bypassed (`mode=4`)** — it's the structural placeholder that var_d superseded. `TrimAudioDuration #2298` has widget `[0.0, 4.0]` (0-4 seconds), which is benchmark-short and must be retargeted for full-song audio-loop. **The audio-freeze pattern in var_d uses `SolidMask #2300` (size 512×512, value=0) + `SetLatentNoiseMask #2301` instead of canonical's `LTXVAudioVideoMask`** — this is a hand-rolled equivalent (mask=0 = freeze) that works but isn't the canonical node. Strip + replace with the canonical pattern.
2. **Dynamic-dimension Set/Get plumbing.** var_d has the full `INTConstant WIDTH=1280` + `INTConstant HEIGHT=720` + `INTConstant LENGTH=15` (seconds) + `PrimitiveFloat FPS=24` + `SimpleCalculatorKJ` computing `((round((a * b - 1) / 8)) * 8) + 1` chain that B lacks. This is structurally closer to canonical's `LTXFramePlanner` SSoT than B's hardcoded widget approach — porting cost is lower. **The `EmptyLTXVLatentVideo #32` widget `[768, 512, 97, 1]` is STALE — overridden by `width`/`height`/`length` Get-Node inputs at runtime.** Flag as `stale-widget bypassed-via-input` during port.
3. **fps already 24.** var_d's `PrimitiveFloat FPS #2076 = 24` matches LTX 2.3 training distribution (`comfy/ldm/lightricks/av_model.py:866`); no fps migration needed at the source (B had 25 widget contention with canonical's pre-2026-05-15 default).
4. **Resolution is `1280×720`.** Confirmed via `INTConstant WIDTH #2080 = 1280` and `INTConstant HEIGHT #2079 = 720`. Hardcoded `EmptyLTXVLatentVideo #32` widget shows `768×512` but is dead (overridden by Set/Get plumbing). When ported, use `LTXFramePlanner` defaults (likely `1280×704` or `1280×720` div-by-32-aligned).
5. **Sigma chain `#5` is orphaned** (same as B). `ManualSigmas #5 "0.909375, 0.725, 0.421875, 0.0"` has `out[0] SIGMAS → []` — dead-code, leftover from a prior pass-2 sigma experiment. `#215` (9-step, pass 1) and `#216` (4-step, pass 2) are the live ones. `LTXVScheduler #2` is also orphaned (out → []).
6. **Same `_cfg_pp` sampler issue as B.** `KSamplerSelect #1 = "euler_ancestral_cfg_pp"` (pass 1) and `KSamplerSelect #4 = "euler_cfg_pp"` (pass 2) — both violate root CLAUDE.md's distilled-path rule ("plain `euler` only, no `euler_ancestral*`"). Both need widget value `"euler"`.
7. **NAG `scale=11`** — same as B, same dial-to-3-7 recommendation. Recommend 5.
8. **Multi-keyframe still active.** Despite var_d carrying "audio input" in its name, the multi-keyframe topology (3 LoadImage + 3 ImageResizeKJv2 + 3 LTXVPreprocess + 3 strength primitives + `LTXVAddGuideMulti #2221` widget `['3', 0, 0.7, 0, 0.25, -1, 1]`) is preserved. The pattern `['3', 0, 0.7, 0, 0.25, -1, 1]` decodes as: 3 guides, guide-1 at frame 0 strength 0.7, guide-2 at frame 0 strength 0.25, guide-3 at frame -1 strength 1.0. Per `workflow_quality_delta_analysis.md`, this triggers the model's `_build_guide_self_attention_mask` codepath. For audio-loop production we strip to single-keyframe (first frame only) — that's the structural difference from B's `_first_keyframe_only` variant.
9. **Two-pass refine identical to B.** `SamplerCustomAdvanced #13` (pass 1, sigmas `#215`, sampler `#1`) → `LTXVSeparateAVLatent #18` → `LTXVCropGuides #2222` → `LTXVLatentUpsampler #25` (with `#182` upscale model) → `LTXVAddGuideMulti #2182` (re-anchor at upscaled res, widget `['2', 0, 1, -1, 1]`) → `SamplerCustomAdvanced #21` (pass 2, sigmas `#216`, sampler `#4`) → `LTXVSeparateAVLatent #146` → `LTXVCropGuides #2156` → `LTXVTiledVAEDecode #149`. Same structure as B.

**Resolution mode confirmation: 1280×720** (per `INTConstant #2080` + `INTConstant #2079`). Stale `EmptyLTXVLatentVideo #32` widget `[768, 512, 97, 1]` is overridden at runtime by the Set/Get chain — flag during port.

---

## §2 Per-node detail (ordered by category)

### reusable (keep as-is)

| Node | Type | WHY |
|---|---|---|
| #187 | `UNETLoader` (`ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors`) | Matches canonical distilled model. |
| #190 | `DualCLIPLoader` (gemma_3_12B_it_fpmixed + ltx-2.3_text_projection_bf16, `ltxv`, `default`) | Canonical text-encoder loader. |
| #181 | `VAELoader` (`LTX2_video_vae_bf16.safetensors`) | Video VAE used in canonical. |
| #175 | `VAELoaderKJ` (`LTX2_audio_vae_bf16.safetensors`, `main_device`, `bf16`) | Canonical audio VAE loader. |
| #182 | `LatentUpscaleModelLoader` (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`) | Required by pass-2 `#25 LTXVLatentUpsampler`. Reusable IF two-pass refine kept. |
| #13 | `SamplerCustomAdvanced` (pass 1) | Core of two-pass quality story — pass-1 at planner res. |
| #21 | `SamplerCustomAdvanced` (pass 2) | Pass-2 at 2× upscaled res. Reusable only if two-pass kept. |
| #215 | `ManualSigmas` `"1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"` (9-step pass 1) | Canonical 8-step distilled chain. |
| #216 | `ManualSigmas` `"0.85, 0.7250, 0.4219, 0.0"` (4-step pass 2) | Refine-stage sigmas — keep with two-pass. |
| #25 | `LTXVLatentUpsampler` (`spatial`) | Spatial upsample between pass 1 and pass 2. |
| #14, #15 | `RandomNoise` (seeds 43, 42, both `fixed`) | One per pass. Reusable. (Will need `AudioLoopController.start_seed` source post-graft.) |
| #8, #36 | `CFGGuider` (cfg=1) | Standard distilled-path guiders. |
| #18, #146 | `LTXVSeparateAVLatent` | A/V splitting — also used by canonical audio-loop. |
| #24, #34 | `LTXVConcatAVLatent` | A/V repacking — also used by canonical. |
| #2222 | `LTXVCropGuides` (between passes) | F3 chain element. |
| #2156 | `LTXVCropGuides` (post pass 2, pre decode) | F3 chain element. |
| #2182 | `LTXVAddGuideMulti` (pass-2 anchors, widget `['2', 0, 1, -1, 1]`) | Re-anchors first frame at upscaled res. Reusable. |
| #2107 | `Power Lora Loader (rgthree)` | Currently empty (no LoRAs configured). Reusable. |
| #149 | `LTXVTiledVAEDecode` `[1,1,1,True,"auto","auto"]` | Canonical 24GB-class config. |
| #150 | `LTXVAudioVAEDecode` | Standard A decode. |
| #43 | `VHS_VideoCombine` | Standard output node. (Tweak below.) |
| #228 | `LTXVChunkFeedForward` `[2, 4096]` | FFN chunking on by default. Keep. |
| #2296 | `AudioLoopHelperSageAttention` `["auto", True, 1024]` | Canonical attention node — exactly what the audio-loop variant wants. |
| #198 | `LTX2SamplingPreviewOverride` `[8]` | Preview override; canonical keeps it. |
| #197 | `LTX2_NAG` (see tweak below) | NAG node itself reusable; widget flagged. |
| #2297 | `LoadAudio` (`your_audio.mp3`) | **The headline win over B**: real audio source. Reusable; user must supply the actual audio file. |
| #2298 | `TrimAudioDuration` `[0.0, 4.0]` | Reusable as node; widget tweak below (0-4s is benchmark-short). |
| #2299 | `LTXVAudioVAEEncode` | Reusable — canonical audio VAE encode. |
| #122, #124, #117, #120, #133, #147, #148, #196, #2155, #2255 | `GetNode` for `model`, `clip`, `vae_audio`, `vae`, `upscale_model`, `vae`, `vae_audio`, `negative`, `vae`, `vae` | Canonical-variable getters that map cleanly to the audio-loop's named map (see §3 for rename plan). |
| #2154, #2163, #2166, #2167, #2259, #2260 | `GetNode` for `negative_guider`, `positive_guider`, `negative_guider2`, `positive_guider2` | Guider plumbing — reusable as-is. |
| #2164, #2165, #2223, #2224, #2233, #2215, #2214 | `SetNode` for `positive_guider2`, `negative_guider2`, `positive_guider`, `negative_guider`, `negative`, `latent_audio`, `Get_latent_audio` | Guider/conditioning routing — reusable (with `latent_audio` source rewired to `#2301 SetLatentNoiseMask` output, which already flows from `LTXVAudioVAEEncode`). |
| #199, #200, #201 | `Set_model_nag` / two `Get_model_nag` | NAG-wrapped model routing — reusable. |
| #230 | `Set_model_with_lora` | Routes patched model out. Reusable. |
| #188, #173, #172 | `Set_clip`, `Set_vae`, `Set_vae_audio` | Loader-output routing — reusable on rename (`vae` → `video_vae`, `vae_audio` → `audio_vae`). |
| #153, #154, #203, #204 | `Set_final_video`, `Set_final_audio`, `Get_final_video`, `Get_final_audio` | Output plumbing — reusable. |

### reusable-with-tweak

| Node | Type | Tweak |
|---|---|---|
| #1 | `KSamplerSelect("euler_ancestral_cfg_pp")` | **Replace widget value with `"euler"`** — root CLAUDE.md forbids `euler_ancestral*` on distilled. |
| #4 | `KSamplerSelect("euler_cfg_pp")` | **Replace widget value with `"euler"`** — same rule; CFG=1 makes `_cfg_pp` variants redundant anyway. |
| #186 | `LoraLoaderModelOnly` (`...lora-dynamic_fro09_avg_rank_111_bf16`, strength 0.6, `mode=4` bypassed) | **Leave bypassed** for canonical audio-loop production. User can toggle on for experimentation but not the default. |
| #197 | `LTX2_NAG` `[scale=11, alpha=0.25, tau=2.5, inplace=True]` | **Dial `scale` down to 3-7** per root CLAUDE.md. Recommend **5** as starting value. |
| #43 | `VHS_VideoCombine` (`frame_rate=24, filename_prefix='LTX-2', save_metadata=True, trim_to_audio=False`) | **Post-graft must run `apply_trim_image_batch_to_audio.py` (F14) + `apply_run_id_layout.py` (F15)** to splice `TrimImageBatchToAudio` into `.images` and `RunIdPrefix` into `.filename_prefix`. Update `filename_prefix` from `"LTX-2"`. Note: `videopreview.params.fullpath` carries a Windows path leak (`'E:\\AI\\ComfyUI\\output\\LTX-2_01647-audio.mp4'`) — scrub before commit. |
| #149 | `LTXVTiledVAEDecode` | Reusable; confirm `audit_workflows.py::vae_decode_no_tile` passes. |
| #228 | `LTXVChunkFeedForward` `[2, 4096]` | Reusable; no tweak strictly required. |
| #2296 | `AudioLoopHelperSageAttention` `["auto", True, 1024]` | Reusable; chain repositioning is the tweak — sage node must sit AFTER any model-mutating node. Current chain: `#187 UNETLoader → #186 LoraLoaderModelOnly (bypassed) → #2296 AudioLoopHelperSageAttention → #226 PathchSageAttentionKJ (bypassed) → #227 (bypassed) → #228 LTXVChunkFeedForward → #229 (bypassed) → #2107 Power Lora Loader → #192 Set_model`. That's correct order. |
| #9 | `LTXVEmptyLatentAudio` `[97, 24, 1]` (`mode=4` bypassed) | **Strip** — bypassed placeholder; var_d's real audio chain (`#2297-#2299`) feeds the live `Set_latent_audio` instead. |
| #2076 | `PrimitiveFloat` (`FPS = 24`) | **Keep at 24.0** — matches canonical (LTX 2.3 training-distribution). Source rewiring: ported workflow should feed fps via `LTXFramePlanner.fps_float` instead of this primitive, but value matches so no migration. |
| #2103 | `PrimitiveStringMultiline` `"video of a man dancing and singing"` | Strip after replacing positive-prompt source with `TimestampPromptScheduleBatchEncode`'s widget. |
| #91, #93, #137 | `GetNode "fps"` (3 copies) | If audio-loop supplies fps via `AudioLoopController.fps`, named-var Get/Set still works — reusable. Their upstream Set (`#2074 Set_fps` ← `#2076 PrimitiveFloat FPS`) needs source rewired to `AudioLoopController.fps` or `LTXFramePlanner.fps_float`. See §3. |
| #2298 | `TrimAudioDuration` `[0.0, 4.0]` | **Tweak widget to `[0.0, 600.0]`** (canonical default per `apply_fix_source_audio_trim_defaults.py`) so full-song audio fits. var_d's 4-second window is benchmark-tight. |

### strip

| Node | Type | WHY |
|---|---|---|
| #2 | `LTXVScheduler` `[8, 2.05, 0.95, True, 0.1]` | **Orphaned** — `output[0] SIGMAS` has no consumers (`→ []`). Dead-code. Strip. |
| #5 | `ManualSigmas` `"0.909375, 0.725, 0.421875, 0.0"` | **Orphaned** — `out[0] SIGMAS → []`. Dead. Strip. |
| #11 | `CLIPTextEncode` (negative prompt) | Replaced by canonical audio-loop's pre-encoded static negative. |
| #16 | `CLIPTextEncode` (positive, single prompt) | Replace by `TimestampPromptScheduleBatchEncode`. Strip the node. |
| #10 | `LTXVConditioning` (frame_rate=24) | Replaced by `TimestampPromptScheduleBatchEncode`'s output, which already stamps `frame_rate`. Strip. |
| #9 | `LTXVEmptyLatentAudio` `mode=4` | Strip — bypassed; real audio chain feeds `Set_latent_audio`. |
| #32 | `EmptyLTXVLatentVideo` `[768, 512, 97, 1]` (widget stale; overridden by Set/Get) | Strip — canonical audio-loop sizes empty video latent via `LTXFramePlanner` outputs. **Stale widget value: `768×512×97` is what the node SHOWS but inputs from `#2192 ComfyMathExpression a/2` (height 720/2=360 → snapped) + `#2191 a/2` (width 1280/2=640 → snapped) + `Get_frames`. Widget irrelevant at runtime — flag during port.** |
| #47, #2172 | `LoadImage` ("MIDDLE FRAME", "LAST FRAME"), both `benchmark_test_frame.png` | Single-keyframe variant — strip both. |
| #48, #2171 | `ImageResizeKJv2` (middle / last) | Strip with their corresponding LoadImage. |
| #49, #2168 | `ResizeImagesByLongerEdge` (middle / last) | Strip. |
| #2174 | `LTXVPreprocess` (middle path, `img_compression=18`) | Strip — middle frame removed. |
| #50 | `LTXVPreprocess` (last path, `img_compression=18`) | Strip — last frame removed. |
| #78, #2169 | `SetNode "middleframe"`, `SetNode "lastframe"` | Strip — only `firstframe` survives. |
| #2173, #2106, #2224 | `GetNode "middleframe"`, `Get "lastframe"` (multiple) | Strip middle/last gets. |
| #2107 (Pwr Lora) note: don't confuse with `#2108`, `#2109`, `#2278` | `PrimitiveFloat LAST FRAME STRENGTH=1`, `PrimitiveFloat MIDDLE FRAME STRENGTH=0.3`, MarkdownNote about strengths | Strip last+middle strength primitives + their note; keep `FIRST FRAME STRENGTH=0.7` (#2110) as the canonical `first_frame_guide_strength` value. |
| #2112, #2113, #2277 | `Set_firstframe_strength`, `Set_lastframe_strength`, `Set_middleframe_strength` | Strip middle/last variants; keep firstframe_strength. |
| #2217, #2218, #2219 | `Set_firstframe_resized`, `Set_middleframe_resized`, `Set_lastframe_resized` | Strip middle/last; keep firstframe. |
| #2185 | `Set_middleframe_count` | Strip. |
| #2187, #2188, #2189, #2276, #2279, #2280, #2281, #2226 | `Get_firstframe_strength` (×3), `Get_lastframe_strength` (×3), `Get_middleframe_strength` (×2) | Consolidate firstframe; strip rest. |
| #2191, #2192, #2216, #92, #2077 | `ComfyMathExpression`, `SimpleCalculatorKJ` (compute width/2, height/2, frames-from-length+fps) | Strip — math becomes implicit in `LTXFramePlanner` + `AudioLoopController` outputs. |
| #2072, #2073, #2074, #2075 | `Set_width`, `Set_height`, `Set_fps`, `Set_frames` | Strip — these are sourced from var_d's `INTConstant` widgets; audio-loop replacement gets these from `LTXFramePlanner`. |
| #2078, #2079, #2080 | `INTConstant LENGTH=15`, `HEIGHT=720`, `WIDTH=1280` | **Strip** — these are the dimension widgets that conflict with `LTXFramePlanner` SSoT. Audio length comes from audio source, not user input; resolution from planner. |
| #2076 | `PrimitiveFloat FPS=24` | Strip after retargeting (fps from planner). |
| #70, #71, #128, #129, #219, #220 | `Get_width` (×3), `Get_height` (×3) | Rewire to `LTXFramePlanner` outputs; redundancy strip — keep one of each at most, or remove the named-var indirection. |
| #127, #2175 | `Get_frames` (×2) | Strip — replaced by `AudioLoopController.frames_per_iter` / `LTXFramePlanner.actual_frames`. |
| #133, #193 | `Get_upscale_model`, `Get_vae_tiny` | `Get_upscale_model` reusable if two-pass kept (consumer of `#182`); `Get_vae_tiny` strip (preview-VAE bypassed). |
| #177 | `Set_vae_tiny` | Strip (preview-VAE source bypassed loader #180). |
| #180 | `VAELoader` `[taeltx2_3.safetensors]` `mode=4` | Strip — bypassed, file not present locally. |
| #189 | `DualCLIPLoaderGGUF` `mode=4` | Strip — alternate loader, bypassed. |
| #191 | `UnetLoaderGGUF` `mode=4` | Strip — alternate loader, bypassed. |
| #226 | `PathchSageAttentionKJ` `mode=4` | **Strip** — bypassed; replaced by `AudioLoopHelperSageAttention #2296`. |
| #227 | `LTX2MemoryEfficientSageAttentionPatch` `mode=4` | Strip — bypassed alternate sage patch. |
| #229 | `LTX2AttentionTunerPatch` `mode=4` | Strip — bypassed tuner. |
| #170, #174, #176, #179, #183, #184, #2109, #9000, #9001 | All `MarkdownNote` (Video settings, Prompting LTX-2, About Models, About User Made Loras, About Size, About Sampler Preview, About Last Frame Strength, Original workflow + credit, Adaptations + benchmark context) | Strip — benchmark-specific (RuneXX credit, sage measurement narrative, frame-strength tips). Replace with audio-loop-appropriate notes. **Note: #9000 + #9001 reference external author + GitHub fork by name; per global instructions ("no external-source attribution in public artifacts") these must NOT survive in the ported public-facing workflow.** |
| #2300 | `SolidMask` `[0, 512, 512]` | **Strip** — var_d's hand-rolled audio-freeze (mask=0 = freeze). Canonical uses `LTXVAudioVideoMask` (Node 606 in canonical) with `audio_start_time = audio_end_time = window_size` pattern. Replace per canonical. |
| #2301 | `SetLatentNoiseMask` | **Strip** — partners with #2300. Canonical noise-mask scheme is handled by `LTXVAudioVideoMask` + `LTXVConcatAVLatent`'s NestedTensor packing, not a separate setter. |

### replace (functionally needed but canonical uses different node)

| var_d node | Replace with | WHY |
|---|---|---|
| #16 `CLIPTextEncode` (positive single prompt) + #2103 `PrimitiveStringMultiline` + #10 `LTXVConditioning` | `TimestampPromptScheduleBatchEncode` (pre-encodes outside loop, `frame_rate` stamped) + `ConditioningSelectByIteration` (per-iter pluck inside loop body) | Root CLAUDE.md: "CLIP must not enter the loop body." |
| #11 `CLIPTextEncode` (negative prompt: "blurry, oversaturated, pixelated, ...") | Negative side of `TimestampPromptScheduleBatchEncode` or a single static negative cond pre-encoded outside loop (canonical pattern: Node 169 + 420 zero-out + 164 + 153 wiring). | Negative is static for audio-loop; pre-encode once. |
| #32 `EmptyLTXVLatentVideo` `[768, 512, 97, 1]` (widget stale) | `LTXFramePlanner → EmptyLTXVLatentVideo` (widgets driven from planner outputs) | Resolution/frames from planner SSoT. |
| #2300 `SolidMask` + #2301 `SetLatentNoiseMask` | `LTXVAudioVideoMask` (canonical Node 606) with `audio_start_time = audio_end_time = window_size` pattern | Canonical noise-mask scheme. Root CLAUDE.md: "Node 606 wiring is intentional — `audio_start_time = audio_end_time = window_size` (empty range keeps audio fixed)." |
| #44 `ImageResizeKJv2 [960, 544, 'lanczos', 'crop', ...]` (firstframe) + #2083 `ResizeImagesByLongerEdge [1536]` | `LTXSmartImageResize` (adaptive multi-stage with bicubic intermediates + lanczos final) | Root CLAUDE.md: aliasing + quantization. |
| #2078, #2079, #2080 INTConstants | `LTXFramePlanner` widgets | SSoT for dimensions per F8 audit (`frame_planner_present`). |
| #92, #2077, #2216, #2191, #2192 math nodes | `LTXFramePlanner` outputs (implicit) + `AudioLoopController` outputs | Math becomes implicit in planner/controller schemas. |
| #14, #15 `RandomNoise` (two passes) | Keep node type, rewire `noise_seed` to derive from `AudioLoopController.iter_seed` (canonical pattern) | Per-iteration seed deterministic from controller. |
| #1, #4 `KSamplerSelect("euler_*_cfg_pp")` | `KSamplerSelect("euler")` (plain) for both passes | Distilled-path rule. |
| #43 `VHS_VideoCombine` | Same node + post-graft `apply_trim_image_batch_to_audio.py` (F14) + `apply_run_id_layout.py` (F15) | F14/F15 invariants. |
| #186 `LoraLoaderModelOnly` (bypassed) | Leave bypassed OR consolidate on `Power Lora Loader (rgthree) #2107` | Two LoRA paths; consolidate. |
| Entire `Set_*`/`Get_*` named-var cluster for `firstframe`/`middleframe`/`lastframe`/`firstframe_strength`/etc. | Audio-loop named vars (`model`, `audio_vae`, `video_vae`, `actual_audio`, `orig_audio`, `full_audio_latent`, `sigmas`, `start_seed`) | See §3 for rename plan. |

### must-add (slots only — companion loop-machinery inventory has the full list)

Mirror of B's must-add list (canonical loop spine is identical for both port targets):

- `AudioLoopController` — upstream of pass-1 `RandomNoise #15` (drives `base_seed`), feeds `frame_rate` + iteration metadata.
- `AudioLoopPlanner` — upstream of `TensorLoopOpen`'s `iterations_in`.
- `LTXFramePlanner` — before `EmptyLTXVLatentVideo` (replacement for #32) + provides `actual_seconds` to `TrimAudioDuration.duration` (F8 autowire).
- `TensorLoopOpen` / `TensorLoopClose` — wrap pass-1 sampler region (per-iter denoise). Decision needed (see §4): loop pass-1 only, or both passes?
- `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` — replace conditioning region.
- `KeyframeLatentScheduleBatchEncode` + `LatentSelectByIteration` — optional, if keyframe timeline desired.
- `LatentContextExtract` / `LatentOverlapTrim` / `LatentConcat` — assembly path.
- `LTXVAudioVideoMask` (Node 606 equivalent) — replaces var_d's `SolidMask` + `SetLatentNoiseMask` pair.
- `ProfileBegin/IterStep/End` — optional, for bench variants.
- `RunIdPrefix`, `TrimImageBatchToAudio`, `TrimVideoLatentToAudio` — F14/F15 post-graft.

---

## §3 Named-variable map: var_d vs canonical

### var_d's Set/Get vocabulary

| Var | var_d's source | Disposition |
|---|---|---|
| `model` | `#192 Set_model` ← `#2107` Power Lora Loader output | **Keep** (canonical uses same `model` name) |
| `clip` | `#188 Set_clip` ← `#190` DualCLIPLoader | **Keep** |
| `vae` | `#173 Set_vae` ← `#181` VAELoader | **RENAME to `video_vae`** (canonical's name; var_d's bare `vae` is ambiguous) |
| `vae_audio` | `#172 Set_vae_audio` ← `#175` VAELoaderKJ | **RENAME to `audio_vae`** (canonical's name) |
| `vae_tiny` | `#177 Set_vae_tiny` ← bypassed `#180` | **STRIP** (preview-VAE bypassed) |
| `upscale_model` | `#171 Set_upscale_model` ← `#182` | **Keep** if two-pass refine kept |
| `model_nag` | `#199 Set_model_nag` ← `#197 LTX2_NAG` | **Keep** |
| `model_with_lora` | `#230 Set_model_with_lora` ← `#2107` rgthree LoRA loader | **Keep** |
| `negative` | `#2233 Set_negative` ← `#11` neg-CLIPTextEncode | **STRIP** (replaced by pre-encoded static negative) |
| `positive_guider`, `negative_guider`, `positive_guider2`, `negative_guider2` | `#2223, #2224, #2164, #2165` | **Keep** (two-pass guider plumbing) |
| `latent_audio` | `#2215 Set_latent_audio` ← `#2301 SetLatentNoiseMask` ← `#2299 LTXVAudioVAEEncode` ← `#2298 TrimAudioDuration` ← `#2297 LoadAudio` | **Keep var name. REWIRE** so source feeds canonical `LTXVAudioVideoMask` flow instead of var_d's `SolidMask`+`SetLatentNoiseMask` pair. |
| `firstframe`, `middleframe`, `lastframe` | `#75, #78, #2169` | **Keep `firstframe`, STRIP middle/last** |
| `firstframe_resized`, `middleframe_resized`, `lastframe_resized` | `#2217, #2218, #2219` | **Keep `firstframe_resized`, STRIP middle/last** |
| `firstframe_strength`, `middleframe_strength`, `lastframe_strength` | `#2112, #2277, #2113` | **Keep `firstframe_strength`, STRIP middle/last** |
| `middleframe_count` | `#2185` | **STRIP** |
| `width`, `height`, `fps`, `frames` | `#2073, #2072, #2074, #2075` (from `INTConstant` + `SimpleCalculatorKJ`) | **STRIP var_d's setters; canonical sources from `LTXFramePlanner`**. The named-var Gets stay; their setter changes to `LTXFramePlanner` output. |
| `final_video`, `final_audio` | `#153, #154` | **Keep** |

### Conflicts + rename plan (same as B)

| Conflict | Resolution |
|---|---|
| var_d `vae` ↔ canonical `video_vae` | Rename `Set_vae` / `Get_vae` (×5) to `Set_video_vae` / `Get_video_vae` |
| var_d `vae_audio` ↔ canonical `audio_vae` | Rename `Set_vae_audio` / `Get_vae_audio` (×2) to `Set_audio_vae` / `Get_audio_vae` |
| var_d has no `sigmas` named-var | Add `Set_sigmas`/`Get_sigmas` if two-pass kept; OR direct-wire. |
| var_d has no `start_seed` named-var | Add — wire `AudioLoopController.start_seed → Set_start_seed`; both `RandomNoise` nodes' `noise_seed` widgets fed via `Get_start_seed`. |
| var_d has no `actual_audio`/`orig_audio`/`full_audio_latent` | Must-add intermediate `Set` nodes wrapping the existing audio chain (`LoadAudio → Set_orig_audio`, `TrimAudioDuration → Set_actual_audio`, `LTXVAudioVAEEncode → Set_full_audio_latent`). |
| var_d's `width`/`height`/`fps`/`frames` setters from `INTConstant` + math | Rewire setters to `LTXFramePlanner` outputs; Gets stay. |

---

## §4 Two-pass refine wiring trace

Pass 1:
```
#15 RandomNoise (seed=42, fixed) ─→ link 16 ─→ #13 SamplerCustomAdvanced.noise
#36 CFGGuider ─→ link 17 ─→ #13 SamplerCustomAdvanced.guider
#1 KSamplerSelect("euler_ancestral_cfg_pp") ─→ link 18 ─→ #13.sampler
#215 ManualSigmas (9-step) ─→ link 308 ─→ #13.sigmas
#34 LTXVConcatAVLatent (assembles initial AV latent from empty video + #2214 Get_latent_audio) ─→ link 20 ─→ #13.latent_image
       ↑
       #2221 LTXVAddGuideMulti ['3', 0, 0.7, 0, 0.25, -1, 1] (3 anchors) ─→ link 4054 ─→ #34.video_latent
       ↑
       (gets positive/negative from #10 LTXVConditioning ← #16 CLIPTextEncode("video of a man dancing and singing"))

#13.output ─→ link 24 ─→ #18 LTXVSeparateAVLatent.av_latent
#18.video_latent ─→ link 3952 ─→ #2222 LTXVCropGuides.latent
#18.audio_latent ─→ link 51 ─→ #34 LTXVConcatAVLatent.audio_latent (cycled back into pass-2 init)
```

Pass-1 → Pass-2 upsample bridge:
```
#2260 Get_positive_guider, #2259 Get_negative_guider ─→ #2222 LTXVCropGuides.positive, .negative
#2222.latent ─→ link 3953 ─→ #25 LTXVLatentUpsampler.samples
#133 Get_upscale_model ─→ link 199 ─→ #25.upscale_model
#120 Get_vae ─→ link 182 ─→ #25.vae
#25.LATENT ─→ link 4097 ─→ #2182 LTXVAddGuideMulti.latent
#2182 LTXVAddGuideMulti ['2', 0, 1, -1, 1] (2 anchors: first + last @ upscaled res) ─→ link 4084 ─→ #34 LTXVConcatAVLatent.video_latent
```

Pass 2:
```
#14 RandomNoise (seed=43, fixed) ─→ link 28 ─→ #21 SamplerCustomAdvanced.noise
#8 CFGGuider ─→ link 29 ─→ #21.guider
#4 KSamplerSelect("euler_cfg_pp") ─→ link 30 ─→ #21.sampler
#216 ManualSigmas (4-step "0.85, 0.7250, 0.4219, 0.0") ─→ link 309 ─→ #21.sigmas
#32 EmptyLTXVLatentVideo (width=Get_width/2, height=Get_height/2, length=Get_frames) ─→ link 4053 ─→ #2221.latent
   (then through #2221 ─→ #34 LTXVConcatAVLatent ─→ #21.latent_image)
   NOTE: #32's widget [768, 512, 97, 1] is STALE — overridden by Set/Get inputs.
   NOTE: width/height go through /2 math (#2191, #2192) — pass-1 runs at HALF the declared resolution, pass-2 doubles back via LTXVLatentUpsampler.

#21.output ─→ link 269 ─→ #146 LTXVSeparateAVLatent.av_latent
#146.video_latent ─→ link 3936 ─→ #2156 LTXVCropGuides.latent
#146.audio_latent ─→ link 219 ─→ #150 LTXVAudioVAEDecode.samples
```

Decode:
```
#2156.latent ─→ link 4099 ─→ #149 LTXVTiledVAEDecode.latents
#147 Get_vae ─→ link 302 ─→ #149.vae
#149.image ─→ link 304 ─→ #153 Set_final_video
#150.Audio ─→ link 230 ─→ #154 Set_final_audio
```

**Sigma routing confirmed**:
- `#215 ManualSigmas (9-step) → #13 SamplerCustomAdvanced (pass 1)` ✓
- `#216 ManualSigmas (4-step) → #21 SamplerCustomAdvanced (pass 2)` ✓
- `#5 ManualSigmas` → orphan (out → []) — DEAD
- `#2 LTXVScheduler` → orphan (out → []) — DEAD

**Resolution chain**:
- `#2080 INTConstant WIDTH=1280` → `Set_width` → `Get_width #220` → `#2191 a/2 = 640` → `#32 EmptyLTXVLatentVideo.width`
- `#2079 INTConstant HEIGHT=720` → `Set_height` → `Get_height #219` → `#2192 a/2 = 360` → `#32.height`
- Pass-1 latent dimensions: **640×360** (half of 1280×720).
- After `LTXVLatentUpsampler`: **1280×720** for pass-2.

**Frames chain**:
- `#2078 INTConstant LENGTH=15` (seconds) → `#2077 SimpleCalculatorKJ` (with `b = #2076 PrimitiveFloat FPS=24`) → formula `((round((15 * 24 - 1) / 8)) * 8) + 1 = ((round(44.875)) * 8) + 1 = (45 * 8) + 1 = 361` frames → `Set_frames` → consumers.

---

## §5 Build-script implications

High-level steps to port var_d → audio-loop production:

1. **Load var_d with `WorkflowEditor`**; deep-copy to working state.
2. **Strip dead defs**: clear `definitions.subgraphs` (cosmetic).
3. **Strip middle/last frame branches** (#47, #2172, #48, #2171, #49, #2168, #2174, #50, #78, #2169, #2173, #2106, #2185, plus their Set/Get of `middleframe_*` and `lastframe_*` and strength primitives #2108, #2278).
4. **Strip alternate loaders**: #180, #189, #191, #177, #193.
5. **Strip bypassed sage patches**: #226, #227, #229. Keep #228 + #2296.
6. **Strip dead sigma/scheduler nodes**: #2, #5.
7. **Strip benchmark MarkdownNotes**: all 9 (including #9000 + #9001 which carry external-attribution content that must not survive porting per global instructions).
8. **Strip conditioning region**: #11, #16, #10, #2103.
9. **Strip empty-latent + INTConstant resolution plumbing**: #9 (bypassed), #32, #2078, #2079, #2080, #2076 (after retargeting), #2072-#2075, #92, #2077, #2191, #2192, #2216.
10. **Strip var_d's hand-rolled audio-freeze**: #2300 `SolidMask`, #2301 `SetLatentNoiseMask`. Replace with canonical `LTXVAudioVideoMask` (Node 606 pattern).
11. **Rename Set/Get vars**: `vae` → `video_vae`, `vae_audio` → `audio_vae` (use `WorkflowEditor` widget rewrites; ~10 Get + 2 Set nodes).
12. **Replace sampler widgets**: #1 `"euler_ancestral_cfg_pp"` → `"euler"`; #4 `"euler_cfg_pp"` → `"euler"`.
13. **Tweak NAG widget**: #197 `scale=11` → `scale=5`.
14. **Tweak TrimAudioDuration**: #2298 `[0.0, 4.0]` → `[0.0, 600.0]` (canonical default).
15. **Add audio-chain `Set` wrappers**: `LoadAudio #2297 → Set_orig_audio`, `TrimAudioDuration #2298 → Set_actual_audio`, `LTXVAudioVAEEncode #2299 → Set_full_audio_latent`. Rewire `Set_latent_audio #2215` source from the (stripped) `SetLatentNoiseMask` to the canonical `LTXVAudioVideoMask`-bearing flow.
16. **Add `LTXFramePlanner`** + wire its outputs to: `EmptyLTXVLatentVideo` (re-added via canonical pattern), `Set_width`/`Set_height`/`Set_fps`/`Set_frames` (or direct-wire), `TrimAudioDuration.duration` (F8 autowire).
17. **Add audio-loop spine**: `AudioLoopController` + `AudioLoopPlanner` + `TensorLoopOpen`/`TensorLoopClose` + `LatentContextExtract` + `LatentOverlapTrim` + `LatentConcat`. Wrap pass-1 sampler (#13). Decision-gated: optionally wrap pass-2 (#21) or leave outside.
18. **Add conditioning region**: `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration`. Wire to both `CFGGuider` positive/negative.
19. **Add `RandomNoise` seed wiring**: rewire #14/#15 `noise_seed` widgets to `AudioLoopController.iter_seed`.
20. **Post-graft apply-script chain** (in order):
    1. `scripts/apply_loop_guide_preprocess_symmetry.py` (F2)
    2. `scripts/apply_loop_cropguides_symmetry.py` (F3)
    3. `scripts/apply_trim_video_latent_to_audio.py` (F14 latent half)
    4. `scripts/apply_trim_image_batch_to_audio.py` (F14 image half)
    5. `scripts/apply_run_id_layout.py` (F15)
    6. `scripts/audit_workflows.py` — must exit 0.
21. **Scrub privacy leaks**: #43 `VHS_VideoCombine.videopreview.params.fullpath` carries a Windows absolute path (`E:\AI\ComfyUI\output\LTX-2_01647-audio.mp4`). Wipe the `videopreview` dict or null its `fullpath`/`filename`. Same for `LoadAudio #2297` widget if user-supplied path is committed.
22. **Layout pass**: run `scripts/apply_layout_polish_audio_loop_latent.py --from-template` (or newer equivalent).
23. **Save to**: `example_workflows/<new_name>.json`. Must pass `audit_workflows.py`.

### What var_d brings that B doesn't (lower port cost)

- **Real audio chain already wired** (`LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode`) — saves grafting that whole chain from canonical. Need only swap the audio-freeze mechanism (`SolidMask`+`SetLatentNoiseMask` → `LTXVAudioVideoMask`) and add the `Set_orig_audio`/`Set_actual_audio`/`Set_full_audio_latent` wrappers.
- **Dynamic-dimension Set/Get plumbing** structurally matches canonical's `LTXFramePlanner` SSoT model — rename `INTConstant`-driven Set/Get to `LTXFramePlanner`-driven and the consumers stay.
- **fps=24 already correct** — no migration vs canonical pre-2026-05-15 default-25 state.

### What's the same as B (same cost)

- Two-pass refine (`#13` + `#21` + `LTXVLatentUpsampler`) — same Option A/B/C decision (loop pass-1 only / drop pass-2 / ship both siblings).
- Sample sampler-violation (`euler_*_cfg_pp` widgets need `"euler"`).
- NAG `scale=11` needs dialing to ~5.
- Multi-keyframe init topology — same strip-to-single-keyframe work.

### Open questions (need user's call)

1. **Two-pass refine: keep or drop?** Same options as B (A: loop pass-1 only; B: drop pass-2; C: ship two siblings).
2. **Pass-1 runs at half-res, pass-2 doubles**: var_d's #2191/#2192 `/2` math means pass-1 operates at 640×360 and pass-2 upsamples to 1280×720. Confirm this is the intended pattern in the ported workflow (it's standard for `LTXVLatentUpsampler`).
3. **`LoraLoaderModelOnly #186`**: leave bypassed (matches var_d default). Flag for user.
4. **`LTX2_NAG.scale`**: 11 → 5 default; user can tune.
5. **Init image**: `benchmark_test_frame.png` placeholder → generic placeholder name + note pointing to user replacement.
6. **Negative prompt**: var_d's #11 has full benchmark negative — preserve or zero-out? Same recommendation as B (zero-out for cleanliness; preserve in comment).
7. **TrimAudioDuration window**: 0-4s for bench → 0-600s for full song (or driven from `LTXFramePlanner.actual_seconds` via F8 autowire).
8. **Subgraphs in `definitions.subgraphs`** (`PROMPT ENHANCER`, `Frames split view`) — dead, strip.
9. **MarkdownNotes #9000 + #9001 carry external-attribution + fork-name + GitHub URL** — strip per global instruction "no external-source attribution in public artifacts." Replace with audio-loop-appropriate notes about prompt scheduling, audio path, F-pair compliance.
10. **Resolution target**: var_d ships 1280×720 (div-by-32 for height? 720/32=22.5 — **not** div-by-32). Per root CLAUDE.md "resolution div-by-32 (single-stage)" the audit will flag 720; planner-snapped value is likely 1280×704. Confirm during port.

Read-only confirmation: nothing was mutated in var_d or any other JSON during this inventory pass.
