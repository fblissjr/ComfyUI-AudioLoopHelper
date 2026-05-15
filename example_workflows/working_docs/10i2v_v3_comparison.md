Last updated: 2026-05-15

# 10i2v v3.3_likeness vs v3_tilersampler — structural comparison

Two scratch single-shot i2v workflows the user is evaluating for patterns to port into the audio-loop family. Both are forks of the same `10Eros_v1-fp8mixed_learned.safetensors` i2v base + same distilled LoRA stack + same two-stage (first-pass STG + tiled-sampler upscale) topology — but they diverge on conditioning routing, init-image anchoring, and per-stage widget values.

Source files (gitignored — private clone only):
- `internal/scratch/10i2v_v3.3_likeness.json` (100.3 KB, 91 nodes, 115 links)
- `internal/scratch/10i2v_v3_tilersampler.json` (96.2 KB, 91 nodes, 115 links)

Privacy: scanned for `/home/`, `/Users/`, `~/`, Windows-style user paths, and the maintainer username. **No private path leakage found** in either workflow JSON. LoRA paths use Windows-style relative paths (`ltx23\\<lora>.safetensors`) — fine for both clones since they resolve against ComfyUI's `models/loras/` root.

## §1 Side-by-side summary table

| Axis | v3.3_likeness | v3_tilersampler | Verdict |
|---|---|---|---|
| **1. Sampler — first pass** | `SamplerCustomAdvanced` #510 + `KSamplerSelect euler_ancestral` #520 + `STGGuiderAdvanced` #653 (cfg=1, scale `2,1.5,1,1,...`) + `Sigmas Easing cubic in_out 0.7` #652 over base sigmas | **(same)** — identical sigmas, sampler, easing, guider config | same |
| **1. Sampler — upscale pass** | `LTXTiledSampler` #802 + `KSamplerSelect euler_ancestral_cfg_pp` #585 + `CFGGuider cfg=1` #583 + `ManualSigmas '0.715, 0.4824, 0.2412, 0.0'` #582 (3-step) | `LTXTiledSampler` #788 + `KSamplerSelect euler_ancestral_cfg_pp` #585 + `CFGGuider cfg=1` #583 + `ManualSigmas '0.85, 0.7250, 0.4219, 0.0'` #582 (3-step) | **differs** — same shape, different upscale-pass sigma profile |
| **1. Base first-pass sigmas** | `1.0,0.99375,0.9875,0.98125,0.9550,0.8925,0.8120,0.7150,0.6030,0.4824,0.3618,0.2412,0.1206,0.0` (13 steps) | **(same)** | same |
| **2. Resolution + frames** | `mxSlider Video Width=1120`, `Height=1344`, `Length=240` (10s @ 24fps), latent canvas `EmptyLTXVLatentVideo [704,64,97,1]` (placeholder; real dims from ImageResizeKJv2 chain) | `Width=1024`, `Height=1344`, `Length=264` (11s @ 24fps), same latent canvas placeholder | **differs** — v3.3 wider (1120) shorter (240F); v3_ts narrower (1024) longer (264F) |
| **2. Frame rate** | `LTXVConditioning frame_rate=24` #523 + `PrimitiveFloat=25` #542 (unused / legacy display) + `VHS_VideoCombine frame_rate=24` | **(same)** | same |
| **3. Init-image / first-frame anchor** | `LTXVImgToVideoInplaceKJ` #770 (`['1', 1, 0]`) + #772 (`['1', 0.75, 0]`) + `mxSlider First Frame Strength=0.77` #797 (unused widget value but title is `First Frame Strength`) | `LTXVImgToVideoInplaceKJ` #770 (`['1', 1, 0]`) + #772 (`['1', 0.75, 0]`) + `mxSlider Conditioning Strength=0.8` #797 wired into `LTXVAddGuide #767.strength` | **differs** — v3_ts wires the mxSlider; v3.3 leaves it as title-only display |
| **3. Likeness / identity anchoring** | `LTXFaceDetector` #808 (threshold=0.15) → `LTXLikenessGuide` #806 (weight=0.9, mode=`silent_reference`+`manual`+`bbox_only`, strength=0.15, fade=24f) → `LTXLikenessAnchor` #827 (strength=0.5, region=`middle`, blend=`directional 0.4`) → STGGuider model input; `LTXLatentAnchorAware` #731 also stacked on model | NO likeness/face nodes; uses `LTXVAddGuide #767` + `LTXVCropGuides #734` (standard core LTX wiring) + `LTXLatentAnchorAware` #731 + `LTXTextAttentionAmplifier` #753 (scale=1.3, threshold=0.15, layers `36-48`) | **unique-to-A** (likeness stack) vs **unique-to-B** (text-attention amplifier) |
| **4. Conditioning routing — first pass** | `LTXVConditioning #523` → `LTXLikenessGuide #806` (positive+negative) → `STGGuiderAdvanced` | `LTXVConditioning #523` → `LTXVAddGuide #767` → `STGGuiderAdvanced` | **differs** — v3.3 routes through likeness module; v3_ts through stock AddGuide |
| **4. Conditioning routing — upscale pass** | `LTXLikenessGuide #806` → `CFGGuider` (same node feeds both stages) | `LTXVAddGuide #767` → `LTXVCropGuides #734` → `CFGGuider` (crop strips guides on second pass) | **differs** — v3_ts uses `LTXVCropGuides` between stages (F3 pattern from our project); v3.3 doesn't |
| **4. Negative prompt** | `CLIPTextEncode #537` 369-char negative ("captions, music, transition, VR, bad quality, subtitles, watermark, link, text, blur...") | `CLIPTextEncode #537` 372-char negative — substantively identical, no `link` token, minor wording | same in shape, near-identical content |
| **4. Positive prompt** | `CLIPTextEncode #536` empty string — positive comes from `LTXLikenessGuide` itself (reference image carries semantics) | `CLIPTextEncode #536` empty string — positive comes from `LTXVAddGuide` (init image carries semantics) | same (both empty-text, image-only conditioning) |
| **5. Attention / sage** | NO `AudioLoopHelperSageAttention`, NO `LTX2_NAG`. `LTXLatentAnchorAware` and `LTXLikenessAnchor` patch the MODEL upstream of guider | NO sage, NO NAG. `LTXLatentAnchorAware` + `LTXTextAttentionAmplifier #753` (layers 36-48 amplification scale 1.3) | **differs** — v3_ts adds `LTXTextAttentionAmplifier` on the CFG path |
| **5. LoRA stack** | 4× `LTX2LoraLoaderAdvanced` (distilled lora 1.1 fro90_ceil72_condsafe, alternating widget patterns `[1,1,0,0,0,1]` and `[1,0,1,1,1,0]`); 1× `Power Lora Loader (rgthree)` with `OmniNFT_converted_lora @ strength 0.8` active | Identical 4× `LTX2LoraLoaderAdvanced` stack; `Power Lora Loader (rgthree)` present but with empty entries (no active LoRA) | **differs** — v3.3 stacks an additional OmniNFT lora @0.8 |
| **5. FFN chunking** | NO chunking node in either | NO chunking node in either | same |
| **6. VAE decode** | `VAEDecode` (core, single-pass — NOT `LTXVTiledVAEDecode`) #552 + #740; `LTXVAudioVAEDecode` #550, #593 for audio | **(same)** | same — both miss our project's F-pair preference for `LTXVTiledVAEDecode [1,1,1]` |
| **7. Tiled sampling** | `LTXTiledSampler #802` widgets `[bypass_tiling=False, tile_axis='auto', n_tiles=2, tile_overlap=8, max_size_for_no_tile=24, audio_pass='tile_carrying', audio_carrier_tile='first', debug=False]` | `LTXTiledSampler #788` widgets `['auto', 2, 4, 38, 'tile_carrying', 'first', False]` (different widget schema — appears to be a newer revision missing the leading `bypass_tiling` slot, i.e. 7 widgets vs 8) | **differs** — v3_ts uses `tile_overlap=4 max_size_for_no_tile=38` (v3.3 uses `8` / `24`); also v3_ts's node may be a newer schema rev |
| **7. Latent upsampler** | `LTXVLatentUpsamplerTiled #744` `[tile_size=11, overlap=6, max_size_for_no_tile=22, rotate_for_landscape=False, debug=False]` | **(same)** | same |
| **7. Spatial super-res** | `RTXVideoSuperResolution #755` (`'scale by multiplier', 1.3, 'ULTRA'`) | `RTXVideoSuperResolution #755` (`'scale by multiplier', 1.3, 'HIGH'`) | **differs** — v3.3 uses `ULTRA` quality preset, v3_ts uses `HIGH` |
| **8. Two-pass refine + upscale pattern** | YES: stage 1 = `SamplerCustomAdvanced` (13-step distilled with `Sigmas Easing cubic in_out 0.7`) at low res; stage 2 = `LTXVLatentUpsamplerTiled` then `LTXTiledSampler` (3-step refine, starting sigma 0.715) | YES: same shape; stage 2 refine uses `'0.85, 0.7250, 0.4219, 0.0'` (starting sigma 0.85 — more refining work) | **differs** — v3_ts upscale pass starts at a higher sigma (0.85 vs 0.715), so does noticeably more re-denoising at upscale |
| **8. Audio handling** | `LTXVEmptyLatentAudio [1,24,1]` (empty audio latent) — both stages have `LTXVAudioVAEEncode`→`LTXVConcatAVLatent` then `LTXVSeparateAVLatent`→`LTXVAudioVAEDecode` | **(same)** | same — both keep audio path intact across both sampler stages (V2A pattern) |
| **9. Misc — face detector** | `LTXFaceDetector #808` (threshold=0.15) wired to both `LTXLikenessGuide` and `LTXLikenessAnchor` for bbox-aware identity locking | absent | unique-to-A |
| **9. Misc — text attention amplifier** | absent | `LTXTextAttentionAmplifier #753` patches MODEL between LoRA stack and CFGGuider (upscale pass only — scale 1.3 on layers 36-48) | unique-to-B |
| **9. Misc — sigmas-easing on first pass** | `Sigmas Easing` #652 (`cubic in_out 0.7`) sits between `ManualSigmas` and `SamplerCustomAdvanced` — easing applied to first-pass sigmas only | **(same)** | same |
| **9. Misc — node title** | `Sigmas Easing` title: `"Sigmas Easing (cubic, sine, quad) don't mess with this tbh"` | **(same)** | same — author left a self-note (treat as load-bearing) |

**Sampler-name violation check**: BOTH workflows use `euler_ancestral` (first pass) and `euler_ancestral_cfg_pp` (upscale pass). Our project's CLAUDE.md forbids `euler_ancestral*` for the audio-loop variants. These scratch workflows are single-shot i2v, not loop-body — the rule may not extend, but flag for the user.

## §2 What's in v3_tilersampler that v3.3_likeness doesn't

The `tilersampler` name turns out to be a **misnomer for the diff** — both workflows have `LTXTiledSampler` for the upscale pass. The actual v3-distinctive shape is the **standard stock-LTX guide topology** in place of the likeness stack:

- **`LTXVAddGuide #767` + `LTXVCropGuides #734` pair**. This is the canonical LTX-Video core conditioning topology — `AddGuide` injects the init-image latent at index 0 with strength 0.75 (widget `[0, 0.7499999999999999]`); `CropGuides` strips those guides between first-pass and upscale-pass. Same pattern our audio-loop variants use inside the loop body (F3 invariant — `LTXVCropGuides` between loop sampling and CFG guider).
- **`LTXTextAttentionAmplifier #753`** on the MODEL just before `CFGGuider`. Widgets `[1.3, 0.15, '36-48', False, False]` — amplifies text cross-attention by 1.3× on transformer layers 36-48 (late layers), threshold 0.15. Only active for the upscale pass (it patches the CFG-pass model, not the STG-pass model).
- **Conditioning Strength mxSlider #797 is actually wired** (to `LTXVAddGuide.strength` via slot s6). v3.3's equivalent mxSlider is title-only / display-only.
- **Upscale sigma profile starts higher**: `'0.85, 0.7250, 0.4219, 0.0'` vs v3.3's `'0.715, 0.4824, 0.2412, 0.0'`. v3_ts does materially more denoising work in the upscale pass.
- **Tile geometry tuned tighter**: `n_tiles=2, tile_overlap=4, max_size_for_no_tile=38` vs v3.3's `2 / 8 / 24`. Lower overlap + higher no-tile threshold = fewer tile seams, more single-pass cases.
- **Lower-resolution canvas** (1024×1344 vs 1120×1344) — likely a paired choice with the higher-effort upscale (start with smaller, refine harder).
- **Longer**: 264 frames (11s @24fps) vs 240 frames (10s).

## §3 What's in v3.3_likeness that v3_tilersampler doesn't

"Likeness" = a dedicated identity-anchoring stack independent of stock LTX guides:

- **`LTXLikenessGuide #806`** replaces `LTXVAddGuide` as the positive/negative conditioning source. Widgets `[0.9, 'silent_reference', 'manual', 'bbox_only', 0.15, 24, 0, 'area', 'center', 1, '', 'passthrough', False]` — weight 0.9, silent-reference mode, bbox-only conditioning, fade-in over 24 frames. Outputs `(positive, negative, latent, reference_info)`.
- **`LTXFaceDetector #808`** (threshold 0.15) runs on the resized init image → emits `face_bbox` → fed to BOTH `LTXLikenessGuide.face_bbox_within_reference` AND `LTXLikenessAnchor.frame_0_bbox`. So identity is bbox-locked, not whole-frame.
- **`LTXLikenessAnchor #827`** patches the MODEL with reference-aware attention. Widgets `[0.5, 'auto', '', 0.5, 0, False, False, False, 'middle', '', 8, '', 0, 'directional', 0.4]` — strength 0.5, region `middle`, blend `directional 0.4`.
- **Stacked over `LTXLatentAnchorAware #731`** — so model is patched twice (first latent-anchor-aware, then likeness-anchor). Order matters for `state_dict()` callers (our CLAUDE.md note).
- **`Power Lora Loader (rgthree)` has an active LoRA**: `OmniNFT_converted_lora @ strength 0.8` (in v3_ts the loader is present but the entry list is empty — disabled).
- **`LTXLatentAnchorAware` configured differently**: v3.3 uses `[..., 'schedule', 2, 50, 0, 'flat', '']` (schedule 2, 50 steps, empty layer string) vs v3_ts's `[..., 'schedule', 1, 432, 0, 'flat', '10-30']` (schedule 1, 432 steps, layers 10-30 targeted). Two materially different schedule policies.
- **`RTXVideoSuperResolution`** at `ULTRA` quality vs v3_ts's `HIGH`.
- **Width 1120** (vs 1024) — pairs with the lower-sigma upscale start (less re-denoising, so start with more pixels).

## §4 Patterns worth considering for the audio-loop variants

| Pattern | Source | Port? | Why / why not |
|---|---|---|---|
| `LTXLikenessGuide` + `LTXFaceDetector` + `LTXLikenessAnchor` stack | v3.3 | **Maybe — high value** | Cross-iteration identity drift is a known audio-loop failure mode (`Illustrated inits drift toward photoreal across iterations`, root CLAUDE.md). A bbox-locked face-aware identity anchor is directly applicable. Open question: do these nodes work with the IC-LoRA video-ref path (F12), and can they be applied once outside the loop or do they need re-evaluation per iteration? |
| `LTXTextAttentionAmplifier` on upscale-pass model | v3_ts | **Defer — narrow use** | Late-layer (36-48) text attention amplification for upscale. Audio-loop variants don't run an upscale pass inline; this would only apply to the `build_upscale_workflow.py` post-loop spatial-upscale. Worth testing there. Loop body itself probably shouldn't amplify text — text token budget is already over-loaded. |
| Two-stage sampler split (low-res 13-step STG + high-res 3-step `LTXTiledSampler` refine) | both | **Defer — different problem class** | Single-shot 10s i2v can afford a high-res refine pass. Audio-loop spends 10× more compute already; adding a second sampler pass inline would push 24GB cards over budget. Reserve for `build_upscale_workflow.py`. |
| `Sigmas Easing cubic in_out 0.7` between `ManualSigmas` and sampler | both | **Test — low-risk** | Our canonical 8-step distilled sigmas may benefit from an easing layer. Note: our chain is shorter (8 steps vs 13 here); easing curve may matter less. Worth a one-prompt A/B. |
| `LTXVCropGuides` between first-pass and upscale-pass sampler (already F3 inside our loop) | v3_ts | **Confirm parity** | This is our F3 invariant, but applied at a different boundary (between sampling stages, not just before CFGGuider). v3_ts shows it generalizes. No change needed for our loop body. |
| `LTXTiledSampler` with `audio_pass='tile_carrying' audio_carrier_tile='first'` for upscale | both | **Adopt in `build_upscale_workflow.py`** | Currently our upscale workflow uses single-pass tiled VAE decode + KSamplerSelect. `LTXTiledSampler` with audio-carrying tiles is the LTX-Video-native way to keep AV alignment across tiles. Worth investigating. |
| 13-step base sigmas `1.0,0.99375,...,0.0` (vs our 8-step `1.0,...,0.0`) | both | **Skip** | We deliberately use the 8-step Lightricks distilled path; 13 steps doubles compute. The first 4 steps match ours (`1.0, 0.99375, 0.9875, 0.98125`); they extended with intermediate values. Our path is correct for distilled. |
| Disable `LoRA` via empty entries in Power Lora Loader instead of removing the node | v3_ts | **Already do this** | We use `mode=4` bypass; same intent. No change. |

## §5 Patterns we should NOT adopt

- **`euler_ancestral` / `euler_ancestral_cfg_pp`**. CLAUDE.md explicitly bans `euler_ancestral*` for our distilled 8-step path; the canonical sampler is `euler`. The ancestral variants inject noise per-step, which destroys the deterministic seam topology that lets our loop-body iterations re-cohere. Single-shot i2v can tolerate ancestral; audio-loops can't.
- **`VAEDecode` (core, single-pass) instead of `LTXVTiledVAEDecode`**. Both scratch workflows use core `VAEDecode`. Our F-pair convention requires `LTXVTiledVAEDecode [1,1,1,true,"auto","auto"]` on 24GB+ (3× faster cold-pass). Don't port the core decoder.
- **`PrimitiveFloat=25` "Base Frame Rate (24 Default)"** widget — dangling unused value that contradicts the `LTXVConditioning frame_rate=24` actual config. Cosmetic-trap; don't replicate in our workflows.
- **`LTXTiledSampler` widget-schema drift** between #802 and #788 (8 vs 7 widgets). One of these node revs has a leading `bypass_tiling` BOOL the other doesn't — if we ever pull this node into a shipped variant, audit the schema first.
- **Windows-style backslash paths in LoRA names** (`'ltx23\\<file>.safetensors'`) — works on both platforms via ComfyUI's path normalization, but inconsistent with our forward-slash convention. Use `/` everywhere.
- **Dual `LTXVConcatAVLatent` + `LTXVSeparateAVLatent` instances per stage** — necessary for two-stage sampling but the audio-loop variants get away with one because the loop body re-concats per iteration. Don't refactor toward two-stage; we'd need it for the upscale path only.
- **`LoadImage` with the specific webp/png filenames currently in the workflows** — both are likely test/reference assets, not anything to commit. (Not actually in our public surface — flagging only.)

## §6 Resolution / dimension findings — direct relevance to 960×544 flip

The user just flipped the audio-loop variants to **960×544**. Neither scratch workflow uses 960×544; both use much larger canvases:

| Workflow | Effective canvas (mxSlider) | Aspect | Latent vol after upscale (rough) |
|---|---|---|---|
| v3.3_likeness | 1120 × 1344 × 240F | 5:6 portrait | ~30M latent voxels at first pass (before upsampler) |
| v3_tilersampler | 1024 × 1344 × 264F | ~3:4 portrait | ~28M latent voxels at first pass |
| our audio-loop (960×544) | 960 × 544 × ~145F per iter | 16:9 landscape | ~3M per iter × N iter |

**Key findings**:

- Both scratch workflows are **portrait orientation, single-character-focused** (matches the i2v + face-anchor use case — full-body or torso-shot). 960×544 is landscape (16:9), explicitly multi-element scene framing. The likeness/face-detector stack is **specifically designed for portrait single-subject** and may not transfer cleanly to landscape multi-subject layouts.
- **The `EmptyLTXVLatentVideo [704, 64, 97, 1]` widget in both workflows is a stale placeholder** — actual dimensions flow through the `ImageResizeKJv2 #531` → `GetImageSize` → `LTXVImgToVideoInplaceKJ` chain from the mxSlider values, not the latent node. Our project's LTXFramePlanner pattern is cleaner; we shouldn't port this placeholder pattern.
- **No latent-volume budget enforcement** in either workflow. They rely on the user catching OOM via mxSlider; ours uses `LTXFramePlanner` snap-to-budget. The 1120×1344×240 canvas at full upscale (1.3× via RTXVSR) is ~6.5M pixels × 240F = ~1.5B pixels — likely needs >24GB or runs offloaded. The scratch workflows don't have any visible guard.
- **Both use 24fps `LTXVConditioning`** (correct per CLAUDE.md F16) but have a `PrimitiveFloat=25` widget titled "Base Frame Rate (24 Default)" — a stale display value that doesn't drive anything. Confirms our F16 sweep was the right call.
- **`Length=240` (10s) in v3.3 vs `Length=264` (11s) in v3_ts** matches naming `10i2v` if the v3.3 is "10s exactly" and v3_ts overshoots by a frame margin (intentional? unclear).

## Which feels more polished / recent

**v3.3_likeness** is the more polished / production-feeling variant:
- Coherent named purpose: identity-anchoring stack is a single integrated subsystem (FaceDetector → LikenessGuide → LikenessAnchor) wired with intent.
- Power Lora Loader has an active entry (vs empty in v3_ts), suggesting active experimentation/tuning.
- Higher canvas (1120×1344) paired with lower-sigma upscale (less re-denoising) reads as deliberate quality-budget allocation.
- "v3.3" minor-version > "v3" — typical for iterative refinement on a stable base.

**v3_tilersampler** feels like an earlier branch:
- Standard core-LTX guide topology (`LTXVAddGuide + LTXVCropGuides`) without the identity-anchoring extension.
- `LTXTextAttentionAmplifier` looks like a one-off experiment (not present in v3.3 — suggests it was tried, then dropped in favor of likeness anchoring).
- Higher upscale-pass starting sigma (0.85) reads as compensating for weaker conditioning (more re-denoising recovers detail).
- The name "tilersampler" suggests the file was named when the user was actively exploring `LTXTiledSampler` widget tuning — that exploration appears settled in v3.3.

Both are clearly **derived from the same parent** (identical 91-node count, identical first-pass sigmas, same LoRA stack, same audio path, near-identical negative prompts, identical TwoWaySwitch / ComfyMathExpression / GetImageSize plumbing). The fork point is the conditioning-routing layer.
