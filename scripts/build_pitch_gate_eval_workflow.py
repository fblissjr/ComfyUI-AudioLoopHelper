"""Build the pitch-gate eval workflow template from the PROVEN stock single-stage A+V graph.

Strategy (see internal/audio_iclora_training/pitch_gate_eval_workflow_build_plan.md):
copy ComfyUI-LTXVideo's LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json (ships + runs,
does joint A+V generation), keep ONLY its distilled sampler chain, drop the quality chain
and the i2v image path, and splice in the audio-reference (a tone wav → LTXVAudioVAEEncode
→ LTXVSetAudioRefTokens) + a constant caption + a LoRA loader (bypassable = base arm).

This is fork-and-prune of a FOREIGN file (not one of our workflows), so it manipulates the
JSON directly + keeps the link array in sync, then runs a self-contained validator that
SIMULATES execution (topo-sort + reachability) — the best end-to-end check possible without
the checkpoint/GPU.

    uv run --group dev python scripts/build_pitch_gate_eval_workflow.py            # build
    uv run --group dev python scripts/build_pitch_gate_eval_workflow.py --validate-only <json>

Output: internal/workflows/pitch_gate_eval/_template.json (gitignored area).
NOTE: structural build only. GENERATION validation needs the trained checkpoint + GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STOCK = (
    Path(__file__).resolve().parents[1].parent
    / "ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json"
)
OUT = Path(__file__).resolve().parents[1] / "internal/workflows/pitch_gate_eval/_template.json"

# --- the distilled chain to KEEP (traced from the stock graph) ----------------------------
# id -> expected node TYPE. STOCK is a LIVE upstream checkout (not vendored), so its node ids
# can renumber on `git pull`. This map is BOTH the keep-set AND a drift guard: assert_stock_ids
# fails loud if any id no longer holds its expected type, instead of silently pruning the wrong
# 24 nodes and producing a structurally-valid but semantically-wrong measurement instrument.
KEEP_TYPES = {
    3940: "CheckpointLoaderSimple",   # repoint -> distilled
    4960: "LTXAVTextEncoderLoader",   # CLIP / Gemma
    2483: "CLIPTextEncode",           # positive -> constant caption
    2612: "CLIPTextEncode",           # negative
    1241: "LTXVConditioning",
    3059: "EmptyLTXVLatentVideo",
    3980: "LTXVEmptyLatentAudio",     # GENERATED audio
    4010: "LTXVAudioVAELoader",       # fp32
    4528: "LTXVConcatAVLatent",
    4922: "LoraLoaderModelOnly",      # trained LoRA; bypass=base arm
    4828: "CFGGuider",                # cfg=1
    4831: "KSamplerSelect",           # -> "euler" (CATCH #1)
    4832: "RandomNoise",
    4971: "ManualSigmas",             # 8-step
    4829: "SamplerCustomAdvanced",    # distilled
    4845: "LTXVSeparateAVLatent",
    4848: "LTXVAudioVAEDecode",       # audio out
    4849: "CreateVideo",
    4852: "SaveVideo",
    4982: "LTXVTiledVAEDecode",       # video, feeds CreateVideo
    # primitives feeding kept nodes:
    4977: "PrimitiveBoolean",         # i2v bypass (kept node #3159 dropped)
    4978: "PrimitiveFloat",           # fps
    4979: "PrimitiveInt",             # length
    4985: "LTXFloatToInt",            # audio frames
}
KEEP = set(KEEP_TYPES)


def assert_stock_ids(wf: dict) -> None:
    """Fail loud if STOCK drifted — a kept id no longer holds its expected type. Protects
    the measurement instrument from silent wrong-graph corruption on an upstream pull."""
    byid = {n["id"]: n.get("type") for n in wf["nodes"]}
    bad = {i: (want, byid.get(i)) for i, want in KEEP_TYPES.items() if byid.get(i) != want}
    if bad:
        lines = "\n".join(f"    #{i}: expected {w}, got {g}" for i, (w, g) in sorted(bad.items()))
        sys.exit(f"stock graph drifted — re-trace KEEP_TYPES against {STOCK.name}:\n{lines}")


def _normalize_link_fields(wf):
    """Rebuild every node's input .link + output .links[] from the authoritative links
    array, so node-body fields can't reference pruned/stale link ids."""
    ins_by = {}   # (tgt, slot) -> lid
    outs_by = {}  # (src, slot) -> [lids]
    for lid, src, ss, tgt, ts, _typ in wf["links"]:
        ins_by[(tgt, ts)] = lid
        outs_by.setdefault((src, ss), []).append(lid)
    for n in wf["nodes"]:
        for i, inp in enumerate(n.get("inputs", [])):
            inp["link"] = ins_by.get((n["id"], i))
        for i, out in enumerate(n.get("outputs", [])):
            out["links"] = outs_by.get((n["id"], i), [])


def prune(wf: dict) -> dict:
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] in KEEP]
    # drop links whose endpoints aren't both kept; splice() re-normalizes node-body fields
    wf["links"] = [L for L in wf["links"] if L[1] in KEEP and L[3] in KEEP]
    return wf


# The links array is the single source of truth; node-body .link / .links[] are DERIVED by
# _normalize_link_fields (called once at the end of splice). So these only touch the array —
# no per-call node-body sync to keep consistent (that desync was the link_integrity bug).
def _add_link(wf, src, ss, tgt, ts, typ):
    lid = max((L[0] for L in wf["links"]), default=0) + 1
    wf["links"].append([lid, src, ss, tgt, ts, typ])
    return lid


def _remove_links(wf, pred):
    wf["links"] = [L for L in wf["links"] if not pred(L)]


def _node(nid, ntype, inputs, outputs, wv=None, pos=(0, 0)):
    """Minimal ComfyUI node dict. inputs/outputs: list of (name, type)."""
    return {
        "id": nid, "type": ntype, "pos": list(pos), "size": [270, 80], "flags": {},
        "order": 0, "mode": 0,
        "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
        "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": wv if wv is not None else [],
    }


def splice(wf: dict, *, caption="a person speaking", res=256, length=121,
           distilled="ltx-2.3-22b-distilled-1.1.safetensors",
           tone_wav="pitch_gate_tone.wav", lora="", lora_on=True,
           prefix="pitch_gate/template") -> dict:
    """Turn the pruned distilled-chain base into the audio-reference eval template.

    Edits (build-plan §EDITS): t2v rewire, constant caption, euler sampler, distilled
    loaders, 256 sizing, stock audio-ref splice (LoadAudio->LTXVAudioVAEEncode->
    LTXVSetAudioRefTokens between conditioning and CFGGuider), LoRA arm toggle, static
    SaveVideo prefix per condition.
    """
    byid = {n["id"]: n for n in wf["nodes"]}

    # CATCH #1: euler, not euler_ancestral
    byid[4831]["widgets_values"] = ["euler"]
    # F16: LTXVConditioning.frame_rate must be 25 (canonical LTX 2.3), not the stock 24.
    # IMPORTANT (code-review H2): 1241.frame_rate has a LIVE link from PrimitiveFloat #4978
    # that SURVIVES prune (both in KEEP) — and a live link WINS over the widget. So setting
    # the 1241 widget alone is DEAD; the real fps source is #4978 (stock value 24). Drive the
    # actual source. (The earlier "fps=25" fix only set the dead widget — this is the real fix.)
    byid[1241]["widgets_values"] = [25]            # belt: correct if the link is ever severed
    if byid.get(4978):
        byid[4978]["widgets_values"] = [25.0]      # suspenders: the LIVE driver of frame_rate
    # vae_decode_no_tile: [1,1,1] single-tile on 24GB+ (3x faster cold-pass)
    if byid.get(4982):
        byid[4982]["widgets_values"] = [1, 1, 1, False, "auto", "auto"]
    # CATCH #2: distilled checkpoint (audio VAE loader left to user-confirm; flagged)
    byid[3940]["widgets_values"] = [distilled]
    # constant caption (positive); negative kept generic
    byid[2483]["widgets_values"] = [caption]
    # CATCH #3: 256 sizing + generated audio length.
    # The prune severs the PrimitiveInt(length)/PrimitiveFloat(fps) -> consumer links, so the
    # video/audio latent + createvideo fps fall back to their OWN widgets — set them explicitly.
    # AUDIO LENGTH: NOT equal to video frames. The stock proven single-stage example pairs
    # 121 VIDEO frames with 97 AUDIO frames (the audio VAE has its own temporal rate). Setting
    # audio=video_frames was a regression — use the proven audio_frames for this video length.
    # (For length != 121 this ratio would need re-deriving; the gate uses the stock 121/97.)
    AUDIO_FRAMES = 97  # proven pairing for length=121 video (stock example)
    # The 121:97 video:audio pairing is the only one this builder has validated. `length` is a
    # parameter but AUDIO_FRAMES is fixed — a mismatched length desyncs audio/video duration and
    # ffmpeg -shortest then clips the F0 measurement window. Fail loud instead of silently
    # mis-pairing (code-review MED-3). Re-derive the audio-frame count before raising `length`.
    if length != 121:
        raise ValueError(
            f"length={length} but AUDIO_FRAMES is hardcoded to 97 (the validated 121:97 "
            f"video:audio pairing). Re-derive the audio-frame count for this video length "
            f"before changing `length`, or audio/video durations desync."
        )
    byid[3059]["widgets_values"] = [res, res, length, 1]            # EmptyLTXVLatentVideo: WxHxLx1
    if byid.get(3980):
        byid[3980]["widgets_values"] = [AUDIO_FRAMES, 25, 1]        # LTXVEmptyLatentAudio: frames,fps,batch
    if byid.get(4979):
        byid[4979]["widgets_values"] = [length, "fixed"]            # PrimitiveInt length (vestigial; keep coherent)
    # LoRA arm: bypass the loader for the base arm
    byid[4922]["mode"] = 0 if lora_on else 4
    if lora:
        wv = byid[4922].get("widgets_values") or ["", 1.0]
        wv[0] = lora
        byid[4922]["widgets_values"] = wv
    # static per-condition prefix
    byid[4852]["widgets_values"] = [prefix, "auto", "auto"]
    # fps: CreateVideo's fps input is unwired after prune (PrimitiveFloat link severed) so it
    # falls back to its own widget — force 25 to match LTXVConditioning/audio-latent fps.
    # At 30 the 121-frame video = 4.03s vs ~4.84s audio, so ffmpeg -shortest would CLIP the
    # audio and shrink the F0 measurement window. (The repo audit's fps_coherence allowlist
    # doesn't cover the comfy-core CreateVideo node, so it missed this — Fred caught it.)
    if byid.get(4849):
        byid[4849]["widgets_values"] = [25]

    # CATCH #4 / the one gap: t2v rewire EmptyLTXVLatentVideo -> Concat.video_latent
    _add_link(wf, 3059, 0, 4528, 0, "LATENT")

    # audio-ref splice: new LoadAudio -> LTXVAudioVAEEncode -> LTXVSetAudioRefTokens
    base = max(n["id"] for n in wf["nodes"])
    load_id, enc_id, ref_id = base + 1, base + 2, base + 3
    wf["nodes"] += [
        _node(load_id, "LoadAudio", [], [("AUDIO", "AUDIO")], wv=[tone_wav], pos=(-400, 600)),
        _node(enc_id, "LTXVAudioVAEEncode",
              [("audio", "AUDIO"), ("audio_vae", "VAE")], [("Audio Latent", "LATENT")], pos=(-100, 600)),
        _node(ref_id, "LTXAudioSetRefTokens",  # debug-instrumented; logs ref latent + token shapes
              [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("audio_latent", "LATENT")],
              [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
              pos=(200, 600)),
    ]
    # wire: LoadAudio->enc.audio ; audioVAE->enc.audio_vae ; enc->ref.audio_latent
    _add_link(wf, load_id, 0, enc_id, 0, "AUDIO")
    _add_link(wf, 4010, 0, enc_id, 1, "VAE")
    _add_link(wf, enc_id, 0, ref_id, 2, "LATENT")
    # reroute conditioning: #1241 LTXVConditioning pos/neg -> ref tokens -> #4828 CFGGuider
    # remove the existing 1241->4828 links (pos slot1, neg slot2), both representations
    _remove_links(wf, lambda L: L[1] == 1241 and L[3] == 4828)
    _add_link(wf, 1241, 0, ref_id, 0, "CONDITIONING")  # pos -> ref.positive
    _add_link(wf, 1241, 1, ref_id, 1, "CONDITIONING")  # neg -> ref.negative
    _add_link(wf, ref_id, 0, 4828, 1, "CONDITIONING")  # ref.pos -> guider.positive
    _add_link(wf, ref_id, 1, 4828, 2, "CONDITIONING")  # ref.neg -> guider.negative
    # Single source of truth: rebuild ALL node-body link fields from the links array, so no
    # hand-sync gap can leave a stale/missing ref (the link_integrity invariant).
    _normalize_link_fields(wf)
    return wf


# Proven loader stack widgets, copied verbatim from the shipped audio-loop workflow
# (example_workflows/audio-loop-music-video_latent.json — the canonical way THIS repo drives
# LTX 2.3). Fork's stock CheckpointLoaderSimple/LTXAVTextEncoderLoader/LTXVAudioVAELoader are
# NOT how the repo runs; swap them for these.
UNET_FILE = "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
VIDEO_VAE = "vae/LTX23_video_vae_bf16.safetensors"
AUDIO_VAE = "vae/LTX23_audio_vae_bf16.safetensors"
GEMMA = "gemma_3_12B_it_fpmixed.safetensors"
TEXT_PROJ = "ltx-2.3_text_projection_bf16.safetensors"
DEFAULT_LORA = "pitch_gate_audio_ref_step02000.safetensors"  # our trained IC-LoRA in models/loras/


def swap_loaders(wf: dict, *, lora=DEFAULT_LORA, lora_on=True) -> dict:
    """Replace the forked stock loaders with the repo's PROVEN stack (Fred's call):
      UNETLoader(fp8 distilled) -> AudioLoopHelperSageAttention -> LTXVChunkFeedForward
        -> LTXICLoRALoaderModelOnly(our LoRA) -> CFGGuider.model
      VAELoaderKJ(video bf16) -> video tiled decode ; VAELoaderKJ(audio bf16) -> audio enc/dec/emptylatent
      DualCLIPLoader(Gemma+proj) -> CLIPTextEncode x2
    The generic LoraLoaderModelOnly is replaced by LTXICLoRALoaderModelOnly (the IC-LoRA
    loader — applies our audio LoRA via load_lora_for_models; same key mapping).
    """
    # who currently consumes each loader output (capture BEFORE deleting)
    audio_vae_consumers = [(L[3], L[4]) for L in wf["links"] if L[1] == 4010]   # (tgt, tgtslot)
    clip_consumers = [(L[3], L[4]) for L in wf["links"] if L[1] == 4960]
    video_vae_consumers = [(L[3], L[4]) for L in wf["links"] if L[1] == 3940 and L[2] == 2]  # VAE out
    guider_model_tgt = (4828, 0)  # CFGGuider.model

    # remove the stock loaders + the generic lora loader, and every link touching them
    drop = {3940, 4960, 4010, 4922}
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in drop]
    wf["links"] = [L for L in wf["links"] if L[1] not in drop and L[3] not in drop]

    nid = max(n["id"] for n in wf["nodes"])
    unet, sage, ffn, iclora = nid + 1, nid + 2, nid + 3, nid + 4
    vvae, avae, clip = nid + 5, nid + 6, nid + 7
    wf["nodes"] += [
        _node(unet, "UNETLoader", [], [("MODEL", "MODEL")], wv=[UNET_FILE, "default"], pos=(700, 560)),
        _node(sage, "AudioLoopHelperSageAttention", [("model", "MODEL")], [("model", "MODEL")],
              wv=["auto", True, 1024], pos=(700, 400)),
        _node(ffn, "LTXVChunkFeedForward", [("model", "MODEL")], [("model", "MODEL")],
              wv=[2, 4096], pos=(700, 1020)),
        _node(iclora, "LTXAudioICLoRALoader", [("model", "MODEL")],  # debug; logs key-match telemetry
              [("model", "MODEL")],
              wv=[lora, 1.0], pos=(1430, 400)),
        _node(vvae, "VAELoaderKJ", [], [("VAE", "VAE")], wv=[VIDEO_VAE, "main_device", "bf16"], pos=(700, 1380)),
        _node(avae, "VAELoaderKJ", [], [("VAE", "VAE")], wv=[AUDIO_VAE, "main_device", "bf16"], pos=(700, 1560)),
        _node(clip, "DualCLIPLoader", [], [("CLIP", "CLIP")], wv=[GEMMA, TEXT_PROJ, "ltxv", "default"], pos=(700, 692)),
    ]
    if not lora_on:
        byid2 = {n["id"]: n for n in wf["nodes"]}
        byid2[iclora]["mode"] = 4  # base arm: bypass the IC-LoRA loader

    # model chain
    _add_link(wf, unet, 0, sage, 0, "MODEL")
    _add_link(wf, sage, 0, ffn, 0, "MODEL")
    _add_link(wf, ffn, 0, iclora, 0, "MODEL")
    _add_link(wf, iclora, 0, guider_model_tgt[0], guider_model_tgt[1], "MODEL")
    # VAE + CLIP rewires onto the new loaders
    for tgt, slot in video_vae_consumers:
        _add_link(wf, vvae, 0, tgt, slot, "VAE")
    for tgt, slot in audio_vae_consumers:
        _add_link(wf, avae, 0, tgt, slot, "VAE")
    for tgt, slot in clip_consumers:
        _add_link(wf, clip, 0, tgt, slot, "CLIP")

    _normalize_link_fields(wf)
    return wf


def validate(wf: dict) -> list[str]:
    """Simulate execution: link integrity + topo-sort (cycle detect) + sink reachability."""
    errs = []
    ids = {n["id"] for n in wf["nodes"]}
    byid = {n["id"]: n for n in wf["nodes"]}

    # 1. link integrity — endpoints exist
    for L in wf["links"]:
        lid, src, _ss, tgt, _ts, _typ = L
        if src not in ids:
            errs.append(f"link {lid}: src #{src} missing")
        if tgt not in ids:
            errs.append(f"link {lid}: tgt #{tgt} missing")

    # 2. every non-loader node has its required inputs wired (input slots with a link)
    incoming = {}
    for L in wf["links"]:
        incoming.setdefault(L[3], set()).add(L[4])
    for n in wf["nodes"]:
        if n.get("mode") == 4:
            continue  # bypassed
        for slot_i, inp in enumerate(n.get("inputs", [])):
            # widget-backed inputs are optional; pure link inputs (no widget) must be fed
            if inp.get("widget") is None and slot_i not in incoming.get(n["id"], set()):
                errs.append(f"#{n['id']} {n.get('type')}: input '{inp.get('name')}' (slot {slot_i}) unwired")

    # 3. cycle detection (topo-sort over active nodes)
    active = {nid for nid, n in byid.items() if n.get("mode") != 4}
    adj = {nid: set() for nid in active}
    indeg = {nid: 0 for nid in active}
    for L in wf["links"]:
        if L[1] in active and L[3] in active and L[3] not in adj[L[1]]:
            adj[L[1]].add(L[3])
    for s in adj:
        for t in adj[s]:
            indeg[t] += 1
    queue = [n for n in active if indeg[n] == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for t in adj[n]:
            indeg[t] -= 1
            if indeg[t] == 0:
                queue.append(t)
    if seen != len(active):
        errs.append(f"CYCLE: topo-sort reached {seen}/{len(active)} active nodes")

    # 4. sink reachability — SaveVideo reachable from a loader
    sinks = [n["id"] for n in wf["nodes"] if n.get("type") in ("SaveVideo", "SaveAudio", "VHS_VideoCombine")]
    if not sinks:
        errs.append("no Save sink present")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", type=Path, help="just validate an existing workflow json")
    args = ap.parse_args()

    if args.validate_only:
        wf = json.loads(args.validate_only.read_text())
        errs = validate(wf)
        print("\n".join(errs) if errs else "VALIDATE OK")
        sys.exit(1 if errs else 0)

    if not STOCK.is_file():
        sys.exit(f"stock graph not found: {STOCK}")
    wf = json.loads(STOCK.read_text())
    print(f"stock: {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    assert_stock_ids(wf)  # fail loud if upstream renumbered (silent wrong-graph guard)
    wf = prune(wf)
    print(f"pruned to distilled chain: {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    wf = splice(wf, lora_on=True, prefix="pitch_gate/lora_template")
    print(f"spliced (audio-ref + t2v rewire + caption): {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    wf = swap_loaders(wf, lora_on=True)
    print(f"swapped to proven loader stack (UNET+sage+ffn+IC-LoRA+VAELoaderKJ+DualCLIP): "
          f"{len(wf['nodes'])} nodes, {len(wf['links'])} links")
    # Local pre-filter: a fast, dependency-free fail-early (cycle / unwired / sink). NOT the
    # authoritative gate — `scripts/audit_workflows.py` is (it caught link-array desync +
    # frame_rate + decode that this pre-filter missed). Always run the audit on the output.
    errs = validate(wf)
    print("=== pre-filter (cycle / unwired / sink) ===")
    print("\n".join(f"  {e}" for e in errs) if errs else "  pre-filter OK")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(wf, indent=2))
    print(f"\nwrote template -> {OUT}")
    if errs:
        print("(pre-filter gaps above — fix before the audit)")
    print(f"AUTHORITATIVE CHECK: uv run --group dev python scripts/audit_workflows.py {OUT}")
    print("Generation validation still needs the trained checkpoint + GPU.")


if __name__ == "__main__":
    main()
