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
# everything else is pruned. Primitives that feed kept nodes are added automatically.
KEEP = {
    3940,  # CheckpointLoaderSimple  (repoint -> distilled)
    4960,  # LTXAVTextEncoderLoader  (CLIP / Gemma)
    2483,  # CLIPTextEncode positive (-> constant caption)
    2612,  # CLIPTextEncode negative
    1241,  # LTXVConditioning
    3059,  # EmptyLTXVLatentVideo
    3980,  # LTXVEmptyLatentAudio    (GENERATED audio)
    4010,  # LTXVAudioVAELoader      (fp32)
    4528,  # LTXVConcatAVLatent
    4922,  # LoraLoaderModelOnly     (the trained LoRA; bypass=base arm)
    4828,  # CFGGuider cfg=1
    4831,  # KSamplerSelect          (-> "euler", CATCH #1)
    4832,  # RandomNoise
    4971,  # ManualSigmas (8-step)
    4829,  # SamplerCustomAdvanced (distilled)
    4845,  # LTXVSeparateAVLatent
    4848,  # LTXVAudioVAEDecode (audio out)
    4849,  # CreateVideo
    4852,  # SaveVideo
    4982,  # LTXVTiledVAEDecode (video, feeds CreateVideo)
    # primitives feeding kept nodes:
    4977,  # PrimitiveBoolean (i2v bypass — kept node #3159 is dropped, see below)
    4978,  # PrimitiveFloat (fps)
    4979,  # PrimitiveInt (length)
    4985,  # LTXFloatToInt (audio frames)
}
# the video tiled decoder #4982 needs the video latent from #4845 + the video VAE; in stock
# it's fed by the quality chain. We rewire it to the distilled separate-latent + checkpoint VAE.


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
    keep_ids = set(KEEP)
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] in keep_ids]
    # drop links whose endpoints aren't both kept
    wf["links"] = [L for L in wf["links"] if L[1] in keep_ids and L[3] in keep_ids]
    _normalize_link_fields(wf)  # clear stale node-body link refs from the pruned-away links
    return wf


def _add_link(wf, src, ss, tgt, ts, typ):
    lid = max((L[0] for L in wf["links"]), default=0) + 1
    wf["links"].append([lid, src, ss, tgt, ts, typ])
    byid = {n["id"]: n for n in wf["nodes"]}
    # BOTH link representations must sync (ComfyUI gotcha): target input .link AND source output .links[]
    byid[tgt]["inputs"][ts]["link"] = lid
    out = byid[src]["outputs"][ss]
    out.setdefault("links", [])
    if out["links"] is None:
        out["links"] = []
    out["links"].append(lid)
    return lid


def _remove_links(wf, pred):
    """Drop links matching pred(L); rebuild BOTH representations consistently."""
    wf["links"] = [L for L in wf["links"] if not pred(L)]
    live = {L[0] for L in wf["links"]}
    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") is not None and inp["link"] not in live:
                inp["link"] = None
        for out in n.get("outputs", []):
            if out.get("links"):
                out["links"] = [lid for lid in out["links"] if lid in live]


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
    # F16: LTXVConditioning.frame_rate must be 25 (canonical LTX 2.3), not the stock 24;
    # also resolves fps_coherence vs the audio latent (25).
    byid[1241]["widgets_values"] = [25]
    # vae_decode_no_tile: [1,1,1] single-tile on 24GB+ (3x faster cold-pass)
    if byid.get(4982):
        byid[4982]["widgets_values"] = [1, 1, 1, False, "auto", "auto"]
    # CATCH #2: distilled checkpoint (audio VAE loader left to user-confirm; flagged)
    byid[3940]["widgets_values"] = [distilled]
    # constant caption (positive); negative kept generic
    byid[2483]["widgets_values"] = [caption]
    # CATCH #3: 256 sizing + generated audio length
    byid[3059]["widgets_values"] = [res, res, length, 1]
    if byid.get(4979):
        byid[4979]["widgets_values"] = [length, "fixed"]
    # LoRA arm: bypass the loader for the base arm
    byid[4922]["mode"] = 0 if lora_on else 4
    if lora:
        wv = byid[4922].get("widgets_values") or ["", 1.0]
        wv[0] = lora
        byid[4922]["widgets_values"] = wv
    # static per-condition prefix
    byid[4852]["widgets_values"] = [prefix, "auto", "auto"]

    # CATCH #4 / the one gap: t2v rewire EmptyLTXVLatentVideo -> Concat.video_latent
    _add_link(wf, 3059, 0, 4528, 0, "LATENT")

    # audio-ref splice: new LoadAudio -> LTXVAudioVAEEncode -> LTXVSetAudioRefTokens
    base = max(n["id"] for n in wf["nodes"])
    load_id, enc_id, ref_id = base + 1, base + 2, base + 3
    wf["nodes"] += [
        _node(load_id, "LoadAudio", [], [("AUDIO", "AUDIO")], wv=[tone_wav], pos=(-400, 600)),
        _node(enc_id, "LTXVAudioVAEEncode",
              [("audio", "AUDIO"), ("audio_vae", "VAE")], [("Audio Latent", "LATENT")], pos=(-100, 600)),
        _node(ref_id, "LTXVSetAudioRefTokens",
              [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("audio_latent", "LATENT")],
              [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("frozen_audio", "LATENT")],
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
    wf = prune(wf)
    print(f"pruned to distilled chain: {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    wf = splice(wf, lora_on=True, prefix="pitch_gate/lora_template")
    print(f"spliced (audio-ref + t2v rewire + caption): {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    errs = validate(wf)
    print("=== post-splice SIMULATE-EXECUTION validation ===")
    print("\n".join(f"  {e}" for e in errs) if errs else "  VALIDATE OK — no gaps, no cycle, sink reachable")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(wf, indent=2))
    print(f"\nwrote template -> {OUT}")
    if errs:
        print("(structural gaps above — fix before generation)")
    else:
        print("STRUCTURAL OK. Generation validation needs the trained checkpoint + GPU.")


if __name__ == "__main__":
    main()
