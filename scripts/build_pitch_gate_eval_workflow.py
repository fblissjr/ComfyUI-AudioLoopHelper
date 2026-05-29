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


def prune(wf: dict) -> dict:
    keep_ids = set(KEEP)
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] in keep_ids]
    # drop links whose endpoints aren't both kept
    wf["links"] = [L for L in wf["links"] if L[1] in keep_ids and L[3] in keep_ids]
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
    errs = validate(wf)
    # the pruned graph WILL have unwired inputs (the cut i2v + audio-ref splices) — report,
    # don't fail; the splice step (next) fills them.
    print("=== post-prune validation (expected gaps = the splice points) ===")
    print("\n".join(f"  {e}" for e in errs) if errs else "  (clean)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(wf, indent=2))
    print(f"\nwrote pruned base -> {OUT}")
    print("NEXT: splice audio-ref + constant caption + RunIdPrefix + t2v rewire (separate step).")


if __name__ == "__main__":
    main()
