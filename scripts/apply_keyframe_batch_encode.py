"""Migrate `audio-loop-music-video_latent_keyframe.json` to use the
pre-loop keyframe-LATENT batch-encode pattern.

Removes the per-iteration `KeyframeImageSchedule + ImageBlend +
top-level VAEEncode` chain and inserts
`KeyframeLatentScheduleBatchEncode` (top-level, runs once) +
`LatentSelectByIteration` (runs per-iter, no VAE dependency). Mirrors
the conditioning-side migration shipped 2026-04-22 via
`apply_batch_encode_fix.py`.

VAE encodes each unique keyframe image exactly once per generation,
regardless of how many iterations share it. Eliminates the
per-iteration VAE round-trip that AudioLoopController re-execution
forces on the legacy chain.

Usage:
    uv run --group dev python scripts/apply_keyframe_batch_encode.py
    uv run --group dev python scripts/apply_keyframe_batch_encode.py --dry-run
    uv run --group dev python scripts/apply_keyframe_batch_encode.py --revert

Idempotent on the output path; re-running produces a byte-identical
file. `--revert` rebuilds the legacy chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path("example_workflows/audio-loop-music-video_latent_keyframe.json")

# Production node IDs in the legacy workflow we replace.
KEYFRAME_IMAGE_SCHEDULE = 1609
IMAGE_BLEND_AUDIO_LOOP = 1610
LEGACY_VAE_ENCODE = 1613
STRIP_IDS = {KEYFRAME_IMAGE_SCHEDULE, IMAGE_BLEND_AUDIO_LOOP, LEGACY_VAE_ENCODE}

# Production node IDs we wire to / read from. Must exist before strip.
IMAGE_BATCH_SOURCE = 1617           # ImageBatch (output: IMAGE batch of keyframes)
TENSOR_LOOP_OPEN = 1539             # TensorLoopOpen (slot 3 = current_iteration)
AUDIO_LOOP_CONTROLLER = 1582        # slot 2=audio_duration, slot 4=stride_seconds
GET_VIDEO_VAE = 619                 # GetNode "video_vae"
SUBGRAPH_INVOKER = 843               # slot 8 = guide_latent (LATENT)
SUBGRAPH_GUIDE_LATENT_SLOT = 8


def _in(name: str, dtype: str) -> dict:
    return {"name": name, "type": dtype, "link": None}


def _out(name: str, dtype: str) -> dict:
    return {"name": name, "type": dtype, "links": []}


def _next_id(wf: dict, key: str = "last_node_id") -> int:
    nid = wf.get(key, 0) + 1
    wf[key] = nid
    return nid


def _next_link_id(wf: dict) -> int:
    return _next_id(wf, "last_link_id")


def _find_node(wf: dict, node_id: int) -> dict | None:
    for n in wf["nodes"]:
        if n["id"] == node_id:
            return n
    return None


def _find_link_to_slot(wf: dict, tgt_node: int, tgt_slot: int) -> list | None:
    for l in wf["links"]:
        if isinstance(l, list) and l[3] == tgt_node and l[4] == tgt_slot:
            return l
    return None


def _remove_link_by_id(wf: dict, link_id: int) -> None:
    wf["links"] = [l for l in wf["links"] if not (isinstance(l, list) and l[0] == link_id)]
    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") == link_id:
                inp["link"] = None
        for out in n.get("outputs", []):
            if out.get("links"):
                out["links"] = [l for l in out["links"] if l != link_id]


def _add_link(wf: dict, src_id: int, src_slot: int, tgt_id: int, tgt_slot: int, dtype: str) -> int:
    lid = _next_link_id(wf)
    wf["links"].append([lid, src_id, src_slot, tgt_id, tgt_slot, dtype])
    src = _find_node(wf, src_id)
    if src and src_slot < len(src.get("outputs", [])):
        src["outputs"][src_slot].setdefault("links", []).append(lid)
    tgt = _find_node(wf, tgt_id)
    if tgt and tgt_slot < len(tgt.get("inputs", [])):
        tgt["inputs"][tgt_slot]["link"] = lid
    return lid


def _remove_node_and_links(wf: dict, node_id: int) -> None:
    to_remove = []
    for l in wf["links"]:
        if isinstance(l, list) and (l[1] == node_id or l[3] == node_id):
            to_remove.append(l[0])
    for lid in to_remove:
        _remove_link_by_id(wf, lid)
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] != node_id]


def _is_already_built(wf: dict) -> bool:
    return any(n["type"] == "KeyframeLatentScheduleBatchEncode" for n in wf["nodes"])


def build(output_path: Path, dry_run: bool = False) -> None:
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if not output_path.exists():
        raise SystemExit(f"Source workflow not found: {output_path}")

    wf = orjson.loads(output_path.read_bytes())
    initial_count = (len(wf["nodes"]), len(wf["links"]))
    print(f"Loaded {output_path.name}: {initial_count[0]} nodes, {initial_count[1]} links")

    if _is_already_built(wf):
        print("  KeyframeLatentScheduleBatchEncode already present — no-op (idempotent).")
        return

    # Verify all target wiring sources exist before mutating.
    for nid, label in (
        (IMAGE_BATCH_SOURCE, "ImageBatch source"),
        (TENSOR_LOOP_OPEN, "TensorLoopOpen"),
        (AUDIO_LOOP_CONTROLLER, "AudioLoopController"),
        (GET_VIDEO_VAE, "Get_video_vae"),
        (SUBGRAPH_INVOKER, "subgraph invoker"),
    ):
        if _find_node(wf, nid) is None:
            raise SystemExit(f"{output_path.name} missing {label} (id={nid}) — workflow shape unexpected")

    # The legacy chain may already be partially stripped on a re-run; only
    # strip what's still present.
    stripped = []
    for nid in list(STRIP_IDS):
        if _find_node(wf, nid):
            _remove_node_and_links(wf, nid)
            stripped.append(nid)
    if stripped:
        print(f"  stripped legacy chain: {sorted(stripped)}")
    print(f"  -> {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    # Add KeyframeLatentScheduleBatchEncode (top-level, outside loop).
    batch_encode = _next_id(wf)
    wf["nodes"].append({
        "id": batch_encode, "type": "KeyframeLatentScheduleBatchEncode",
        "pos": [-1100, 1900], "size": [380, 200], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            _in("vae", "VAE"),
            _in("images", "IMAGE"),
            _in("stride_seconds", "FLOAT"),
            _in("audio_duration", "FLOAT"),
            {"name": "schedule", "type": "STRING",
             "widget": {"name": "schedule"}, "link": None},
            {"name": "snap_boundaries", "type": "BOOLEAN",
             "widget": {"name": "snap_boundaries"}, "link": None},
        ],
        "outputs": [
            _out("latent_list", "*"),
            _out("iteration_count", "INT"),
        ],
        "properties": {"Node name for S&R": "KeyframeLatentScheduleBatchEncode"},
        "widgets_values": ["0:00+: 0", True],
        "title": "Keyframe Latent Schedule (Batch Encode)",
    })

    # Add LatentSelectByIteration (inside loop).
    select = _next_id(wf)
    wf["nodes"].append({
        "id": select, "type": "LatentSelectByIteration",
        "pos": [-650, 1900], "size": [320, 100], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            _in("latent_list", "*"),
            _in("current_iteration", "INT"),
        ],
        "outputs": [_out("latent", "LATENT")],
        "properties": {"Node name for S&R": "LatentSelectByIteration"},
        "widgets_values": [],
        "title": "Latent Select (by Iteration)",
    })

    # Wire batch encoder inputs.
    _add_link(wf, GET_VIDEO_VAE, 0, batch_encode, 0, "VAE")
    _add_link(wf, IMAGE_BATCH_SOURCE, 0, batch_encode, 1, "IMAGE")
    _add_link(wf, AUDIO_LOOP_CONTROLLER, 4, batch_encode, 2, "FLOAT")  # stride_seconds
    _add_link(wf, AUDIO_LOOP_CONTROLLER, 2, batch_encode, 3, "FLOAT")  # audio_duration

    # Wire batch encoder -> selector.
    _add_link(wf, batch_encode, 0, select, 0, "*")
    _add_link(wf, TENSOR_LOOP_OPEN, 3, select, 1, "INT")  # current_iteration

    # Wire selector -> subgraph invoker's guide_latent slot.
    existing = _find_link_to_slot(wf, SUBGRAPH_INVOKER, SUBGRAPH_GUIDE_LATENT_SLOT)
    if existing:
        _remove_link_by_id(wf, existing[0])
    _add_link(wf, select, 0, SUBGRAPH_INVOKER, SUBGRAPH_GUIDE_LATENT_SLOT, "LATENT")

    print(f"  added KeyframeLatentScheduleBatchEncode({batch_encode}) + LatentSelectByIteration({select})")
    print(f"  rewired: ImageBatch({IMAGE_BATCH_SOURCE}) + AudioLoopController({AUDIO_LOOP_CONTROLLER}) -> batch_encode -> select -> subgraph({SUBGRAPH_INVOKER}).slot{SUBGRAPH_GUIDE_LATENT_SLOT}")
    print(f"  final: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    if dry_run:
        print(f"\n[DRY-RUN] would write {output_path}")
        return

    output_path.write_bytes(orjson.dumps(wf, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {output_path}")
    print("\nVerify with:")
    print(f"  uv run --group dev python scripts/audit_workflows.py")
    print(f"  uv run --group dev python scripts/analyze_workflow_dag.py {output_path} --format ascii | tail -50")
    print("\nIn ComfyUI: edit the schedule widget on the new "
          "KeyframeLatentScheduleBatchEncode node to map song sections "
          "to image indices (e.g. '0:00-0:42: 0\\n0:42-1:28: 1\\n1:28+: 2').")


def revert(output_path: Path) -> None:
    """Rebuild the legacy KeyframeImageSchedule + ImageBlend + VAEEncode chain.

    Re-applies the original node shape so a workflow can be rolled back
    if the new pattern misbehaves. Source of truth is git history; this
    revert only restores the topological shape, not the exact link IDs.
    """
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if not output_path.exists():
        print(f"{output_path} does not exist; nothing to revert.")
        return

    wf = orjson.loads(output_path.read_bytes())
    if not _is_already_built(wf):
        print(f"{output_path.name} has no KeyframeLatentScheduleBatchEncode; already reverted or never migrated.")
        return

    # Strip the new nodes
    new_nodes = [n["id"] for n in wf["nodes"]
                 if n["type"] in ("KeyframeLatentScheduleBatchEncode", "LatentSelectByIteration")]
    for nid in new_nodes:
        _remove_node_and_links(wf, nid)

    # Re-add legacy chain at the original IDs (KEYFRAME_IMAGE_SCHEDULE, etc.)
    # so wire IDs in any saved workflow JSON snapshot align.
    legacy_nodes = [
        {
            "id": KEYFRAME_IMAGE_SCHEDULE, "type": "KeyframeImageSchedule",
            "pos": [-1100, 1900], "size": [340, 200], "flags": {}, "order": 0, "mode": 0,
            "inputs": [
                _in("images", "IMAGE"),
                _in("current_iteration", "INT"),
                _in("stride_seconds", "FLOAT"),
                {"name": "schedule", "type": "STRING",
                 "widget": {"name": "schedule"}, "link": None},
                {"name": "blend_seconds", "type": "FLOAT",
                 "widget": {"name": "blend_seconds"}, "link": None},
            ],
            "outputs": [
                _out("image", "IMAGE"),
                _out("next_image", "IMAGE"),
                _out("blend_factor", "FLOAT"),
                _out("current_time", "FLOAT"),
                _out("image_index", "INT"),
            ],
            "properties": {"Node name for S&R": "KeyframeImageSchedule"},
            "widgets_values": ["0:00+: 0", 0.0],
            "title": "Keyframe Image Schedule",
        },
        {
            "id": IMAGE_BLEND_AUDIO_LOOP, "type": "ImageBlend_AudioLoop",
            "pos": [-720, 1900], "size": [280, 100], "flags": {}, "order": 0, "mode": 0,
            "inputs": [
                _in("image_a", "IMAGE"),
                _in("image_b", "IMAGE"),
                _in("blend_factor", "FLOAT"),
            ],
            "outputs": [_out("image", "IMAGE")],
            "properties": {"Node name for S&R": "ImageBlend_AudioLoop"},
            "widgets_values": [],
            "title": "Image Blend",
        },
        {
            "id": LEGACY_VAE_ENCODE, "type": "VAEEncode",
            "pos": [-400, 1900], "size": [220, 46], "flags": {}, "order": 0, "mode": 0,
            "inputs": [_in("pixels", "IMAGE"), _in("vae", "VAE")],
            "outputs": [_out("LATENT", "LATENT")],
            "properties": {"Node name for S&R": "VAEEncode"},
            "widgets_values": [],
            "title": "VAE Encode (init image → guide latent)",
        },
    ]
    wf["nodes"].extend(legacy_nodes)
    # Bump last_node_id past the highest legacy ID we just inserted.
    wf["last_node_id"] = max(wf.get("last_node_id", 0), LEGACY_VAE_ENCODE)

    # Rewire legacy chain.
    _add_link(wf, IMAGE_BATCH_SOURCE, 0, KEYFRAME_IMAGE_SCHEDULE, 0, "IMAGE")
    _add_link(wf, TENSOR_LOOP_OPEN, 3, KEYFRAME_IMAGE_SCHEDULE, 1, "INT")
    _add_link(wf, AUDIO_LOOP_CONTROLLER, 4, KEYFRAME_IMAGE_SCHEDULE, 2, "FLOAT")
    _add_link(wf, KEYFRAME_IMAGE_SCHEDULE, 0, IMAGE_BLEND_AUDIO_LOOP, 0, "IMAGE")
    _add_link(wf, KEYFRAME_IMAGE_SCHEDULE, 1, IMAGE_BLEND_AUDIO_LOOP, 1, "IMAGE")
    _add_link(wf, KEYFRAME_IMAGE_SCHEDULE, 2, IMAGE_BLEND_AUDIO_LOOP, 2, "FLOAT")
    _add_link(wf, IMAGE_BLEND_AUDIO_LOOP, 0, LEGACY_VAE_ENCODE, 0, "IMAGE")
    _add_link(wf, GET_VIDEO_VAE, 0, LEGACY_VAE_ENCODE, 1, "VAE")
    _add_link(wf, LEGACY_VAE_ENCODE, 0, SUBGRAPH_INVOKER, SUBGRAPH_GUIDE_LATENT_SLOT, "LATENT")

    output_path.write_bytes(orjson.dumps(wf, option=orjson.OPT_INDENT_2))
    print(f"Reverted {output_path.name} to legacy KeyframeImageSchedule chain.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="Workflow path to mutate in place (default: %(default)s)")
    ap.add_argument("--revert", action="store_true",
                    help="Rebuild the legacy KeyframeImageSchedule+ImageBlend+VAEEncode chain.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the build diff without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        revert(output_path)
        return
    build(output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
