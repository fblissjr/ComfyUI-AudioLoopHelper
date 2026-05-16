"""One-shot cleanup applied to LATENT-variant example workflows (2026-04-23).

Three orthogonal transforms bundled because they touch the same
node-ID layout and all needed to land together:

**Transform 1 — dead-node removal.** Remove:
- `#1590 VAEEncode` + `#1597 LTXVTiledVAEDecode` (mode=4 skeleton from
  the deferred in-workflow upscale path; input unwired).
- `Reroute #618` (input wired, output empty).
- `Note #1585` (stale prompt-schedule copy from pre-batch-encode days;
  schedule now lives on `TimestampPromptScheduleBatchEncode.schedule`).

**Transform 2 — preview decode upgrade.** `#1318 VAEDecode` →
`LTXVTiledVAEDecode` (widgets `[2, 2, 1, True, "auto", "auto"]`).
Defensive: prevents OOM on the preview path at higher resolutions.

**Transform 3 — subgraph VAE move.** Move the per-iteration init-image
`VAEEncode` out of the subgraph.
   The subgraph was VAE-encoding the SAME init image every iteration
   even though `LTXVImgToVideoInplaceKJ` already encodes it once at
   graph setup. New top-level `VAEEncode` encodes once; subgraph input
   slot 8 changes from IMAGE (`num_guides.image_1`) to LATENT
   (`guide_latent`) and feeds `LTXVAddLatentGuide.guiding_latent`
   directly.

Targets LATENT-variant workflows only. The legacy `_image.json` is
excluded (reference-only per CLAUDE.md; its subgraph shape is different).

Usage: `uv run --group dev python scripts/apply_vae_and_cleanup.py`
(idempotent; safe to re-run).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from workflow_utils import WorkflowEditor  # noqa: E402

TARGET_WORKFLOWS = [
    "audio-loop-music-video_latent.json",
    "audio-loop-music-video_latent_keyframe.json",
    "audio-loop-music-video_latent_stg.json",
    "audio-loop-music-video_latent_validator.json",
    "audio-loop-music-video_image_adain_perstep.json",
]


def _remove_dead_nodes(ed: WorkflowEditor) -> list[str]:
    changes = []
    for nid in (1590, 1597, 618, 1585):
        try:
            node = ed.find_node(nid)
        except ValueError:
            continue
        ntype = node.get("type")
        changes.append(f"removed #{nid} {ntype}")
        ed.remove_node_and_links(nid)
    return changes


def _upgrade_preview_decode(ed: WorkflowEditor) -> list[str]:
    try:
        node = ed.find_node(1318)
    except ValueError:
        return []
    if node.get("type") != "VAEDecode":
        # Already upgraded or replaced; leave it.
        return []
    node["type"] = "LTXVTiledVAEDecode"
    node["widgets_values"] = [2, 2, 1, True, "auto", "auto"]
    # Update cnr/property tags so ComfyUI resolves the right class.
    props = node.setdefault("properties", {})
    props["cnr_id"] = "comfyui-ltxvideo"
    props["Node name for S&R"] = "LTXVTiledVAEDecode"
    # LTXVTiledVAEDecode swaps input order relative to VAEDecode:
    #   VAEDecode(samples, vae)  →  LTXVTiledVAEDecode(vae, latents)
    inputs = node.get("inputs", [])
    by_name = {inp.get("name"): inp for inp in inputs}
    samples_inp = by_name.pop("samples", None)
    vae_inp = by_name.pop("vae", None)
    if samples_inp and vae_inp:
        samples_inp["name"] = "latents"
        node["inputs"] = [
            {**vae_inp, "name": "vae"},
            {**samples_inp, "name": "latents"},
        ]
    return ["upgraded #1318 VAEDecode → LTXVTiledVAEDecode"]


def _move_subgraph_vae_out(ed: WorkflowEditor) -> list[str]:
    """Surgery: subgraph `#1520 VAEEncode` → top-level VAEEncode.

    Shape change:
      - Add top-level VAEEncode, wire Get_input_image → pixels,
        Get_video_vae → vae.
      - Subgraph input slot 8 changes from IMAGE ('num_guides.image_1')
        to LATENT ('guide_latent').
      - Subgraph-internal: remove #1520, re-route distributor-slot-8
        directly into #1519 LTXVAddLatentGuide.guiding_latent (input 4).
    """
    sg = ed.get_subgraph(0)
    if sg is None:
        return ["(no subgraph; skipped VAE move)"]

    # Note: `remove_subgraph_link` replaces `sg["links"]` with a new list
    # (filter comprehension), so any local ref to `sg["links"]` goes stale
    # after the first removal. Always dereference `sg["links"]` fresh.
    sg_nodes = sg.get("nodes", [])
    sg_inputs = sg.get("inputs", [])

    # Check idempotency: if slot 8 is already LATENT named guide_latent,
    # this migration has already been applied.
    if len(sg_inputs) > 8 and sg_inputs[8].get("type") == "LATENT":
        return ["(VAE move already applied; skipped)"]

    # --- Locate the internal nodes we need ---
    sg_vae = next((n for n in sg_nodes if n.get("type") == "VAEEncode"), None)
    if sg_vae is None:
        return ["(no subgraph VAEEncode; skipped)"]
    sg_vae_id = sg_vae["id"]
    add_guide = next(
        (n for n in sg_nodes if n.get("type") == "LTXVAddLatentGuide"), None,
    )
    if add_guide is None:
        return ["(no LTXVAddLatentGuide in subgraph; skipped)"]
    add_guide_id = add_guide["id"]

    # --- Find the outer subgraph container ---
    # Subgraph-instance `type` is a per-workflow UUID, so we can't scan by
    # type. Scan by structural shape: slot 8 == `num_guides.image_1`,
    # slot 3 == `vae`. If an input-8-matching container has an unexpected
    # slot 3, fail loudly — silently skipping could leave the subgraph
    # half-migrated on a future schema variant.
    container = None
    for n in ed.wf.get("nodes", []):
        inputs = n.get("inputs", [])
        if len(inputs) > 8 and inputs[8].get("name") == "num_guides.image_1":
            if inputs[3].get("name") != "vae":
                raise RuntimeError(
                    f"Subgraph container #{n.get('id')} has "
                    f"num_guides.image_1 at slot 8 but slot 3 is "
                    f"{inputs[3].get('name')!r} (expected 'vae'). Shape "
                    "changed; update the migration."
                )
            container = n
            break
    if container is None:
        return ["(subgraph container not found; skipped)"]

    # Resolve the outer links feeding slot 8 (image) and slot 3 (vae).
    img_link_id = container["inputs"][8].get("link")
    vae_link_id = container["inputs"][3].get("link")
    if img_link_id is None:
        return ["(subgraph slot 8 image input unwired; skipped)"]
    img_link = next(L for L in ed.wf["links"] if L[0] == img_link_id)
    img_src_node, img_src_slot = img_link[1], img_link[2]
    vae_link = next(L for L in ed.wf["links"] if L[0] == vae_link_id) if vae_link_id else None

    # --- Add top-level VAEEncode ---
    # Position: near Get_input_image to keep visual flow intuitive.
    img_src = ed.find_node(img_src_node)
    src_pos = img_src.get("pos", [0, 0])
    new_vae_pos = [float(src_pos[0]) + 260.0, float(src_pos[1])]
    new_vae_id = ed.add_top_level_node(
        node_type="VAEEncode",
        pos=new_vae_pos,
        size=[210, 46],
        inputs=[
            {"name": "pixels", "type": "IMAGE", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
        ],
        outputs=[
            {"name": "LATENT", "type": "LATENT", "links": []},
        ],
        widgets_values=[],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "VAEEncode"},
        title="VAE Encode (init image → guide latent)",
    )

    # --- Rewire outer links ---
    # `remove_link` rebinds `ed.wf["links"]` (same rebind-not-mutate
    # pattern as `remove_subgraph_link`). img_link / vae_link were
    # resolved to tuples above; we only read slot numbers from them.
    # If you add more link-object work below, re-resolve first.
    ed.remove_link(img_link_id)
    ed.add_link(img_src_node, img_src_slot, new_vae_id, 0, "IMAGE")
    # VAE source: reuse whatever feeds subgraph slot 3 (Get_video_vae).
    if vae_link:
        ed.add_link(vae_link[1], vae_link[2], new_vae_id, 1, "VAE")
    ed.add_link(new_vae_id, 0, container["id"], 8, "LATENT")

    # --- Subgraph schema: slot 8 becomes LATENT/guide_latent ---
    slot8 = sg_inputs[8]
    slot8["name"] = "guide_latent"
    slot8["type"] = "LATENT"
    # Outer-side slot metadata on the container
    container["inputs"][8]["name"] = "guide_latent"
    container["inputs"][8]["type"] = "LATENT"

    # --- Subgraph internals ---
    def _link_key(L):
        if isinstance(L, dict):
            return (L.get("origin_id"), L.get("target_id"), L.get("id"))
        return (L[1], L[3], L[0])

    # Two classes of links feed or leave the dying VAEEncode:
    #   A. distributor → #1520 (pixels, vae) — remove when target is it
    #   B. #1520 → #1519 guiding_latent — remove when origin is it
    # Both get dropped; different conditions, not redundant.
    to_remove_link_ids: list[int] = []
    for L in list(sg["links"]):
        origin_id, target_id, link_id = _link_key(L)
        if target_id == sg_vae_id:            # class A
            to_remove_link_ids.append(link_id)
        elif origin_id == sg_vae_id and target_id == add_guide_id:  # class B
            to_remove_link_ids.append(link_id)

    for link_id in to_remove_link_ids:
        ed.remove_subgraph_link(link_id)

    # Remove the VAEEncode node itself.
    sg["nodes"] = [n for n in sg_nodes if n.get("id") != sg_vae_id]

    # Add a new subgraph-internal link: distributor-slot-8 → #1519 guiding_latent
    # guiding_latent is input slot 4 on LTXVAddLatentGuide.
    # IMPORTANT: re-fetch sg["links"] — remove_subgraph_link replaced it.
    live_links = sg["links"]
    def _lid(L):
        return L[0] if isinstance(L, list) else L.get("id", 0)
    new_link_id = (max(_lid(L) for L in live_links) + 1) if live_links else 1
    live_links.append({
        "id": new_link_id,
        "origin_id": -10,
        "origin_slot": 8,
        "target_id": add_guide_id,
        "target_slot": 4,
        "type": "LATENT",
    })
    # Reflect on the LTXVAddLatentGuide.guiding_latent input link ref.
    for inp in add_guide.get("inputs", []):
        if inp.get("name") == "guiding_latent":
            inp["link"] = new_link_id

    return [f"moved subgraph VAEEncode (#{sg_vae_id}) → new top-level VAEEncode (#{new_vae_id}); subgraph slot 8 IMAGE → LATENT"]


def apply(wf_path: Path) -> list[str]:
    ed = WorkflowEditor(wf_path)
    changes: list[str] = []
    changes += _remove_dead_nodes(ed)
    changes += _upgrade_preview_decode(ed)
    changes += _move_subgraph_vae_out(ed)
    if changes:
        ed.save()
    return changes


def main() -> int:
    root = REPO_ROOT / "example_workflows"
    any_changes = False
    for name in TARGET_WORKFLOWS:
        wf_path = root / name
        if not wf_path.exists():
            print(f"skip (missing): {name}")
            continue
        changes = apply(wf_path)
        if changes:
            any_changes = True
            print(f"{name}:")
            for c in changes:
                print(f"  {c}")
        else:
            print(f"{name}: no changes")
    if not any_changes:
        print("no workflows changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
