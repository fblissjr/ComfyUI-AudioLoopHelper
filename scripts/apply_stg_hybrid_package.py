"""Generate an STG-hybrid variant of the latent workflow.

Keeps the authoritative distilled-1.1 sigma schedule (verified
bit-exact against `coderef/LTX-2/packages/ltx-pipelines/src/
ltx_pipelines/utils/constants.py:DISTILLED_SIGMAS` via
`BasicScheduler linear_quadratic, 8, 1` + `ModelSamplingSD3 shift=13`),
and only swaps in `MultimodalGuider` + two `GuiderParameters` nodes
for STG (Spatial-Temporal Guidance) quality lift.

What changes from `audio-loop-music-video_latent.json`:
  - Replace `CFGGuider` with `MultimodalGuider skip_blocks="28"`.
  - Add `GuiderParameters` x2:
      AUDIO: cfg=1, stg=1, rescale=0.7, modality_scale=1
      VIDEO: cfg=1, stg=1, rescale=0.9, modality_scale=1
    cfg=1 disables CFG branch (distilled path uses no guidance);
    modality_scale=1 disables modality split. Only STG contributes,
    via the perturbed-attention path in the transformer (skip block
    28) — a pure quality lift on top of the distilled noise prediction.
  - Bypass `LTX2_NAG` (mode 0 → 4); STG replaces NAG's quality role.

What STAYS:
  - `BasicScheduler linear_quadratic, 8, 1`
  - `ModelSamplingSD3 shift=13`
  - `KSamplerSelect: euler`
Together these produce the bit-exact distilled-1.1 sigma schedule
`[1.0, 0.994, 0.988, 0.981, 0.975, 0.909, 0.725, 0.422, 0.0]`.

Output: `audio-loop-music-video_latent_stg.json`. Run via:
    uv run python scripts/apply_stg_hybrid_package.py
"""

from workflow_utils import WorkflowEditor

SRC = "example_workflows/audio-loop-music-video_latent.json"
DST = "example_workflows/audio-loop-music-video_latent_stg.json"

# Source-workflow node IDs (stable post-DR1).
NODE_ID_LTX2_NAG = 508
NODE_ID_CFG_GUIDER = 153
NODE_ID_SAMPLER = 161  # SamplerCustomAdvanced
NODE_ID_SET_GUIDER = 575
NODE_ID_LTXV_CONDITIONING = 164
NODE_ID_GET_MODEL = 572  # Set_model feeds this; CFGGuider.model was wired from here


def _remove_node_and_links(ed: WorkflowEditor, node_id: int):
    for link in list(ed.wf["links"]):
        if not isinstance(link, list):
            continue
        lid, src, _, tgt, _, _ = link
        if src == node_id or tgt == node_id:
            ed.remove_link(lid)
    ed.wf["nodes"] = [n for n in ed.wf["nodes"] if n["id"] != node_id]


def _add_top_level_node(
    ed: WorkflowEditor,
    node_type: str,
    pos: list,
    size: list,
    inputs: list,
    outputs: list,
    widgets_values: list,
    properties: dict | None = None,
    title: str | None = None,
) -> int:
    nid = ed.next_node_id()
    node = {
        "id": nid,
        "type": node_type,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": properties or {"Node name for S&R": node_type},
        "widgets_values": widgets_values,
    }
    if title:
        node["title"] = title
    ed.add_node(node)
    return nid


def transform(src_path: str, dst_path: str) -> None:
    ed = WorkflowEditor(src_path)

    # LTXVConditioning output slots: 0=positive, 1=negative.
    cond_pos_slot = 0
    cond_neg_slot = 1
    set_guider_slot = 0
    sampler = ed.find_node(NODE_ID_SAMPLER)
    sampler_guider_slot = next(
        i for i, inp in enumerate(sampler["inputs"]) if inp["name"] == "guider"
    )

    # 1. Bypass NAG — STG replaces its quality role.
    ed.find_node(NODE_ID_LTX2_NAG)["mode"] = 4

    # 2. Drop CFGGuider; its model/pos/neg inputs are re-taken by MultimodalGuider,
    # and its GUIDER output is re-emitted from MultimodalGuider.
    _remove_node_and_links(ed, NODE_ID_CFG_GUIDER)

    # 3. Add GuiderParameters (AUDIO) then (VIDEO) chained. cfg=1 disables CFG;
    # stg=1 enables STG; modality_scale=1 disables modality split. Net effect
    # is STG-only on top of the distilled model's deterministic prediction.
    audio_params_id = _add_top_level_node(
        ed,
        node_type="GuiderParameters",
        pos=[-31, 4600],
        size=[300, 226],
        inputs=[
            {"name": "parameters", "shape": 7, "type": "GUIDER_PARAMETERS", "link": None},
        ],
        outputs=[
            {"name": "GUIDER_PARAMETERS", "type": "GUIDER_PARAMETERS", "links": []},
        ],
        widgets_values=["AUDIO", 1, 1, True, 0.7, 1, 0, True],
        properties={"Node name for S&R": "GuiderParameters"},
        title="GuiderParameters (AUDIO, STG-only)",
    )
    video_params_id = _add_top_level_node(
        ed,
        node_type="GuiderParameters",
        pos=[290, 4600],
        size=[300, 226],
        inputs=[
            {"name": "parameters", "shape": 7, "type": "GUIDER_PARAMETERS", "link": None},
        ],
        outputs=[
            {"name": "GUIDER_PARAMETERS", "type": "GUIDER_PARAMETERS", "links": []},
        ],
        widgets_values=["VIDEO", 1, 1, True, 0.9, 1, 0, True],
        properties={"Node name for S&R": "GuiderParameters"},
        title="GuiderParameters (VIDEO, STG-only)",
    )
    ed.add_link(audio_params_id, 0, video_params_id, 0, "GUIDER_PARAMETERS")

    # 4. Add MultimodalGuider, wired to Get_model + LTXVConditioning (same sources
    # CFGGuider had) and to the chained GuiderParameters.
    mm_guider_id = _add_top_level_node(
        ed,
        node_type="MultimodalGuider",
        pos=[-31, 4886],
        size=[270, 148],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "parameters", "type": "GUIDER_PARAMETERS", "link": None},
        ],
        outputs=[
            {"name": "GUIDER", "type": "GUIDER", "links": []},
        ],
        widgets_values=["28"],
        properties={"Node name for S&R": "MultimodalGuider"},
    )
    ed.add_link(NODE_ID_GET_MODEL, 0, mm_guider_id, 0, "MODEL")
    ed.add_link(NODE_ID_LTXV_CONDITIONING, cond_pos_slot, mm_guider_id, 1, "CONDITIONING")
    ed.add_link(NODE_ID_LTXV_CONDITIONING, cond_neg_slot, mm_guider_id, 2, "CONDITIONING")
    ed.add_link(video_params_id, 0, mm_guider_id, 3, "GUIDER_PARAMETERS")
    ed.add_link(mm_guider_id, 0, NODE_ID_SET_GUIDER, set_guider_slot, "GUIDER")
    ed.add_link(mm_guider_id, 0, NODE_ID_SAMPLER, sampler_guider_slot, "GUIDER")

    ed.save(dst_path)


if __name__ == "__main__":
    transform(SRC, DST)
    print("Done.")
