"""Generate an upstream-sampling-package variant of the latent workflow.

Takes `audio-loop-music-video_latent.json` as the source and produces
`audio-loop-music-video_latent_upstream.json` which swaps the sampling
stack for the one Lightricks ships in
`LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json` (their 2026-04-13
distilled-1.1 release). That means:

  - Remove `ModelSamplingSD3` (shift=13) — LTXVScheduler bakes shift in.
  - Remove `BasicScheduler (linear_quadratic, 8, 1)` — replaced by
    `LTXVScheduler [15, 2.05, 0.95, True, 0.1]` (15 steps, dynamic shift
    interpolating max_shift 2.05 → base_shift 0.95, stretch_terminal_snr,
    terminal sigma 0.1).
  - Remove `CFGGuider (CFG=1)` — replaced by `MultimodalGuider` with
    two `GuiderParameters` nodes chained (AUDIO cfg=7, VIDEO cfg=3,
    both stg=1) and STG skip_blocks="28".
  - Change `KSamplerSelect` widget `euler` → `euler_ancestral_cfg_pp`.
  - Bypass `LTX2_NAG` (mode 0 → 4) — MultimodalGuider handles guidance
    natively via STG, no need for NAG on top.

Re-runnable. Destination is overwritten. Run via:
    uv run python scripts/apply_upstream_sampling_package.py

Then delete + re-add the affected nodes in ComfyUI UI (JSON slot
indices are baked at save time per CLAUDE.md — but since this script
builds from scratch, a fresh load in ComfyUI picks them up).
"""

from pathlib import Path

from workflow_utils import WorkflowEditor

SRC = "example_workflows/audio-loop-music-video_latent.json"
DST = "example_workflows/audio-loop-music-video_latent_upstream.json"


# Source-workflow node IDs we depend on (stable post-DR1).
NODE_ID_LTX2_NAG = 508
NODE_ID_KSAMPLER_SELECT = 154
NODE_ID_CFG_GUIDER = 153
NODE_ID_BASIC_SCHEDULER = 1421
NODE_ID_MODEL_SAMPLING_SD3 = 1513
NODE_ID_VISUALIZE_SIGMAS = 1422
NODE_ID_SAMPLER = 161  # SamplerCustomAdvanced
NODE_ID_SET_GUIDER = 575
NODE_ID_LTXV_CONDITIONING = 164
NODE_ID_GET_MODEL = 572  # Set_model feeds this; CFGGuider.model was wired from here
NODE_ID_AV_CONCAT = 350  # LTXVConcatAVLatent — source for LTXVScheduler's latent input


def _remove_node_and_links(ed: WorkflowEditor, node_id: int):
    """Drop a node plus every link that touches it. Link IDs on surviving
    nodes are left dangling only if we don't re-wire explicitly — callers
    must re-wire before using a dropped input/output slot."""
    for link in list(ed.wf["links"]):
        if not isinstance(link, list):
            continue
        lid, src, _, tgt, _, _ = link
        if src == node_id or tgt == node_id:
            ed.remove_link(lid)
    ed.wf["nodes"] = [n for n in ed.wf["nodes"] if n["id"] != node_id]


def _set_node_mode(ed: WorkflowEditor, node_id: int, mode: int):
    ed.find_node(node_id)["mode"] = mode


def _set_widget_value(ed: WorkflowEditor, node_id: int, index: int, value):
    ed.find_node(node_id)["widgets_values"][index] = value


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
    # Set_guider has one input; VisualizeSigmasKJ has one input.
    set_guider_slot = 0
    vis_sigmas_input_slot = 0
    # Index of SamplerCustomAdvanced's "guider" input (schema may reorder).
    sampler = ed.find_node(NODE_ID_SAMPLER)
    sampler_guider_slot = next(
        i for i, inp in enumerate(sampler["inputs"]) if inp["name"] == "guider"
    )

    # --- 1. Bypass LTX2_NAG (MultimodalGuider handles guidance natively).
    _set_node_mode(ed, NODE_ID_LTX2_NAG, 4)

    # --- 2. Change sampler widget.
    _set_widget_value(ed, NODE_ID_KSAMPLER_SELECT, 0, "euler_ancestral_cfg_pp")

    # --- 3. Drop the old sampling-package nodes. This severs the model chain
    # from LTX2SamplingPreviewOverride → ModelSamplingSD3 → BasicScheduler
    # and the guider chain CFGGuider → SamplerCustomAdvanced / Set_guider.
    _remove_node_and_links(ed, NODE_ID_CFG_GUIDER)
    _remove_node_and_links(ed, NODE_ID_BASIC_SCHEDULER)
    _remove_node_and_links(ed, NODE_ID_MODEL_SAMPLING_SD3)

    # --- 4. Add LTXVScheduler (replaces BasicScheduler + ModelSamplingSD3).
    # Takes LATENT as input (from LTXVConcatAVLatent) — shift is internal
    # via max_shift/base_shift widgets.
    ltxv_sched_id = _add_top_level_node(
        ed,
        node_type="LTXVScheduler",
        pos=[55, 5300],
        size=[270, 154],
        inputs=[
            {"name": "latent", "shape": 7, "type": "LATENT", "link": None},
        ],
        outputs=[
            {"name": "SIGMAS", "type": "SIGMAS", "links": []},
        ],
        widgets_values=[15, 2.05, 0.95, True, 0.1],
        properties={"cnr_id": "comfy-core", "Node name for S&R": "LTXVScheduler"},
    )
    # Wire LTXVScheduler.latent ← LTXVConcatAVLatent.latent output.
    av_concat = ed.find_node(NODE_ID_AV_CONCAT)
    av_out_slot = next(
        i for i, out in enumerate(av_concat["outputs"]) if out["name"] == "latent"
    )
    ed.add_link(NODE_ID_AV_CONCAT, av_out_slot, ltxv_sched_id, 0, "LATENT")
    # Wire LTXVScheduler.SIGMAS → VisualizeSigmasKJ input.
    ed.add_link(ltxv_sched_id, 0, NODE_ID_VISUALIZE_SIGMAS, vis_sigmas_input_slot, "SIGMAS")

    # --- 5. Add GuiderParameters (AUDIO) + GuiderParameters (VIDEO) chained.
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
        widgets_values=["AUDIO", 7, 1, True, 0.7, 3, 0, True],
        properties={"Node name for S&R": "GuiderParameters"},
        title="GuiderParameters (AUDIO, cfg=7)",
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
        widgets_values=["VIDEO", 3, 1, True, 0.9, 3, 0, True],
        properties={"Node name for S&R": "GuiderParameters"},
        title="GuiderParameters (VIDEO, cfg=3)",
    )
    ed.add_link(audio_params_id, 0, video_params_id, 0, "GUIDER_PARAMETERS")

    # --- 6. Add MultimodalGuider. Model comes from Get_model (same source
    # CFGGuider was using), conditioning from LTXVConditioning.
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
    # Wire MultimodalGuider inputs.
    ed.add_link(NODE_ID_GET_MODEL, 0, mm_guider_id, 0, "MODEL")
    ed.add_link(NODE_ID_LTXV_CONDITIONING, cond_pos_slot, mm_guider_id, 1, "CONDITIONING")
    ed.add_link(NODE_ID_LTXV_CONDITIONING, cond_neg_slot, mm_guider_id, 2, "CONDITIONING")
    ed.add_link(video_params_id, 0, mm_guider_id, 3, "GUIDER_PARAMETERS")
    # Wire MultimodalGuider.GUIDER → Set_guider AND → SamplerCustomAdvanced.guider.
    ed.add_link(mm_guider_id, 0, NODE_ID_SET_GUIDER, set_guider_slot, "GUIDER")
    ed.add_link(mm_guider_id, 0, NODE_ID_SAMPLER, sampler_guider_slot, "GUIDER")

    # --- 7. Save.
    ed.save(dst_path)


if __name__ == "__main__":
    Path(__file__).parent  # ensure relative paths work from repo root
    transform(SRC, DST)
    print("Done.")
