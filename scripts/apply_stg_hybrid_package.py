"""Generate `audio-loop-music-video_latent_stg.json` from the baseline
latent workflow. Keeps the authoritative distilled-1.1 sigma chain
(`BasicScheduler linear_quadratic, 8, 1` + `ModelSamplingSD3 shift=13`
+ `KSamplerSelect euler`) and swaps `CFGGuider` for `MultimodalGuider`
+ two `GuiderParameters` (AUDIO/VIDEO both cfg=1, stg=1, modality=1)
so only STG (Spatial-Temporal Guidance) contributes — pure quality
lift on top of the correct noise prediction. Bypasses `LTX2_NAG`.
Re-runnable: `uv run python scripts/apply_stg_hybrid_package.py`.
"""

from workflow_utils import WorkflowEditor

SRC = "example_workflows/audio-loop-music-video_latent.json"
DST = "example_workflows/audio-loop-music-video_latent_stg.json"

# Source-workflow node IDs, stable post-DR1.
NODE_ID_LTX2_NAG = 508
NODE_ID_CFG_GUIDER = 153
NODE_ID_SAMPLER = 161  # SamplerCustomAdvanced
NODE_ID_SET_GUIDER = 575
NODE_ID_LTXV_CONDITIONING = 164
NODE_ID_GET_MODEL = 572  # Set_model feeds this; CFGGuider.model was wired from here


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
    ed.remove_node_and_links(NODE_ID_CFG_GUIDER)

    # 3. Add GuiderParameters (AUDIO) then (VIDEO) chained. cfg=1 disables CFG;
    # stg=1 enables STG; modality_scale=1 disables modality split. Net effect
    # is STG-only on top of the distilled model's deterministic prediction.
    audio_params_id = ed.add_top_level_node(
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
    video_params_id = ed.add_top_level_node(
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
    mm_guider_id = ed.add_top_level_node(
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
