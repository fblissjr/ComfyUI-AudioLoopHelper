"""Add LoopConfigValidator + PreviewAny to a latent workflow.

Loads example_workflows/audio-loop-music-video_latent.json, adds the
validator node wired to the same audio / window source that feeds
AudioLoopController, plus widget inputs for length/schedule/etc. the
user edits directly. Saves as a new file.

Run:
    uv run python scripts/apply_config_validator.py

Produces: example_workflows/audio-loop-music-video_latent_validator.json
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor


SRC = "example_workflows/audio-loop-music-video_latent.json"
OUT = "example_workflows/audio-loop-music-video_latent_validator.json"


def _link_from(ed: WorkflowEditor, link_id: int) -> list:
    """Find top-level link entry [lid, src_node, src_slot, tgt_node, tgt_slot, type]."""
    for link in ed.wf["links"]:
        if isinstance(link, list) and link[0] == link_id:
            return link
    raise ValueError(f"link {link_id} not found")


def _node_input_source(ed: WorkflowEditor, node_id: int, input_name: str) -> tuple[int, int]:
    """Return (src_node_id, src_slot) feeding `input_name` on `node_id`."""
    node = ed.find_node(node_id)
    for inp in node.get("inputs", []):
        if inp.get("name") == input_name:
            link_id = inp.get("link")
            if link_id is None:
                raise ValueError(f"{node['type']}.{input_name} is unlinked")
            link = _link_from(ed, link_id)
            return link[1], link[2]
    raise ValueError(f"{node['type']} has no input named {input_name}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ed = WorkflowEditor(root / SRC)

    # Resolve source nodes by type/link rather than hardcoded IDs so this
    # script doesn't silently wire the wrong nodes if the baseline workflow
    # gets re-exported with different IDs.
    alc = ed.find_nodes_by_type("AudioLoopController")[0]
    audio_src_node, audio_src_slot = _node_input_source(ed, alc["id"], "audio")
    window_src_node, window_src_slot = _node_input_source(ed, alc["id"], "window_seconds")

    empty_ltxv = ed.find_nodes_by_type("EmptyLTXVLatentVideo")[0]
    length_src_node, length_src_slot = _node_input_source(ed, empty_ltxv["id"], "length")

    resize = ed.find_nodes_by_type("ImageResizeKJv2")[0]

    print(
        f"Resolved sources: audio={audio_src_node}.{audio_src_slot}, "
        f"window={window_src_node}.{window_src_slot}, "
        f"length={length_src_node}.{length_src_slot}, "
        f"resize={resize['id']}"
    )

    validator_pos = [2150, 3850]
    preview_pos = [2520, 3850]

    validator_inputs = [
        {"name": "audio", "type": "AUDIO", "link": None},
        {"name": "window_seconds", "type": "FLOAT", "link": None,
         "widget": {"name": "window_seconds"}},
        {"name": "length", "type": "INT", "link": None,
         "widget": {"name": "length"}},
        {"name": "width", "type": "INT", "link": None,
         "widget": {"name": "width"}},
        {"name": "height", "type": "INT", "link": None,
         "widget": {"name": "height"}},
    ]
    validator_outputs = [
        {"name": "report", "type": "STRING", "links": []},
        {"name": "ok", "type": "BOOLEAN", "links": None},
        {"name": "warnings", "type": "INT", "links": None},
        {"name": "errors", "type": "INT", "links": None},
        {"name": "effective_stride_seconds", "type": "FLOAT", "links": None},
    ]
    # widgets_values order matches non-linked schema inputs:
    # window_seconds (default, overridden by link), overlap_seconds, fps,
    # length (default, overridden by link), width (ditto), height (ditto),
    # schedule, resolution_rule, seam_tolerance_seconds.
    validator_widgets = [
        19.88,
        2.0,
        25,
        0,
        0,
        0,
        "",
        "div_by_32",
        0.2,
    ]
    validator_id = ed.add_top_level_node(
        node_type="LoopConfigValidator",
        pos=validator_pos,
        size=[340, 280],
        inputs=validator_inputs,
        outputs=validator_outputs,
        widgets_values=validator_widgets,
        properties={
            "Node name for S&R": "LoopConfigValidator",
            "aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
        },
        title="Loop Config Validator",
    )

    ed.add_link(audio_src_node, audio_src_slot, validator_id, 0, "AUDIO")
    ed.add_link(window_src_node, window_src_slot, validator_id, 1, "FLOAT")
    ed.add_link(length_src_node, length_src_slot, validator_id, 2, "INT")
    # ImageResizeKJv2 outputs: 0=IMAGE, 1=width, 2=height.
    ed.add_link(resize["id"], 1, validator_id, 3, "INT")
    ed.add_link(resize["id"], 2, validator_id, 4, "INT")

    preview_id = ed.add_top_level_node(
        node_type="PreviewAny",
        pos=preview_pos,
        size=[440, 320],
        inputs=[{"name": "source", "type": "*", "link": None}],
        outputs=[],
        widgets_values=[None, None, False],
        properties={
            "Node name for S&R": "PreviewAny",
            "cnr_id": "comfy-core",
            "ver": "0.18.5",
        },
        title="Config Validator Report",
    )
    ed.add_link(validator_id, 0, preview_id, 0, "STRING")

    out_path = root / OUT
    ed.save(out_path)
    print(
        f"Added validator (node {validator_id}) + preview (node {preview_id}) "
        f"to {out_path.name}"
    )


if __name__ == "__main__":
    main()
