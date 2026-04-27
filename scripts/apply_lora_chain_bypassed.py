"""apply_lora_chain_bypassed.

Last updated: 2026-04-27

Adds a three-loader LoRA chain to `audio-loop-music-video_latent.json`,
bypassed by default. User flips `mode: 0` per loader to enable any of:
  - ID-LoRA file (LoRA weights for audio-conditioned identity transfer per
    AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K, paper arxiv:2603.10256)
  - IC-LoRA file (e.g. MergeGreen, Outpaint, Cameraman, Motion-Track) —
    visual reference adapters; pair with `LTXAddVideoICLoRAGuide` on the
    conditioning side via `apply_iclora_initial_render.py` for full effect.
  - Style/Generic LoRA file (any standard LTX 2.3-compatible LoRA)

Splice site: model chain at #503 LTX2SamplingPreviewOverride -> #572
SetNode("model"). After splice:
    503 -> ID-LoRA -> IC-LoRA -> Style -> 572
The Set/Get(654) pair feeds both the initial-render CFGGuider AND the loop
subgraph's MODEL slot via LoopIterationStamp(1618), so any patched MODEL
flows everywhere.

Bypass semantics (CLAUDE.md "ComfyUI gotchas"): mode=4 nodes pass inputs
to outputs of the same TYPE. All three loaders take MODEL in / MODEL out
so the chain is a no-op when fully bypassed. The `LTXICLoRALoaderModelOnly`
also emits a `latent_downscale_factor` output — when bypassed this
dead-ends, which is fine because no consumer is wired by default.

To use any single LoRA:
  1. Open the workflow in ComfyUI.
  2. Right-click the loader you want -> "Bypass" (toggle off mode=4).
  3. Set the `lora_name` widget to the .safetensors filename under
     ComfyUI/models/loras/.
  4. Set the strength widget (default 1.0).
  5. For IC-LoRA: separately run `scripts/apply_iclora_initial_render.py`
     to add the `LTXAddVideoICLoRAGuide` on the conditioning side; un-bypass
     it too.
  6. For ID-LoRA full reference-audio pipeline (vs. just LoRA weight
     patching): the runtime `LTXVReferenceAudio` node is not wired by this
     script — that's a separate follow-up that requires reference_audio +
     audio_vae wires. Loading just the LoRA weights via the bypassed
     ID-LoRA loader gives you the model's trained-in identity bias without
     the per-render reference clip.

Compatibility:
  - Strict superset of `apply_id_lora_initial_render.py` and
    `apply_iclora_initial_render.py` — those produce STAGED variants under
    `internal/scratch/` and use `LoraLoaderModelOnly` only. This script
    edits the production workflow in place with the proper LTX-specific
    `LTXICLoRALoaderModelOnly` for the IC-LoRA slot.
  - Independent of F2/F3/F4/F5/F6/F7. No interaction with apply_sage_mode
    or apply_iterations_autowire.

Usage:
    uv run --group dev python scripts/apply_lora_chain_bypassed.py
    uv run --group dev python scripts/apply_lora_chain_bypassed.py --revert
    uv run --group dev python scripts/apply_lora_chain_bypassed.py --dry-run

Idempotent. Re-run is no-op. `--revert` removes the three loaders and
restores the direct 503 -> 572 wire.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

LTX2_PREVIEW_OVERRIDE_ID = 503
SETNODE_MODEL_ID = 572

# Title markers — used as idempotence signal because LoraLoaderModelOnly is
# a generic core node and could match an unrelated existing instance.
TITLE_ID_LORA = "ID-LoRA File (audio-conditioned identity)"
TITLE_IC_LORA = "IC-LoRA File (visual reference adapter)"
TITLE_STYLE_LORA = "Style/Generic LoRA"

# Default widget values. User overrides these in the ComfyUI UI when
# enabling. .safetensors paths are placeholders documenting the canonical
# files — workflow ships with bypass on, so the placeholders never load.
DEFAULT_ID_LORA_FILE = "LTX-2.3-ID-LoRA-CelebVHQ-3K/lora_weights.safetensors"
DEFAULT_IC_LORA_FILE = "MergeGreen_IC-lora_ltx2.3.safetensors"
DEFAULT_STYLE_LORA_FILE = "your_style_lora.safetensors"
DEFAULT_STRENGTH = 1.0

# Layout: positioned vertically under the LTX2_NAG / LTX2SamplingPreviewOverride
# stack so the chain is visible when the workflow is loaded. X aligned with
# the existing model-chain nodes; Y just below the preview override.
_X = -300
_Y_BASE = 5500
_Y_STEP = 130


def _is_already_built(ed: WorkflowEditor) -> tuple[bool, list[dict]]:
    """Return (built, [id_node, ic_node, style_node]) where built means all
    three titled markers exist."""
    titles = (TITLE_ID_LORA, TITLE_IC_LORA, TITLE_STYLE_LORA)
    found: dict[str, dict] = {}
    for n in ed.wf.get("nodes", []):
        title = n.get("title")
        if title in titles:
            found[title] = n
    if len(found) == 3:
        return True, [found[t] for t in titles]
    return False, []


def _add_loader(ed: WorkflowEditor, *, node_type: str, title: str, y: int,
                widgets: list, extra_outputs: list[dict] | None = None) -> int:
    """Add a bypassed (mode=4) LoRA loader node. All three loaders share
    the (model in, model out) shape; LTXICLoRALoaderModelOnly additionally
    emits a latent_downscale_factor output (covered via extra_outputs).

    `add_top_level_node` doesn't accept a `mode` kwarg, so set it after
    creation. mode=4 means bypass — the node appears in the graph but
    passes inputs to outputs of the same type without executing."""
    outputs = [{"name": "model", "type": "MODEL", "links": []}]
    if extra_outputs:
        outputs.extend(extra_outputs)
    nid = ed.add_top_level_node(
        node_type=node_type,
        pos=[_X, y],
        size=[360, 90],
        inputs=[{"name": "model", "type": "MODEL", "link": None}],
        outputs=outputs,
        widgets_values=widgets,
        properties={"Node name for S&R": node_type},
        title=title,
    )
    ed.find_node(nid)["mode"] = 4
    return nid


def _apply(ed: WorkflowEditor) -> str:
    built, _ = _is_already_built(ed)
    if built:
        return "no change (chain already present)"

    existing = ed.find_link_to_slot(SETNODE_MODEL_ID, 0)
    if existing is None:
        return f"skip (#{SETNODE_MODEL_ID}.in[0] has no inbound link)"
    existing_link_id, src_node, src_slot, *_ = existing
    if src_node != LTX2_PREVIEW_OVERRIDE_ID:
        return (
            f"skip (#{SETNODE_MODEL_ID}.in[0] inbound from #{src_node}, "
            f"expected #{LTX2_PREVIEW_OVERRIDE_ID})"
        )

    id_id = _add_loader(
        ed,
        node_type="LoraLoaderModelOnly",
        title=TITLE_ID_LORA,
        y=_Y_BASE,
        widgets=[DEFAULT_ID_LORA_FILE, DEFAULT_STRENGTH],
    )
    ic_id = _add_loader(
        ed,
        node_type="LTXICLoRALoaderModelOnly",
        title=TITLE_IC_LORA,
        y=_Y_BASE + _Y_STEP,
        widgets=[DEFAULT_IC_LORA_FILE, DEFAULT_STRENGTH],
        extra_outputs=[
            {"name": "latent_downscale_factor", "type": "FLOAT", "links": []},
        ],
    )
    style_id = _add_loader(
        ed,
        node_type="LoraLoaderModelOnly",
        title=TITLE_STYLE_LORA,
        y=_Y_BASE + 2 * _Y_STEP,
        widgets=[DEFAULT_STYLE_LORA_FILE, DEFAULT_STRENGTH],
    )

    # Splice: 503 -> 572 becomes 503 -> ID-LoRA -> IC-LoRA -> Style -> 572.
    # remove_link cleans both the link record and the source's output list.
    ed.remove_link(existing_link_id)
    ed.add_link(LTX2_PREVIEW_OVERRIDE_ID, src_slot, id_id, 0, "MODEL")
    ed.add_link(id_id, 0, ic_id, 0, "MODEL")
    ed.add_link(ic_id, 0, style_id, 0, "MODEL")
    ed.add_link(style_id, 0, SETNODE_MODEL_ID, 0, "MODEL")

    return f"chain inserted (#{id_id} ID-LoRA -> #{ic_id} IC-LoRA -> #{style_id} Style)"


def _revert(ed: WorkflowEditor) -> str:
    built, nodes = _is_already_built(ed)
    if not built:
        return "already reverted (no chain found)"
    id_n, _ic_n, style_n = nodes

    # Restore direct 503 -> 572 link, then strip the three loader nodes.
    incoming = ed.find_link_to_slot(id_n["id"], 0)
    if incoming is None:
        return "skip (revert: ID-LoRA loader has no inbound link)"
    _link_id, src_node, src_slot, *_ = incoming

    # Find the link from style -> 572 to know where to reattach.
    setnode_in = ed.find_link_to_slot(SETNODE_MODEL_ID, 0)
    if setnode_in is None:
        return "skip (revert: SetNode model has no inbound link)"
    _, style_src, _, *_ = setnode_in
    if style_src != style_n["id"]:
        return f"skip (revert: SetNode inbound is #{style_src}, not the Style loader)"

    for n in nodes:
        # remove_node_and_links lives in WorkflowEditor; falls back to
        # manual cleanup if not available.
        for inp in n.get("inputs") or []:
            lid = inp.get("link")
            if lid is not None:
                ed.remove_link(lid)
        for out in n.get("outputs") or []:
            for lid in list(out.get("links") or []):
                ed.remove_link(lid)
    # Strip nodes
    keep_ids = {n["id"] for n in nodes}
    ed.wf["nodes"] = [n for n in ed.wf["nodes"] if n["id"] not in keep_ids]

    ed.add_link(src_node, src_slot, SETNODE_MODEL_ID, 0, "MODEL")
    return "reverted (3 loaders removed, direct 503 -> 572 restored)"


def apply(revert: bool, dry_run: bool, wf_path: Path) -> int:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        print(f"load error: {e}")
        return 1

    if dry_run:
        # Snapshot then discard.
        result = _revert(ed) if revert else _apply(ed)
        print(f"  {wf_path.relative_to(REPO_ROOT)}: would {result}")
        return 0

    result = _revert(ed) if revert else _apply(ed)
    print(f"  {wf_path.relative_to(REPO_ROOT)}: {result}")
    if "no change" not in result and "already" not in result and "skip" not in result:
        ed.save(wf_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW),
                    help="Path to workflow JSON (default: latent workflow)")
    ap.add_argument("--revert", action="store_true",
                    help="Remove the chain and restore direct 503 -> 572 wire")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run, Path(args.workflow))


if __name__ == "__main__":
    sys.exit(main())
