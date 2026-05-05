"""apply_id_lora_runtime.

Last updated: 2026-04-27

Adds the LTXVReferenceAudio runtime + reference-slice trim to
audio-loop-music-video_latent.json. Three new nodes, all bypassed (mode=4)
by default. User flips mode to enable.

Mechanism: ID-LoRA (paper arxiv:2603.10256, model
AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K) requires both LoRA weights AND a
runtime "reference audio" + identity-guidance pass. The LoRA-weights side
is shipped via apply_lora_chain_bypassed.py (the bypassed
LoraLoaderModelOnly titled "ID-LoRA File"). This script adds the runtime
side: LTXVReferenceAudio nodes that inject ref_audio tokens into
conditioning and install a per-step CFG-like extra forward pass (gated by
identity_guidance_scale).

Two LTXVReferenceAudio instances are needed because the initial render
and loop body have separately-rooted conditioning trees:
  - Initial render: positive/negative from LTXVConditioning(164)
  - Loop body positive: from ConditioningSelectByIteration(1616) (per-iter
    batch-encoded prompt)
  - Loop body negative: from Get_base_cond_neg(648) (which is a Set/Get
    of LTXVConditioning(164).negative)

A single instance can cover at most three of these (initial+loop neg via
164's outputs); loop-body positive routes through a different source and
must be patched separately. The post_cfg_function reads only the positive
conditioning to detect ref_audio, so leaving loop-body positive un-patched
makes ID-LoRA inert for loop iterations — equivalent to "initial render
only", which has the well-known iter-0-vs-iter-N identity-jump problem.

Two parallel instances solve this without post_cfg_function stacking:
each lives on its own model branch, each clones the model independently,
each adds one post_cfg_function that fires only when its branch's CFGGuider
samples. Net cost when un-bypassed: 2x sampling forward passes (the
identity-guidance pass), comparable to a regular CFG run with cfg=3.

Conditioning splice ORDER (post-LTXVConditioning, not pre):
The canonical RuneXX reference workflow
(internal/ref_workflows/RuneXX_LTX-2.3-Workflows/Custom-Audio/
LTX-2.3_-_I2V_T2V_Basic_ID-Lora_reference_audio.json) places
LTXVReferenceAudio BEFORE LTXVConditioning (raw CLIP -> RefAudio -> LTXVConditioning).
We splice AFTER LTXVConditioning(164) instead. Functionally equivalent:
LTXVReferenceAudio.execute calls
`conditioning_set_values(cond, {"ref_audio": ref_audio})` which adds the
key without reading frame_rate — both orderings produce cond containing
both `frame_rate` (from LTXVConditioning) and `ref_audio` (from RefAudio).
Why we deviate: 164 has fanout to 381(LTXVCropGuides) + 153(CFGGuider) +
Set_base_cond_neg(646), and pre-164 splicing would leak ref_audio into
loop body's negative path (via Set_base_cond_neg -> Get(648) -> subgraph
slot 7), which we explicitly want to keep on the parallel-branch loop_id
splice for symmetry. Post-164 splicing isolates the splices to the wires
we name explicitly.

Splice points (verified against current latent.json link IDs):
  Initial render branch (LTXVReferenceAudio_INITIAL):
    in.model       <- replaces link 1572 (SetNode(572) Set_model -> 153)
    in.positive    <- replaces link 1509 (LTXVConditioning(164) -> 153)
    in.negative    <- replaces link 1510 (LTXVConditioning(164) -> 153)
  Loop body branch (LTXVReferenceAudio_LOOP):
    in.model       <- replaces link 3053 (Get_model(654) -> LoopIterationStamp(1618))
    in.positive    <- replaces link 3048 (CondSelectByIter(1616) -> subgraph slot 6)
    in.negative    <- replaces link 2999 (Get_base_cond_neg(648) -> subgraph slot 7)
  Shared inputs:
    reference_audio <- new TrimAudioDuration("ID-LoRA Reference Slice")
                       sourced from Get_orig_audio(604), widgets [30.0, 5.0]
    audio_vae       <- Get_audio_vae(254) (existing virtual GetNode, can tee)

Default widgets:
  TrimAudioDuration:    [start_index=30.0, duration=5.0]
                        first 30s often instrumental intro; 30-35s usually
                        in the first vocal section. User-configurable.
  LTXVReferenceAudio:   [identity_guidance_scale=3.0, start_percent=0.0,
                         end_percent=1.0]
                        Schema defaults; matches coderef/ID-LoRA-2.3
                        examples/one_stage/args.json.

Compatibility:
  - Requires apply_lora_chain_bypassed.py to have been applied first (we
    rely on the "ID-LoRA File" loader being present so the LoRA weights
    are loaded into the model BEFORE LTXVReferenceAudio's post_cfg_function
    fires). Errors out with a clear message if not.
  - Independent of F2/F3/F4/F5/F6/F7 invariants. No subgraph-internal edits.
  - Audit pair-check `id_lora_runtime_consistent` warns (not ERR) on
    half-enabled state — bypassing only one of the two instances produces
    iter-0-vs-loop drift that the user might intentionally want for
    debugging, hence WARN not ERR.

Usage:
    uv run --group dev python scripts/apply_id_lora_runtime.py
    uv run --group dev python scripts/apply_id_lora_runtime.py --revert
    uv run --group dev python scripts/apply_id_lora_runtime.py --dry-run

Idempotent. Re-run is no-op.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

# Existing node IDs we splice between (verified from current latent.json).
SETNODE_MODEL_ID = 572                  # Set_model -> CFGGuider(153) directly
CFGGUIDER_INITIAL_ID = 153              # initial render's guider
LTXVCONDITIONING_ID = 164               # initial render's positive/negative source
GETNODE_MODEL_ID = 654                  # Get_model -> LoopIterationStamp(1618)
LOOP_ITERATION_STAMP_ID = 1618          # loop's model entry
COND_SELECT_BY_ITER_ID = 1616           # loop positive (per-iter batch encoder selector)
GETNODE_BASE_COND_NEG_ID = 648          # loop negative (static base via Set/Get)
SUBGRAPH_INVOKER_ID = 843               # the loop subgraph instance
GETNODE_ORIG_AUDIO_ID = 604             # full song's audio (untrimmed)
GETNODE_AUDIO_VAE_ID = 254              # audio VAE

LORA_CHAIN_TITLE_MARKER = "Style/Generic LoRA"  # set by apply_lora_chain_bypassed.py

# Title markers for our new nodes (idempotence + audit identification).
TITLE_REFERENCE_TRIM = "ID-LoRA Reference Slice"
TITLE_REFAUDIO_INITIAL = "LTXV Reference Audio (ID-LoRA initial render)"
TITLE_REFAUDIO_LOOP = "LTXV Reference Audio (ID-LoRA loop body)"

# Default widget values
REF_TRIM_START_S = 30.0
REF_TRIM_DURATION_S = 5.0
ID_GUIDANCE_SCALE = 3.0
ID_GUIDANCE_START_PCT = 0.0
ID_GUIDANCE_END_PCT = 1.0

# Layout: position the new nodes near the existing LoRA chain so the
# ID-LoRA pipeline reads as a unit on the canvas.
_X_TRIM = -750
_Y_TRIM = 6650
_X_REFAUDIO = -300
_Y_INITIAL = 5900
_Y_LOOP = 6050


def _is_already_built(ed: WorkflowEditor) -> tuple[bool, list[dict]]:
    titles = (TITLE_REFERENCE_TRIM, TITLE_REFAUDIO_INITIAL, TITLE_REFAUDIO_LOOP)
    found: dict[str, dict] = {}
    for n in ed.wf.get("nodes", []):
        if n.get("title") in titles:
            found[n["title"]] = n
    if len(found) == 3:
        return True, [found[t] for t in titles]
    return False, []


def _require_lora_chain(ed: WorkflowEditor) -> bool:
    for n in ed.wf.get("nodes", []):
        if n.get("title") == LORA_CHAIN_TITLE_MARKER:
            return True
    return False


def _add_node(ed: WorkflowEditor, *, node_type: str, title: str, pos: list,
              size: list, inputs: list, outputs: list,
              widgets_values: list, properties: dict | None = None) -> int:
    """Add a top-level node and stamp mode=4 (bypass) post-creation.
    `add_top_level_node` doesn't accept a mode kwarg — see CLAUDE.md note."""
    nid = ed.add_top_level_node(
        node_type=node_type, pos=pos, size=size,
        inputs=inputs, outputs=outputs,
        widgets_values=widgets_values,
        properties=properties or {"Node name for S&R": node_type},
        title=title,
    )
    ed.find_node(nid)["mode"] = 4
    return nid


def _apply(ed: WorkflowEditor) -> tuple[bool, str]:
    if not _require_lora_chain(ed):
        return False, (
            "skip (apply_lora_chain_bypassed.py must be applied first — "
            f"no node titled '{LORA_CHAIN_TITLE_MARKER}' found)"
        )

    built, _ = _is_already_built(ed)
    if built:
        return False, "no change (ID-LoRA runtime already wired)"

    # Required source nodes
    missing = ed.require_nodes((
        SETNODE_MODEL_ID, CFGGUIDER_INITIAL_ID, LTXVCONDITIONING_ID,
        GETNODE_MODEL_ID, LOOP_ITERATION_STAMP_ID, COND_SELECT_BY_ITER_ID,
        GETNODE_BASE_COND_NEG_ID, SUBGRAPH_INVOKER_ID,
        GETNODE_ORIG_AUDIO_ID, GETNODE_AUDIO_VAE_ID,
    ))
    if missing:
        return False, f"skip (missing required nodes: {missing})"

    # 1. Reference-slice TrimAudioDuration
    trim_id = _add_node(
        ed,
        node_type="TrimAudioDuration",
        title=TITLE_REFERENCE_TRIM,
        pos=[_X_TRIM, _Y_TRIM], size=[270, 82],
        inputs=[
            {"name": "audio", "type": "AUDIO", "link": None},
            {"name": "start_index", "type": "FLOAT",
             "widget": {"name": "start_index"}, "link": None},
            {"name": "duration", "type": "FLOAT",
             "widget": {"name": "duration"}, "link": None},
        ],
        outputs=[{"name": "AUDIO", "type": "AUDIO", "links": []}],
        widgets_values=[REF_TRIM_START_S, REF_TRIM_DURATION_S],
        properties={"cnr_id": "comfy-core",
                    "Node name for S&R": "TrimAudioDuration"},
    )
    # Wire trim's audio input from Get_orig_audio
    ed.add_link(GETNODE_ORIG_AUDIO_ID, 0, trim_id, 0, "AUDIO")

    # Helper: define both LTXVReferenceAudio instances with the same shape
    def _add_refaudio(title: str, y: int) -> int:
        return _add_node(
            ed,
            node_type="LTXVReferenceAudio",
            title=title,
            pos=[_X_REFAUDIO, y], size=[400, 200],
            inputs=[
                {"name": "model", "type": "MODEL", "link": None},
                {"name": "positive", "type": "CONDITIONING", "link": None},
                {"name": "negative", "type": "CONDITIONING", "link": None},
                {"name": "reference_audio", "type": "AUDIO", "link": None},
                {"label": "Audio VAE", "name": "audio_vae", "type": "VAE",
                 "link": None},
                {"name": "identity_guidance_scale", "type": "FLOAT",
                 "widget": {"name": "identity_guidance_scale"}, "link": None},
                {"name": "start_percent", "type": "FLOAT",
                 "widget": {"name": "start_percent"}, "link": None},
                {"name": "end_percent", "type": "FLOAT",
                 "widget": {"name": "end_percent"}, "link": None},
            ],
            outputs=[
                {"name": "MODEL", "type": "MODEL", "links": []},
                {"name": "positive", "type": "CONDITIONING", "links": []},
                {"name": "negative", "type": "CONDITIONING", "links": []},
            ],
            widgets_values=[ID_GUIDANCE_SCALE, ID_GUIDANCE_START_PCT,
                            ID_GUIDANCE_END_PCT],
            properties={"cnr_id": "comfy-core",
                        "Node name for S&R": "LTXVReferenceAudio"},
        )

    initial_id = _add_refaudio(TITLE_REFAUDIO_INITIAL, _Y_INITIAL)
    loop_id = _add_refaudio(TITLE_REFAUDIO_LOOP, _Y_LOOP)

    # Wire shared inputs (audio + audio_vae) on both instances
    for refaudio_id in (initial_id, loop_id):
        ed.add_link(trim_id, 0, refaudio_id, 3, "AUDIO")              # reference_audio
        ed.add_link(GETNODE_AUDIO_VAE_ID, 0, refaudio_id, 4, "VAE")   # audio_vae

    # Splice INITIAL branch:
    # Replace SetNode(572) -> CFGGuider(153).model with chain through initial_id
    _splice(ed, src_node=SETNODE_MODEL_ID, src_slot=0,
            tgt_node=CFGGUIDER_INITIAL_ID, tgt_slot=0,
            via_node=initial_id, via_in_slot=0, via_out_slot=0, dtype="MODEL")
    # Replace 164.positive -> 153.positive with chain through initial_id.positive
    _splice(ed, src_node=LTXVCONDITIONING_ID, src_slot=0,
            tgt_node=CFGGUIDER_INITIAL_ID, tgt_slot=1,
            via_node=initial_id, via_in_slot=1, via_out_slot=1, dtype="CONDITIONING")
    # Replace 164.negative -> 153.negative with chain through initial_id.negative
    _splice(ed, src_node=LTXVCONDITIONING_ID, src_slot=1,
            tgt_node=CFGGUIDER_INITIAL_ID, tgt_slot=2,
            via_node=initial_id, via_in_slot=2, via_out_slot=2, dtype="CONDITIONING")

    # Splice LOOP branch:
    # Get_model(654) -> LoopIterationStamp(1618).model becomes via loop_id
    _splice(ed, src_node=GETNODE_MODEL_ID, src_slot=0,
            tgt_node=LOOP_ITERATION_STAMP_ID, tgt_slot=0,
            via_node=loop_id, via_in_slot=0, via_out_slot=0, dtype="MODEL")
    # CondSelectByIter(1616) -> subgraph(843).positive (slot 6) becomes via loop_id
    _splice(ed, src_node=COND_SELECT_BY_ITER_ID, src_slot=0,
            tgt_node=SUBGRAPH_INVOKER_ID, tgt_slot=6,
            via_node=loop_id, via_in_slot=1, via_out_slot=1, dtype="CONDITIONING")
    # Get_base_cond_neg(648) -> subgraph(843).negative (slot 7) becomes via loop_id
    _splice(ed, src_node=GETNODE_BASE_COND_NEG_ID, src_slot=0,
            tgt_node=SUBGRAPH_INVOKER_ID, tgt_slot=7,
            via_node=loop_id, via_in_slot=2, via_out_slot=2, dtype="CONDITIONING")

    return True, (
        f"wired (#{trim_id} ref slice -> #{initial_id} initial RefAudio + "
        f"#{loop_id} loop RefAudio); all 3 bypassed (mode=4)"
    )


def _splice(ed: WorkflowEditor, *, src_node: int, src_slot: int,
            tgt_node: int, tgt_slot: int,
            via_node: int, via_in_slot: int, via_out_slot: int,
            dtype: str) -> None:
    """Reroute the existing src->tgt link to go src->via->tgt."""
    existing = ed.find_link_to_slot(tgt_node, tgt_slot)
    if existing is None:
        raise RuntimeError(
            f"expected an existing link to #{tgt_node}.in[{tgt_slot}] but found none"
        )
    existing_link_id, existing_src, existing_src_slot, *_ = existing
    if existing_src != src_node or existing_src_slot != src_slot:
        raise RuntimeError(
            f"link to #{tgt_node}.in[{tgt_slot}] expected src "
            f"#{src_node}.out[{src_slot}] but found "
            f"#{existing_src}.out[{existing_src_slot}]"
        )
    ed.remove_link(existing_link_id)
    ed.add_link(src_node, src_slot, via_node, via_in_slot, dtype)
    ed.add_link(via_node, via_out_slot, tgt_node, tgt_slot, dtype)


def _revert(ed: WorkflowEditor) -> tuple[bool, str]:
    built, nodes = _is_already_built(ed)
    if not built:
        return False, "already reverted (no ID-LoRA runtime nodes found)"
    _trim, initial, loop = nodes

    # For each spliced edge, walk the via_node back to the src and reattach
    # directly to the tgt. Order matters: do initial first, then loop, then
    # delete the bypassed nodes.
    splices = [
        # (via_node, via_out_slot, tgt_node, tgt_slot, dtype)
        (initial["id"], 0, CFGGUIDER_INITIAL_ID, 0, "MODEL"),
        (initial["id"], 1, CFGGUIDER_INITIAL_ID, 1, "CONDITIONING"),
        (initial["id"], 2, CFGGUIDER_INITIAL_ID, 2, "CONDITIONING"),
        (loop["id"], 0, LOOP_ITERATION_STAMP_ID, 0, "MODEL"),
        (loop["id"], 1, SUBGRAPH_INVOKER_ID, 6, "CONDITIONING"),
        (loop["id"], 2, SUBGRAPH_INVOKER_ID, 7, "CONDITIONING"),
    ]
    for via_node, via_out_slot, tgt_node, tgt_slot, dtype in splices:
        # Find the upstream that feeds via_node's matching input slot
        in_slot = via_out_slot  # for LTXVReferenceAudio, in slot N matches out slot N for the 3 branches
        upstream = ed.find_link_to_slot(via_node, in_slot)
        if upstream is None:
            continue
        _, src, src_slot, *_ = upstream
        # Reattach src -> tgt directly
        ed.add_link(src, src_slot, tgt_node, tgt_slot, dtype)

    # Strip the three bypassed nodes (and their remaining links).
    for n in nodes:
        ed.remove_node_and_links(n["id"])

    return True, "reverted (3 nodes removed, original links restored)"


def apply(revert: bool, dry_run: bool, wf_path: Path) -> int:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        print(f"load error: {e}")
        return 1

    changed, message = _revert(ed) if revert else _apply(ed)
    prefix = "would " if dry_run and changed else ""
    print(f"  {wf_path.relative_to(REPO_ROOT)}: {prefix}{message}")
    if changed and not dry_run:
        ed.save(wf_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW))
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run, Path(args.workflow))


if __name__ == "__main__":
    sys.exit(main())
