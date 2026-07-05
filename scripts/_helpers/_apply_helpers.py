"""Shared raw-orjson helpers for fork-and-rewire apply scripts.

Apply scripts that mutate `example_workflows/*.json` deliberately use raw
orjson + these inline helpers (NOT `WorkflowEditor`) per the canonical
fork-and-strip pattern: a debug tool stays usable when the editor it
audits has a bug.

Promoted from `apply_spectrogram_iclora_minimal.py`,
`apply_audio_loop_retake.py`, and `apply_keyframe_batch_encode.py` per
CLAUDE.md "Promote helpers at the 3rd call site, not the 2nd."

Workflow-JSON shape touched:
- `wf["nodes"]` — list of `{id, type, inputs[], outputs[], ...}`
- `wf["links"]` — list of `[link_id, src_id, src_slot, tgt_id, tgt_slot, dtype]`
- `wf["last_node_id"]` / `wf["last_link_id"]` — monotonic counters
- node `inputs[i]` carry `link: int | None`
- node `outputs[i]` carry `links: list[int] | None`
"""

from __future__ import annotations


def in_(name: str, dtype: str) -> dict:
    """Build an input slot dict. (`in_` because `in` is a keyword.)"""
    return {"name": name, "type": dtype, "link": None}


def out(name: str, dtype: str) -> dict:
    return {"name": name, "type": dtype, "links": []}


def widget_in(name: str, dtype: str) -> dict:
    """A widget-converted input slot (carries the widget descriptor).

    Third member of the in_/out family; raw-helper style keeps the explicit
    ``"link": None`` (unlike WorkflowEditor.io_in which omits it when unset).
    """
    return {"name": name, "type": dtype, "widget": {"name": name}, "link": None}


def next_id(wf: dict, key: str = "last_node_id") -> int:
    nid = wf.get(key, 0) + 1
    wf[key] = nid
    return nid


def next_link_id(wf: dict) -> int:
    return next_id(wf, "last_link_id")


def find_node(wf: dict, node_id: int) -> dict | None:
    for n in wf["nodes"]:
        if n["id"] == node_id:
            return n
    return None


def find_link_to_slot(wf: dict, tgt_node: int, tgt_slot: int) -> list | None:
    for l in wf["links"]:
        if isinstance(l, list) and l[3] == tgt_node and l[4] == tgt_slot:
            return l
    return None


def remove_link_by_id(wf: dict, link_id: int) -> None:
    wf["links"] = [l for l in wf["links"] if not (isinstance(l, list) and l[0] == link_id)]
    for n in wf["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") == link_id:
                inp["link"] = None
        for o in n.get("outputs", []):
            if o.get("links"):
                o["links"] = [l for l in o["links"] if l != link_id]


def add_link(wf: dict, src_id: int, src_slot: int, tgt_id: int, tgt_slot: int, dtype: str) -> int:
    lid = next_link_id(wf)
    wf["links"].append([lid, src_id, src_slot, tgt_id, tgt_slot, dtype])
    src = find_node(wf, src_id)
    if src and src_slot < len(src.get("outputs", [])):
        src["outputs"][src_slot].setdefault("links", []).append(lid)
    tgt = find_node(wf, tgt_id)
    if tgt and tgt_slot < len(tgt.get("inputs", [])):
        tgt["inputs"][tgt_slot]["link"] = lid
    return lid


def remove_node_and_links(wf: dict, node_id: int) -> None:
    to_remove = []
    for l in wf["links"]:
        if isinstance(l, list) and (l[1] == node_id or l[3] == node_id):
            to_remove.append(l[0])
    for lid in to_remove:
        remove_link_by_id(wf, lid)
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] != node_id]


def find_input_slot(node: dict, name: str) -> int:
    for i, inp in enumerate(node.get("inputs", [])):
        if inp.get("name") == name:
            return i
    raise ValueError(f"No input {name!r} on node {node.get('id')}")
