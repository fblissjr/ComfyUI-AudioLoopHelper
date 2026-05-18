"""Verify TensorLoop containment for a flat-canvas loop workflow.

Walks forward from ``TensorLoopOpen`` data outputs (previous_value,
accumulated_count, current_iteration — NOT flow_control) and backward
from ``TensorLoopClose`` data inputs (processed, stop — NOT flow_control).
Every node forward-reachable from TLO must also be backward-reachable
from TLC. The intersection is what ComfyUI-NativeLooping's
``_WhileLoopClose._explore_dependencies`` clones per iter. Side
branches that depend on TLO state but never feed TLC will be
considered iter-independent and execute ONCE statically — the
canonical "first iter looks right, subsequent iters frozen" bug.

KJNodes ``SetNode`` / ``GetNode`` buses are traversed as virtual edges
(same bus name = single dataflow). Bypassed nodes (``mode == 4``) are
excluded from both directions.

Usage::

    uv run python scripts/verify_loop_containment.py <workflow.json>

Exits 0 if all containment OK, 1 if violations found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# TLO output slots that carry per-iter DATA (not the flow-control wire).
_TLO_DATA_OUT_SLOTS = {1, 2, 3}  # previous_value, accumulated_count, current_iteration

# TLC input slots that carry per-iter DATA (not the flow-control wire).
_TLC_DATA_IN_SLOTS = {1, 2}      # processed, stop


def _build_graph(wf: dict) -> tuple[dict[int, dict], dict[int, set], dict[int, set]]:
    """Build node table + forward/reverse adjacency. Each adjacency entry is a
    tuple (neighbor_id, src_slot, tgt_slot, dtype). SetNode → GetNode (matched
    by bus name) gets a virtual edge with src_slot=tgt_slot=0, dtype='BUS'.
    Bypassed nodes are traversed (ComfyUI bypass passes inputs to outputs of
    matching type, so dataflow continues), but bypassed bus endpoints are
    skipped (an active SetNode → bypassed GetNode is a dead bus).
    """
    nodes = {n["id"]: n for n in wf.get("nodes", [])}
    forward: dict[int, set] = defaultdict(set)
    reverse: dict[int, set] = defaultdict(set)

    for link in wf.get("links", []):
        if not isinstance(link, list) or len(link) < 6:
            continue
        _, src, src_slot, tgt, tgt_slot, dtype = link
        forward[src].add((tgt, src_slot, tgt_slot, dtype))
        reverse[tgt].add((src, src_slot, tgt_slot, dtype))

    sets_by_bus: dict[str, list[int]] = defaultdict(list)
    gets_by_bus: dict[str, list[int]] = defaultdict(list)
    for n in wf.get("nodes", []):
        if n.get("mode") == 4:
            continue
        wv = n.get("widgets_values") or []
        if not wv:
            continue
        if n.get("type") == "SetNode":
            sets_by_bus[wv[0]].append(n["id"])
        elif n.get("type") == "GetNode":
            gets_by_bus[wv[0]].append(n["id"])

    for bus, set_ids in sets_by_bus.items():
        for sid in set_ids:
            for gid in gets_by_bus.get(bus, ()):
                forward[sid].add((gid, 0, 0, "BUS"))
                reverse[gid].add((sid, 0, 0, "BUS"))

    return nodes, forward, reverse


def _walk(
    start_ids: list[int],
    adjacency: dict[int, set],
    nodes: dict[int, dict],
    stop_at: int | None = None,
) -> set[int]:
    """DFS through an adjacency map. Halts (doesn't expand past) ``stop_at`` so
    the loop-boundary nodes don't pull in nodes on the far side. Bypassed
    nodes are TRAVERSED (ComfyUI bypass passes data through matching-type
    slots), since otherwise a bypassed AttentionTunerPatch in the middle of
    a model patch chain would falsely sever the chain.
    """
    seen: set[int] = set()
    stack = list(start_ids)
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        if nid == stop_at:
            continue
        for neighbor, _, _, _ in adjacency.get(nid, ()):
            if neighbor in nodes:
                stack.append(neighbor)
    return seen


def verify(wf: dict) -> tuple[int, list[dict]]:
    """Return (exit_code, violations). 0 = OK, 1 = containment violation."""
    nodes, forward, reverse = _build_graph(wf)

    tlos = [n["id"] for n in wf.get("nodes", [])
            if n.get("type") == "TensorLoopOpen" and n.get("mode") != 4]
    tlcs = [n["id"] for n in wf.get("nodes", [])
            if n.get("type") == "TensorLoopClose" and n.get("mode") != 4]

    if not tlos and not tlcs:
        print("No TensorLoopOpen/Close on canvas — nothing to verify.")
        return 0, []
    if not tlos:
        print(f"ERR: TensorLoopClose #{tlcs[0]} present but no active TensorLoopOpen")
        return 1, []
    if not tlcs:
        print(f"ERR: TensorLoopOpen #{tlos[0]} present but no active TensorLoopClose")
        return 1, []
    if len(tlos) > 1 or len(tlcs) > 1:
        print(f"WARN: multiple TLO ({len(tlos)}) / TLC ({len(tlcs)}); using first of each")

    tlo, tlc = tlos[0], tlcs[0]

    forward_seeds = [tgt for tgt, src_slot, _, _ in forward.get(tlo, ())
                     if src_slot in _TLO_DATA_OUT_SLOTS]
    backward_seeds = [src for src, _, tgt_slot, _ in reverse.get(tlc, ())
                      if tgt_slot in _TLC_DATA_IN_SLOTS]

    iter_dep = _walk(forward_seeds, forward, nodes, stop_at=tlc)
    reaches_tlc = _walk(backward_seeds, reverse, nodes, stop_at=tlo)

    # Exclude the loop-boundary nodes from violations — they're the boundary,
    # not violators. Also exclude bypassed nodes (they don't execute, so
    # "executes once vs per-iter" is moot).
    violation_ids = sorted(iter_dep - reaches_tlc - {tlo, tlc})

    real = []      # active non-bus nodes — these are correctness bugs
    bus_dead = []  # SetNode/GetNode with no live consumer — benchmark cruft
    bypassed = []  # mode=4 — won't execute regardless
    for nid in violation_ids:
        n = nodes[nid]
        info = {"id": nid, "type": n.get("type"), "title": n.get("title") or ""}
        if n.get("mode") == 4:
            bypassed.append(info)
        elif n.get("type") in ("SetNode", "GetNode"):
            bus_dead.append(info)
        else:
            real.append(info)

    print(f"TensorLoopOpen #{tlo}  TensorLoopClose #{tlc}")
    print(f"  iter-dependent (forward from TLO data outputs):  {len(iter_dep)} nodes")
    print(f"  reaches TLC (backward from TLC data inputs):     {len(reaches_tlc)} nodes")
    print(f"  intersection (cloned per iter):                  {len(iter_dep & reaches_tlc)} nodes")

    def _fmt(entries):
        return "\n".join(f"  #{e['id']:<5} {e['type']:<35} {e['title']}" for e in entries)

    if real:
        print(f"\nERR: {len(real)} active iter-dependent node(s) do NOT reach TLC")
        print(f"     (will execute ONCE statically; per-iter state will freeze):")
        print(_fmt(real))
    if bus_dead:
        print(f"\nWARN: {len(bus_dead)} orphan SetNode/GetNode bus endpoint(s) "
              f"(cosmetic; no live consumer):")
        print(_fmt(bus_dead))
    if bypassed:
        print(f"\nINFO: {len(bypassed)} bypassed node(s) on side branches "
              f"(don't execute):")
        print(_fmt(bypassed))

    if not real:
        print("\nOK: every active iter-dependent node reaches TensorLoopClose.")
        return 0, []
    return 1, real


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "workflow", nargs="?",
        default="example_workflows/experimental/fml2v_var_d_audio_loop.json",
        help="Path to workflow JSON (default: %(default)s)",
    )
    args = parser.parse_args()

    path = Path(args.workflow)
    if not path.exists():
        print(f"ERR: workflow not found: {path}")
        return 1
    wf = json.loads(path.read_text())
    print(f"Verifying {path}")
    exit_code, _ = verify(wf)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
