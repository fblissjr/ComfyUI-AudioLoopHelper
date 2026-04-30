#!/usr/bin/env python3
"""Structural diff of two ComfyUI workflow JSON files.

Filters noise: link IDs, node positions, node ordering. Reports what
actually differs behaviorally — added/removed nodes, widget-value
changes on nodes matched by ID, and link-topology changes (which
source/target node+slot each link connects, not the link IDs).

Scopes: top-level graph AND every subgraph definition.

Usage:
    uv run --group dev python \\
      .claude/skills/compare-workflows/diff_workflows.py \\
      <workflow_a.json> <workflow_b.json>

Exit code 0 regardless of diff count — this is a report tool.

Author: 2026-04-23.
"""
from __future__ import annotations

import sys
from pathlib import Path

import orjson


def _load(path: str) -> dict:
    return orjson.loads(Path(path).read_bytes())


def _link_fingerprint(link: list | dict, node_types_by_id: dict[int, str]) -> str:
    """Canonical string for a link: (src_type#id:slot → tgt_type#id:slot, dtype).
    Ignores the link's numeric id (which is workflow-local and noisy).
    """
    if isinstance(link, dict):
        sid, sslot = link.get("origin_id"), link.get("origin_slot")
        tid, tslot = link.get("target_id"), link.get("target_slot")
        dtype = link.get("type", "?")
    else:
        _, sid, sslot, tid, tslot, dtype = link[:6]
    # -10 = subgraph input distributor, -20 = subgraph output collector
    virtual = {-10: "<sg_input>", -20: "<sg_output>"}
    src_label = virtual.get(sid) or f"{node_types_by_id.get(sid, '?')}#{sid}"
    tgt_label = virtual.get(tid) or f"{node_types_by_id.get(tid, '?')}#{tid}"
    return f"{src_label}:{sslot} → {tgt_label}:{tslot} ({dtype})"


def _nodes_by_id(nodes: list[dict]) -> dict[int, dict]:
    return {n["id"]: n for n in nodes}


def _types_by_id(nodes: list[dict]) -> dict[int, str]:
    return {n["id"]: n.get("type", "?") for n in nodes}


def _compare_widget_values(a: list, b: list) -> list[str]:
    """Return human-readable diffs between two widgets_values lists."""
    if a == b:
        return []
    diffs: list[str] = []
    for i in range(max(len(a), len(b))):
        av = a[i] if i < len(a) else "<missing>"
        bv = b[i] if i < len(b) else "<missing>"
        if av != bv:
            av_repr = repr(av)[:80]
            bv_repr = repr(bv)[:80]
            diffs.append(f"widget[{i}]: {av_repr} → {bv_repr}")
    return diffs


def _compare_scope(
    label: str,
    nodes_a: list[dict],
    links_a: list,
    nodes_b: list[dict],
    links_b: list,
) -> list[str]:
    """Compare one scope (top-level or a specific subgraph). Return report lines."""
    lines: list[str] = []

    by_id_a = _nodes_by_id(nodes_a)
    by_id_b = _nodes_by_id(nodes_b)
    types_a = _types_by_id(nodes_a)
    types_b = _types_by_id(nodes_b)

    # --- Nodes: added / removed / modified ---
    only_in_a = sorted(set(by_id_a) - set(by_id_b))
    only_in_b = sorted(set(by_id_b) - set(by_id_a))
    common = sorted(set(by_id_a) & set(by_id_b))

    if only_in_a:
        lines.append(f"  nodes removed (only in A):")
        for nid in only_in_a:
            t = types_a[nid]
            lines.append(f"    -#{nid} {t}")
    if only_in_b:
        lines.append(f"  nodes added (only in B):")
        for nid in only_in_b:
            t = types_b[nid]
            lines.append(f"    +#{nid} {t}")

    # --- Matched nodes: type change / widget diff / mode (bypass) change ---
    for nid in common:
        na, nb = by_id_a[nid], by_id_b[nid]
        changes: list[str] = []
        if na.get("type") != nb.get("type"):
            changes.append(f"type: {na.get('type')} → {nb.get('type')}")
        if na.get("mode", 0) != nb.get("mode", 0):
            changes.append(f"mode: {na.get('mode', 0)} → {nb.get('mode', 0)}")
        changes.extend(
            _compare_widget_values(
                na.get("widgets_values", []),
                nb.get("widgets_values", []),
            )
        )
        if changes:
            lines.append(f"  #{nid} {types_a[nid]} changed:")
            for c in changes:
                lines.append(f"    {c}")

    # --- Links: topology diff (by fingerprint, ignoring link IDs) ---
    fp_a = {_link_fingerprint(L, types_a) for L in links_a}
    fp_b = {_link_fingerprint(L, types_b) for L in links_b}
    links_removed = sorted(fp_a - fp_b)
    links_added = sorted(fp_b - fp_a)
    if links_removed:
        lines.append(f"  links removed (only in A):")
        for fp in links_removed:
            lines.append(f"    -{fp}")
    if links_added:
        lines.append(f"  links added (only in B):")
        for fp in links_added:
            lines.append(f"    +{fp}")

    if lines:
        lines.insert(0, f"[{label}]")
    return lines


def _compare_subgraph_schemas(a: dict, b: dict) -> list[str]:
    """Compare subgraph input/output slot metadata (names + types)."""
    lines: list[str] = []
    for key in ("inputs", "outputs"):
        ia = a.get(key, [])
        ib = b.get(key, [])
        for i in range(max(len(ia), len(ib))):
            ax = ia[i] if i < len(ia) else None
            bx = ib[i] if i < len(ib) else None
            if ax != bx:
                if ax and bx and (ax.get("name") != bx.get("name") or ax.get("type") != bx.get("type")):
                    lines.append(
                        f"  subgraph {key}[{i}]: "
                        f"{ax.get('name')!r}/{ax.get('type')!r} → "
                        f"{bx.get('name')!r}/{bx.get('type')!r}"
                    )
                elif ax is None:
                    lines.append(f"  subgraph {key}[{i}] added: {bx.get('name')!r}/{bx.get('type')!r}")
                elif bx is None:
                    lines.append(f"  subgraph {key}[{i}] removed: {ax.get('name')!r}/{ax.get('type')!r}")
    return lines


def diff(path_a: str, path_b: str) -> list[str]:
    wf_a, wf_b = _load(path_a), _load(path_b)
    lines: list[str] = [
        f"A: {path_a}",
        f"B: {path_b}",
        "",
    ]

    top = _compare_scope(
        "TOP LEVEL",
        wf_a.get("nodes", []), wf_a.get("links", []),
        wf_b.get("nodes", []), wf_b.get("links", []),
    )
    if top:
        lines.extend(top)
        lines.append("")

    # Subgraphs: compare by index (expects aligned ordering; most workflows have 1).
    sgs_a = wf_a.get("definitions", {}).get("subgraphs", [])
    sgs_b = wf_b.get("definitions", {}).get("subgraphs", [])
    for i in range(max(len(sgs_a), len(sgs_b))):
        sg_a = sgs_a[i] if i < len(sgs_a) else None
        sg_b = sgs_b[i] if i < len(sgs_b) else None
        if sg_a is None:
            lines.append(f"[SUBGRAPH {i}] added in B")
            continue
        if sg_b is None:
            lines.append(f"[SUBGRAPH {i}] removed from A")
            continue

        label = f"SUBGRAPH {i} ({sg_a.get('name', '?')})"
        schema_diffs = _compare_subgraph_schemas(sg_a, sg_b)
        if schema_diffs:
            lines.append(f"[{label} schema]")
            lines.extend(schema_diffs)
            lines.append("")

        body = _compare_scope(
            label,
            sg_a.get("nodes", []), sg_a.get("links", []),
            sg_b.get("nodes", []), sg_b.get("links", []),
        )
        if body:
            lines.extend(body)
            lines.append("")

    if len(lines) <= 3:
        lines.append("(no structural differences)")
    return lines


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: diff_workflows.py <workflow_a.json> <workflow_b.json>", file=sys.stderr)
        return 2
    for line in diff(sys.argv[1], sys.argv[2]):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
