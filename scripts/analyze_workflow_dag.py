"""Static DAG + execution-order analyzer for a ComfyUI workflow JSON.

Answers: what nodes exist, how are they wired, what order does the
executor run them in, and where do the loop / subgraph boundaries fall?

Outputs a topologically-sorted execution plan plus a graph rendering
(mermaid, graphviz dot, ASCII, or JSON). Static — does not run
anything. Combines well with scripts/trace_node_source.py: this script
tells you the execution order; that script tells you what each node
actually does.

Usage:

    uv run --group dev python scripts/analyze_workflow_dag.py \
      example_workflows/audio-loop-music-video_latent.json \
      --format mermaid \
      --output internal/analysis/dag_latent.md

Formats:
  mermaid  -- markdown-embeddable flowchart, color-coded by type
  dot      -- graphviz dot (render with: `dot -Tsvg file.dot > file.svg`)
  ascii    -- plain-text topological listing (fastest, no viewer needed)
  json     -- structured dict for programmatic consumption

Flags:
  --subgraph 0         include subgraph index 0 internals
  --include-bypassed   show mode=4 nodes (default hidden from renderings)
  --collapse-setget    treat Set_X / Get_X pairs as implicit edges
  --filter-types T1,T2 only include nodes of these types + their neighbors
"""

from __future__ import annotations

import argparse
import heapq
import sys
from collections import defaultdict
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402


# ComfyUI "mode" field: 0=active, 2=mute, 4=bypass.
MODE_ACTIVE = 0
MODE_BYPASS = 4

# Node categories → fill color. Order matters: first-match wins, so put
# specific categories (conditioning, latent) before generic ones (helper).
_CATEGORY_PREFIXES: dict[str, tuple[str, ...]] = {
    "loader": ("loader", "unetloader", "vaeload", "cliploader"),
    "sampler": ("sampler", "scheduler", "noise"),
    "conditioning": ("condition", "cliptextencode", "batchencode", "textencode"),
    "latent": ("latent",),
    "audio": ("audio", "trimaudio", "melband"),
    "loop": ("tensorloop",),
    "subgraph": ("subgraph",),
}
_HELPER_EXACT_TYPES = frozenset({
    "SetNode", "GetNode", "PrimitiveNode", "Reroute", "Note", "FloatConstant",
})
_CATEGORY_COLORS = {
    "loader": "#A8D5BA",
    "sampler": "#F5B86B",
    "conditioning": "#8EB9D4",
    "latent": "#C8A7D4",
    "audio": "#D4A5A5",
    "loop": "#E5D27C",
    "subgraph": "#B0B0B0",
    "helper": "#DCDCDC",
    "dead": "#F0A0A0",
}


def _categorize(node: dict) -> str:
    if node["type"] in _HELPER_EXACT_TYPES:
        return "helper"
    t = node["type"].lower()
    for category, prefixes in _CATEGORY_PREFIXES.items():
        if any(p in t for p in prefixes):
            return category
    return "helper"


def _build_edges(ed: WorkflowEditor, *, collapse_setget: bool) -> list[tuple[int, int, str]]:
    """Return (src_node, tgt_node, type_label) edges from top-level links.

    When `collapse_setget` is True, Set_X / Get_X pairs by key name get
    a synthetic edge so the graph reflects actual data flow.
    """
    edges: list[tuple[int, int, str]] = list(ed.iter_edges())

    if collapse_setget:
        set_by_key: dict[str, int] = {}
        get_by_key: dict[str, list[int]] = defaultdict(list)
        for n in ed.wf["nodes"]:
            if n["type"] == "SetNode":
                key = (n.get("widgets_values") or [None])[0]
                if isinstance(key, str):
                    set_by_key[key] = n["id"]
            elif n["type"] == "GetNode":
                key = (n.get("widgets_values") or [None])[0]
                if isinstance(key, str):
                    get_by_key[key].append(n["id"])
        for key, sid in set_by_key.items():
            for gid in get_by_key.get(key, []):
                edges.append((sid, gid, f"SETGET({key})"))
    return edges


def _topo_sort(
    nodes: list[dict], edges: list[tuple[int, int, str]],
) -> tuple[list[int], list[int]]:
    """Return (execution_order, cycle_node_ids). Kahn's algorithm with a
    min-heap ready-queue so tie-breaking is by lowest node id (stable
    output across runs) in O(N log N) total rather than O(N² log N)."""
    in_deg: dict[int, int] = {n["id"]: 0 for n in nodes}
    out_adj: dict[int, list[int]] = defaultdict(list)
    node_ids = set(in_deg)
    for src, tgt, _ in edges:
        if src in node_ids and tgt in node_ids and src != tgt:
            in_deg[tgt] = in_deg.get(tgt, 0) + 1
            out_adj[src].append(tgt)

    ready: list[int] = [nid for nid, d in in_deg.items() if d == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        nid = heapq.heappop(ready)
        order.append(nid)
        for nxt in out_adj[nid]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                heapq.heappush(ready, nxt)

    cycle = [nid for nid, d in in_deg.items() if d > 0]
    return order, cycle


def _dead_nodes(ed: WorkflowEditor) -> set[int]:
    """Nodes whose outputs go nowhere (unless they're terminal sinks like VideoCombine)."""
    terminal_types = {"VHS_VideoCombine", "PreviewImage", "SaveImage",
                      "SaveAudio", "Note"}
    dead: set[int] = set()
    for n in ed.wf["nodes"]:
        if n["type"] in terminal_types or not n.get("outputs"):
            continue
        if not any((o.get("links") or []) for o in n.get("outputs", [])):
            dead.add(n["id"])
    return dead


def _node_label(n: dict) -> str:
    title = n.get("title") or ""
    if title and title != n["type"]:
        return f"{n['id']} {n['type']}\\n({title})"
    return f"{n['id']} {n['type']}"


def render_mermaid(
    nodes: list[dict], edges: list[tuple[int, int, str]],
    order: list[int], dead: set[int],
    *, include_bypassed: bool,
) -> str:
    nodes_by_id = {n["id"]: n for n in nodes}
    order_rank = {nid: i for i, nid in enumerate(order)}

    def label(nid: int) -> str:
        n = nodes_by_id[nid]
        rank = order_rank.get(nid, "?")
        lbl = _node_label(n).replace("\\n", "<br/>")
        return f"{nid}[\"#{rank} {lbl}\"]"

    def style(nid: int) -> str:
        n = nodes_by_id[nid]
        if nid in dead:
            color = _CATEGORY_COLORS["dead"]
        else:
            color = _CATEGORY_COLORS[_categorize(n)]
        stroke = "dashed" if n.get("mode") == MODE_BYPASS else "solid"
        return f"style {nid} fill:{color},stroke-dasharray: {stroke}"

    lines = ["```mermaid", "flowchart TD"]
    shown = set()
    for n in nodes:
        if n.get("mode") == MODE_BYPASS and not include_bypassed:
            continue
        lines.append(f"    {label(n['id'])}")
        shown.add(n["id"])
    lines.append("")
    for src, tgt, dtype in edges:
        if src not in shown or tgt not in shown:
            continue
        lines.append(f"    {src} -->|{dtype}| {tgt}")
    lines.append("")
    for nid in shown:
        lines.append(f"    {style(nid)}")
    lines.append("```")
    return "\n".join(lines)


def render_dot(
    nodes: list[dict], edges: list[tuple[int, int, str]],
    order: list[int], dead: set[int],
    *, include_bypassed: bool,
) -> str:
    order_rank = {nid: i for i, nid in enumerate(order)}

    def esc(s: str) -> str:
        return s.replace('"', '\\"').replace("\n", "\\n")

    lines = ["digraph workflow {", "    rankdir=TB;", "    node [shape=box];"]
    for n in nodes:
        if n.get("mode") == MODE_BYPASS and not include_bypassed:
            continue
        nid = n["id"]
        color = (_CATEGORY_COLORS["dead"] if nid in dead
                 else _CATEGORY_COLORS[_categorize(n)])
        style = "dashed" if n.get("mode") == MODE_BYPASS else "filled"
        rank = order_rank.get(nid, "?")
        label = esc(f"#{rank} {_node_label(n)}")
        lines.append(
            f'    n{nid} [label="{label}", fillcolor="{color}", style="{style}"];'
        )
    lines.append("")
    shown = {n["id"] for n in nodes
             if include_bypassed or n.get("mode") != MODE_BYPASS}
    for src, tgt, dtype in edges:
        if src in shown and tgt in shown:
            lines.append(f'    n{src} -> n{tgt} [label="{dtype}"];')
    lines.append("}")
    return "\n".join(lines)


def render_ascii(
    nodes: list[dict], edges: list[tuple[int, int, str]],
    order: list[int], dead: set[int], cycle: list[int],
    *, include_bypassed: bool,
) -> str:
    nodes_by_id = {n["id"]: n for n in nodes}
    incoming: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for src_tgt_dtype in edges:
        src, tgt, dtype = src_tgt_dtype
        incoming[tgt].append((src, dtype))

    lines = [f"# Execution order ({len(order)} nodes)", ""]
    for i, nid in enumerate(order):
        n = nodes_by_id[nid]
        if n.get("mode") == MODE_BYPASS and not include_bypassed:
            continue
        flags = []
        if n.get("mode") == MODE_BYPASS:
            flags.append("BYPASS")
        if nid in dead:
            flags.append("DEAD")
        flag_str = f"  [{','.join(flags)}]" if flags else ""
        title = n.get("title") or ""
        title_str = f" ({title})" if title and title != n["type"] else ""
        lines.append(f"#{i:3d}  [{nid:5d}] {n['type']}{title_str}{flag_str}")
        for src, dtype in incoming[nid]:
            src_n = nodes_by_id.get(src)
            src_label = src_n["type"] if src_n else "?"
            lines.append(f"         <- {src:5d} {src_label}  ({dtype})")
    if cycle:
        lines.append("")
        lines.append(f"# WARNING: cycle detected, {len(cycle)} nodes:")
        for nid in cycle:
            n = nodes_by_id[nid]
            lines.append(f"  {nid}  {n['type']}")
    return "\n".join(lines)


def analyze(
    path: Path,
    *,
    fmt: str = "mermaid",
    include_bypassed: bool = False,
    collapse_setget: bool = True,
    filter_types: list[str] | None = None,
    include_subgraph: int | None = None,
) -> str:
    ed = WorkflowEditor(path)
    nodes = list(ed.wf["nodes"])
    edges = _build_edges(ed, collapse_setget=collapse_setget)

    if include_subgraph is not None:
        sg = ed.get_subgraph(include_subgraph)
        if sg:
            # Subgraph internal nodes get IDs offset into a high range
            # to avoid collision with top-level node IDs, and we link
            # them via virtual edges reflecting subgraph inputs.
            for sn in sg.get("nodes", []):
                virt = dict(sn)
                virt["id"] = 1000000 + int(sn["id"])
                virt["title"] = f"[sg] {virt.get('title', '')}".strip()
                nodes.append(virt)
            for link in sg.get("links", []):
                if isinstance(link, dict):
                    edges.append((
                        1000000 + link["origin_id"] if link["origin_id"] > 0 else -10,
                        1000000 + link["target_id"] if link["target_id"] > 0 else -20,
                        link.get("type", "SG"),
                    ))

    if filter_types:
        keep = {int(n["id"]) for n in nodes if n["type"] in filter_types}
        keep |= {src for src, tgt, _ in edges if tgt in keep}
        keep |= {tgt for src, tgt, _ in edges if src in keep}
        nodes = [n for n in nodes if n["id"] in keep]
        edges = [e for e in edges if e[0] in keep and e[1] in keep]

    order, cycle = _topo_sort(nodes, edges)
    dead = _dead_nodes(ed)

    if fmt == "mermaid":
        return render_mermaid(nodes, edges, order, dead,
                              include_bypassed=include_bypassed)
    if fmt == "dot":
        return render_dot(nodes, edges, order, dead,
                          include_bypassed=include_bypassed)
    if fmt == "ascii":
        return render_ascii(nodes, edges, order, dead, cycle,
                            include_bypassed=include_bypassed)
    if fmt == "json":
        order_rank = {nid: i for i, nid in enumerate(order)}
        payload = {
            "workflow": str(path),
            "execution_order": [
                {
                    "rank": order_rank[n["id"]],
                    "id": n["id"],
                    "type": n["type"],
                    "title": n.get("title", ""),
                    "mode": n.get("mode", 0),
                    "bypassed": n.get("mode") == MODE_BYPASS,
                    "dead": n["id"] in dead,
                    "widgets": n.get("widgets_values"),
                }
                for n in sorted(nodes, key=lambda x: order_rank.get(x["id"], 1e9))
                if include_bypassed or n.get("mode") != MODE_BYPASS
            ],
            "edges": [
                {"src": src, "tgt": tgt, "type": dtype}
                for src, tgt, dtype in edges
            ],
            "cycle": cycle,
        }
        return orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()
    raise ValueError(f"Unknown format: {fmt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workflow", help="Path to workflow JSON")
    ap.add_argument("--format", choices=("mermaid", "dot", "ascii", "json"),
                    default="mermaid")
    ap.add_argument("--output", help="Write to file instead of stdout")
    ap.add_argument("--include-bypassed", action="store_true",
                    help="Include mode=4 (bypassed) nodes in rendering")
    ap.add_argument("--no-collapse-setget", action="store_true",
                    help="Do not add synthetic edges for Set_/Get_ pairs")
    ap.add_argument("--filter-types", default=None,
                    help="Comma-separated node types to include (plus neighbors)")
    ap.add_argument("--subgraph", type=int, default=None,
                    help="Include definitions.subgraphs[N] internal nodes")
    args = ap.parse_args()

    filter_types = args.filter_types.split(",") if args.filter_types else None
    out = analyze(
        Path(args.workflow),
        fmt=args.format,
        include_bypassed=args.include_bypassed,
        collapse_setget=not args.no_collapse_setget,
        filter_types=filter_types,
        include_subgraph=args.subgraph,
    )
    if args.output:
        Path(args.output).write_text(out)
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
