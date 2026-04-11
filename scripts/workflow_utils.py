"""Reusable utilities for programmatic ComfyUI workflow JSON editing.

Usage:
    from workflow_utils import WorkflowEditor
    ed = WorkflowEditor("path/to/workflow.json")
    ed.find_node(843)
    ed.trace_node_inputs(843)
    ed.add_node(...)
    ed.rewire(old_src, old_tgt, new_src, new_tgt)
    ed.save("path/to/output.json")

Handles the three link representations that must stay in sync:
  1. Top-level links array (array format)
  2. Node input/output link/links fields
  3. Subgraph internal links (dict format) with linkIds on inputs
"""

import json
from pathlib import Path


class WorkflowEditor:
    """Load, inspect, modify, and save ComfyUI workflow JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.wf = json.loads(self.path.read_text())

    def save(self, path: str | Path | None = None):
        """Write workflow to disk. Defaults to original path."""
        out = Path(path) if path else self.path
        out.write_text(json.dumps(self.wf, indent=2))
        print(f"Saved to {out}")

    # --- Node operations ---

    def find_node(self, node_id: int) -> dict:
        for n in self.wf["nodes"]:
            if n["id"] == node_id:
                return n
        raise ValueError(f"Node {node_id} not found")

    def find_nodes_by_type(self, node_type: str) -> list[dict]:
        return [n for n in self.wf["nodes"] if n["type"] == node_type]

    def add_node(self, node: dict):
        """Add a node dict to the workflow. Updates last_node_id."""
        self.wf["nodes"].append(node)
        if node["id"] > self.wf.get("last_node_id", 0):
            self.wf["last_node_id"] = node["id"]

    def next_node_id(self) -> int:
        nid = self.wf.get("last_node_id", 0) + 1
        self.wf["last_node_id"] = nid
        return nid

    # --- Link operations (top-level) ---

    def next_link_id(self) -> int:
        lid = self.wf.get("last_link_id", 0) + 1
        self.wf["last_link_id"] = lid
        return lid

    def add_link(self, src_node: int, src_slot: int, tgt_node: int, tgt_slot: int, dtype: str) -> int:
        """Add a top-level link. Updates source output links and target input link. Returns link ID."""
        lid = self.next_link_id()
        self.wf["links"].append([lid, src_node, src_slot, tgt_node, tgt_slot, dtype])

        # Update source node output
        src = self.find_node(src_node)
        if src_slot < len(src.get("outputs", [])):
            links_list = src["outputs"][src_slot].get("links") or []
            links_list.append(lid)
            src["outputs"][src_slot]["links"] = links_list

        # Update target node input
        tgt = self.find_node(tgt_node)
        if tgt_slot < len(tgt.get("inputs", [])):
            tgt["inputs"][tgt_slot]["link"] = lid

        return lid

    def remove_link(self, link_id: int):
        """Remove a top-level link. Cleans up source output and target input references."""
        # Find the link details before removing
        link_data = None
        for l in self.wf["links"]:
            if isinstance(l, list) and l[0] == link_id:
                link_data = l
                break
        if not link_data:
            raise ValueError(f"Link {link_id} not found")

        src_node_id, src_slot, tgt_node_id, tgt_slot = link_data[1], link_data[2], link_data[3], link_data[4]

        # Remove from top-level array
        self.wf["links"] = [l for l in self.wf["links"] if not (isinstance(l, list) and l[0] == link_id)]

        # Clean source output
        try:
            src = self.find_node(src_node_id)
            if src_slot < len(src.get("outputs", [])):
                links = src["outputs"][src_slot].get("links") or []
                src["outputs"][src_slot]["links"] = [l for l in links if l != link_id]
        except ValueError:
            pass

        # Clean target input
        try:
            tgt = self.find_node(tgt_node_id)
            if tgt_slot < len(tgt.get("inputs", [])):
                if tgt["inputs"][tgt_slot].get("link") == link_id:
                    tgt["inputs"][tgt_slot]["link"] = None
        except ValueError:
            pass

    def find_link(self, src_node: int, tgt_node: int) -> int | None:
        """Find link ID between two nodes. Returns None if not found."""
        for l in self.wf["links"]:
            if isinstance(l, list) and l[1] == src_node and l[3] == tgt_node:
                return l[0]
        return None

    def find_links_to(self, tgt_node: int) -> list:
        """Find all links targeting a node."""
        return [l for l in self.wf["links"] if isinstance(l, list) and l[3] == tgt_node]

    def find_links_from(self, src_node: int) -> list:
        """Find all links originating from a node."""
        return [l for l in self.wf["links"] if isinstance(l, list) and l[1] == src_node]

    # --- Subgraph operations ---

    def get_subgraph(self, index: int = 0) -> dict | None:
        """Get a subgraph definition by index."""
        defs = self.wf.get("definitions", {})
        if isinstance(defs, dict):
            sgs = defs.get("subgraphs", [])
            if index < len(sgs):
                return sgs[index]
        return None

    def find_subgraph_node(self, node_id: int, sg_index: int = 0) -> dict | None:
        """Find a node inside a subgraph."""
        sg = self.get_subgraph(sg_index)
        if sg:
            for n in sg.get("nodes", []):
                if n["id"] == node_id:
                    return n
        return None

    def find_subgraph_link(self, link_id: int, sg_index: int = 0) -> dict | None:
        """Find an internal link inside a subgraph."""
        sg = self.get_subgraph(sg_index)
        if sg:
            for l in sg.get("links", []):
                if l["id"] == link_id:
                    return l
        return None

    def remove_subgraph_link(self, link_id: int, sg_index: int = 0):
        """Remove an internal link from a subgraph. Cleans up node inputs and input linkIds."""
        sg = self.get_subgraph(sg_index)
        if not sg:
            raise ValueError(f"Subgraph {sg_index} not found")

        # Find and remove the link
        link_data = None
        for l in sg["links"]:
            if l["id"] == link_id:
                link_data = l
                break
        if not link_data:
            raise ValueError(f"Subgraph link {link_id} not found")

        sg["links"] = [l for l in sg["links"] if l["id"] != link_id]

        # Clean target node input
        tgt_id = link_data["target_id"]
        tgt_slot = link_data["target_slot"]
        for n in sg.get("nodes", []):
            if n["id"] == tgt_id:
                inputs = n.get("inputs", [])
                if tgt_slot < len(inputs) and inputs[tgt_slot].get("link") == link_id:
                    inputs[tgt_slot]["link"] = None
                break

        # Clean subgraph input linkIds
        for inp in sg.get("inputs", []):
            link_ids = inp.get("linkIds", [])
            if link_id in link_ids:
                inp["linkIds"] = [l for l in link_ids if l != link_id]

    # --- Inspection ---

    def trace_node_inputs(self, node_id: int) -> list[dict]:
        """Trace all inputs of a node to their sources."""
        n = self.find_node(node_id)
        results = []
        for i, inp in enumerate(n.get("inputs", [])):
            link_id = inp.get("link")
            if link_id:
                for l in self.wf["links"]:
                    if isinstance(l, list) and l[0] == link_id:
                        try:
                            src = self.find_node(l[1])
                            src_type = src["type"]
                        except ValueError:
                            src_type = "?"
                        results.append({
                            "slot": i, "name": inp["name"], "link": link_id,
                            "src_node": l[1], "src_slot": l[2], "src_type": src_type,
                        })
                        break
            else:
                results.append({"slot": i, "name": inp["name"], "link": None})
        return results

    def trace_forward(self, node_id: int, slot: int = 0, max_depth: int = 10) -> list[dict]:
        """Trace output links forward from a node."""
        chain = []
        for _ in range(max_depth):
            try:
                n = self.find_node(node_id)
            except ValueError:
                break
            outs = n.get("outputs", [])
            if slot >= len(outs) or not outs[slot].get("links"):
                break
            link_id = outs[slot]["links"][0]
            for l in self.wf["links"]:
                if isinstance(l, list) and l[0] == link_id:
                    try:
                        tgt = self.find_node(l[3])
                        tgt_type = tgt["type"]
                    except ValueError:
                        tgt_type = "?"
                    chain.append({
                        "src": node_id, "src_slot": slot,
                        "tgt": l[3], "tgt_slot": l[4], "tgt_type": tgt_type,
                    })
                    node_id = l[3]
                    slot = 0
                    break
            else:
                break
        return chain

    def print_node_summary(self, node_id: int):
        """Print a human-readable summary of a node and its connections."""
        n = self.find_node(node_id)
        print(f"Node {n['id']} ({n['type']}) '{n.get('title', '')}'")
        print(f"  widgets: {n.get('widgets_values', [])}")
        for r in self.trace_node_inputs(node_id):
            if r["link"]:
                print(f"  in[{r['slot']}] {r['name']}: <- Node {r['src_node']} ({r['src_type']})")
            else:
                print(f"  in[{r['slot']}] {r['name']}: (no link)")
        for i, out in enumerate(n.get("outputs", [])):
            if out.get("links"):
                print(f"  out[{i}] {out['name']}: -> {out['links']}")

    # --- Convenience builders ---

    @staticmethod
    def make_get_node(node_id: int, var_name: str, dtype: str, pos: list, title: str | None = None) -> dict:
        """Create a KJNodes GetNode dict."""
        return {
            "id": node_id, "type": "GetNode",
            "pos": pos, "size": [210, 58], "flags": {}, "order": 20, "mode": 0,
            "inputs": [],
            "outputs": [{"name": dtype, "type": dtype, "links": []}],
            "title": title or f"Get_{var_name}",
            "properties": {"Node name for S&R": "GetNode", "aux_id": "kijai/ComfyUI-KJNodes"},
            "widgets_values": [var_name],
        }

    @staticmethod
    def make_node(node_id: int, node_type: str, pos: list, widgets: list | None = None,
                  title: str | None = None, inputs: list | None = None,
                  outputs: list | None = None) -> dict:
        """Create a generic node dict."""
        n = {
            "id": node_id, "type": node_type,
            "pos": pos, "size": [270, 100], "flags": {}, "order": 90, "mode": 0,
            "inputs": inputs or [], "outputs": outputs or [],
            "properties": {"cnr_id": "comfy-core", "ver": "0.18.5", "Node name for S&R": node_type},
            "widgets_values": widgets or [],
        }
        if title:
            n["title"] = title
        return n


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python workflow_utils.py <workflow.json> [node_id]")
        sys.exit(1)

    ed = WorkflowEditor(sys.argv[1])
    if len(sys.argv) >= 3:
        nid = int(sys.argv[2])
        ed.print_node_summary(nid)
        print("\nForward trace:")
        for step in ed.trace_forward(nid):
            print(f"  Node {step['src']}[{step['src_slot']}] -> Node {step['tgt']} ({step['tgt_type']}) [{step['tgt_slot']}]")
    else:
        print(f"Workflow: {ed.path}")
        print(f"  Nodes: {len(ed.wf['nodes'])}")
        print(f"  Links: {len(ed.wf['links'])}")
        print(f"  last_node_id: {ed.wf.get('last_node_id')}")
        print(f"  last_link_id: {ed.wf.get('last_link_id')}")
        sg = ed.get_subgraph()
        if sg:
            print(f"  Subgraph: {sg.get('name', sg.get('id', '?'))}")
            print(f"    Internal nodes: {len(sg.get('nodes', []))}")
            print(f"    Internal links: {len(sg.get('links', []))}")
