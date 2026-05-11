"""Reusable utilities for programmatic ComfyUI workflow JSON editing.

Usage:
    from workflow_utils import WorkflowEditor
    ed = WorkflowEditor("path/to/workflow.json")
    ed.find_node(843)
    ed.trace_node_inputs(843)
    ed.add_node(...)
    ed.rewire_input(tgt_node, tgt_slot, new_src, new_src_slot, dtype)
    ed.save("path/to/output.json")

Handles the three link representations that must stay in sync:
  1. Top-level links array (array format)
  2. Node input/output link/links fields
  3. Subgraph internal links (dict format) with linkIds on inputs
"""

import os
from datetime import datetime
from pathlib import Path

import orjson


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_WORKFLOWS_DIR = REPO_ROOT / "example_workflows"
INTERNAL_WORKFLOWS_DIR = REPO_ROOT / "internal" / "workflows"
RUNS_DIR = REPO_ROOT / "internal" / "analysis" / "runs"
DATA_RUNS_DIR = REPO_ROOT / "data" / "runs"


def iter_all_workflows() -> list[Path]:
    """Return every workflow JSON the apply-script family touches:
    ``example_workflows/`` (shipped, public) + ``internal/workflows/``
    (drafts, gitignored). Sorted within each dir for deterministic
    iteration; deterministic across runs.

    Promoted from inline ``_iter_workflows`` in 4 apply scripts
    (``apply_trim_image_batch_to_audio``, ``apply_trim_video_latent_to_audio``,
    ``apply_run_id_layout``, ``apply_fix_source_audio_trim_defaults``)
    per CLAUDE.md "promote at 3rd call site." Skips dirs that don't
    exist (some clones have no internal/workflows).
    """
    paths: list[Path] = []
    for d in (EXAMPLE_WORKFLOWS_DIR, INTERNAL_WORKFLOWS_DIR):
        if d.exists():
            paths.extend(sorted(d.rglob("*.json")))
    return paths

# Decoders that produce IMAGE from a video LATENT. Shared vocabulary so
# `apply_trim_video_latent_to_audio.py` (splice-target detection) and
# `audit_workflows.py::_check_trim_video_latent_to_audio_present` (F14
# invariant) stay in sync. Add new types here when a workflow uses one.
DECODER_TYPES = frozenset({"LTXVTiledVAEDecode", "VAEDecodeTiled", "VAEDecode"})


_RUN_TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"  # lexicographic-sortable; verify_sage_iteration_trace.sh depends on this shape


def timestamped_run_path(subdir: str, prefix: str, ext: str) -> Path:
    """Build `<project>/internal/analysis/runs/<subdir>/<prefix>_YYYY-MM-DD_HHMMSS.<ext>`.

    Shared helper for debug tools that dump timestamped artifacts
    (DAG dumps, exec logs, profiler traces). Creates the parent dir if
    missing. Gitignored by the project's internal/ rule.

    Legacy path. New per-run artifacts should use `run_artifact_path` so
    they correlate via a shared `RUN_ID` env var.
    """
    out_dir = RUNS_DIR / subdir if subdir else RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime(_RUN_TIMESTAMP_FMT)
    return out_dir / f"{prefix}_{ts}.{ext}"


def _current_run_id() -> str | None:
    """Single source of truth for reading the RUN_ID env var.

    Strips whitespace; returns None for unset/empty rather than ""
    so callers can branch with `if run_id is None`. Read at call time
    (not module load) so wrapper scripts and tests can mutate the env
    before triggering loggers. Documented in
    `docs/reference/environment.md`.
    """
    raw = os.environ.get("RUN_ID", "").strip()
    return raw or None


_PER_PROMPT_ENV = "AUDIOLOOPHELPER_PER_PROMPT"
_PER_PROMPT_TRUTHY = {"1", "true", "yes", "on", "auto"}


def _per_prompt_enabled() -> bool:
    """True iff the operator opted into per-prompt subdirectory routing.

    Off by default — keeps `data/runs/${RUN_ID}/` flat for tools that
    submit one prompt per ComfyUI launch (the original assumption).
    Multi-prompt-per-session bench tools (sage-fork's bench_e2e_ltx)
    set this so each prompt's telemetry lands in its own subdir without
    needing to restart ComfyUI between bench iterations.
    """
    return os.environ.get(_PER_PROMPT_ENV, "").strip().lower() in _PER_PROMPT_TRUTHY


def _current_prompt_id() -> str | None:
    """Read the active prompt_id from ComfyUI's executing-context.

    ComfyUI exposes the currently-running prompt via a contextvar in
    `comfy_execution.utils.get_executing_context()`. Returns None when:
      - we're outside a /prompt execution (e.g. graph-build, tests),
      - ComfyUI isn't importable (module loaded standalone), OR
      - the contextvar is set but its prompt_id is None.

    Lazy import so non-ComfyUI consumers (pytest, debug scripts) don't
    pay the import cost or fail when ComfyUI isn't on sys.path. Same
    pattern as `nodes_sage._prompt_id_from_kwargs`.
    """
    try:
        from comfy_execution.utils import get_executing_context
    except ImportError:
        return None
    ctx = get_executing_context()
    if ctx is None:
        return None
    pid = getattr(ctx, "prompt_id", None)
    return str(pid) if pid is not None else None


def _run_artifact_root() -> Path | None:
    """Resolve the per-run (and optionally per-prompt) root dir under
    `data/runs/`. Returns None when RUN_ID is unset (caller falls back
    to the legacy `internal/analysis/runs/` path).

    Centralizes the per-prompt-routing logic so every caller of
    `run_artifact_path` and `run_artifact_dir` honors the same toggle —
    that's the holistic-fix property: opting in once benefits every
    current and future telemetry consumer.
    """
    run_id = _current_run_id()
    if run_id is None:
        return None
    root = DATA_RUNS_DIR / run_id
    if _per_prompt_enabled():
        prompt_id = _current_prompt_id()
        if prompt_id is not None:
            root = root / prompt_id
    return root


def run_artifact_path(category: str, ext: str) -> Path:
    """Path for a per-render artifact, honoring the `RUN_ID` env var.

    With `RUN_ID` set: `data/runs/${RUN_ID}/<category>.<ext>`. Every
    logger called during the same render lands artifacts under the same
    directory, making cross-system correlation (exec_log + sage trace +
    profiler + output mp4) trivial.

    With `RUN_ID` + `AUDIOLOOPHELPER_PER_PROMPT=1` AND an active
    executing-context: `data/runs/${RUN_ID}/${prompt_id}/<category>.<ext>`.
    Lets multi-prompt-per-session bench tools keep traces from different
    prompts isolated without restarting ComfyUI.

    Without `RUN_ID`: falls back to the legacy
    `timestamped_run_path(category, category, ext)` shape so existing
    tooling that runs without the experiment harness keeps working.

    Diagnosed 2026-04-26 — three loggers stamping `time.time()` at
    different startup moments produced filenames that looked unrelated
    despite coming from the same render. Single env var fixes it.
    Per-prompt routing added 2026-04-27 for sage-fork's bench harness.
    """
    root = _run_artifact_root()
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{category}.{ext}"
    return timestamped_run_path(category, category, ext)


def run_artifact_dir(subdir: str = "") -> Path:
    """Directory under the per-render artifact root, honoring `RUN_ID`.

    Companion to `run_artifact_path` for tools that produce a directory
    of files (profiler traces with trace.json + summary.txt +
    memory_timeline.html, frame sequences, etc.).

    With `RUN_ID` set: `data/runs/${RUN_ID}/<subdir>/` (or
    `data/runs/${RUN_ID}/` when subdir is empty). When
    `AUDIOLOOPHELPER_PER_PROMPT=1` and an executing-context is active,
    inserts `${prompt_id}` before `<subdir>`. Without RUN_ID, falls
    back to a legacy timestamped dir under `internal/analysis/runs/`.
    Creates the directory if missing.
    """
    root = _run_artifact_root()
    if root is not None:
        target = root / subdir if subdir else root
        target.mkdir(parents=True, exist_ok=True)
        return target
    return timestamped_run_dir(RUNS_DIR / subdir if subdir else RUNS_DIR)


def timestamped_run_dir(base: Path) -> Path:
    """Build `<base>/YYYY-MM-DD_HHMMSS/` and create it. Companion to
    `timestamped_run_path` for tools that produce a directory of
    artifacts (PNG frame sequences, profiler traces, multi-file runs).
    """
    ts = datetime.now().strftime(_RUN_TIMESTAMP_FMT)
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def is_active(node: dict) -> bool:
    """True iff the node will execute at runtime. ComfyUI uses
    `mode=4` to mark a node as bypassed; bypass passes inputs through
    to outputs of matching type (and dead-ends inputs with no
    matching output type). All other modes (0, 2) are active."""
    return node.get("mode", 0) != 4


def resolve_repo_path(p: str | Path) -> Path:
    """Resolve a CLI-supplied path: absolute paths kept as-is, relative
    paths re-rooted at the repo root. Used by every apply_*.py for its
    --input / --output / --workflow flags."""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


_EMPTY_WORKFLOW_SKELETON = {
    "revision": 0,
    "last_node_id": 0,
    "last_link_id": 0,
    "nodes": [],
    "links": [],
    "groups": [],
    "definitions": {"subgraphs": []},
    "config": {},
    "extra": {"ds": {"scale": 0.5, "offset": [0, 0]}},
    "version": 0.4,
}


class WorkflowEditor:
    """Load, inspect, modify, and save ComfyUI workflow JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.wf = orjson.loads(self.path.read_bytes())

    @classmethod
    def from_scratch(cls, output_path: str | Path) -> "WorkflowEditor":
        """Build an editor around an empty in-memory workflow skeleton.

        For apply scripts that construct a workflow from zero rather than
        forking an existing one. Call `.save()` (or pass `output_path` to
        `save()`) to write the result. The skeleton uses a fresh uuid
        per instance so saved workflows have distinct IDs.
        """
        import copy
        import uuid
        instance = cls.__new__(cls)
        instance.path = Path(output_path)
        instance.wf = copy.deepcopy(_EMPTY_WORKFLOW_SKELETON)
        instance.wf["id"] = str(uuid.uuid4())
        return instance

    def save(self, path: str | Path | None = None, *, verbose: bool = False):
        """Write workflow to disk. Defaults to original path."""
        out = Path(path) if path else self.path
        out.write_bytes(orjson.dumps(self.wf, option=orjson.OPT_INDENT_2))
        if verbose:
            print(f"Saved to {out}")

    # --- Node operations ---

    def find_node(self, node_id: int) -> dict:
        for n in self.wf["nodes"]:
            if n["id"] == node_id:
                return n
        raise ValueError(f"Node {node_id} not found")

    def has_node(self, node_id: int) -> bool:
        """True iff the workflow contains a top-level node with this id."""
        return any(n["id"] == node_id for n in self.wf["nodes"])

    def require_nodes(self, node_ids) -> list[int]:
        """Return the subset of `node_ids` NOT present in the workflow.

        Empty list means every requested node exists. Callers typically
        early-return (skip-workflow) or raise depending on severity.
        """
        return [nid for nid in node_ids if not self.has_node(nid)]

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

    def remove_node_and_links(self, node_id: int):
        """Remove a node plus every top-level link touching it.

        Callers must re-wire any surviving input/output slots before use —
        this helper only detaches.
        """
        for link in list(self.wf["links"]):
            if not isinstance(link, list):
                continue
            lid, src, _, tgt, _, _ = link
            if src == node_id or tgt == node_id:
                self.remove_link(lid)
        self.wf["nodes"] = [n for n in self.wf["nodes"] if n["id"] != node_id]

    @staticmethod
    def io_in(name: str, dtype: str, link: int | None = None) -> dict:
        """Build a non-widget input slot dict in the shape ComfyUI expects.

        Centralizes the slot shape so workflow builders don't open-code it.
        Pair with `widget_in` for inputs that also surface as widgets, and
        `out` for output slots.
        """
        d: dict = {"name": name, "type": dtype}
        if link is not None:
            d["link"] = link
        return d

    @staticmethod
    def widget_in(name: str, dtype: str, link: int | None = None) -> dict:
        """Build an input slot that is also surfaced as a widget on the node."""
        d = WorkflowEditor.io_in(name, dtype, link)
        d["widget"] = {"name": name}
        return d

    @staticmethod
    def out(name: str, dtype: str) -> dict:
        """Build an output slot dict in the shape ComfyUI expects."""
        return {"name": name, "type": dtype, "links": []}

    def add_top_level_node(
        self,
        node_type: str,
        pos: list,
        size: list,
        inputs: list,
        outputs: list,
        widgets_values: list | dict,
        properties: dict | None = None,
        title: str | None = None,
    ) -> int:
        """Append a new top-level node with the given shape. Returns assigned ID."""
        nid = self.next_node_id()
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
        self.add_node(node)
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
        """Find the FIRST link id between two nodes. Returns None if not found.

        Two nodes can be connected by multiple links (different slot pairs);
        this returns only the first match in link-array order. For inbound
        lookups prefer `find_link_to_slot` (slot-precise, single-result).
        """
        for l in self.wf["links"]:
            if isinstance(l, list) and l[1] == src_node and l[3] == tgt_node:
                return l[0]
        return None

    def find_links_to(self, tgt_node: int) -> list:
        """Find all links targeting a node."""
        return [l for l in self.wf["links"] if isinstance(l, list) and l[3] == tgt_node]

    def find_link_to_slot(self, tgt_node: int, tgt_slot: int) -> list | None:
        """Find the top-level link feeding a specific input slot. ComfyUI
        allows only one inbound link per input, so there is at most one."""
        for l in self.wf["links"]:
            if isinstance(l, list) and len(l) >= 6 and l[3] == tgt_node and l[4] == tgt_slot:
                return l
        return None

    def rewire_input(
        self, tgt_node: int, tgt_slot: int,
        new_src: int, new_src_slot: int, dtype: str,
    ) -> int:
        """Replace whatever feeds `tgt_node[tgt_slot]` with `new_src[new_src_slot]`.

        If an inbound link exists it's removed; a new link is then added.
        Returns the new link id. No-ops if the same wiring is already in place
        (still returns the existing link id, so callers can chain).
        """
        existing = self.find_link_to_slot(tgt_node, tgt_slot)
        if existing is not None:
            if existing[1] == new_src and existing[2] == new_src_slot:
                return existing[0]
            self.remove_link(existing[0])
        return self.add_link(new_src, new_src_slot, tgt_node, tgt_slot, dtype)

    @staticmethod
    def find_input_slot(node: dict, name: str) -> int:
        """Return the index of a named input slot on a node. Raises if missing."""
        for i, inp in enumerate(node.get("inputs", [])):
            if inp.get("name") == name:
                return i
        raise ValueError(f"Node {node.get('id')} has no input named {name!r}.")

    def find_links_from(self, src_node: int) -> list:
        """Find all links originating from a node."""
        return [l for l in self.wf["links"] if isinstance(l, list) and l[1] == src_node]

    def iter_edges(self):
        """Yield (src_node, tgt_node, type_label) edges from top-level links.

        Shared helper so tools don't hand-parse the `[id, src, src_slot, tgt,
        tgt_slot, type]` list format.
        """
        for link in self.wf["links"]:
            if isinstance(link, list) and len(link) >= 6:
                yield link[1], link[3], link[5]

    # --- Subgraph operations ---

    def get_subgraph(self, index: int = 0) -> dict | None:
        """Get a subgraph definition by index."""
        defs = self.wf.get("definitions", {})
        if isinstance(defs, dict):
            sgs = defs.get("subgraphs", [])
            if index < len(sgs):
                return sgs[index]
        return None

    def find_subgraph_invoker(self, sg_index: int = 0) -> dict | None:
        """Return the top-level node whose `type` is the subgraph UUID at
        `sg_index`, i.e. the node that invokes the subgraph as a loop body.
        Returns None if no invoker is wired."""
        sg = self.get_subgraph(sg_index)
        if not sg:
            return None
        sg_id = sg.get("id")
        if not sg_id:
            return None
        for n in self.wf["nodes"]:
            if n.get("type") == sg_id:
                return n
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
        """Find an internal link inside a subgraph by link id."""
        sg = self.get_subgraph(sg_index)
        if sg:
            for l in sg.get("links", []):
                if l["id"] == link_id:
                    return l
        return None

    def find_subgraph_link_to_slot(
        self, tgt_node: int, tgt_slot: int, sg_index: int = 0,
    ) -> dict | None:
        """Find the internal subgraph link feeding a specific input slot.
        Mirrors top-level `find_link_to_slot` but returns the dict-format
        subgraph link. ComfyUI allows only one inbound link per input, so
        there is at most one."""
        sg = self.get_subgraph(sg_index)
        if not sg:
            return None
        return next(
            (l for l in sg.get("links", [])
             if l.get("target_id") == tgt_node and l.get("target_slot") == tgt_slot),
            None,
        )

    def remove_subgraph_link(self, link_id: int, sg_index: int = 0):
        """Remove an internal link from a subgraph. Cleans target input.link,
        source output.links, and the virtual boundary node's linkIds."""
        sg = self.get_subgraph(sg_index)
        if not sg:
            raise ValueError(f"Subgraph {sg_index} not found")

        link_data = next((l for l in sg["links"] if l["id"] == link_id), None)
        if not link_data:
            raise ValueError(f"Subgraph link {link_id} not found")

        sg["links"] = [l for l in sg["links"] if l["id"] != link_id]

        tgt_id = link_data["target_id"]
        tgt_slot = link_data["target_slot"]
        src_id = link_data["origin_id"]
        src_slot = link_data["origin_slot"]

        for n in sg.get("nodes", []):
            if n["id"] == tgt_id:
                inputs = n.get("inputs", [])
                if tgt_slot < len(inputs) and inputs[tgt_slot].get("link") == link_id:
                    inputs[tgt_slot]["link"] = None
                break

        for n in sg.get("nodes", []):
            if n["id"] == src_id:
                outputs = n.get("outputs", [])
                if src_slot < len(outputs):
                    existing = outputs[src_slot].get("links") or []
                    cleaned = [l for l in existing if l != link_id]
                    outputs[src_slot]["links"] = cleaned if cleaned else None
                break

        for inp in sg.get("inputs", []):
            link_ids = inp.get("linkIds", [])
            if link_id in link_ids:
                inp["linkIds"] = [l for l in link_ids if l != link_id]

        for out in sg.get("outputs", []):
            link_ids = out.get("linkIds", [])
            if link_id in link_ids:
                out["linkIds"] = [l for l in link_ids if l != link_id]

    def add_subgraph_link(
        self, src_node: int, src_slot: int, tgt_node: int, tgt_slot: int,
        dtype: str, sg_index: int = 0,
    ) -> int:
        """Add an internal link inside a subgraph. Mirrors top-level `add_link`.
        Updates source node output.links and target node input.link. Returns
        new link id (pulled from the top-level `last_link_id` counter, which
        is shared across top-level and subgraph links in ComfyUI).

        Handles virtual collector node ids:
          - `tgt_node == -20`: updates `sg["outputs"][tgt_slot]["linkIds"]`
            (output collector — not in `sg["nodes"]`).
          - `src_node == -10`: updates `sg["inputs"][src_slot]["linkIds"]`
            (input distributor — not in `sg["nodes"]`).
        """
        sg = self.get_subgraph(sg_index)
        if not sg:
            raise ValueError(f"Subgraph {sg_index} not found")
        lid = self.next_link_id()
        sg["links"].append({
            "id": lid,
            "origin_id": src_node,
            "origin_slot": src_slot,
            "target_id": tgt_node,
            "target_slot": tgt_slot,
            "type": dtype,
        })

        if src_node == -10:
            ins = sg.get("inputs", [])
            if src_slot < len(ins):
                existing = ins[src_slot].get("linkIds") or []
                if lid not in existing:
                    existing.append(lid)
                ins[src_slot]["linkIds"] = existing
        else:
            for n in sg.get("nodes", []):
                if n["id"] == src_node:
                    outs = n.get("outputs", [])
                    if src_slot < len(outs):
                        existing = outs[src_slot].get("links") or []
                        if lid not in existing:
                            existing.append(lid)
                        outs[src_slot]["links"] = existing
                    break

        if tgt_node == -20:
            outs = sg.get("outputs", [])
            if tgt_slot < len(outs):
                existing = outs[tgt_slot].get("linkIds") or []
                if lid not in existing:
                    existing.append(lid)
                outs[tgt_slot]["linkIds"] = existing
        else:
            for n in sg.get("nodes", []):
                if n["id"] == tgt_node:
                    ins = n.get("inputs", [])
                    if tgt_slot < len(ins):
                        ins[tgt_slot]["link"] = lid
                    break

        return lid

    def add_subgraph_node(
        self,
        node_type: str,
        pos: list,
        size: list,
        inputs: list,
        outputs: list,
        properties: dict | None = None,
        widgets_values: list | dict | None = None,
        title: str | None = None,
        order: int = 0,
        mode: int = 0,
        sg_index: int = 0,
    ) -> int:
        """Append a new node into a subgraph. Mirrors `add_top_level_node`.
        Returns assigned node ID.

        Note: subgraph nodes share the top-level `last_node_id` counter."""
        sg = self.get_subgraph(sg_index)
        if not sg:
            raise ValueError(f"Subgraph {sg_index} not found")
        nid = self.next_node_id()
        node = {
            "id": nid,
            "type": node_type,
            "pos": pos,
            "size": size,
            "flags": {},
            "order": order,
            "mode": mode,
            "inputs": inputs,
            "outputs": outputs,
            "properties": properties or {"Node name for S&R": node_type},
        }
        if widgets_values is not None:
            node["widgets_values"] = widgets_values
        if title:
            node["title"] = title
        sg["nodes"].append(node)
        return nid

    def rewire_subgraph_input(
        self, tgt_node: int, tgt_slot: int,
        new_src: int, new_src_slot: int, dtype: str,
        sg_index: int = 0,
    ) -> int:
        """Replace whatever feeds `tgt_node[tgt_slot]` (inside the subgraph)
        with `new_src[new_src_slot]`. Mirrors top-level `rewire_input`.
        Returns the new link id (or the existing one if wiring already matches)."""
        if self.get_subgraph(sg_index) is None:
            raise ValueError(f"Subgraph {sg_index} not found")
        existing = self.find_subgraph_link_to_slot(tgt_node, tgt_slot, sg_index)
        if existing is not None:
            if existing["origin_id"] == new_src and existing["origin_slot"] == new_src_slot:
                return existing["id"]
            self.remove_subgraph_link(existing["id"], sg_index)
        return self.add_subgraph_link(new_src, new_src_slot, tgt_node, tgt_slot, dtype, sg_index)

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
