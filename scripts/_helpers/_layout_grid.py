"""Reusable column-grid layout for ComfyUI workflow JSON.

Last updated: 2026-05-08

Extracted from `scripts/apply_intro_workflow.py::_layout_workflow` so multiple
apply scripts can share the same layout primitives.

Concepts
--------

- **Group**: a visual rectangle on the canvas. Each group has an origin
  (x, y), a color, a title, and a derived `bounding` box `[x, y, w, h]`
  computed from member nodes' actual positions + sizes after layout.
- **Node-group binding**: tells the layout which nodes belong to which
  group. Bindings come from two sources:
    1. an explicit id-mapping passed in the LayoutSpec (`node_groups`)
    2. runtime-tagged nodes that carry their group key in
       `node["properties"][group_tag_key]` — used for nodes inserted by
       the apply script that don't have stable IDs at design time
- **Note anchor**: a Note node positioned by an offset from a group's
  bounding-box origin. Notes track their group when the grid reflows so
  layout edits don't strand them in space.
- **Pill**: a collapsed Get/Set reroute. Pills sort to the top of their
  group with `COLLAPSED_GAP` packing, forming a thin plumbing strip at
  the column head.

Usage
-----

Apply scripts build a LayoutSpec and call `apply_layout(wf, spec)`:

    from _layout_grid import LayoutSpec, GroupSpec, NoteAnchor, apply_layout

    spec = LayoutSpec(
        groups={
            "inputs_required": GroupSpec(origin=(0, 200), color="#29699c", title="REQUIRED"),
            "inputs_common":   GroupSpec(origin=(0, 700), color="#3f789e", title="COMMON"),
            ...
        },
        node_groups={565: "inputs_required", 444: "inputs_required", ...},
        note_anchors={
            "readme": NoteAnchor(group="inputs_required", dx=0, dy=-540, w=660, h=600),
        },
    )
    apply_layout(wf, spec)

Template extraction (--from-template golden.json):

    spec = extract_template(golden_wf)
    apply_layout(target_wf, spec)

Reference
---------
- Pattern source: `scripts/apply_intro_workflow.py::_layout_workflow`
- v0 -> v1 history: `internal/design/intro_workflow_design.md` ("v1 layout fix")
- Group bounding shape `[x, y, w, h]` verified against
  `example_workflows/audio-loop-music-video_latent.json` group entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Module-scope constants. Kept compatible with apply_intro_workflow's
# original values so a follow-up refactor produces byte-identical layout.
INTRA_NODE_GAP = 50      # vertical gap between full-size nodes
COLLAPSED_GAP = 12       # vertical gap between collapsed pills
NODE_X_OFFSET = 30       # node x within group (from group origin)
NODE_Y_OFFSET = 60       # node y within group (room for title bar)
GROUP_PAD = 40           # horizontal/vertical padding around group bounds
GROUP_BANNER_H = 30      # banner above group content (group origin sits below banner)

# Default tag keys. Apply scripts can override via LayoutSpec for
# co-existence with other tagging conventions.
DEFAULT_GROUP_TAG_KEY = "_alh_group"
DEFAULT_NOTE_KEY_TAG = "_alh_note_key"


@dataclass
class GroupSpec:
    """Per-group origin + visual attributes.

    `origin` is (x, y) of the group's content area (banner sits above).
    Bounds are computed at apply time from member-node positions + sizes,
    so changing `origin` is sufficient to move the whole column.
    """
    origin: tuple[float, float]
    color: str
    title: str
    font_size: int = 24


@dataclass
class NoteAnchor:
    """Position a Note node relative to a group's bounding box.

    `dx` and `dy` are offsets from the anchor group's `[x, y]` origin.
    `dy < 0` puts the note above the group; `dx > group_width` puts it
    to the right. `w` and `h` set the note's rendered size.
    """
    group: str
    dx: float
    dy: float
    w: float
    h: float


@dataclass
class LayoutSpec:
    """Full layout specification consumed by `apply_layout`.

    `groups` is render-order-significant: it determines the array order
    in `wf["groups"]` which maps to z-order on canvas. Earlier entries
    render below later ones, so put parent / background groups first.
    """
    groups: dict[str, GroupSpec]
    node_groups: dict[int, str] = field(default_factory=dict)
    note_anchors: dict[str, NoteAnchor] = field(default_factory=dict)
    group_tag_key: str = DEFAULT_GROUP_TAG_KEY
    note_key_tag: str = DEFAULT_NOTE_KEY_TAG


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def is_pill(node: dict) -> bool:
    """Collapsed Get/Set reroute (KJNodes virtual wires)."""
    return (
        node.get("flags", {}).get("collapsed", False)
        and node.get("type") in ("GetNode", "SetNode")
    )


def _resolve_group(node: dict, spec: LayoutSpec) -> str | None:
    """Group binding precedence: properties tag > id mapping."""
    tag = (node.get("properties") or {}).get(spec.group_tag_key)
    if tag is not None:
        return tag
    nid = node.get("id")
    if nid is None:
        return None
    return spec.node_groups.get(nid)


def _bin_nodes(
    nodes: list[dict],
    spec: LayoutSpec,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Split nodes into per-group bins + a notes-by-key map.

    Notes are recognized by carrying `note_key_tag` in their properties;
    they're held aside so `apply_layout` can position them after group
    bounds are known.
    """
    bins: dict[str, list[dict]] = {gkey: [] for gkey in spec.groups}
    notes_by_key: dict[str, dict] = {}
    for node in nodes:
        note_key = (node.get("properties") or {}).get(spec.note_key_tag)
        if note_key is not None:
            notes_by_key[note_key] = node
            continue
        gkey = _resolve_group(node, spec)
        if gkey is not None and gkey in bins:
            bins[gkey].append(node)
    return bins, notes_by_key


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def apply_layout(wf: dict, spec: LayoutSpec) -> None:
    """Reposition every classified node into its group column; compute
    per-group bounding boxes; position note nodes; replace `wf["groups"]`.

    Mutates `wf` in place. Nodes/notes not bound to any group keep their
    existing `pos` (no-op for unclassified nodes — useful when the spec
    only covers a subset of the workflow).
    """
    bins, notes_by_key = _bin_nodes(wf.get("nodes", []), spec)

    # Sort within each bin: pills first (sorted by id), then full nodes
    # (sorted by id). Stable order makes apply runs deterministic.
    for gkey in bins:
        bins[gkey].sort(key=lambda n: (0 if is_pill(n) else 1, n.get("id", 0)))

    group_bounds: dict[str, tuple[float, float, float, float]] = {}
    for gkey, gspec in spec.groups.items():
        nodes = bins[gkey]
        if not nodes:
            continue
        gx, gy = gspec.origin
        cur_y = gy + NODE_Y_OFFSET
        max_w = 0.0
        for node in nodes:
            sz = node.get("size") or [280, 80]
            w, h = float(sz[0]), float(sz[1])
            node["pos"] = [gx + NODE_X_OFFSET, cur_y]
            cur_y += h + (COLLAPSED_GAP if is_pill(node) else INTRA_NODE_GAP)
            if w > max_w:
                max_w = w
        gw = max_w + 2 * GROUP_PAD + NODE_X_OFFSET
        gh = (cur_y - gy) + GROUP_PAD
        group_bounds[gkey] = (gx, gy - GROUP_BANNER_H, gw, gh)

    # Position notes via anchor offsets from group bounding origins.
    for note_key, anchor in spec.note_anchors.items():
        note = notes_by_key.get(note_key)
        bounds = group_bounds.get(anchor.group)
        if note is None or bounds is None:
            continue
        bx, by = bounds[0], bounds[1]
        note["pos"] = [bx + anchor.dx, by + anchor.dy]
        note["size"] = [anchor.w, anchor.h]

    # Rewrite the groups array. Render-order = spec insertion order.
    new_groups: list[dict] = []
    for i, (gkey, gspec) in enumerate(spec.groups.items(), start=1):
        bounds = group_bounds.get(gkey)
        if bounds is None:
            continue
        bx, by, bw, bh = bounds
        new_groups.append({
            "id": i,
            "title": gspec.title,
            "bounding": [bx, by, bw, bh],
            "color": gspec.color,
            "font_size": gspec.font_size,
            "flags": {},
        })
    wf["groups"] = new_groups


def assigned_nodes(spec: LayoutSpec, wf: dict) -> dict[str, list[int]]:
    """Return `{group_key: [node_ids]}` per the spec, for diagnostics.

    Useful in --dry-run output to surface which tier each node lands in.
    """
    bins: dict[str, list[int]] = {gkey: [] for gkey in spec.groups}
    for node in wf.get("nodes", []):
        if (node.get("properties") or {}).get(spec.note_key_tag) is not None:
            continue
        gkey = _resolve_group(node, spec)
        if gkey is not None and gkey in bins:
            bins[gkey].append(node.get("id", -1))
    for gkey in bins:
        bins[gkey].sort()
    return bins


def unassigned_node_ids(spec: LayoutSpec, wf: dict) -> list[int]:
    """Node ids not bound to any group. Surface in --dry-run to catch
    drift: a fresh apply on a workflow that's grown new nodes since the
    spec was authored will list them here so the spec author can decide
    whether to bin them or leave them unmanaged."""
    out: list[int] = []
    for node in wf.get("nodes", []):
        if (node.get("properties") or {}).get(spec.note_key_tag) is not None:
            continue
        if _resolve_group(node, spec) is None:
            out.append(node.get("id", -1))
    out.sort()
    return out


# --------------------------------------------------------------------------
# Template extraction (for `--from-template <golden.json>`)
# --------------------------------------------------------------------------

def extract_template(wf: dict) -> LayoutSpec:
    """Build a LayoutSpec from a hand-laid-out workflow.

    Reads existing `wf["groups"]` for origins/colors/titles, and assigns
    each node to whichever group's bounding box contains its `pos`. The
    resulting spec, when applied to the same workflow, reproduces the
    layout (modulo `cur_y` packing — re-applying snaps nodes to the
    canonical y-cadence).

    Node-tag and note-tag keys default to ALH conventions; override on
    the returned spec if you're consuming a different tagging scheme.
    """
    groups: dict[str, GroupSpec] = {}
    bounds: dict[str, tuple[float, float, float, float]] = {}
    for g in wf.get("groups", []):
        title = g.get("title") or f"group_{g.get('id', '?')}"
        gkey = title.lower().replace(" ", "_").replace(".", "")
        b = g.get("bounding") or [0, 0, 0, 0]
        bx, by, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        # Origin is content area (below banner), not the bounding box top.
        groups[gkey] = GroupSpec(
            origin=(bx, by + GROUP_BANNER_H),
            color=g.get("color") or "#3f789e",
            title=title,
            font_size=int(g.get("font_size") or 24),
        )
        bounds[gkey] = (bx, by, bw, bh)

    node_groups: dict[int, str] = {}
    for node in wf.get("nodes", []):
        if node.get("type") == "Note":
            continue
        pos = node.get("pos") or [0, 0]
        nx, ny = float(pos[0]), float(pos[1])
        for gkey, (bx, by, bw, bh) in bounds.items():
            if bx <= nx <= bx + bw and by <= ny <= by + bh:
                node_groups[node["id"]] = gkey
                break

    return LayoutSpec(groups=groups, node_groups=node_groups)


def summarize(spec: LayoutSpec, wf: dict) -> str:
    """Human-readable spec summary for --dry-run."""
    lines = [f"LayoutSpec: {len(spec.groups)} groups, {len(spec.node_groups)} pinned ids, {len(spec.note_anchors)} note anchors"]
    bins = assigned_nodes(spec, wf)
    for gkey, gspec in spec.groups.items():
        ids = bins.get(gkey, [])
        ox, oy = gspec.origin
        lines.append(f"  {gkey:24s} @ ({ox:>5.0f}, {oy:>5.0f})  {len(ids):>3d} nodes  {gspec.title}")
    unassigned = unassigned_node_ids(spec, wf)
    if unassigned:
        lines.append(f"  unassigned ({len(unassigned)}): {unassigned}")
    return "\n".join(lines)


__all__ = [
    "LayoutSpec",
    "GroupSpec",
    "NoteAnchor",
    "apply_layout",
    "extract_template",
    "is_pill",
    "assigned_nodes",
    "unassigned_node_ids",
    "summarize",
    "INTRA_NODE_GAP",
    "COLLAPSED_GAP",
    "NODE_X_OFFSET",
    "NODE_Y_OFFSET",
    "GROUP_PAD",
    "GROUP_BANNER_H",
]
