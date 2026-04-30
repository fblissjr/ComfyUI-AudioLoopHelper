---
name: workflow-edit
description: Edit ComfyUI workflow JSON using WorkflowEditor utility. Use when modifying nodes, links, widget values, or subgraph internals in workflow files.
---

# Workflow Edit

Edit ComfyUI workflow JSON files programmatically using the WorkflowEditor utility.

## Always use WorkflowEditor

```python
import sys
from pathlib import Path

# Resolve scripts/ relative to the project root (works regardless of CWD)
_project_root = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(_project_root / 'scripts'))
from workflow_utils import WorkflowEditor

ed = WorkflowEditor('example_workflows/audio-loop-music-video_image.json')
```

Or if running inline via `python3 -c`:
```python
import sys
sys.path.insert(0, 'scripts')
from workflow_utils import WorkflowEditor

ed = WorkflowEditor('example_workflows/audio-loop-music-video_image.json')
```

## Common operations

### Inspect a node
```python
ed.print_node_summary(843)
ed.trace_forward(1507)  # follow output chain
ed.trace_node_inputs(161)  # show all inputs
```

### Modify widget values
```python
n = ed.find_node(531)
n['widgets_values'] = ['2', 1, 0, 1, -1]
```

### Bypass/activate nodes
```python
ed.find_node(1558)['mode'] = 4  # bypass
ed.find_node(1588)['mode'] = 0  # activate
```

### Rewire links

Prefer `rewire_input` for the common "replace whatever feeds an input slot" pattern — it's idempotent and handles the remove+add in one call:
```python
ed.rewire_input(tgt_node=1587, tgt_slot=0, new_src=1588, new_src_slot=0, dtype="CONDITIONING")
```

Manual splice only when you need to insert a new node between two existing ones (remove one link, add two):
```python
ed.remove_link(2906)
ed.add_link(1588, 0, new_node_id, 0, "CONDITIONING")
ed.add_link(new_node_id, 0, 1587, 0, "CONDITIONING")
```

Lookup first if you don't have the link id:
```python
row = ed.find_link_to_slot(tgt_node=1587, tgt_slot=0)  # returns [id, src, src_slot, tgt, tgt_slot, type] or None
```

### Subgraph edits
```python
sg_node = ed.find_subgraph_node(606)  # find node inside subgraph
ed.remove_subgraph_link(1782)  # remove internal link
```

### Add nodes
```python
n = WorkflowEditor.make_node(1589, "LatentUpscaleModelLoader", [4500, 5800],
    widgets=["ltx-2.3-spatial-upscaler-x2-1.0.safetensors"],
    title="Load Spatial Upscaler 2x")
ed.add_node(n)
```

### Remove nodes
```python
ed.remove_node_and_links(1596)  # detaches every link touching the node, then removes it
```
Callers must re-wire any surviving slots before use — the helper only detaches.

## Always save and validate
```python
ed.save()                              # quiet
ed.save('out.json', verbose=True)      # prints "Saved to ..."
```

Validation runs automatically via PostToolUse hook on workflow JSON writes. To run manually:
```bash
python3 scripts/test_workflow_integrity.py example_workflows/*.json
```

## Key facts
- Top-level links: array format `[id, src, src_slot, tgt, tgt_slot, type]`
- Subgraph links: dict format `{id, origin_id, origin_slot, target_id, target_slot, type}`
- Subgraph defs: `wf['definitions']['subgraphs'][0]`
- Subgraph input node: -10, output node: -20
- Node mode: 0 = active, 4 = bypassed
- DynamicCombo widget format (LTXVImgToVideoInplaceKJ): `[num, str1, str2, ..., idx1, idx2, ...]`
  Strengths first, indices second. NOT interleaved. Example: `['2', 1.0, 0.5, 0, -1]`
