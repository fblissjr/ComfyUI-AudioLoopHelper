"""Trace node source: given a ComfyUI workflow + node_id, extract and report Python source.

Given a workflow.json and node_id, this utility:
1. Resolves node type to source file + class definition
2. Extracts full class body (INPUT_TYPES/execute/methods)
3. Builds call graph from execute() to depth N
4. Reports workflow wiring (--include-inputs)
5. Flags common issues (bypassed nodes, widget overrides, object_patches, etc.)
6. Outputs in markdown/json/text format

Usage:
    uv run --group dev python scripts/trace_node_source.py <workflow.json> <node_id> [options]

Options:
    --depth N               Levels of call-graph recursion (default 2)
    --format {text|markdown|json}  Output format (default markdown)
    --include-inputs        Include workflow wiring analysis
    --output PATH           Write to file instead of stdout
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import orjson
except ImportError:
    orjson = None


# Resolve paths relative to project root (ComfyUI-AudioLoopHelper)
PROJECT_ROOT = Path(__file__).parent.parent
COMFYUI_ROOT = PROJECT_ROOT.parent.parent
CUSTOM_NODES_ROOT = PROJECT_ROOT.parent


def find_node_class_source(node_type: str) -> tuple[Optional[Path], Optional[int]]:
    """Resolve node_type to source file and line number.
    
    Search order:
    1. ComfyUI extras (comfy_extras/*.py)
    2. ComfyUI core (nodes.py)
    3. Custom nodes (recursive grep for class definition, prefer files with class def)
    4. ComfyUI core (comfy/*.py)
    
    Returns: (path, line_number) or (None, None) if not found.
    """
    # Try comfy_extras
    extras_dir = COMFYUI_ROOT / "comfy_extras"
    if extras_dir.exists():
        for py_file in extras_dir.glob("*.py"):
            line = _find_class_line_or_mapping(py_file, node_type)
            if line:
                return py_file, line
    
    # Try core nodes.py
    core_nodes = COMFYUI_ROOT / "nodes.py"
    if core_nodes.exists():
        line = _find_class_line_or_mapping(core_nodes, node_type)
        if line:
            return core_nodes, line
    
    # Try custom_nodes recursively — prefer actual class definitions
    if CUSTOM_NODES_ROOT.exists():
        # First pass: look for direct class definitions
        for py_file in CUSTOM_NODES_ROOT.rglob("*.py"):
            if py_file.parent.name == "__pycache__":
                continue
            line = _find_class_line_exact(py_file, node_type)
            if line:
                return py_file, line
        
        # Second pass: look for NODE_CLASS_MAPPINGS or node_id entries
        for py_file in CUSTOM_NODES_ROOT.rglob("*.py"):
            if py_file.parent.name == "__pycache__":
                continue
            line = _find_class_line_or_mapping(py_file, node_type)
            if line:
                return py_file, line
    
    # Try comfy core
    comfy_dir = COMFYUI_ROOT / "comfy"
    if comfy_dir.exists():
        for py_file in comfy_dir.rglob("*.py"):
            line = _find_class_line_or_mapping(py_file, node_type)
            if line:
                return py_file, line
    
    return None, None


def _find_class_line_exact(filepath: Path, node_type: str) -> Optional[int]:
    """Find line number of exact class definition. Returns None if not found."""
    try:
        text = filepath.read_text()
        for i, line in enumerate(text.split("\n"), 1):
            if re.match(rf"^\s*class\s+{re.escape(node_type)}\b", line):
                return i
        return None
    except Exception:
        return None


def _find_class_line_or_mapping(filepath: Path, node_type: str) -> Optional[int]:
    """Find class definition or NODE_CLASS_MAPPINGS entry. Returns line number or None."""
    try:
        text = filepath.read_text()
        
        # Check for exact class definition
        for i, line in enumerate(text.split("\n"), 1):
            if re.match(rf"^\s*class\s+{re.escape(node_type)}\b", line):
                return i
        
        # Check for NODE_CLASS_MAPPINGS["NodeType"] or node_id="NodeType"
        if (f'"{node_type}"' in text or f"'{node_type}'" in text):
            if "NODE_CLASS_MAPPINGS" in text or "node_id=" in text:
                # Return line where NODE_CLASS_MAPPINGS or node_id is mentioned
                for i, line in enumerate(text.split("\n"), 1):
                    if f'"{node_type}"' in line or f"'{node_type}'" in line:
                        return i
        
        return None
    except Exception:
        return None


def extract_class_from_file(filepath: Path, class_name: str) -> Optional[str]:
    """Extract full class definition (including all methods) using ast."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Get class definition source by line numbers
                start_line = node.lineno - 1
                end_line = node.end_lineno or len(source.split("\n"))
                lines = source.split("\n")[start_line:end_line]
                return "\n".join(lines)
        
        return None
    except Exception as e:
        return f"# Error extracting class: {e}"


def build_call_graph(source: str, filepath: Path, depth: int) -> dict:
    """Extract execute() method and any called functions (depth-limited recursion).
    
    Returns dict with keys: execute_source, called_functions (list of dicts with
    name, source, call_type, file_hint).
    """
    result = {
        "execute_source": None,
        "called_functions": [],
        "warnings": [],
        "depth": depth,
    }
    
    try:
        tree = ast.parse(source)
        
        # Find execute method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                execute_start = node.lineno - 1
                execute_end = node.end_lineno or len(source.split("\n"))
                lines = source.split("\n")[execute_start:execute_end]
                result["execute_source"] = "\n".join(lines)
                
                # Extract function calls within execute
                calls = _extract_calls(node, depth)
                result["called_functions"] = calls
                break
        
        # Check for object_patches usage
        if "object_patches" in source or "add_object_patch" in source:
            result["warnings"].append(
                "Node uses object_patches.add_object_patch() — captures state "
                "in closures; sensitive to model offload/reload cycles (see "
                "CLAUDE.md NAG issue)"
            )
        
        # Check for captured tensors in closures
        if ".to(" in source and "nag_cond" in source:
            result["warnings"].append(
                "Found tensor.to(device, dtype) pattern — likely captures tensors "
                "in closure context"
            )
        
        return result
    except Exception as e:
        result["warnings"].append(f"Error parsing call graph: {e}")
        return result


def _extract_calls(func_node: ast.FunctionDef, depth: int, current_depth: int = 0) -> list:
    """Recursively extract function calls from a function node."""
    calls = []
    
    if current_depth >= depth:
        return calls
    
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            call_name = None
            call_type = "unknown"
            
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
                call_type = "function"
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "self":
                        call_name = node.func.attr
                        call_type = "method"
                    elif node.func.value.id == "torch":
                        call_name = f"torch.{node.func.attr}"
                        call_type = "torch"
                    else:
                        call_name = f"{node.func.value.id}.{node.func.attr}"
                        call_type = "module"
                else:
                    call_name = node.func.attr
                    call_type = "method"
            
            if call_name and call_name not in ["range", "len", "str", "int", "float", "list", "dict"]:
                calls.append({
                    "name": call_name,
                    "type": call_type,
                    "source": None,  # Would need full file context to extract
                })
    
    return calls


def analyze_workflow_wiring(workflow_path: str, node_id: int) -> dict:
    """Analyze workflow-level wiring of node (requires WorkflowEditor)."""
    try:
        # Import locally to avoid hard dependency
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from workflow_utils import WorkflowEditor
        
        ed = WorkflowEditor(workflow_path)
        node = ed.find_node(node_id)
        
        wiring = {
            "node_id": node_id,
            "node_type": node.get("type"),
            "inputs": [],
            "outputs": [],
            "warnings": [],
        }
        
        # Trace inputs
        for inp in ed.trace_node_inputs(node_id):
            wiring["inputs"].append({
                "slot": inp["slot"],
                "name": inp["name"],
                "source_node": inp.get("src_node"),
                "source_slot": inp.get("src_slot"),
                "source_type": inp.get("src_type"),
                "wired": inp["link"] is not None,
            })
        
        # Trace outputs
        if node.get("outputs"):
            for i, out in enumerate(node["outputs"]):
                targets = []
                if out.get("links"):
                    for link_id in out["links"]:
                        for link in ed.wf["links"]:
                            if isinstance(link, list) and link[0] == link_id:
                                tgt = ed.find_node(link[3])
                                targets.append({
                                    "target_node": link[3],
                                    "target_slot": link[4],
                                    "target_type": tgt.get("type"),
                                })
                                break
                
                wiring["outputs"].append({
                    "slot": i,
                    "name": out.get("name"),
                    "targets": targets,
                })
        
        # Check for widget overrides (widget value present + input wired)
        for inp in wiring["inputs"]:
            if inp["wired"]:
                if inp["slot"] < len(node.get("widgets_values", [])):
                    val = node["widgets_values"][inp["slot"]]
                    if val is not None and val != "":
                        wiring["warnings"].append(
                            f"Input [{inp['slot']}] {inp['name']}: widget value present "
                            f"but also wired — ComfyUI uses wired value; widget is default"
                        )
        
        # Check for bypassed node
        if node.get("mode") == 4:
            wiring["warnings"].append(
                "Node is bypassed (mode=4) — will be skipped during execution"
            )
        
        return wiring
    except Exception as e:
        return {
            "error": str(e),
            "inputs": [],
            "outputs": [],
            "warnings": [],
        }


def trace(node_id: int, workflow_path: str, depth: int = 2, 
          include_inputs: bool = False, format_type: str = "markdown") -> dict:
    """Main trace function — programmatically callable.
    
    Returns dict with keys: node_id, node_type, source_file, source_line,
    class_source, call_graph, wiring, warnings, runtime_summary, format.
    """
    result = {
        "node_id": node_id,
        "workflow_path": str(workflow_path),
        "node_type": None,
        "source_file": None,
        "source_line": None,
        "class_source": None,
        "call_graph": None,
        "wiring": None,
        "warnings": [],
        "runtime_summary": None,
        "format": format_type,
    }
    
    # Parse workflow
    try:
        with open(workflow_path) as f:
            wf = json.load(f)
    except Exception as e:
        result["warnings"].append(f"Failed to load workflow: {e}")
        return result
    
    # Find node
    node = None
    for n in wf.get("nodes", []):
        if n.get("id") == node_id:
            node = n
            break
    
    if not node:
        result["warnings"].append(f"Node {node_id} not found in workflow")
        return result
    
    result["node_type"] = node.get("type")
    
    # Resolve source
    source_file, source_line = find_node_class_source(node["type"])
    if not source_file:
        result["warnings"].append(
            f"Could not locate source for node type '{node['type']}'. "
            f"Searched: comfy_extras/*.py, {COMFYUI_ROOT}/nodes.py, "
            f"custom_nodes/*/**, comfy/**"
        )
        return result
    
    result["source_file"] = str(source_file)
    result["source_line"] = source_line
    
    # Extract class
    class_source = extract_class_from_file(source_file, node["type"])
    if not class_source:
        result["warnings"].append(f"Could not extract class '{node['type']}' from {source_file}")
        return result
    
    result["class_source"] = class_source
    
    # Build call graph
    call_graph = build_call_graph(class_source, source_file, depth)
    result["call_graph"] = call_graph if call_graph else {}
    result["warnings"].extend(call_graph.get("warnings", []) if call_graph else [])
    
    # Analyze wiring
    if include_inputs:
        wiring = analyze_workflow_wiring(workflow_path, node_id)
        result["wiring"] = wiring
        result["warnings"].extend(wiring.get("warnings", []))
    
    # Generate runtime summary
    result["runtime_summary"] = _generate_summary(class_source, call_graph if call_graph else {})
    
    return result


def _generate_summary(class_source: str, call_graph: dict) -> str:
    """Generate plain-English runtime summary from class source and call graph."""
    lines = []
    
    # Look for docstring
    try:
        tree = ast.parse(class_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    first_line = docstring.strip().split("\n")[0]
                    if first_line:
                        lines.append(first_line)
                    break
    except Exception:
        pass
    
    # Look for execute method docstring
    try:
        tree = ast.parse(class_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                docstring = ast.get_docstring(node)
                if docstring:
                    first_line = docstring.strip().split("\n")[0]
                    if first_line and first_line not in lines:
                        lines.append(first_line)
                break
    except Exception:
        pass
    
    # Extract key operations from execute source
    exec_src = call_graph.get("execute_source", "")
    if exec_src and "return" in exec_src:
        # Count return values to infer output type
        returns = re.findall(r"return\s+(.+?)(?:,|$)", exec_src, re.MULTILINE)
        if returns:
            lines.append(f"Returns {len(returns)} value(s).")
    
    if call_graph.get("called_functions"):
        func_names = ", ".join([f["name"] for f in call_graph["called_functions"][:3]])
        lines.append(f"Calls: {func_names}.")
    
    return " ".join(lines) if lines else "(No summary available)"


def format_output(result: dict, format_type: str) -> str:
    """Format result dict as markdown, json, or text."""
    if format_type == "json":
        if orjson:
            return orjson.dumps(result, default=str, option=orjson.OPT_INDENT_2).decode()
        else:
            return json.dumps(result, indent=2, default=str)
    
    elif format_type == "text":
        lines = [
            f"Node ID: {result['node_id']}",
            f"Node Type: {result['node_type']}",
            f"Source: {result['source_file']}:{result['source_line']}",
            "",
            "Class Source:",
            result.get("class_source") or "(not found)",
            "",
        ]
        if result.get("wiring"):
            lines.extend([
                "Inputs:",
            ])
            for i in result["wiring"].get("inputs", []):
                if i:
                    lines.append(f"  [{i.get('slot')}] {i.get('name')}: {i.get('source_type') if i.get('wired') else '(no link)'}")
            
            lines.extend(["", "Outputs:"])
            for i in result["wiring"].get("outputs", []):
                if i:
                    lines.append(f"  [{i.get('slot')}] {i.get('name')}: {len(i.get('targets', []))} target(s)")
            lines.append("")
        
        if result.get("warnings"):
            lines.extend(["Warnings:"] + [f"  - {w}" for w in result["warnings"]] + [""])
        
        return "\n".join(lines)
    
    else:  # markdown (default)
        lines = [
            "---",
            f"node_id: {result['node_id']}",
            f"node_type: {result['node_type']}",
            f"source: {result['source_file']}:{result['source_line']}",
            f"workflow: {result['workflow_path']}",
            "---",
            "",
        ]
        
        if result.get("wiring"):
            lines.extend([
                "## Wiring",
                "",
                "### Inputs",
            ])
            for inp in result["wiring"].get("inputs", []):
                if inp:
                    if inp.get("wired"):
                        lines.append(
                            f"- `[{inp['slot']}]` **{inp['name']}**: "
                            f"← Node {inp['source_node']} ({inp['source_type']}) slot {inp['source_slot']}"
                        )
                    else:
                        lines.append(f"- `[{inp['slot']}]` **{inp['name']}**: (no link)")
            
            lines.extend(["", "### Outputs"])
            for out in result["wiring"].get("outputs", []):
                if out:
                    targets = out.get("targets", [])
                    if targets:
                        targets_str = "; ".join(
                            [f"Node {t['target_node']} ({t['target_type']}) slot {t['target_slot']}"
                             for t in targets if t]
                        )
                        lines.append(f"- `[{out['slot']}]` **{out['name']}**: → {targets_str}")
                    else:
                        lines.append(f"- `[{out['slot']}]` **{out['name']}**: (no targets)")
            lines.append("")
        
        lines.extend([
            "## Source",
            "",
            "```python",
            result.get("class_source") or "(not found)",
            "```",
            "",
        ])
        
        call_graph = result.get("call_graph") or {}
        if call_graph.get("execute_source"):
            lines.extend([
                "## Execute Method",
                "",
                "```python",
                call_graph["execute_source"],
                "```",
                "",
            ])
        
        if call_graph.get("called_functions"):
            lines.extend([
                "## Call Graph (Depth " + str(call_graph.get("depth", 2)) + ")",
                "",
            ])
            for call in call_graph["called_functions"]:
                lines.append(f"- **{call['name']}** ({call['type']})")
            lines.append("")
        
        if result.get("warnings"):
            lines.extend([
                "## Warnings",
                "",
            ] + [f"- {w}" for w in result["warnings"]] + [""])
        
        lines.extend([
            "## Runtime Summary",
            "",
            result.get("runtime_summary") or "(no summary)",
            "",
        ])
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Trace ComfyUI node source: find class, extract methods, build call graph."
    )
    parser.add_argument("workflow", help="Path to workflow.json")
    parser.add_argument("node_id", type=int, help="Node ID to trace")
    parser.add_argument(
        "--depth", type=int, default=2,
        help="Depth of call-graph recursion (default 2)"
    )
    parser.add_argument(
        "--format", choices=["text", "markdown", "json"], default="markdown",
        help="Output format (default markdown)"
    )
    parser.add_argument(
        "--include-inputs", action="store_true",
        help="Include workflow wiring analysis"
    )
    parser.add_argument(
        "--output", help="Write to file instead of stdout"
    )
    
    args = parser.parse_args()
    
    # Resolve workflow path
    wf_path = Path(args.workflow)
    if not wf_path.is_absolute():
        wf_path = Path.cwd() / wf_path
    
    # Trace
    result = trace(
        args.node_id,
        str(wf_path),
        depth=args.depth,
        include_inputs=args.include_inputs,
        format_type=args.format,
    )
    
    # Format
    output = format_output(result, args.format)
    
    # Write
    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
