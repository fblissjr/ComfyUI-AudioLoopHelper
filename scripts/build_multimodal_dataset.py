"""build_multimodal_dataset.

Last updated: 2026-05-31

Point this at a folder of ComfyUI renders (the `<base>.png` / `<base>.mp4` /
`<base>-audio.mp4` triplets that VHS_VideoCombine writes) and get a schema'd
multimodal dataset out: one JSONL row per render, plus a `dataset/media/` folder
of SYMLINKS to the media (absolute targets — see `--media`), plus a
`dataset_card.md` documenting the schema.

Why this works with zero re-rendering: ComfyUI embeds the executed graph in the
PNG's `prompt` tEXt chunk (API format). We read it, follow the node links, and
flatten the bits that matter — the positive/negative prompt, the reference
audio file, the LoRA + strength, and the generation params (resolution, length,
fps, seed, sampler, cfg, sigmas/steps). Reference-audio files that can't be
located on disk are recorded with `found: false` (never silently dropped) —
the embedded workflow always knows the FILENAME even when the file moved.

The output JSONL is directly loadable:
    from datasets import load_dataset
    ds = load_dataset("json", data_files="dataset/dataset.jsonl")

Usage:
    uv run --group analysis python scripts/build_multimodal_dataset.py <renders_dir> -o <out_dir>
    uv run --group analysis python scripts/build_multimodal_dataset.py <renders_dir> \
        -o <out_dir> --audio-root /path/to/comfy/input --audio-root ./data/audio
    uv run --group analysis python scripts/build_multimodal_dataset.py <renders_dir> --media reference

`--media`:
    symlink   (default) — dataset/media/ holds symlinks (absolute targets) to the
                          source files. Row paths are dataset-relative ("media/..");
                          the dataset survives being moved, but is NOT self-contained
                          (sources must stay put / same machine).
    reference          — rows carry the source ABSOLUTE paths in place; no media/
                          folder. Fast, but the JSONL then contains absolute paths
                          (do not share/commit it — path-privacy footgun).

Media probing (duration / sample_rate) uses stdlib `wave` for .wav and `ffprobe`
for everything else; both degrade gracefully to null when unavailable.
"""

from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import shutil
import subprocess
import sys
import urllib.parse
import wave
from dataclasses import dataclass
from pathlib import Path

import orjson
from PIL import Image


# --------------------------------------------------------------------------- #
# Graph parsing (pure — the testable core)
# --------------------------------------------------------------------------- #

# Negative-prompt lexicon: used only as a fallback to classify two CLIPTextEncode
# nodes when no LTXVConditioning is present to disambiguate by link.
_NEG_LEXICON = (
    "blurry", "deformed", "low quality", "watermark", "distorted",
    "bad anatomy", "artifacts", "text, subtitles", "low resolution",
)

# Node types whose primary scalar we know how to read when a link points at them.
_CONSTANT_VALUE_KEYS = {
    "INTConstant": "value",
    "FloatConstant": "value",
    "PrimitiveNode": "value",
    "PrimitiveInt": "value",
    "PrimitiveFloat": "value",
    "Int Literal": "int",
}

# Terminal samplers + the guiders/loaders we trace backward from to find the
# conditioning, model, and latent that ACTUALLY produced the render (vs. picking
# an arbitrary node by graph order).
_SAMPLER_TYPES = {"SamplerCustomAdvanced", "SamplerCustom", "KSampler", "KSamplerAdvanced"}
_GUIDER_TYPES = {"CFGGuider", "MultimodalGuider", "BasicGuider", "DualCFGGuider"}
_LORA_LOADER_TYPES = {
    "LTXAudioICLoRALoader", "LTXICLoRALoaderModelOnly", "LoraLoaderModelOnly", "LoraLoader",
}
# CONDITIONING passthrough nodes: output slot -> input name to keep tracing along.
# Lets the trace follow the POSITIVE branch through a node without wandering into
# its negative/reference inputs (the slot-unaware bug).
_COND_PASSTHROUGH = {
    "LTXVConditioning": {0: "positive", 1: "negative"},
    "LTXAddAudioICLoRAGuide": {0: "positive", 1: "negative"},
    "LTXAddVideoICLoRAGuide": {0: "positive", 1: "negative"},
    "LTXAudioSetRefTokens": {0: "positive", 1: "negative"},
    "LTXVSetAudioRefTokens": {0: "positive", 1: "negative"},
}
_REF_AUDIO_NODES = {"LTXAddAudioICLoRAGuide", "LTXAudioSetRefTokens", "LTXVSetAudioRefTokens"}


def _is_link(value) -> bool:
    """API-format links are [node_id:str, slot:int]."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _resolve_scalar(graph: dict, value, _depth: int = 0):
    """Return a literal scalar, following a link through known constant nodes.

    Literals pass through unchanged. A link to an INTConstant/FloatConstant (or
    a CM_FloatToInt that itself links to one) is resolved to its value. A link to
    a node we can't interpret returns None rather than crashing — the caller
    records the absence instead of a bogus number.
    """
    if not _is_link(value):
        return value          # literal — pass through
    if _depth > 8:
        return None           # link chain too deep / cyclic — give up
    node = graph.get(value[0])
    if not node:
        return None
    ct = node.get("class_type", "")
    inputs = node.get("inputs", {})
    if ct in _CONSTANT_VALUE_KEYS:
        return _resolve_scalar(graph, inputs.get(_CONSTANT_VALUE_KEYS[ct]), _depth + 1)
    if ct == "CM_FloatToInt":
        v = _resolve_scalar(graph, inputs.get("a"), _depth + 1)
        return int(v) if isinstance(v, (int, float)) else v
    return None


def _find(graph: dict, class_type: str) -> list[tuple[str, dict]]:
    return [(nid, n) for nid, n in graph.items() if n.get("class_type") == class_type]


def _first(graph: dict, class_type: str) -> dict | None:
    return next((n for n in graph.values() if n.get("class_type") == class_type), None)


def _text_at_link(graph: dict, link) -> str | None:
    """Follow a CONDITIONING link to its CLIPTextEncode and return the text."""
    if not _is_link(link):
        return None
    node = graph.get(link[0])
    if not node:
        return None
    inp = node.get("inputs", {})
    if node.get("class_type") == "CLIPTextEncode":
        return inp.get("text")
    # one hop of indirection (e.g. through a passthrough conditioning node)
    for v in inp.values():
        if _is_link(v):
            t = _text_at_link(graph, v)
            if t is not None:
                return t
    return None


def _looks_negative(text: str) -> bool:
    low = text.lower()
    return sum(tok in low for tok in _NEG_LEXICON) >= 2


def _prompts(graph: dict) -> tuple[str | None, str | None]:
    """(positive, negative). Prefer LTXVConditioning's explicit links; fall back
    to lexicon classification of the CLIPTextEncode nodes."""
    cond = _first(graph, "LTXVConditioning")
    if cond:
        pos = _text_at_link(graph, cond["inputs"].get("positive"))
        neg = _text_at_link(graph, cond["inputs"].get("negative"))
        if pos is not None or neg is not None:
            return pos, neg
    texts = [n["inputs"].get("text", "") for _, n in _find(graph, "CLIPTextEncode")]
    texts = [t for t in texts if isinstance(t, str) and t]
    pos = next((t for t in texts if not _looks_negative(t)), None)
    neg = next((t for t in texts if _looks_negative(t)), None)
    return pos, neg


def _reference_audio(graph: dict) -> tuple[str | None, str | None]:
    """(filename, audioUI hint). Follow the IC-LoRA guide's reference_audio link
    to its LoadAudio; fall back to the first LoadAudio."""
    guide = _first(graph, "LTXAddAudioICLoRAGuide")
    load = None
    if guide:
        ref = guide["inputs"].get("reference_audio")
        if _is_link(ref):
            load = graph.get(ref[0])
    if load is None or load.get("class_type") != "LoadAudio":
        loads = _find(graph, "LoadAudio")
        load = loads[0][1] if loads else None
    if not load:
        return None, None
    inp = load.get("inputs", {})
    return inp.get("audio"), inp.get("audioUI")


def _loras_fallback(graph: dict) -> list[dict]:
    node = (_first(graph, "LTXAudioICLoRALoader") or _first(graph, "LTXICLoRALoaderModelOnly")
            or _first(graph, "LoraLoaderModelOnly"))
    if not node:
        return []
    li = node["inputs"]
    name = li.get("lora_name")
    return [{"name": name, "strength": li.get("strength_model")}] if name else []


# --- backward trace from the terminal sampler (the "right way") ------------- #

def _build_consumers(graph: dict) -> dict:
    """node_id -> [downstream node ids that consume one of its outputs]."""
    consumers: dict[str, list[str]] = {}
    for nid, n in graph.items():
        for v in n.get("inputs", {}).values():
            if _is_link(v):
                consumers.setdefault(v[0], []).append(nid)
    return consumers


def _reachable_forward(consumers: dict, start: str) -> set:
    seen: set[str] = set()
    stack = [start]
    while stack:
        for c in consumers.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _find_terminal_sampler(graph: dict) -> str | None:
    """The sampler that actually produced the final latent — i.e. the one with no
    other sampler downstream of it (handles two-stage upscale graphs)."""
    samplers = [nid for nid, n in graph.items() if n.get("class_type") in _SAMPLER_TYPES]
    if not samplers:
        return None
    if len(samplers) == 1:
        return samplers[0]
    consumers = _build_consumers(graph)
    sset = set(samplers)
    terminal = [s for s in samplers if not (_reachable_forward(consumers, s) & (sset - {s}))]
    return terminal[-1] if terminal else samplers[-1]


def _guider_anchor(graph: dict, samp: str | None) -> tuple:
    """(positive_link, negative_link, model_link, cfg_value) from the terminal
    sampler's guider (or the sampler directly for KSampler-style), or a standalone
    guider. (None, None, None, None) when there's no anchor. `samp` is the terminal
    sampler id (computed once by the caller and reused for dims)."""
    node = None
    if samp:
        si = graph[samp].get("inputs", {})
        if _is_link(si.get("guider")):
            node = graph.get(si["guider"][0])
        else:
            return (si.get("positive"), si.get("negative"), si.get("model"), si.get("cfg"))
    if node is None:
        gids = [nid for nid, n in graph.items() if n.get("class_type") in _GUIDER_TYPES]
        node = graph.get(gids[0]) if gids else None
    if node is None:
        return (None, None, None, None)
    gi = node.get("inputs", {})
    if node.get("class_type") == "BasicGuider":
        return (gi.get("conditioning"), None, gi.get("model"), None)
    return (gi.get("positive"), gi.get("negative"), gi.get("model"), gi.get("cfg"))


def _trace_cond_text(graph: dict, link, _seen=None, _depth=0) -> str | None:
    """Follow a CONDITIONING link to its CLIPTextEncode, staying on the correct
    branch through known passthrough nodes (slot-aware)."""
    if not _is_link(link) or _depth > 32:
        return None
    nid, slot = link[0], link[1]
    _seen = _seen or set()
    if nid in _seen:
        return None
    _seen.add(nid)
    node = graph.get(nid)
    if not node:
        return None
    ct, inp = node.get("class_type", ""), node.get("inputs", {})
    if ct == "CLIPTextEncode":
        return inp.get("text")
    if ct in _COND_PASSTHROUGH:
        nxt = _COND_PASSTHROUGH[ct].get(slot)
        if nxt is not None and _is_link(inp.get(nxt)):
            return _trace_cond_text(graph, inp[nxt], _seen, _depth + 1)
    for name in ("positive", "conditioning", "cond"):
        if _is_link(inp.get(name)):
            t = _trace_cond_text(graph, inp[name], _seen, _depth + 1)
            if t is not None:
                return t
    for v in inp.values():  # unknown node: any conditioning-ish upstream
        if _is_link(v):
            t = _trace_cond_text(graph, v, _seen, _depth + 1)
            if t is not None:
                return t
    return None


def _trace_ref_audio(graph: dict, link, _seen=None, _depth=0) -> tuple:
    """Find the reference-audio LoadAudio on the positive conditioning path (via the
    IC-LoRA guide). Returns (filename, audioUI) or (None, None) — None correctly means
    'no reference attached' (e.g. the guide is bypassed/absent)."""
    if not _is_link(link) or _depth > 32:
        return None, None
    nid, slot = link[0], link[1]
    _seen = _seen or set()
    if nid in _seen:
        return None, None
    _seen.add(nid)
    node = graph.get(nid)
    if not node:
        return None, None
    ct, inp = node.get("class_type", ""), node.get("inputs", {})
    if ct in _REF_AUDIO_NODES and _is_link(inp.get("reference_audio")):
        la = graph.get(inp["reference_audio"][0])
        if la and la.get("class_type") == "LoadAudio":
            return la["inputs"].get("audio"), la["inputs"].get("audioUI")
    if ct in _COND_PASSTHROUGH:
        nxt = _COND_PASSTHROUGH[ct].get(slot)
        if nxt is not None and _is_link(inp.get(nxt)):
            return _trace_ref_audio(graph, inp[nxt], _seen, _depth + 1)
    for name in ("positive", "conditioning", "cond"):
        if _is_link(inp.get(name)):
            r = _trace_ref_audio(graph, inp[name], _seen, _depth + 1)
            if r[0] is not None:
                return r
    return None, None


def _trace_loras(graph: dict, model_link) -> list[dict]:
    """LoRAs actually applied along the model chain, base-first, skipping empty-name
    placeholder loaders."""
    loras: list[dict] = []
    seen: set[str] = set()

    def walk(link, depth):
        if not _is_link(link) or depth > 64 or link[0] in seen:
            return
        seen.add(link[0])
        node = graph.get(link[0])
        if not node:
            return
        ct, inp = node.get("class_type", ""), node.get("inputs", {})
        if _is_link(inp.get("model")):
            walk(inp["model"], depth + 1)  # base first
        if ct in _LORA_LOADER_TYPES and inp.get("lora_name"):
            loras.append({"name": inp["lora_name"], "strength": inp.get("strength_model")})

    walk(model_link, 0)
    return loras


def _trace_to_class(graph: dict, link, target: str, _seen=None, _depth=0) -> dict | None:
    """First node of `target` reachable backward from `link`."""
    if not _is_link(link) or _depth > 64:
        return None
    nid = link[0]
    _seen = _seen or set()
    if nid in _seen:
        return None
    _seen.add(nid)
    node = graph.get(nid)
    if not node:
        return None
    if node.get("class_type") == target:
        return node
    for v in node.get("inputs", {}).values():
        if _is_link(v):
            r = _trace_to_class(graph, v, target, _seen, _depth + 1)
            if r is not None:
                return r
    return None


def parse_prompt_graph(graph: dict) -> dict:
    """Flatten an API-format prompt graph into the dataset's metadata fields by
    tracing backward from the terminal sampler (the conditioning/model/latent that
    actually produced the render), with heuristic fallbacks for degenerate graphs
    and a `warnings` list for anything that couldn't be resolved cleanly."""
    warnings: list[str] = []
    samp = _find_terminal_sampler(graph)  # computed once; reused for the anchor + dims
    pos_link, neg_link, model_link, cfg_v = _guider_anchor(graph, samp)

    prompt = _trace_cond_text(graph, pos_link) if pos_link else None
    negative = _trace_cond_text(graph, neg_link) if neg_link else None
    ref_audio, ref_ui = _trace_ref_audio(graph, pos_link) if pos_link else (None, None)
    loras = _trace_loras(graph, model_link) if model_link else []

    # Fallbacks when the trace can't anchor (no sampler/guider, or a missing node).
    if prompt is None or negative is None:
        fpos, fneg = _prompts(graph)
        prompt = prompt if prompt is not None else fpos
        negative = negative if negative is not None else fneg
    if ref_audio is None:
        ref_audio, ref_ui = _reference_audio(graph)
    if not loras:
        loras = _loras_fallback(graph)

    unet = _first(graph, "UNETLoader")
    ckpt = _first(graph, "CheckpointLoaderSimple")
    base_model = (unet["inputs"].get("unet_name") if unet
                  else ckpt["inputs"].get("ckpt_name") if ckpt else None)

    # dims: from the sampler's latent_image (so two-stage graphs pick the right one),
    # else the first EmptyLTXVLatentVideo. width/height that resolve to None here are
    # graph-unresolvable (e.g. LTXFramePlanner-linked); build_row owns recovering them
    # from the output video and the resulting warning (single warning owner).
    empty = None
    if samp and _is_link(graph[samp]["inputs"].get("latent_image")):
        empty = _trace_to_class(graph, graph[samp]["inputs"]["latent_image"], "EmptyLTXVLatentVideo")
    empty = empty or _first(graph, "EmptyLTXVLatentVideo")
    width = height = length = None
    if empty:
        ei = empty["inputs"]
        width = _resolve_scalar(graph, ei.get("width"))
        height = _resolve_scalar(graph, ei.get("height"))
        length = _resolve_scalar(graph, ei.get("length"))

    # fps from the SAMPLED conditioning, not an arbitrary LTXVConditioning.
    cond_on_path = _trace_to_class(graph, pos_link, "LTXVConditioning") if pos_link else None
    fps = _resolve_scalar(graph, cond_on_path["inputs"].get("frame_rate")) if cond_on_path else None
    if fps is None:
        c = _first(graph, "LTXVConditioning")
        fps = _resolve_scalar(graph, c["inputs"].get("frame_rate")) if c else None

    noise = _first(graph, "RandomNoise")
    seed = _resolve_scalar(graph, noise["inputs"].get("noise_seed")) if noise else None
    sampler_node = _first(graph, "KSamplerSelect")
    sampler = sampler_node["inputs"].get("sampler_name") if sampler_node else None
    cfg = _resolve_scalar(graph, cfg_v) if cfg_v is not None else None
    if cfg is None:
        guider = _first(graph, "CFGGuider")
        cfg = _resolve_scalar(graph, guider["inputs"].get("cfg")) if guider else None

    sig_node = _first(graph, "ManualSigmas")
    sigmas = sig_node["inputs"].get("sigmas") if sig_node else None
    steps = None
    if isinstance(sigmas, str):
        vals = [s for s in sigmas.split(",") if s.strip()]
        steps = max(len(vals) - 1, 0) if vals else None

    if prompt is None:
        warnings.append("positive prompt not found")
    if ref_audio is None:
        warnings.append("no reference audio (IC-LoRA guide absent/bypassed)")

    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "reference_audio_filename": ref_audio,
        "reference_audio_ui": ref_ui,
        "loras": loras,
        "base_model": base_model,
        "generation": {
            "width": width,
            "height": height,
            "length_frames": length,
            "fps": fps,
            "seed": seed,
            "sampler": sampler,
            "cfg": cfg,
            "steps": steps,
            "sigmas": sigmas,
        },
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Filesystem: discovery, audio resolution, PNG chunk
# --------------------------------------------------------------------------- #

@dataclass
class Render:
    id: str
    png: Path
    video_audio: Path | None
    video_silent: Path | None


def discover_renders(folder: Path) -> list[Render]:
    """Group a folder's files into renders keyed by PNG basename. A render needs
    a PNG (the metadata carrier); the `-audio.mp4` / `.mp4` siblings are optional."""
    folder = Path(folder)
    renders: list[Render] = []
    for png in sorted(folder.glob("*.png")):
        base = png.stem
        va = folder / f"{base}-audio.mp4"
        vs = folder / f"{base}.mp4"
        renders.append(Render(
            id=base,
            png=png,
            video_audio=va if va.exists() else None,
            video_silent=vs if vs.exists() else None,
        ))
    return renders


def read_png_prompt(png_path: Path) -> dict | None:
    """Extract + parse the API-format `prompt` tEXt chunk from a ComfyUI PNG."""
    try:
        with Image.open(png_path) as im:
            raw = im.info.get("prompt")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None


def _wav_meta(path: Path) -> tuple[float | None, int | None]:
    try:
        with wave.open(str(path), "r") as w:
            fr = w.getframerate()
            n = w.getnframes()
            return (n / fr if fr else None), fr
    except Exception:
        return None, None


def _ffprobe_meta(path: Path) -> tuple[float | None, int | None]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "a:0", str(path)],
            capture_output=True, timeout=20,
        )
        if out.returncode != 0:
            return None, None
        data = orjson.loads(out.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None, None
        s = streams[0]
        dur = s.get("duration")
        sr = s.get("sample_rate")
        return (float(dur) if dur else None), (int(sr) if sr else None)
    except Exception:
        return None, None


def _probe_video_dims(path: Path) -> dict:
    """Read width/height (and nb_frames when reported) from a video via ffprobe.
    Ground truth for dims that the graph computed at runtime (e.g. LTXFramePlanner)."""
    out: dict = {"width": None, "height": None, "nb_frames": None}
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(path)],
            capture_output=True, timeout=20,
        )
        if r.returncode != 0:
            return out
        streams = (orjson.loads(r.stdout).get("streams") or [])
        if not streams:
            return out
        s = streams[0]
        out["width"] = int(s["width"]) if s.get("width") else None
        out["height"] = int(s["height"]) if s.get("height") else None
        nbf = s.get("nb_frames")
        out["nb_frames"] = int(nbf) if nbf and str(nbf).isdigit() else None
    except Exception:
        pass
    return out


def _audio_hint_subfolder(audio_ui: str | None) -> str | None:
    """Pull the `subfolder=` value out of a LoadAudio.audioUI /api/view URL."""
    if not audio_ui or "subfolder=" not in audio_ui:
        return None
    try:
        q = urllib.parse.urlparse(audio_ui).query
        sub = urllib.parse.parse_qs(q).get("subfolder", [""])[0]
        return sub or None
    except Exception:
        return None


def resolve_audio(filename: str | None, search_roots: list[Path],
                  audio_ui: str | None = None, cache: dict | None = None) -> dict:
    """Locate the reference-audio file across search roots and probe its metadata.
    Always returns the filename; `found: false` when the file can't be located.

    Resolution order is exact-first: every root's direct `root[/sub]/filename` is
    tried (across ALL roots) before any recursive search, so an exact hit in a
    later root always beats a stray basename match buried in an earlier root.
    Only on a total miss do we fall back to a recursive basename search — with the
    name glob-escaped so metacharacters (`[ ] * ?`) match literally. `cache` (a
    dict keyed by (filename, audio_ui)) lets a batch resolve each distinct
    reference once instead of re-walking per render."""
    info = {"filename": filename, "path": None, "found": False,
            "duration_s": None, "sample_rate": None}
    if not filename:
        return info

    key = (filename, audio_ui)
    if cache is not None and key in cache:
        return dict(cache[key])

    sub = _audio_hint_subfolder(audio_ui)
    roots = [Path(r) for r in search_roots]
    name = Path(filename).name

    # 1) exact direct paths, across all roots (cheap, deterministic)
    hit = None
    for root in roots:
        direct = ([root / sub / filename] if sub else []) + [root / filename]
        hit = next((c for c in direct if c.is_file()), None)
        if hit:
            break

    # 2) last-ditch recursive basename search — escaped so [ ] * ? are literal
    if hit is None:
        pat = _glob.escape(name)
        for root in roots:
            hit = next((c for c in root.rglob(pat) if c.is_file()), None)
            if hit:
                break

    if hit is not None:
        info["found"] = True
        info["path"] = str(hit)
        dur, sr = (_wav_meta(hit) if hit.suffix.lower() == ".wav" else _ffprobe_meta(hit))
        info["duration_s"] = dur
        info["sample_rate"] = sr

    if cache is not None:
        cache[key] = dict(info)
    return info


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #

def _graph_sha256(graph: dict) -> str:
    """SHA-256 of the canonicalized workflow graph (sorted keys) — identifies the
    GRAPH, independent of node ordering or the PNG's pixels. (Hashing the PNG file
    instead would change on every re-render of the same graph and re-read MBs.)"""
    return hashlib.sha256(orjson.dumps(graph, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _symlink_media(src: Path, link_path: Path) -> str:
    """Create (or refresh) a symlink at `link_path` pointing at the ABSOLUTE source,
    and return the dataset-relative path string ("media/<name>"). Absolute target =
    the link survives moving the dataset folder, but the dataset is NOT self-contained
    (the source must stay put / be on the same machine). Use `--media copy` if you
    need a relocatable bundle."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(Path(src).resolve())
    return link_path.name


def build_row(render: Render, graph: dict, search_roots: list[Path],
              out_dir: Path, media_mode: str, audio_cache: dict | None = None) -> dict:
    meta = parse_prompt_graph(graph)
    audio = resolve_audio(meta["reference_audio_filename"], search_roots,
                          meta.get("reference_audio_ui"), cache=audio_cache)

    gen = meta["generation"]
    warnings = list(meta["warnings"])
    # build_row is the sole owner of the resolution warning: parse_prompt_graph leaves
    # graph-unresolvable dims as None (no warning), and here — the only place with the
    # rendered video — we recover them and report the outcome exactly once.
    if gen["width"] is None or gen["height"] is None:
        vid = render.video_audio or render.video_silent
        dims = _probe_video_dims(vid) if vid else {"width": None, "height": None, "nb_frames": None}
        gen["width"] = gen["width"] if gen["width"] is not None else dims["width"]
        gen["height"] = gen["height"] if gen["height"] is not None else dims["height"]
        if gen["length_frames"] is None and dims["nb_frames"]:
            gen["length_frames"] = dims["nb_frames"]
        if gen["width"] is not None and gen["height"] is not None:
            warnings.append("resolution probed from output video")
        else:
            warnings.append("resolution unresolved (not in graph; output video missing/unprobed)")

    outputs: dict[str, str | None] = {}
    media_dir = out_dir / "media"
    for key, src in (("thumbnail", render.png),
                     ("video_audio", render.video_audio),
                     ("video_silent", render.video_silent)):
        if not src:
            outputs[key] = None
        elif media_mode == "symlink":
            outputs[key] = f"media/{_symlink_media(src, media_dir / src.name)}"
        else:  # reference
            outputs[key] = str(src)

    if media_mode == "symlink" and audio["found"]:
        ap = Path(audio["path"])
        audio = {**audio, "path": f"media/{_symlink_media(ap, media_dir / ap.name)}"}

    return {
        "id": render.id,
        "prompt": meta["prompt"],
        "negative_prompt": meta["negative_prompt"],
        "reference_audio": {
            "filename": audio["filename"],
            "path": audio["path"],
            "found": audio["found"],
            "duration_s": audio["duration_s"],
            "sample_rate": audio["sample_rate"],
        },
        "loras": meta["loras"],
        "base_model": meta["base_model"],
        "generation": gen,
        "outputs": outputs,
        "warnings": warnings,
        "provenance": {
            "source_png": render.png.name,
            "workflow_sha256": _graph_sha256(graph),
        },
    }


_DATASET_CARD = """# Multimodal dataset — audio reference -> LTX 2.3 video

Last updated: {date}

Built by `scripts/build_multimodal_dataset.py` from a folder of ComfyUI renders.
One JSONL row per render in `dataset.jsonl`. Media mode: **{media_mode}**. In
`symlink` mode `media/` holds symlinks whose targets are ABSOLUTE — the dataset
survives being moved but is not self-contained (sources must stay put). In
`reference` mode rows carry absolute source paths and no `media/` is written.

Load directly:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="dataset.jsonl")
```

## Row schema

| field | type | meaning |
|---|---|---|
| `id` | str | render basename |
| `prompt` | str\\|null | positive prompt (CLIPTextEncode via LTXVConditioning) |
| `negative_prompt` | str\\|null | negative prompt |
| `reference_audio.filename` | str\\|null | LoadAudio filename fed to the IC-LoRA guide |
| `reference_audio.path` | str\\|null | `media/..` (symlink mode) or absolute source (reference mode); null if not found |
| `reference_audio.found` | bool | whether the source audio file was located |
| `reference_audio.duration_s` | float\\|null | reference duration (wave/ffprobe) |
| `reference_audio.sample_rate` | int\\|null | reference sample rate |
| `loras` | list | LoRAs applied along the model chain (base-first): `[{{name, strength}}]`; empty-name placeholders excluded |
| `base_model` | str | UNETLoader unet_name (or checkpoint) |
| `generation.{{width,height,length_frames,fps,seed,sampler,cfg,steps,sigmas}}` | mixed | render params; width/height fall back to probing the output video when graph-unresolvable |
| `outputs.{{thumbnail,video_audio,video_silent}}` | str\\|null | media paths |
| `warnings` | list[str] | per-row notes (e.g. resolution probed from video, prompt not found, no reference audio) |
| `provenance.source_png` | str | source PNG basename |
| `provenance.workflow_sha256` | str | SHA-256 of the canonical graph (sorted keys) — identifies the GRAPH, not the pixels |

Values are traced backward from the terminal sampler (the conditioning / model /
latent that actually produced the render), so multi-conditioning, multi-LoadAudio,
and stacked-LoRA graphs resolve to the SAMPLED node rather than an arbitrary one.

`reference_audio.found == false` means the workflow recorded the filename but the
file wasn't on any `--audio-root`. The row is kept — re-point `--audio-root` and
rebuild to fill it in.

## Build summary

- renders discovered: {n_total}
- rows written: {n_rows}
- reference audio located: {n_found} / {n_rows}
- rows with warnings: {n_warn} / {n_rows}
"""


def build_dataset(renders_dir: Path, out_dir: Path, search_roots: list[Path],
                  media_mode: str, date: str = "") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear prior media symlinks so a re-run can't leave ghosts/dangling links for
    # renders that have since been removed or renamed (#8). JSONL is rewritten below.
    if media_mode == "symlink":
        shutil.rmtree(out_dir / "media", ignore_errors=True)
    renders = discover_renders(renders_dir)

    rows: list[dict] = []
    skipped: list[str] = []
    audio_cache: dict = {}
    for r in renders:
        graph = read_png_prompt(r.png)
        if graph is None:
            skipped.append(r.id)
            continue
        rows.append(build_row(r, graph, search_roots, out_dir, media_mode, audio_cache))

    jsonl = out_dir / "dataset.jsonl"
    with open(jsonl, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")

    n_found = sum(1 for r in rows if r["reference_audio"]["found"])
    n_warn = sum(1 for r in rows if r["warnings"])
    (out_dir / "dataset_card.md").write_text(_DATASET_CARD.format(
        date=date or "(undated)", media_mode=media_mode,
        n_total=len(renders), n_rows=len(rows), n_found=n_found, n_warn=n_warn,
    ))

    return {"renders": len(renders), "rows": len(rows), "audio_found": n_found,
            "warnings": n_warn, "skipped": skipped, "jsonl": str(jsonl)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _default_audio_roots(renders_dir: Path) -> list[Path]:
    """Best-effort defaults: ComfyUI input dir (walk up for an `input/` sibling of
    a `custom_nodes` tree) + the renders dir itself."""
    roots: list[Path] = [Path(renders_dir)]
    for parent in Path(renders_dir).resolve().parents:
        cand = parent / "input"
        if cand.is_dir():
            roots.append(cand)
            break
    return roots


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a multimodal dataset from a folder of ComfyUI renders.")
    ap.add_argument("renders_dir", type=Path, help="folder of <base>.png / .mp4 / -audio.mp4 renders")
    ap.add_argument("-o", "--out", type=Path, default=Path("dataset"), help="output dataset dir")
    ap.add_argument("--audio-root", type=Path, action="append", default=[],
                    help="root(s) to search for reference-audio files (repeatable)")
    ap.add_argument("--media", choices=("symlink", "reference"), default="symlink",
                    help="symlink media into dataset/media (default) or reference source paths in place")
    ap.add_argument("--date", default="", help="last-updated date for the dataset card (YYYY-MM-DD)")
    args = ap.parse_args(argv)

    if not args.renders_dir.is_dir():
        print(f"error: {args.renders_dir} is not a directory", file=sys.stderr)
        return 2

    roots = list(args.audio_root) or _default_audio_roots(args.renders_dir)
    summary = build_dataset(args.renders_dir, args.out, roots, args.media, args.date)

    print(f"renders discovered : {summary['renders']}")
    print(f"rows written       : {summary['rows']}  -> {summary['jsonl']}")
    print(f"reference audio hit : {summary['audio_found']}/{summary['rows']}")
    print(f"rows with warnings : {summary['warnings']}/{summary['rows']}")
    if summary["skipped"]:
        shown = summary["skipped"][:5]
        more = "" if len(summary["skipped"]) <= 5 else f", +{len(summary['skipped']) - 5} more"
        print(f"skipped (no embedded prompt): {len(summary['skipped'])} ({', '.join(shown)}{more})")
    print(f"audio search roots : {', '.join(str(r) for r in roots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
