"""extract_workflow_from_png.

Last updated: 2026-04-26

Dump the embedded ComfyUI workflow JSON from a PNG (or batch of PNGs).

ComfyUI saves two tEXt chunks on its preview/output PNGs:
  - `workflow`  - UI-format graph (nodes/links/widgets), what we use for diffing.
  - `prompt`    - API-format flattened graph (what the executor actually ran).

By default we emit `workflow`. Use `--prompt` to emit the API-format payload
(useful when the workflow chunk is missing on a renamed/transformed PNG and the
executor-format graph still survived).

Usage:
    uv run --group analysis python scripts/extract_workflow_from_png.py <png>
    uv run --group analysis python scripts/extract_workflow_from_png.py <png> -o out.json
    uv run --group analysis python scripts/extract_workflow_from_png.py <png1> <png2> ... -d out_dir/
    uv run --group analysis python scripts/extract_workflow_from_png.py <png> --prompt
    uv run --group analysis python scripts/extract_workflow_from_png.py <png> --stdout

When invoked on multiple PNGs without `-d`, files land alongside each PNG as
`<basename>.workflow.json` (or `.prompt.json` with `--prompt`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson
from PIL import Image


def _extract_chunk(png_path: Path, key: str) -> bytes:
    with Image.open(png_path) as im:
        raw = im.info.get(key)
    if raw is None:
        available = sorted(Image.open(png_path).info.keys())
        raise SystemExit(
            f"{png_path}: no '{key}' tEXt chunk found. Available keys: {available}"
        )
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return raw


def _pretty(payload: bytes) -> bytes:
    obj = orjson.loads(payload)
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2)


def _emit(png_path: Path, out_path: Path | None, key: str, stdout: bool, pretty: bool) -> None:
    payload = _extract_chunk(png_path, key)
    if pretty:
        payload = _pretty(payload)
    if stdout:
        sys.stdout.buffer.write(payload)
        if not payload.endswith(b"\n"):
            sys.stdout.buffer.write(b"\n")
        return
    if out_path is None:
        out_path = png_path.with_suffix(f".{key}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    print(f"{png_path.name} -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("pngs", nargs="+", type=Path, help="PNG file(s) to extract from")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output file path (single-PNG mode only)")
    ap.add_argument("-d", "--out-dir", type=Path, default=None,
                    help="Output directory (multi-PNG mode); filenames mirror inputs")
    ap.add_argument("--prompt", action="store_true",
                    help="Extract API-format 'prompt' chunk instead of 'workflow'")
    ap.add_argument("--stdout", action="store_true",
                    help="Write to stdout instead of a file (single-PNG mode only)")
    ap.add_argument("--raw", action="store_true",
                    help="Emit raw chunk bytes verbatim (skip pretty-print round-trip)")
    args = ap.parse_args(argv)

    key = "prompt" if args.prompt else "workflow"
    pretty = not args.raw

    if args.stdout and len(args.pngs) != 1:
        ap.error("--stdout requires exactly one PNG")
    if args.out is not None and len(args.pngs) != 1:
        ap.error("--out requires exactly one PNG; use --out-dir for batches")

    for png in args.pngs:
        if not png.is_file():
            print(f"skip: {png} (not a file)", file=sys.stderr)
            continue
        if args.stdout:
            _emit(png, None, key, stdout=True, pretty=pretty)
        elif args.out is not None:
            _emit(png, args.out, key, stdout=False, pretty=pretty)
        elif args.out_dir is not None:
            out = args.out_dir / f"{png.stem}.{key}.json"
            _emit(png, out, key, stdout=False, pretty=pretty)
        else:
            _emit(png, None, key, stdout=False, pretty=pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
