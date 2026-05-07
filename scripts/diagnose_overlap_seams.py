"""Diagnose seam-zone artifacts in an assembled loop output latent.

Last updated: 2026-05-04

Phase A of the seam-zone refinement design (`internal/design/polish_passes_design.md §P5`).

Loads a video file (MP4) or a saved latent tensor and runs a per-frame
ghost-residual scan: `|f[t] - (f[t-1] + f[t+1]) / 2|`. Frames that
look like a linear blend of their neighbors score LOW (because the
prediction matches them); frames with independent content score HIGH.
For the seam-zone investigation we invert this: a LOW residual at a
position that *should* have independent content suggests blending
artifact.

Reports:
  1. Top-K frames by ghost score (1 - normalized residual).
  2. For each iteration boundary derivable from `--iteration-count`,
     `--window-latents`, and `--overlap-latents`, the ghost score in
     a band of `--seam-band-latents` around the boundary.
  3. A noise-floor baseline (median ghost score over all frames) so
     the boundary-zone scores can be read against it.

This is the gating evidence for Phase B (the `LatentSeamZoneMask`
node + corrective pass workflow). If real renders show boundary-zone
scores well above the noise floor, build Phase B. If they don't, P5
stays parked.

Usage:

    uv run --group dev python scripts/diagnose_overlap_seams.py \\
        --latent /tmp/loop.latent.pt \\
        --iteration-count 5 --window-latents 16 --overlap-latents 4

To capture a latent for the diagnostic, stage the canonical loop
workflow with `scripts/apply_save_video_latent.py` (inserts a
`SaveLatent` node wired to `LTXVSeparateAVLatent.video_latent`),
render once, then point `--latent` at the resulting
`output/seam_diag/loop_video_latent_NNNNN_.latent` file.

Accepted formats: `.pt` (dict with `samples` or bare Tensor),
`.safetensors` / `.latent` (`samples`/`latent`/`video_latent`/
`latent_tensor` keys). Cheap to run on CPU (~seconds per minute of video).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
LTX_TEMPORAL_SCALE = 8


def _load_latent(latent_path: Path) -> torch.Tensor:
    """Load a saved latent tensor. Accepts .pt or .safetensors.

    Expects shape `[B, C, F, H, W]` (LTX video latent). Returns the
    tensor as-is; further squeezing is the caller's job.
    """
    if not latent_path.exists():
        raise SystemExit(f"Latent file not found: {latent_path}")
    if latent_path.suffix == ".pt":
        obj = torch.load(latent_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "samples" in obj:
            return obj["samples"]
        if isinstance(obj, torch.Tensor):
            return obj
        raise SystemExit(
            f"Unexpected .pt content; expected dict with 'samples' or a Tensor, got {type(obj)}."
        )
    if latent_path.suffix in (".safetensors", ".latent"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise SystemExit("safetensors not installed; install or use a .pt file.")
        obj = load_file(latent_path)
        # `latent_tensor` is the key ComfyUI core's SaveLatent node writes;
        # `samples` / `latent` / `video_latent` cover hand-saved variants.
        for key in ("samples", "latent", "video_latent", "latent_tensor"):
            if key in obj:
                return obj[key]
        raise SystemExit(
            f"No samples/latent/video_latent/latent_tensor key in {latent_path.suffix}; "
            f"saw {list(obj.keys())}."
        )
    raise SystemExit(f"Unrecognized latent extension: {latent_path.suffix}")


def _ghost_residual(samples: torch.Tensor) -> torch.Tensor:
    """Per-frame ghost residual `|f[t] - (f[t-1] + f[t+1]) / 2|`.

    Returns a tensor of shape `[B, F]` (mean across channels + spatial
    dims). Edge frames replicate to avoid boundary distortion of the
    residual at frame 0 and frame F-1.
    """
    if samples.dim() != 5:
        raise SystemExit(
            f"Expected 5D latent [B, C, F, H, W], got {tuple(samples.shape)}."
        )
    prev = torch.cat([samples[:, :, :1], samples[:, :, :-1]], dim=2)
    nxt = torch.cat([samples[:, :, 1:], samples[:, :, -1:]], dim=2)
    pred = (prev + nxt) * 0.5
    residual = (samples - pred).abs().mean(dim=(1, 3, 4))  # [B, F]
    return residual


def _ghost_score(residual: torch.Tensor) -> torch.Tensor:
    """Normalize and invert: HIGH score = ghost-like (low residual).

    Per-batch min-max normalization. Range [0, 1] after the inversion.
    """
    r_min = residual.amin(dim=1, keepdim=True)
    r_max = residual.amax(dim=1, keepdim=True)
    spread = (r_max - r_min).clamp(min=1e-8)
    r_norm = (residual - r_min) / spread
    return (1.0 - r_norm).clamp(0.0, 1.0)


def _internal_seam_latents(
    iteration_count: int, window_latents: int, overlap_latents: int
) -> list[int]:
    """Latent-frame indices of internal iteration boundaries.

    For N iterations, there are N-1 internal seams at multiples of the
    stride (window - overlap). Matches AudioLoopController's emitted
    integer-latent counts.
    """
    stride = window_latents - overlap_latents
    if stride <= 0:
        raise SystemExit(
            f"window_latents ({window_latents}) must exceed overlap_latents ({overlap_latents})."
        )
    return [stride * i for i in range(1, iteration_count)]


def diagnose(
    samples: torch.Tensor,
    iteration_count: int,
    window_latents: int,
    overlap_latents: int,
    seam_band_latents: int,
    top_k: int,
) -> dict:
    """Run the residual scan and assemble a report.

    Returns a dict for programmatic use; `main()` formats for stdout.
    """
    residual = _ghost_residual(samples)  # [B, F]
    ghost = _ghost_score(residual)  # [B, F]
    # Use batch 0 for display; batch dim is always 1 in shipped workflows.
    g = ghost[0]

    F = g.shape[0]
    seams = _internal_seam_latents(iteration_count, window_latents, overlap_latents)
    half_band = max(1, seam_band_latents // 2)

    # Top-K ghost frames overall
    _, top_idx = g.topk(min(top_k, F))
    top_idx_sorted = top_idx.sort().values
    top_table = [(int(i), float(g[i])) for i in top_idx_sorted.tolist()]

    # Per-seam band scores
    seam_table: list[dict] = []
    for s in seams:
        lo = max(0, s - half_band)
        hi = min(F, s + half_band)
        if hi <= lo:
            continue
        band = g[lo:hi]
        seam_table.append(
            {
                "boundary_latent": s,
                "band_lo": lo,
                "band_hi": hi,
                "max_score": float(band.max()),
                "mean_score": float(band.mean()),
                "argmax_offset_from_seam": int(band.argmax().item() + lo - s),
            }
        )

    noise_floor = float(g.median())

    return {
        "total_frames": int(F),
        "noise_floor_median": noise_floor,
        "top_k_frames": top_table,
        "seam_bands": seam_table,
        "seams_total": len(seams),
        "seam_band_latents": seam_band_latents,
    }


def _format_report(report: dict, samples_shape: tuple[int, ...]) -> str:
    lines: list[str] = []
    lines.append(f"Latent shape: {samples_shape}")
    lines.append(f"Total latent frames: {report['total_frames']}")
    lines.append(
        f"Noise floor (median ghost score): {report['noise_floor_median']:.4f}"
    )
    lines.append("")
    lines.append(f"Top {len(report['top_k_frames'])} frames by ghost score:")
    lines.append(f"  {'frame':>6}  {'score':>7}")
    for i, score in report["top_k_frames"]:
        lines.append(f"  {i:>6d}  {score:>7.4f}")
    lines.append("")
    lines.append(
        f"Per-seam band ({report['seams_total']} internal boundaries, "
        f"band ±{report['seam_band_latents'] // 2} latents):"
    )
    if not report["seam_bands"]:
        lines.append("  (no internal boundaries — single iteration)")
    else:
        lines.append(
            f"  {'seam@':>6}  {'band':>10}  {'max':>7}  {'mean':>7}  {'argmax_off':>10}"
        )
        for row in report["seam_bands"]:
            lines.append(
                f"  {row['boundary_latent']:>6d}  "
                f"[{row['band_lo']:>3d},{row['band_hi']:>3d})  "
                f"{row['max_score']:>7.4f}  "
                f"{row['mean_score']:>7.4f}  "
                f"{row['argmax_offset_from_seam']:>+10d}"
            )
    lines.append("")
    lines.append("Verdict guidance:")
    lines.append(
        "  - If max seam-band score > 1.5x the noise floor, real artifact likely."
    )
    lines.append(
        "  - argmax_offset_from_seam near 0 = artifact aligns with the seam (high signal)."
    )
    lines.append(
        "  - argmax far from 0 = high-ghost frame elsewhere; likely content, not seam."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--latent", type=Path, required=True,
                    help="Saved latent tensor (.pt or .safetensors). To capture one from a "
                         "render, hook the workflow to torch.save the dict at the "
                         "LTXVSeparateAVLatent.video_latent output.")

    ap.add_argument("--iteration-count", type=int, required=True,
                    help="Number of loop iterations stitched.")
    ap.add_argument("--window-latents", type=int, required=True,
                    help="Latents per window (from AudioLoopController.window_latents output).")
    ap.add_argument("--overlap-latents", type=int, required=True,
                    help="Overlap latents per window (from AudioLoopController.overlap_latents output).")
    ap.add_argument("--seam-band-latents", type=int, default=4,
                    help="Width (in latents) of the band centered on each seam (default: 4).")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of top-ghost frames to print (default: 20).")
    args = ap.parse_args()

    samples = _load_latent(args.latent)
    if samples.dim() != 5:
        if samples.dim() == 4:
            samples = samples.unsqueeze(0)
        else:
            raise SystemExit(f"Latent shape {tuple(samples.shape)} not handled (need 5D).")

    report = diagnose(
        samples,
        iteration_count=args.iteration_count,
        window_latents=args.window_latents,
        overlap_latents=args.overlap_latents,
        seam_band_latents=args.seam_band_latents,
        top_k=args.top_k,
    )
    print(_format_report(report, tuple(samples.shape)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
