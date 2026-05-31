Last updated: 2026-05-31

# Building a multimodal dataset from a folder of renders

`scripts/build_multimodal_dataset.py` turns a folder of ComfyUI renders into a
schema'd multimodal dataset — one JSONL row per render — without re-rendering
anything. ComfyUI embeds the executed graph in each PNG's `prompt` tEXt chunk;
the script reads it, traces the graph, and flattens the bits that matter (prompt,
reference audio, LoRA + strength, generation params) alongside the output media.

Primary use case: turning an audio-reference IC-LoRA sweep (many renders across
strengths / references / prompts) into queryable data — see
[`../audio_iclora/audio_only_ic_lora.md`](../audio_iclora/audio_only_ic_lora.md).

## Run it

```bash
uv run --group analysis python scripts/build_multimodal_dataset.py <renders_dir> \
    -o <out_dir> \
    --audio-root <comfy_input> \
    --audio-root <audio_data_dir>
```

- `<renders_dir>` — a folder of `<base>.png` / `<base>.mp4` / `<base>-audio.mp4`
  triplets (what `VHS_VideoCombine` writes). The PNG is the metadata carrier; the
  mp4 siblings are optional.
- `--audio-root` (repeatable) — where to look for the reference-audio files named
  in each workflow. Defaults to the renders dir plus an auto-detected ComfyUI
  `input/`.
- `--media symlink` (default) — `<out_dir>/media/` holds symlinks (absolute
  targets) to the source media; row paths are dataset-relative `media/..`, so the
  dataset survives being moved but is not self-contained (sources must stay put).
  `--media reference` writes absolute source paths into the JSONL instead — fast,
  but **do not share that JSONL**, it carries absolute paths.

Output: `<out_dir>/dataset.jsonl` + `<out_dir>/dataset_card.md` (the full schema) +
`<out_dir>/media/`. Load it directly:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="<out_dir>/dataset.jsonl")
```

## How values are resolved (and why it's trustworthy)

The extractor traces backward from the **terminal sampler** (the one with no
sampler downstream — so two-stage / upscale graphs pick the final pass), through
its guider, to the conditioning / model / latent that actually produced the
render. This is why multi-conditioning, multi-`LoadAudio`, and stacked-LoRA graphs
resolve to the *sampled* node rather than an arbitrary one:

- **prompt / negative** — followed through the guider's positive/negative links
  (slot-aware), so the loop/init prompt actually sampled is the one recorded, not
  whichever `CLIPTextEncode` happens to come first in the graph.
- **reference audio** — the `LoadAudio` feeding the IC-LoRA guide on the positive
  path. If no guide is present (bypassed), `reference_audio.found` reflects that
  and the filename is `null` — it does **not** fall back to an unrelated `LoadAudio`
  (e.g. the full song).
- **loras** — every LoRA applied along the model chain (base-first), empty-name
  placeholders excluded.
- **dimensions** — read from the sampler's `EmptyLTXVLatentVideo`; when width/height
  are wired from a node the graph can't resolve statically (e.g. `LTXFramePlanner`
  outputs), they're recovered by probing the output video. A per-row `warnings`
  list flags anything left unresolved.

Reference-audio files that can't be located on any `--audio-root` are recorded with
`found: false` (filename preserved) rather than dropped — re-point `--audio-root`
and rebuild to fill them in. ComfyUI display-name vs on-disk-name mismatches (e.g.
an uploaded `"My Song (Final).mp3"` saved on disk under a slug) surface as
`found: false`; that's honest, not a miss.

The full per-row schema is documented in the generated `dataset_card.md`.
