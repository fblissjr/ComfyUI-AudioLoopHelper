# scripts/startup/

Last updated: 2026-05-13

Holds the canonical **deploy template** for ComfyUI's launcher.

## Contents

- `start.sh` — copy to `<comfyui_root>/start.sh` on a fresh deploy.
  Defines six modes (`default`, `safe`, `extreme`, `minimal`, `nodynvram`,
  `highvram`) and forwards trailing flags to `main.py`. Customize via
  env vars (`COMFYUI_OUTPUT_DIR`, `COMFYUI_INPUT_DIR`, `COMFYUI_TEMP_DIR`,
  `COMFYUI_PORT`) without editing the script. Defaults resolve to
  ComfyUI's standard relative layout (`./output`, `./input`, `./temp`).

## Asymmetry — where's the experiment wrapper?

The matching telemetry wrapper (`start_experiment.sh`) lives at this
repo's root, not here. The asymmetry is intentional:

- `start.sh` lives **outside** this repo (in the user's ComfyUI install
  at `<comfyui_root>/start.sh`), so we ship a template to be copied out
  at deploy time.
- `start_experiment.sh` lives **inside** this repo at the plugin root,
  so it's already at its canonical production location — no template
  needed; a fresh clone gets it.

## Deploy recipe

```bash
# On a fresh ComfyUI install or new machine:
cp scripts/startup/start.sh <comfyui_root>/start.sh
chmod +x <comfyui_root>/start.sh

# Optional: set custom paths if you don't use ComfyUI's relative layout
# (e.g. in your shell rc file or at invocation time):
export COMFYUI_OUTPUT_DIR=/path/to/your/output
export COMFYUI_INPUT_DIR=/path/to/your/input

# Then either launch ComfyUI directly:
bash <comfyui_root>/start.sh default

# Or via the experiment wrapper (telemetry + sage trace):
bash <plugin_root>/start_experiment.sh default
```

## Bench methodology

For the `nodynvram` mode and the full kernel-OOM testing recipe, see
`docs/reference/benchmarking_memory_pressure.md`.
