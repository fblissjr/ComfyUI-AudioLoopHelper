---
name: comfyui-test
description: Run the AudioLoopHelper test suite with correct uv groups, rootdir, and ComfyUI loader verification
---

Last updated: 2026-04-30

Run the full test suite for ComfyUI-AudioLoopHelper. This requires the project's own venv (not ComfyUI's global venv) with both `dev` and `analysis` dependency groups.

## Steps

1. Run all tests from the project root:

```bash
cd "$(git rev-parse --show-toplevel)" && uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.
```

2. Verify ComfyUI can load the extension (catches import chain failures). This
   requires being one directory above this repo (`<comfyui_root>/custom_nodes/`
   is where ComfyUI looks). If `${COMFYUI_ROOT}` is set, use it; otherwise
   ask the user where their ComfyUI install lives:

```bash
cd "${COMFYUI_ROOT:?set COMFYUI_ROOT or run from your ComfyUI install dir}" && python -c "
import asyncio, nodes as n
r = asyncio.run(n.load_custom_node('custom_nodes/ComfyUI-AudioLoopHelper', module_parent='custom_nodes'))
print('ComfyUI loader: OK' if r else 'COMFYUI LOADER FAILED')
exit(0 if r else 1)
"
```

## What the tests cover

- **test_audio_features.py** (61 tests): Offline librosa extraction -- BPM, key, chromagram, mel spectrogram, F0, structure, schedule generation
- **test_audio_analysis_nodes.py** (9 tests): Runtime torchaudio nodes -- _slice_audio_window, AudioPitchDetect, vocal fraction, male/female classification
- **test_workflows.py** (1 test): Workflow JSON structural validation for both IMAGE and LATENT variants

## Common failures

- `ModuleNotFoundError: torch` -- forgot `--group dev`
- `ImportError: analyze_audio_features` -- forgot `--rootdir=.` or not in project dir
- `Group 'dev' is not defined` -- running from ComfyUI root instead of project dir
- `comfy_entrypoint not found` -- __init__.py guard swallowed a real error
