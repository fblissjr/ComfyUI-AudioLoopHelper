"""Tests for scripts/build_multimodal_dataset.py.

The dataset builder reads the embedded API-format `prompt` graph out of each
render's PNG and flattens it into a schema'd JSONL row (prompt, reference
audio, lora+strength, generation params) alongside symlinked media. The
graph-parsing logic is the load-bearing, pure, testable core — it must follow
node links (CLIPTextEncode via LTXVConditioning, reference audio via
LTXAddAudioICLoRAGuide) and resolve constant-node links (INTConstant length).

Fixtures mirror the REAL embedded-prompt key names captured from a pitch_gate
render (ManualSigmas.sigmas, RandomNoise.noise_seed, LTXVConditioning.frame_rate,
LTXAudioICLoRALoader.strength_model, etc.) so the parser is tested against the
shape it actually meets in the wild, not an invented one.
"""

from __future__ import annotations

import wave
from pathlib import Path

import build_multimodal_dataset as bmd


# A minimal API-format prompt graph using the real class_type + input key names.
# `length` is a link to an INTConstant to exercise constant-link resolution.
GRAPH = {
    "1570": {"class_type": "UNETLoader",
             "inputs": {"unet_name": "ltx-2.3-distilled.safetensors", "weight_dtype": "default"}},
    "1842": {"class_type": "RandomNoise", "inputs": {"noise_seed": 43}},
    "1857": {"class_type": "ManualSigmas",
             "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
    "1853": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "1621": {"class_type": "CLIPTextEncode",
             "inputs": {"text": 'A person says, "Hello"', "clip": ["1562", 0]}},
    "1626": {"class_type": "CLIPTextEncode",
             "inputs": {"text": "blurry, deformed, low quality, watermark", "clip": ["1562", 0]}},
    "164": {"class_type": "LTXVConditioning",
            "inputs": {"frame_rate": 25.0, "positive": ["1621", 0], "negative": ["1626", 0]}},
    "1975": {"class_type": "LoadAudio",
             "inputs": {"audio": "pitch_gate_tone.wav",
                        "audioUI": "/api/view?filename=pitch_gate_tone.wav&type=input&subfolder=&rand=0.5"}},
    "1976": {"class_type": "LTXAddAudioICLoRAGuide",
             "inputs": {"positive": ["164", 0], "negative": ["164", 1],
                        "audio_vae": ["1567", 0], "reference_audio": ["1975", 0]}},
    "1977": {"class_type": "LTXAudioICLoRALoader",
             "inputs": {"lora_name": "pitch_gate_audio_ref_step02000.safetensors",
                        "strength_model": 0.5, "model": ["1570", 0]}},
    "1856": {"class_type": "CFGGuider",
             "inputs": {"cfg": 1.0, "model": ["1977", 0], "positive": ["1976", 0], "negative": ["1976", 1]}},
    "27": {"class_type": "INTConstant", "inputs": {"value": 121}},
    "344": {"class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 960, "height": 544, "length": ["27", 0], "batch_size": 1}},
    "1936": {"class_type": "LTXVEmptyLatentAudio",
             "inputs": {"frames_number": ["27", 0], "frame_rate": 25, "batch_size": 1, "audio_vae": ["1567", 0]}},
}


# A realistic multi-node graph that encodes the disambiguation traps the
# dict-order heuristic gets WRONG. The correct values are only reachable by
# tracing backward from the terminal sampler:
#   - an INIT prompt + a SAMPLED (loop) prompt; only the sampled one feeds the guider
#   - a full-song LoadAudio (earlier in dict order) + the real reference on the guide
#   - a dead empty-name LoRA off the model chain + the real LoRA on it
#   - width/height linked to LTXFramePlanner (not a constant) -> unresolvable from graph
# Off-path / decoy nodes are placed EARLIER in insertion order on purpose.
SAMPLED_GRAPH = {
    "song": {"class_type": "LoadAudio", "inputs": {"audio": "full_song.mp3"}},  # decoy, first
    "p_init": {"class_type": "CLIPTextEncode", "inputs": {"text": "INIT PROMPT off path", "clip": ["clip", 0]}},
    "cond_init": {"class_type": "LTXVConditioning",
                  "inputs": {"frame_rate": 24.0, "positive": ["p_init", 0], "negative": ["neg", 0]}},  # decoy
    "dead_lora": {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"lora_name": "", "strength_model": 1.0, "model": ["unet", 0]}},  # decoy off-chain
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltx.safetensors", "weight_dtype": "default"}},
    "real_lora": {"class_type": "LTXAudioICLoRALoader",
                  "inputs": {"lora_name": "pitch.safetensors", "strength_model": 0.5, "model": ["unet", 0]}},
    "p_loop": {"class_type": "CLIPTextEncode", "inputs": {"text": "SAMPLED PROMPT", "clip": ["clip", 0]}},
    "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, deformed, low quality, watermark", "clip": ["clip", 0]}},
    "cond_loop": {"class_type": "LTXVConditioning",
                  "inputs": {"frame_rate": 25.0, "positive": ["p_loop", 0], "negative": ["neg", 0]}},
    "ref": {"class_type": "LoadAudio", "inputs": {"audio": "reference_tone.wav"}},
    "guide": {"class_type": "LTXAddAudioICLoRAGuide",
              "inputs": {"positive": ["cond_loop", 0], "negative": ["cond_loop", 1],
                         "audio_vae": ["avae", 0], "reference_audio": ["ref", 0]}},
    "g": {"class_type": "CFGGuider",
          "inputs": {"cfg": 1.0, "model": ["real_lora", 0], "positive": ["guide", 0], "negative": ["guide", 1]}},
    "ks": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "ms": {"class_type": "ManualSigmas",
           "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
    "rn": {"class_type": "RandomNoise", "inputs": {"noise_seed": 7}},
    "fp": {"class_type": "LTXFramePlanner", "inputs": {"aspect": "16:9", "long_edge": 960}},
    "len": {"class_type": "INTConstant", "inputs": {"value": 121}},
    "ev": {"class_type": "EmptyLTXVLatentVideo",
           "inputs": {"width": ["fp", 0], "height": ["fp", 1], "length": ["len", 0], "batch_size": 1}},
    "ea": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": ["len", 0], "frame_rate": 25, "batch_size": 1}},
    "cat": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["ev", 0], "audio_latent": ["ea", 0]}},
    "samp": {"class_type": "SamplerCustomAdvanced",
             "inputs": {"noise": ["rn", 0], "guider": ["g", 0], "sampler": ["ks", 0],
                        "sigmas": ["ms", 0], "latent_image": ["cat", 0]}},
}


class TestBackwardTrace:
    def test_prompt_is_the_sampled_conditioning_not_an_off_path_one(self):
        # #11: traces guider.positive -> guide -> cond_loop, not the earlier cond_init.
        assert bmd.parse_prompt_graph(SAMPLED_GRAPH)["prompt"] == "SAMPLED PROMPT"

    def test_reference_audio_is_on_guide_path_not_first_loadaudio(self):
        # #10: the decoy 'song' LoadAudio is first in dict order; the guide's ref must win.
        assert bmd.parse_prompt_graph(SAMPLED_GRAPH)["reference_audio_filename"] == "reference_tone.wav"

    def test_loras_are_the_applied_chain_excluding_empty(self):
        m = bmd.parse_prompt_graph(SAMPLED_GRAPH)
        assert m["loras"] == [{"name": "pitch.safetensors", "strength": 0.5}]

    def test_fps_comes_from_the_sampled_conditioning(self):
        # cond_loop=25.0 is on the path; cond_init=24.0 is the decoy.
        assert bmd.parse_prompt_graph(SAMPLED_GRAPH)["generation"]["fps"] == 25.0

    def test_scalar_params_from_sampler_chain(self):
        g = bmd.parse_prompt_graph(SAMPLED_GRAPH)["generation"]
        assert g["seed"] == 7
        assert g["cfg"] == 1.0
        assert g["steps"] == 8
        assert g["length_frames"] == 121

    def test_planner_linked_dims_are_null_in_pure_parse(self):
        # #9: width/height link to LTXFramePlanner (not a constant) -> unresolvable from
        # the graph alone. parse_prompt_graph reports null (the fact); the resolution
        # WARNING is owned by build_row (the only layer that can probe the video).
        m = bmd.parse_prompt_graph(SAMPLED_GRAPH)
        assert m["generation"]["width"] is None
        assert m["generation"]["height"] is None
        assert not any("resolution" in w.lower() for w in m["warnings"])


class TestProbeVideoDims:
    def test_probe_reads_dims_from_video(self, tmp_path):
        import shutil as _sh
        if _sh.which("ffmpeg") is None or _sh.which("ffprobe") is None:
            import pytest
            pytest.skip("ffmpeg/ffprobe not available")
        import subprocess
        mp4 = tmp_path / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x96:d=0.2:r=25",
             "-pix_fmt", "yuv420p", str(mp4)],
            capture_output=True, check=True,
        )
        dims = bmd._probe_video_dims(mp4)
        assert dims["width"] == 160
        assert dims["height"] == 96


def _wav(path, sample_rate, seconds):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def _png_with_prompt(path, graph):
    import orjson
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    meta = PngInfo()
    meta.add_text("prompt", orjson.dumps(graph).decode())
    Image.new("RGB", (8, 8)).save(str(path), pnginfo=meta)


class TestParsePromptGraph:
    def test_positive_and_negative_via_ltxvconditioning(self):
        m = bmd.parse_prompt_graph(GRAPH)
        assert m["prompt"] == 'A person says, "Hello"'
        assert m["negative_prompt"].startswith("blurry")

    def test_reference_audio_followed_through_guide(self):
        m = bmd.parse_prompt_graph(GRAPH)
        assert m["reference_audio_filename"] == "pitch_gate_tone.wav"
        # the audioUI hint is captured for downstream path resolution
        assert "type=input" in (m["reference_audio_ui"] or "")

    def test_loras_list_name_and_strength(self):
        m = bmd.parse_prompt_graph(GRAPH)
        assert m["loras"] == [{"name": "pitch_gate_audio_ref_step02000.safetensors", "strength": 0.5}]

    def test_base_model(self):
        assert bmd.parse_prompt_graph(GRAPH)["base_model"] == "ltx-2.3-distilled.safetensors"

    def test_generation_params(self):
        g = bmd.parse_prompt_graph(GRAPH)["generation"]
        assert g["width"] == 960 and g["height"] == 544
        assert g["length_frames"] == 121  # resolved through the INTConstant link
        assert g["fps"] == 25.0
        assert g["seed"] == 43
        assert g["sampler"] == "euler"
        assert g["cfg"] == 1.0
        assert g["steps"] == 8  # nine sigma values -> eight steps

    def test_negative_lexicon_fallback_without_conditioning(self):
        # Drop LTXVConditioning so the parser must classify the two prompts by lexicon.
        g = {k: v for k, v in GRAPH.items() if v["class_type"] != "LTXVConditioning"}
        m = bmd.parse_prompt_graph(g)
        assert m["prompt"] == 'A person says, "Hello"'
        assert m["negative_prompt"].startswith("blurry")

    def test_reference_audio_fallback_first_loadaudio(self):
        # Drop the guide node so reference resolution falls back to the first LoadAudio.
        g = {k: v for k, v in GRAPH.items() if v["class_type"] != "LTXAddAudioICLoRAGuide"}
        assert bmd.parse_prompt_graph(g)["reference_audio_filename"] == "pitch_gate_tone.wav"


class TestResolveScalar:
    def test_literal_passthrough(self):
        assert bmd._resolve_scalar(GRAPH, 960) == 960

    def test_follows_intconstant_link(self):
        assert bmd._resolve_scalar(GRAPH, ["27", 0]) == 121

    def test_unresolvable_link_returns_none(self):
        # link to a node that isn't a known constant -> None, not a crash
        assert bmd._resolve_scalar(GRAPH, ["1570", 0]) is None


class TestResolveAudio:
    def test_found_wav_reports_duration_and_rate(self, tmp_path):
        p = tmp_path / "tone.wav"
        with wave.open(str(p), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)  # 1.0 s
        info = bmd.resolve_audio("tone.wav", [tmp_path])
        assert info["found"] is True
        assert info["sample_rate"] == 16000
        assert abs(info["duration_s"] - 1.0) < 0.01

    def test_missing_file_flagged_not_dropped(self, tmp_path):
        info = bmd.resolve_audio("nope.mp3", [tmp_path])
        assert info["found"] is False
        assert info["filename"] == "nope.mp3"


class TestDiscoverRenders:
    def test_groups_triplet_by_base_id(self, tmp_path):
        for suffix in (".png", ".mp4", "-audio.mp4"):
            (tmp_path / f"pitch_gate_iclora_00045{suffix}").write_bytes(b"x")
        renders = bmd.discover_renders(tmp_path)
        assert len(renders) == 1
        r = renders[0]
        assert r.id == "pitch_gate_iclora_00045"
        assert r.png.name == "pitch_gate_iclora_00045.png"
        assert r.video_audio.name == "pitch_gate_iclora_00045-audio.mp4"
        assert r.video_silent.name == "pitch_gate_iclora_00045.mp4"

    def test_skips_base_without_png(self, tmp_path):
        (tmp_path / "orphan.mp4").write_bytes(b"x")
        assert bmd.discover_renders(tmp_path) == []


class TestResolveAudioOrdering:
    def test_exact_match_in_later_root_beats_deep_match_in_earlier_root(self, tmp_path):
        # #1: an exact root/filename hit must win over a deep basename match in an
        # earlier root — otherwise the wrong audio is recorded as found.
        r1 = tmp_path / "r1" / "sub"
        r1.mkdir(parents=True)
        r2 = tmp_path / "r2"
        r2.mkdir()
        _wav(r1 / "tone.wav", 8000, 0.5)   # wrong file, buried under r1
        _wav(r2 / "tone.wav", 16000, 1.0)  # intended file, directly in r2
        info = bmd.resolve_audio("tone.wav", [tmp_path / "r1", r2])
        assert info["found"] is True
        assert info["sample_rate"] == 16000  # exact-in-r2, not deep-in-r1

    def test_glob_metachars_in_filename_match_literally(self, tmp_path):
        # #2: a filename with [ ] must not be treated as a glob char-class.
        name = "track[remix].wav"
        deep = tmp_path / "deep"
        deep.mkdir()
        _wav(deep / name, 16000, 0.5)  # not at root/name -> requires rglob
        info = bmd.resolve_audio(name, [tmp_path])
        assert info["found"] is True
        assert Path(info["path"]).name == name

    def test_cache_reuses_resolution(self, tmp_path):
        # #3: a shared cache resolves a repeated filename only once.
        _wav(tmp_path / "tone.wav", 16000, 1.0)
        cache: dict = {}
        a = bmd.resolve_audio("tone.wav", [tmp_path], cache=cache)
        b = bmd.resolve_audio("tone.wav", [tmp_path], cache=cache)
        assert a == b
        assert ("tone.wav", None) in cache


class TestWorkflowHash:
    def test_hashes_graph_not_pixels_and_is_order_independent(self):
        # #5: provenance hash is over the canonical graph, not the PNG file, so two
        # different node orderings of the same graph hash identically.
        h1 = bmd._graph_sha256(GRAPH)
        h2 = bmd._graph_sha256(dict(reversed(list(GRAPH.items()))))
        assert h1 == h2
        assert len(h1) == 64


class TestMultiNodeDisambiguation:
    def test_reference_audio_follows_guide_over_first_loadaudio(self):
        # The reference must come from the IC-LoRA guide's link even when a second
        # LoadAudio (the full song) appears EARLIER in graph order.
        g = {"9001": {"class_type": "LoadAudio", "inputs": {"audio": "full_song.mp3"}}, **GRAPH}
        assert bmd.parse_prompt_graph(g)["reference_audio_filename"] == "pitch_gate_tone.wav"


class TestBuildDataset:
    def test_end_to_end_row_and_symlink(self, tmp_path):
        renders = tmp_path / "renders"
        renders.mkdir()
        _png_with_prompt(renders / "r_00001.png", GRAPH)
        (renders / "r_00001.mp4").write_bytes(b"v")
        (renders / "r_00001-audio.mp4").write_bytes(b"va")
        out = tmp_path / "ds"
        summary = bmd.build_dataset(renders, out, [tmp_path], "symlink")
        assert summary["rows"] == 1
        import orjson
        rows = [orjson.loads(line) for line in (out / "dataset.jsonl").read_bytes().splitlines()]
        assert rows[0]["prompt"] == 'A person says, "Hello"'
        assert (out / "media" / "r_00001.png").is_symlink()

    def test_rerun_clears_stale_symlinks(self, tmp_path):
        # #8: a render dropped from the source must not leave a ghost symlink behind.
        renders = tmp_path / "renders"
        renders.mkdir()
        out = tmp_path / "ds"
        _png_with_prompt(renders / "a.png", GRAPH)
        bmd.build_dataset(renders, out, [tmp_path], "symlink")
        assert (out / "media" / "a.png").is_symlink()
        (renders / "a.png").unlink()
        _png_with_prompt(renders / "b.png", GRAPH)
        bmd.build_dataset(renders, out, [tmp_path], "symlink")
        # is_symlink() is True for a dangling link, so this catches a stale leftover
        # that .exists() (which follows the now-broken link) would mask.
        assert not (out / "media" / "a.png").is_symlink()
        assert (out / "media" / "b.png").is_symlink()

    def test_dims_probed_from_video_when_graph_cannot_resolve(self, tmp_path):
        # #9: SAMPLED_GRAPH's dims are planner-linked (null from graph); build_row
        # must recover width/height from the rendered video.
        import shutil as _sh
        if _sh.which("ffmpeg") is None:
            import pytest
            pytest.skip("ffmpeg not available")
        import subprocess
        renders = tmp_path / "renders"
        renders.mkdir()
        _png_with_prompt(renders / "r.png", SAMPLED_GRAPH)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x96:d=0.2:r=25",
             "-pix_fmt", "yuv420p", str(renders / "r-audio.mp4")],
            capture_output=True, check=True,
        )
        out = tmp_path / "ds"
        bmd.build_dataset(renders, out, [tmp_path], "symlink")
        import orjson
        row = orjson.loads((out / "dataset.jsonl").read_bytes().splitlines()[0])
        assert row["generation"]["width"] == 160
        assert row["generation"]["height"] == 96

    def test_unresolvable_dims_without_video_warn_in_row(self, tmp_path):
        # build_row owns the resolution warning: planner-linked dims + NO video -> warn.
        renders = tmp_path / "renders"
        renders.mkdir()
        _png_with_prompt(renders / "r.png", SAMPLED_GRAPH)  # no sibling mp4
        out = tmp_path / "ds"
        bmd.build_dataset(renders, out, [tmp_path], "symlink")
        import orjson
        row = orjson.loads((out / "dataset.jsonl").read_bytes().splitlines()[0])
        assert row["generation"]["width"] is None
        assert any("resolution" in w.lower() for w in row["warnings"])


class TestModuleInvariants:
    def test_ref_audio_nodes_are_a_subset_of_cond_passthrough(self):
        # The ref-audio trace only reaches these nodes if they're also passthrough
        # nodes the conditioning trace walks; keep the two tables from desyncing.
        assert bmd._REF_AUDIO_NODES <= set(bmd._COND_PASSTHROUGH.keys())

    def test_dataset_card_documents_every_row_field(self):
        # The card's schema table is maintained by hand; pin that every top-level row
        # key and every generation.* key appears as a token in the card so a rename
        # can't silently drift the docs.
        card = bmd._DATASET_CARD
        top_keys = ("id", "prompt", "negative_prompt", "reference_audio", "loras",
                    "base_model", "generation", "outputs", "warnings", "provenance")
        for k in top_keys:
            assert f"`{k}" in card, f"row key {k!r} not documented in dataset_card"
        for k in bmd.parse_prompt_graph(GRAPH)["generation"]:
            assert k in card, f"generation.{k} not documented in dataset_card"
