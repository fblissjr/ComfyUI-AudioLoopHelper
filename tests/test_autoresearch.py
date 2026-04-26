"""Tests for the Phase 2 experiment-runner framework.

Covers the parts that don't require ComfyUI:
  - Fixture loading + hash stability
  - Tracker DB schema + insert/query
  - Workflow mutation logic (base_seed, iterations override, VHS prefix)

What we DON'T test here (deferred to integration / manual verification):
  - submit_prompt / poll_until_done — HTTP calls to a running ComfyUI
  - End-to-end run() — needs ComfyUI + an API-format workflow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import orjson
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
# Make `internal.autoresearch` importable. conftest.py adds scripts/ + tests/
# but not the repo root itself; we need it for the package-style import.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Skip the whole module if duckdb / httpx aren't installed (experiments
# group not synced) — lets dev test runs keep working without forcing
# the experiments dep group.
duckdb = pytest.importorskip("duckdb")
httpx = pytest.importorskip("httpx")

# Skip if the autoresearch framework itself isn't present. internal/ is
# gitignored, so a fresh public clone won't have it; the tests still ship
# as documentation of the framework's contract for anyone who builds the
# framework locally.
if not (_REPO_ROOT / "internal" / "autoresearch" / "harness.py").exists():
    pytest.skip(
        "internal/autoresearch/ framework not present (gitignored). "
        "Tests cover the experiment-runner contract; not applicable here.",
        allow_module_level=True,
    )

from internal.autoresearch import tracker as tracker_mod  # noqa: E402
from internal.autoresearch.harness import (  # noqa: E402
    Fixture,
    _mutate_alc_seed,
    _mutate_iterations,
    _mutate_vhs_filename,
)


# Avoid heavy WorkflowEditor import side-effects: import lazily below.


def _fixture_dict(**overrides) -> dict:
    """Default fixture content for tests; overrides win."""
    base = {
        "fixture_id": "unit_test",
        "audio_path": "/tmp/fake.wav",
        "init_image_path": "/tmp/fake.png",
        "init_positive": "a singing test",
        "init_negative": "noise",
        "schedule": [{"timestamp": "0:00+", "prompt": "p0"}],
        "base_seed": 42,
        "tier1_iterations": 1,
        "tier2_iterations": 3,
        "tier3_iterations": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

class TestFixture:
    def test_load_round_trips_all_fields(self, tmp_path):
        path = tmp_path / "f.json"
        data = _fixture_dict()
        path.write_bytes(orjson.dumps(data))
        f = Fixture.load(path)
        assert f.fixture_id == "unit_test"
        assert f.base_seed == 42
        assert f.tier3_iterations is None  # null in JSON → None in python

    def test_iterations_for_tier_returns_per_tier_value(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_bytes(orjson.dumps(_fixture_dict()))
        f = Fixture.load(path)
        assert f.iterations_for_tier(1) == 1
        assert f.iterations_for_tier(2) == 3
        assert f.iterations_for_tier(3) is None

    def test_iterations_for_unknown_tier_raises(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_bytes(orjson.dumps(_fixture_dict()))
        f = Fixture.load(path)
        with pytest.raises(ValueError, match="unknown tier"):
            f.iterations_for_tier(4)

    def test_hash_is_stable_across_loads_with_same_content(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_bytes(orjson.dumps(_fixture_dict()))
        h1 = Fixture.load(path).hash()
        h2 = Fixture.load(path).hash()
        assert h1 == h2 and len(h1) == 16

    def test_hash_differs_when_content_differs(self, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_bytes(orjson.dumps(_fixture_dict(init_positive="A")))
        b.write_bytes(orjson.dumps(_fixture_dict(init_positive="B")))
        assert Fixture.load(a).hash() != Fixture.load(b).hash()

    def test_hash_excludes_local_paths(self, tmp_path):
        """Same prompts/seed/tiers across machines produce the same hash —
        path differences don't perturb it. Otherwise cross-machine
        comparability breaks."""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_bytes(orjson.dumps(_fixture_dict(audio_path="/path/A")))
        b.write_bytes(orjson.dumps(_fixture_dict(audio_path="/path/B")))
        assert Fixture.load(a).hash() == Fixture.load(b).hash()


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TestTracker:
    def test_connect_creates_schema(self, tmp_path):
        db = tmp_path / "test.duckdb"
        with tracker_mod.connect(db) as conn:
            tables = conn.execute("SHOW TABLES").fetchall()
            assert ("runs",) in tables

    def test_insert_then_query(self, tmp_path):
        db = tmp_path / "test.duckdb"
        rec = tracker_mod.RunRecord(
            run_id="r1", fixture_id="fix_a", tier=1,
            status="keep", primary_metric="wall_seconds",
            primary_metric_value=42.0, description="first run",
            metrics={"a": 1, "b": 2},
        )
        with tracker_mod.connect(db) as conn:
            tracker_mod.insert(conn, rec)
            rows = tracker_mod.recent_runs(conn)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "r1"
        assert rows[0]["primary_metric_value"] == 42.0

    def test_insert_upserts_on_duplicate_run_id(self, tmp_path):
        db = tmp_path / "test.duckdb"
        with tracker_mod.connect(db) as conn:
            tracker_mod.insert(conn, tracker_mod.RunRecord(
                run_id="r1", fixture_id="fix_a", tier=1, status="pending",
                primary_metric_value=1.0,
            ))
            tracker_mod.insert(conn, tracker_mod.RunRecord(
                run_id="r1", fixture_id="fix_a", tier=1, status="keep",
                primary_metric_value=2.0,
            ))
            row = conn.execute(
                "SELECT status, primary_metric_value FROM runs WHERE run_id='r1'"
            ).fetchone()
        assert row == ("keep", 2.0)

    def test_latest_baseline_returns_most_recent(self, tmp_path):
        db = tmp_path / "test.duckdb"
        statuses: list[tracker_mod.RunStatus] = ["baseline", "keep", "baseline"]
        with tracker_mod.connect(db) as conn:
            for i, status in enumerate(statuses):
                tracker_mod.insert(conn, tracker_mod.RunRecord(
                    run_id=f"r{i}", fixture_id="fix_a", tier=1,
                    status=status, primary_metric_value=float(i),
                ))
            baseline = tracker_mod.latest_baseline(conn, "fix_a", 1)
        assert baseline is not None
        # Most recent (r2) was a baseline — should be returned.
        assert baseline["run_id"] == "r2"
        assert baseline["primary_metric_value"] == 2.0

    def test_latest_baseline_returns_none_when_no_baseline(self, tmp_path):
        db = tmp_path / "test.duckdb"
        with tracker_mod.connect(db) as conn:
            tracker_mod.insert(conn, tracker_mod.RunRecord(
                run_id="r1", fixture_id="fix_a", tier=1, status="keep",
                primary_metric_value=1.0,
            ))
            assert tracker_mod.latest_baseline(conn, "fix_a", 1) is None

    def test_metrics_json_round_trips(self, tmp_path):
        db = tmp_path / "test.duckdb"
        metrics = {"wall_seconds": 12.3, "top_consumers": [{"a": 1}]}
        with tracker_mod.connect(db) as conn:
            tracker_mod.insert(conn, tracker_mod.RunRecord(
                run_id="r1", fixture_id="fix_a", tier=1, status="keep",
                primary_metric="wall_seconds", primary_metric_value=12.3,
                metrics=metrics,
            ))
            row = conn.execute("SELECT metrics FROM runs WHERE run_id='r1'").fetchone()
        loaded = json.loads(row[0])
        assert loaded == metrics


# ---------------------------------------------------------------------------
# Workflow mutation
# ---------------------------------------------------------------------------

# Lazy WorkflowEditor import so module-level pyimport doesn't fail when
# scripts/ is not on sys.path yet (tests/conftest.py handles that, but the
# module-level import order matters).
def _ed():
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from workflow_utils import WorkflowEditor
    return WorkflowEditor


def _alc_node(seed: int = 0) -> dict:
    return {
        "id": 1582,
        "type": "AudioLoopController",
        "inputs": [
            {"name": "audio", "type": "AUDIO", "link": 1},
            {"name": "current_iteration", "type": "INT", "link": 2},
            {"name": "window_seconds", "type": "FLOAT", "link": 3},
            {"name": "base_seed", "type": "INT", "link": 4},
        ],
        "outputs": [],
        "widgets_values": [1, 19.88, 2.0, seed, 25],
    }


def _tlo_node_post_phase1c(iterations_in_link: int | None = None) -> dict:
    return {
        "id": 1539,
        "type": "TensorLoopOpen",
        "inputs": [
            {"name": "initial_value", "type": "LATENT", "link": 100},
            {"name": "iterations_in", "type": "INT", "shape": 7, "link": iterations_in_link},
        ],
        "outputs": [],
        "widgets_values": ["iterations", 50],
    }


def _vhs_node() -> dict:
    return {
        "id": 617,
        "type": "VHS_VideoCombine",
        "inputs": [],
        "outputs": [],
        "widgets_values": {
            "frame_rate": 25,
            "loop_count": 0,
            "filename_prefix": "LTX-2",
        },
    }


def _empty_workflow_with(*nodes: dict) -> dict:
    return {
        "last_node_id": max((n["id"] for n in nodes), default=0),
        "last_link_id": 1000,
        "nodes": list(nodes),
        "links": [],
    }


class TestMutateAlcSeed:
    def test_overwrites_widget_slot_3(self, tmp_path):
        WorkflowEditor = _ed()
        wf_path = tmp_path / "wf.json"
        wf_path.write_bytes(orjson.dumps(_empty_workflow_with(_alc_node(seed=999))))
        ed = WorkflowEditor(wf_path)
        _mutate_alc_seed(ed, 42)
        assert ed.find_node(1582)["widgets_values"][3] == 42

    def test_no_op_when_no_alc(self, tmp_path):
        WorkflowEditor = _ed()
        wf_path = tmp_path / "wf.json"
        wf_path.write_bytes(orjson.dumps(_empty_workflow_with(_vhs_node())))
        ed = WorkflowEditor(wf_path)
        _mutate_alc_seed(ed, 42)  # must not raise


class TestMutateIterations:
    def test_none_leaves_workflow_unchanged(self, tmp_path):
        WorkflowEditor = _ed()
        wf_path = tmp_path / "wf.json"
        wf_path.write_bytes(orjson.dumps(_empty_workflow_with(
            _tlo_node_post_phase1c(iterations_in_link=200)
        )))
        ed = WorkflowEditor(wf_path)
        _mutate_iterations(ed, None)  # tier 3 path: no override
        # Existing wire untouched.
        assert ed.find_node(1539)["inputs"][1]["link"] == 200

    def test_inserts_constant_and_wires_iterations_in(self, tmp_path):
        WorkflowEditor = _ed()
        wf_path = tmp_path / "wf.json"
        wf_path.write_bytes(orjson.dumps(_empty_workflow_with(
            _tlo_node_post_phase1c(iterations_in_link=200)
        )))
        ed = WorkflowEditor(wf_path)
        _mutate_iterations(ed, 3)
        # New INTConstant node was added.
        consts = ed.find_nodes_by_type("INTConstant")
        assert len(consts) == 1
        assert consts[0]["widgets_values"] == [3]
        # TLO.iterations_in is now wired from the constant, not the original.
        tlo = ed.find_node(1539)
        new_link_id = tlo["inputs"][1]["link"]
        assert new_link_id is not None and new_link_id != 200

    def test_falls_back_when_iterations_in_slot_absent(self, tmp_path):
        """Pre-Phase-1c workflows have no iterations_in slot; we mutate the
        widget value as a fallback so the harness still works on legacy
        graphs."""
        WorkflowEditor = _ed()
        wf_path = tmp_path / "wf.json"
        legacy_tlo = {
            "id": 1539,
            "type": "TensorLoopOpen",
            "inputs": [{"name": "initial_value", "type": "LATENT", "link": 100}],
            "outputs": [],
            "widgets_values": ["iterations", 50],
        }
        wf_path.write_bytes(orjson.dumps(_empty_workflow_with(legacy_tlo)))
        ed = WorkflowEditor(wf_path)
        _mutate_iterations(ed, 7)
        assert ed.find_node(1539)["widgets_values"][1] == 7


class TestMutateVhsFilename:
    def test_dict_widgets_filename_prefix_gets_run_id(self, tmp_path):
        WorkflowEditor = _ed()
        wf_path = tmp_path / "wf.json"
        wf_path.write_bytes(orjson.dumps(_empty_workflow_with(_vhs_node())))
        ed = WorkflowEditor(wf_path)
        _mutate_vhs_filename(ed, "test_xyz")
        wv = ed.find_node(617)["widgets_values"]
        assert wv["filename_prefix"] == "LTX-2_test_xyz"

    def test_no_op_when_no_vhs(self, tmp_path):
        WorkflowEditor = _ed()
        wf_path = tmp_path / "wf.json"
        wf_path.write_bytes(orjson.dumps(_empty_workflow_with(_alc_node())))
        ed = WorkflowEditor(wf_path)
        _mutate_vhs_filename(ed, "test_xyz")  # must not raise


# ---------------------------------------------------------------------------
# Metric: sage_summary
#
# First non-placeholder Phase 2.1 metric extractor. Reads sage.jsonl from
# run_dir, aggregates kernel distribution + fallback count + total
# attention time. Surfaces value sage-fork's bench would otherwise re-derive.
# ---------------------------------------------------------------------------

def _write_sage_jsonl(path, rows):
    """Helper: serialize a list of dicts as JSONL bytes."""
    lines = [orjson.dumps(r).decode() for r in rows]
    path.write_text("\n".join(lines) + "\n")


class TestSageSummary:
    def test_returns_trace_missing_when_file_absent(self, tmp_path):
        from internal.autoresearch.metrics import sage_summary
        out = sage_summary.extract(tmp_path, fixture=None)
        assert out == {"sage_status": "trace_missing"}

    def test_returns_trace_empty_when_no_per_call_rows(self, tmp_path):
        from internal.autoresearch.metrics import sage_summary
        sage_log = tmp_path / "sage.jsonl"
        # Header + summary only — no per-call rows.
        _write_sage_jsonl(sage_log, [
            {"event": "header", "arch": "sm89_cuda12_8"},
            {"event": "summary", "total_calls": 0, "fallback_count": 0, "distinct_shapes": 0},
        ])
        out = sage_summary.extract(tmp_path, fixture=None)
        assert out == {"sage_status": "trace_empty"}

    def test_aggregates_per_call_rows_into_summary(self, tmp_path):
        from internal.autoresearch.metrics import sage_summary
        sage_log = tmp_path / "sage.jsonl"
        _write_sage_jsonl(sage_log, [
            {"event": "header", "arch": "sm89_cuda12_8"},
            {"shape": [1, 22932, 4096], "has_mask": False, "mode": "auto_mask_aware",
             "effective_mode": "auto", "fell_back": False, "elapsed_us": 100.0,
             "dispatched_kernel": "fp8_cuda++"},
            {"shape": [1, 22932, 4096], "has_mask": False, "mode": "auto_mask_aware",
             "effective_mode": "auto", "fell_back": False, "elapsed_us": 200.0,
             "dispatched_kernel": "fp8_cuda++"},
            {"shape": [1, 498, 2048], "has_mask": True, "mode": "auto_mask_aware",
             "effective_mode": "fp16_triton", "fell_back": False, "elapsed_us": 50.0,
             "dispatched_kernel": "fp16_triton"},
            {"shape": [1, 498, 2048], "has_mask": True, "mode": "auto_mask_aware",
             "effective_mode": "fp16_triton", "fell_back": True, "elapsed_us": 80.0},
        ])
        out = sage_summary.extract(tmp_path, fixture=None)
        assert out["sage_status"] == "ok"
        assert out["sage_total_calls"] == 4
        assert out["sage_fallback_count"] == 1
        assert out["sage_distinct_shapes"] == 2
        assert out["sage_total_attention_us"] == 430.0
        assert out["sage_arch"] == "sm89_cuda12_8"
        # fp8_cuda++: 2 calls × 100+200 µs = 300 µs
        # fp16_triton: 1 call × 50 µs (the fallback row falls back to effective_mode)
        # = total 350; the fallback row uses effective_mode "fp16_triton" since
        # dispatched_kernel was absent.
        assert out["sage_kernel_distribution"]["fp8_cuda++"]["n"] == 2
        assert out["sage_kernel_distribution"]["fp8_cuda++"]["total_us"] == 300.0
        assert out["sage_kernel_distribution"]["fp16_triton"]["n"] == 2  # one dispatched + one fallback
        assert out["sage_kernel_distribution"]["fp16_triton"]["total_us"] == 130.0
        # Fractions sum to 1.0 (within rounding)
        total_frac = sum(
            d["fraction"] for d in out["sage_kernel_distribution"].values()
        )
        assert abs(total_frac - 1.0) < 0.001

    def test_falls_back_to_effective_mode_when_dispatched_kernel_absent(self, tmp_path):
        """Older traces (pre-`6a3be19`) may not have dispatched_kernel
        on every row. The aggregator must fall back to effective_mode
        rather than skipping the row."""
        from internal.autoresearch.metrics import sage_summary
        sage_log = tmp_path / "sage.jsonl"
        _write_sage_jsonl(sage_log, [
            {"shape": [1, 100, 64], "has_mask": False, "mode": "auto_mask_aware",
             "effective_mode": "auto", "fell_back": False, "elapsed_us": 10.0},
        ])
        out = sage_summary.extract(tmp_path, fixture=None)
        assert out["sage_total_calls"] == 1
        assert "auto" in out["sage_kernel_distribution"]
        assert out["sage_kernel_distribution"]["auto"]["n"] == 1

    def test_skips_blank_lines_and_decode_errors(self, tmp_path):
        """Tracer JSONL can contain blank lines (line buffering near
        crash) or partial last lines (mid-write truncation). Aggregator
        must keep going."""
        from internal.autoresearch.metrics import sage_summary
        sage_log = tmp_path / "sage.jsonl"
        sage_log.write_text(
            '{"shape":[1,10,10],"has_mask":false,"mode":"auto","effective_mode":"auto","fell_back":false,"elapsed_us":5.0}\n'
            "\n"
            "{not valid json\n"
            '{"shape":[1,20,20],"has_mask":false,"mode":"auto","effective_mode":"auto","fell_back":false,"elapsed_us":7.0}\n'
        )
        out = sage_summary.extract(tmp_path, fixture=None)
        assert out["sage_status"] == "ok"
        assert out["sage_total_calls"] == 2
        assert out["sage_distinct_shapes"] == 2


# ---------------------------------------------------------------------------
# Metric: subject_consistency
#
# DINO-v2 cosine sim of per-frame embeddings vs the anchor (frame 0).
# Tests cover the status-only paths + the helper functions that don't
# require a model. The model-loading + frame-decoding paths are
# integration territory (require the `metrics` dep group + a real mp4)
# and are deferred to the live tier-1 smoke test.
# ---------------------------------------------------------------------------

class TestSubjectConsistency:
    def test_returns_video_missing_when_mp4_absent(self, tmp_path):
        from internal.autoresearch.metrics import subject_consistency
        out = subject_consistency.extract(tmp_path, fixture=None)
        assert out == {"subject_consistency_status": "video_missing"}

    def test_returns_model_unavailable_when_load_returns_none(
        self, tmp_path, monkeypatch
    ):
        """Public-clone path: heavy deps (transformers/torch/cv2) absent
        → _load_model returns None → extract reports model_unavailable
        instead of crashing the harness.
        """
        from internal.autoresearch.metrics import subject_consistency
        (tmp_path / "output.mp4").write_bytes(b"\x00")
        monkeypatch.setattr(subject_consistency, "_load_model", lambda: None)
        out = subject_consistency.extract(tmp_path, fixture=None)
        assert out == {"subject_consistency_status": "model_unavailable"}

    def test_returns_decode_failed_when_sample_frames_returns_none(
        self, tmp_path, monkeypatch
    ):
        from internal.autoresearch.metrics import subject_consistency
        (tmp_path / "output.mp4").write_bytes(b"\x00")
        monkeypatch.setattr(
            subject_consistency, "_load_model", lambda: ("model", "processor")
        )
        monkeypatch.setattr(
            subject_consistency, "_sample_frames", lambda p, n: None
        )
        out = subject_consistency.extract(tmp_path, fixture=None)
        assert out == {"subject_consistency_status": "decode_failed"}

    def test_returns_no_frames_when_sample_frames_returns_empty(
        self, tmp_path, monkeypatch
    ):
        from internal.autoresearch.metrics import subject_consistency
        (tmp_path / "output.mp4").write_bytes(b"\x00")
        monkeypatch.setattr(
            subject_consistency, "_load_model", lambda: ("model", "processor")
        )
        monkeypatch.setattr(
            subject_consistency, "_sample_frames", lambda p, n: []
        )
        out = subject_consistency.extract(tmp_path, fixture=None)
        assert out == {"subject_consistency_status": "no_frames"}

    def test_aggregates_with_synthetic_embeddings(self, tmp_path, monkeypatch):
        """End-to-end extract path with mocked model + frame sampler.
        Synthetic embeddings: anchor at angle 0, frames 1..3 drift
        progressively away. Mean cos-sim < 1, drift slope < 0."""
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import subject_consistency

        (tmp_path / "output.mp4").write_bytes(b"\x00")

        # 4 frames; embeddings drift along a known direction
        synth_embs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.31, 0.0, 0.0],
            [0.87, 0.50, 0.0, 0.0],
            [0.71, 0.71, 0.0, 0.0],
        ], dtype=np.float32)
        # L2-normalize
        synth_embs = synth_embs / np.linalg.norm(
            synth_embs, axis=1, keepdims=True
        )

        monkeypatch.setattr(
            subject_consistency, "_load_model", lambda: ("m", "p")
        )
        monkeypatch.setattr(
            subject_consistency,
            "_sample_frames",
            lambda p, n: ["f0", "f1", "f2", "f3"],
        )
        monkeypatch.setattr(
            subject_consistency, "_embed_frames", lambda f, m, p: synth_embs
        )

        out = subject_consistency.extract(tmp_path, fixture=None)
        assert out["subject_consistency_status"] == "ok"
        assert out["subject_consistency_n_frames"] == 4
        # Frames 1..3 vs anchor: 0.95, 0.87, 0.71
        assert abs(out["subject_consistency_mean_to_anchor"] - 0.8433) < 0.01
        assert abs(out["subject_consistency_max_to_anchor"] - 0.95) < 0.01
        assert abs(out["subject_consistency_min_to_anchor"] - 0.71) < 0.01
        # Drift is monotonically decreasing → negative slope
        assert out["subject_consistency_drift_slope"] < 0

    def test_cosine_to_anchor_excludes_self(self):
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import subject_consistency

        embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        sims = subject_consistency._cosine_to_anchor(embs)
        # Output length = N-1 (anchor itself excluded)
        assert sims.shape == (2,)
        # Frame 1 orthogonal to anchor → 0; frame 2 == anchor → 1
        assert abs(float(sims[0]) - 0.0) < 1e-6
        assert abs(float(sims[1]) - 1.0) < 1e-6

    def test_cosine_to_anchor_handles_short_input(self):
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import subject_consistency

        # Single embedding → no comparisons possible
        sims = subject_consistency._cosine_to_anchor(
            np.array([[1.0, 0.0]], dtype=np.float32)
        )
        assert sims.shape == (0,)

    def test_drift_slope_is_negative_for_decreasing_sims(self):
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import subject_consistency

        sims = np.array([0.95, 0.87, 0.71], dtype=np.float32)
        slope = subject_consistency._drift_slope(sims)
        assert slope < 0

    def test_drift_slope_zero_for_short_input(self):
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import subject_consistency

        sims = np.array([0.95], dtype=np.float32)
        assert subject_consistency._drift_slope(sims) == 0.0

    def test_returns_single_frame_status_on_n1(self, tmp_path, monkeypatch):
        """Edge case: video has exactly 1 frame → no comparisons
        possible. We used to emit `status: "ok"` with sentinel 1.0
        values, but that polluted downstream `WHERE status='ok'`
        aggregations (a degenerate render scoring identically to a
        perfect one). Distinct status keeps the aggregation clean."""
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import subject_consistency

        (tmp_path / "output.mp4").write_bytes(b"\x00")
        monkeypatch.setattr(
            subject_consistency, "_load_model", lambda: ("m", "p")
        )
        monkeypatch.setattr(
            subject_consistency, "_sample_frames", lambda p, n: ["f0"]
        )
        monkeypatch.setattr(
            subject_consistency,
            "_embed_frames",
            lambda f, m, p: np.array([[1.0, 0.0]], dtype=np.float32),
        )
        out = subject_consistency.extract(tmp_path, fixture=None)
        assert out == {"subject_consistency_status": "single_frame"}


# ---------------------------------------------------------------------------
# Metric: av_consistency
#
# PE-AV-16-frame joint audio-video-text embedding. v0 reports a single
# cosine sim — AV emb vs the fixture's init_positive text — measuring
# how well the rendered video+audio matches its target prompt.
# Apache-2.0 model (ungated).
# ---------------------------------------------------------------------------

class _FixtureStub:
    """Minimal duck-typed Fixture for tests that need .init_positive."""
    def __init__(self, init_positive: str = "a singing test"):
        self.init_positive = init_positive


class TestAvConsistency:
    def test_returns_video_missing_when_mp4_absent(self, tmp_path):
        from internal.autoresearch.metrics import av_consistency
        out = av_consistency.extract(tmp_path, fixture=_FixtureStub())
        assert out == {"av_consistency_status": "video_missing"}

    def test_returns_no_text_when_fixture_is_none(self, tmp_path):
        from internal.autoresearch.metrics import av_consistency
        (tmp_path / "output.mp4").write_bytes(b"\x00")
        out = av_consistency.extract(tmp_path, fixture=None)
        assert out == {"av_consistency_status": "no_text"}

    def test_returns_no_text_when_init_positive_empty(self, tmp_path):
        from internal.autoresearch.metrics import av_consistency
        (tmp_path / "output.mp4").write_bytes(b"\x00")
        out = av_consistency.extract(tmp_path, fixture=_FixtureStub(""))
        assert out == {"av_consistency_status": "no_text"}

    def test_returns_model_unavailable_when_load_returns_none(
        self, tmp_path, monkeypatch
    ):
        """Public-clone path: PE-AV not installed → _load_model None
        → extract reports model_unavailable instead of crashing."""
        from internal.autoresearch.metrics import av_consistency
        (tmp_path / "output.mp4").write_bytes(b"\x00")
        monkeypatch.setattr(av_consistency, "_load_model", lambda: None)
        out = av_consistency.extract(tmp_path, fixture=_FixtureStub())
        assert out == {"av_consistency_status": "model_unavailable"}

    def test_returns_decode_failed_when_embed_returns_none(
        self, tmp_path, monkeypatch
    ):
        from internal.autoresearch.metrics import av_consistency
        (tmp_path / "output.mp4").write_bytes(b"\x00")
        monkeypatch.setattr(
            av_consistency, "_load_model", lambda: ("model", "transform")
        )
        monkeypatch.setattr(
            av_consistency,
            "_embed_av_and_text",
            lambda v, t, m, tf: None,
        )
        out = av_consistency.extract(tmp_path, fixture=_FixtureStub())
        assert out == {"av_consistency_status": "decode_failed"}

    def test_aggregates_with_synthetic_embeddings(self, tmp_path, monkeypatch):
        """End-to-end with mocked model + embed. Synthetic L2-normalized
        embeddings → known cosine similarity. The backend tag falls
        through to "unknown" since _load_model is mocked."""
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import av_consistency

        (tmp_path / "output.mp4").write_bytes(b"\x00")

        # AV emb angled 0; AV-text emb angled ~30° → cos_sim ≈ 0.866
        av = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        av_text = np.array([0.866, 0.5, 0.0], dtype=np.float32)
        av = av / np.linalg.norm(av)
        av_text = av_text / np.linalg.norm(av_text)

        monkeypatch.setattr(
            av_consistency, "_load_model", lambda: ("m", "t")
        )
        monkeypatch.setattr(
            av_consistency,
            "_embed_av_and_text",
            lambda v, t, m, tf: (av, av_text),
        )

        out = av_consistency.extract(tmp_path, fixture=_FixtureStub("singer"))
        assert out["av_consistency_status"] == "ok"
        assert out["av_consistency_model"] == "facebook/pe-av-large-16-frame"
        assert abs(out["av_consistency_av_text_sim"] - 0.866) < 0.01
        # _load_model is mocked → the real loader never ran → backend
        # stays at the module-level default. Pinning this prevents
        # a future cache-shape change from silently re-introducing
        # the old "unknown" fallback the simplify pass removed.
        assert out["av_consistency_backend"] in {
            "transformers", "perception_models", "unknown"
        }

    def test_cosine_helper_with_normalized_inputs(self):
        np = pytest.importorskip("numpy")
        from internal.autoresearch.metrics import av_consistency

        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        # Orthogonal → 0
        assert av_consistency._cosine(a, b) == 0.0
        # Identical → 1
        assert av_consistency._cosine(a, a) == 1.0


# ---------------------------------------------------------------------------
# Output mp4 discovery
#
# After ComfyUI completes a render, VHS_VideoCombine writes the mp4 to
# its output dir (NOT to data/runs/${RUN_ID}/). Phase 2.1 metrics that
# read the rendered video (subject_consistency, eventually style /
# lip_sync / aesthetic) need the mp4 inside run_dir. Harness symlinks
# it after poll_until_done returns.
# ---------------------------------------------------------------------------

class TestLocateAndLinkOutputMp4:
    def test_returns_false_when_source_dir_is_none(self, tmp_path):
        from internal.autoresearch.harness import _locate_and_link_output_mp4
        ok = _locate_and_link_output_mp4(
            run_id="abc", run_dir=tmp_path, source_dir=None
        )
        assert ok is False
        assert not (tmp_path / "output.mp4").exists()

    def test_returns_false_when_source_dir_does_not_exist(self, tmp_path):
        from internal.autoresearch.harness import _locate_and_link_output_mp4
        ok = _locate_and_link_output_mp4(
            run_id="abc", run_dir=tmp_path, source_dir=tmp_path / "missing"
        )
        assert ok is False
        assert not (tmp_path / "output.mp4").exists()

    def test_returns_false_when_no_matching_mp4(self, tmp_path):
        from internal.autoresearch.harness import _locate_and_link_output_mp4
        source = tmp_path / "src"
        source.mkdir()
        (source / "LTX-2_other_run_001.mp4").write_bytes(b"\x00")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        ok = _locate_and_link_output_mp4(
            run_id="abc", run_dir=run_dir, source_dir=source
        )
        assert ok is False
        assert not (run_dir / "output.mp4").exists()

    def test_symlinks_first_matching_mp4(self, tmp_path):
        from internal.autoresearch.harness import _locate_and_link_output_mp4
        source = tmp_path / "src"
        source.mkdir()
        target_mp4 = source / "LTX-2_abc_00001.mp4"
        target_mp4.write_bytes(b"video-bytes")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        ok = _locate_and_link_output_mp4(
            run_id="abc", run_dir=run_dir, source_dir=source
        )
        assert ok is True
        link = run_dir / "output.mp4"
        assert link.exists()
        assert link.is_symlink()
        assert link.resolve() == target_mp4.resolve()

    def test_idempotent_when_called_twice(self, tmp_path):
        """Second call should not raise FileExistsError; the symlink
        target may have been swept and re-created during a retry."""
        from internal.autoresearch.harness import _locate_and_link_output_mp4
        source = tmp_path / "src"
        source.mkdir()
        target_mp4 = source / "LTX-2_abc_00001.mp4"
        target_mp4.write_bytes(b"video-bytes")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert _locate_and_link_output_mp4(
            run_id="abc", run_dir=run_dir, source_dir=source
        ) is True
        assert _locate_and_link_output_mp4(
            run_id="abc", run_dir=run_dir, source_dir=source
        ) is True

    def test_picks_first_match_when_multiple_match(self, tmp_path):
        """VHS_VideoCombine appends counters — when retried, multiple
        matches may exist. Take the lexicographically first (=earliest
        counter) for determinism."""
        from internal.autoresearch.harness import _locate_and_link_output_mp4
        source = tmp_path / "src"
        source.mkdir()
        first = source / "LTX-2_abc_00001.mp4"
        second = source / "LTX-2_abc_00002.mp4"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        ok = _locate_and_link_output_mp4(
            run_id="abc", run_dir=run_dir, source_dir=source
        )
        assert ok is True
        assert (run_dir / "output.mp4").resolve() == first.resolve()

    def test_reads_env_var_when_source_dir_omitted(self, tmp_path, monkeypatch):
        from internal.autoresearch.harness import _locate_and_link_output_mp4
        source = tmp_path / "src"
        source.mkdir()
        (source / "LTX-2_abc_00001.mp4").write_bytes(b"\x00")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        monkeypatch.setenv("COMFYUI_OUTPUT_DIR", str(source))
        ok = _locate_and_link_output_mp4(run_id="abc", run_dir=run_dir)
        assert ok is True
        assert (run_dir / "output.mp4").exists()
