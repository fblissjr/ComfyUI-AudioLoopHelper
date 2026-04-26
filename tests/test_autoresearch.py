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
        with tracker_mod.connect(db) as conn:
            for i, status in enumerate(["baseline", "keep", "baseline"]):
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
