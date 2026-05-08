"""Tests for `scripts/_helpers/_layout_classifications.py`.

Locks in the typo-guard behavior on `compose()` — the failure mode it
catches is silent node-drops when a `function_to_group` key doesn't
match any functional column in `SHARED_NODE_FUNCTIONS`.
"""

from __future__ import annotations

import pytest

from _helpers._layout_classifications import (
    FUNCTIONAL_COLUMNS,
    SHARED_NODE_FUNCTIONS,
    compose,
)


def test_functional_columns_match_shared_table():
    """The exposed FUNCTIONAL_COLUMNS set must equal the unique values in
    SHARED_NODE_FUNCTIONS — derived once at import, not hand-maintained."""
    assert FUNCTIONAL_COLUMNS == frozenset(SHARED_NODE_FUNCTIONS.values())


def test_compose_raises_on_unknown_column():
    with pytest.raises(KeyError, match="Unknown functional column"):
        compose({"inputs_typo": "X"})


def test_compose_partial_mapping_is_allowed():
    """Scripts can omit columns they don't care about; matching nodes are
    simply unclassified rather than triggering a guard."""
    out = compose({"inputs": "G"})
    assert all(group == "G" for group in out.values())
    assert all(SHARED_NODE_FUNCTIONS[nid] == "inputs" for nid in out)


def test_compose_overrides_win():
    """An override pins a specific node id regardless of its functional
    column, even when that column isn't in function_to_group."""
    sample_node_id = next(iter(SHARED_NODE_FUNCTIONS))
    out = compose({}, overrides={sample_node_id: "OVERRIDE_GROUP"})
    assert out[sample_node_id] == "OVERRIDE_GROUP"
