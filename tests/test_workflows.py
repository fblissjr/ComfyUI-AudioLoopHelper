"""Tests that workflow JSON files pass structural validation.

Runs the workflow integrity checker against all example workflows.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "internal" / "scripts"))
from test_workflow_integrity import validate


WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "example_workflows"


def _get_workflows():
    return sorted(WORKFLOW_DIR.glob("*.json"))


@pytest.mark.parametrize("wf_path", _get_workflows(), ids=lambda p: p.name)
def test_workflow_integrity(wf_path):
    errors = validate(str(wf_path))
    assert errors == [], f"Workflow {wf_path.name} has errors:\n" + "\n".join(errors)
