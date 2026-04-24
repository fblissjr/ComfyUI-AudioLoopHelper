"""Tests that workflow JSON files pass structural validation.

Runs the workflow integrity checker against all example workflows.
"""

import pytest

from test_workflow_integrity import validate
from workflow_utils import EXAMPLE_WORKFLOWS_DIR


def _get_workflows():
    return sorted(EXAMPLE_WORKFLOWS_DIR.glob("*.json"))


@pytest.mark.parametrize("wf_path", _get_workflows(), ids=lambda p: p.name)
def test_workflow_integrity(wf_path):
    errors = validate(str(wf_path))
    assert errors == [], f"Workflow {wf_path.name} has errors:\n" + "\n".join(errors)
