"""Test configuration. Adds scripts/ and tests/ to sys.path so tests
can import analyze_audio_features and the shared _fakes module."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent / "scripts"))
sys.path.insert(0, str(_TESTS_DIR))
