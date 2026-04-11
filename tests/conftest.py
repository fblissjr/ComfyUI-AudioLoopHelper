"""Test configuration. Adds scripts/ to sys.path for imports."""

import sys
from pathlib import Path

# Add scripts/ so tests can import analyze_audio_features directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
