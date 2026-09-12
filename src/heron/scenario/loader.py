"""Reading a scenario file.

Thin by design: a scenario is data, and everything that could be wrong about it
beyond its own schema -- a missing profile, a robot without an adapter for the
world backend -- is a cross-document question and therefore the validator's, not
the loader's.
"""

from __future__ import annotations

from pathlib import Path

from ..core.loading import load_model
from ..core.models import Scenario


def load_scenario(path: Path) -> Scenario:
    """Load and schema-check a scenario document."""
    return load_model(Path(path), Scenario)
