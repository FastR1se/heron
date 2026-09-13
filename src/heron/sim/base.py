"""What every simulation backend shares with the command that starts it.

``sim run`` knows no backend by name (ADR-0021). It loads one from the plugin
group by ``world.backend`` and constructs it with the arguments below, so this
signature is part of the extension point as much as ``SimBackend`` is. The
launcher is handed in, not created inside: the command decides whether a run is
a run or a plan, and a backend that could start processes on its own would make
a dry run a promise instead of a fact (ADR-0011).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..core.protocols import SimBackend
from ..registry.registry import Registry
from ..validator.report import Report
from .process import Launcher


class SimulationError(RuntimeError):
    """The world cannot be built or started."""

    def __init__(self, message: str, report: Report | None = None) -> None:
        self.report = report
        if report is not None:
            message = "\n".join([message, *(str(f) for f in report.errors)])
        super().__init__(message)


class SimBackendFactory(Protocol):
    """How ``sim run`` constructs a backend it found in the plugin group."""

    def __call__(
        self,
        registry: Registry,
        *,
        into: Path,
        launcher: Launcher,
        scenario_path: Path | None,
        gui: bool,
        rviz: bool,
    ) -> SimBackend: ...
