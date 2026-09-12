"""Simulation backends.

One scenario has exactly one owner of its physics (ADR-0002), and a backend is
an orchestrator of processes rather than a handle on a live scene (ADR-0011).
Nothing here imports a simulator: the core installs without vendor dependencies
(section 10), and these modules speak in argument vectors and generated files.
"""

from .gazebo import GazeboBackend, GazeboRobot, GazeboWorld, SimulationError
from .process import Launcher, PlanningLauncher, Process, SubprocessLauncher

__all__ = [
    "GazeboBackend",
    "GazeboRobot",
    "GazeboWorld",
    "Launcher",
    "PlanningLauncher",
    "Process",
    "SimulationError",
    "SubprocessLauncher",
]
