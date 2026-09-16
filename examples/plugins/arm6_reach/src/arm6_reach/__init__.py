"""Training task for the example arm: reach a joint pose (ADR-0040, ADR-0043).

A package like any a user would write for their own robot and their own task.
The platform finds it through an entry point and imports nothing of it by path;
deleting this directory leaves the core building and passing its tests.
"""

from .task import Arm6ReachTask

__all__ = ["Arm6ReachTask"]
