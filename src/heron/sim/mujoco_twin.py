"""What a twin contributes to a MuJoCo world (ADR-0024, ADR-0030).

The platform starts no MuJoCo of its own. A MuJoCo world is a vendor's
simulator that loads a scene, steps the physics and answers in the robot's
protocol -- a twin that owns the physics of its scene (ADR-0021). What such a
twin contributes is therefore not elements of a description or nodes of a launch,
as in Gazebo, but processes: the commands that start it.

They are values, like everything a Gazebo twin returns (ADR-0011): a dry run
prints them, a test asserts on them, and the backend starts exactly them through
the launcher it was handed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..core.mjcf import MUJOCO
from ..core.models import RobotProfile
from ..core.protocols import Twin

__all__ = ["MUJOCO", "MujocoScene", "MujocoTwin", "TwinProcess"]


@dataclass(frozen=True, slots=True)
class TwinProcess:
    """One process of a scene: a name to report it by, and the command starting it."""

    name: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError(f"процесс {self.name!r}: пустая команда")


@dataclass(frozen=True, slots=True)
class MujocoScene:
    """What a twin is told about the scene it is asked to start."""

    robot: str
    """The robot's identifier in the scenario."""

    profile: RobotProfile
    params: Mapping[str, Any]
    """``params`` of the profile's adapter under ``sim.mujoco``. The link is read
    from here and never declared by the twin: a domain described twice is a
    domain two sides can disagree on (ADR-0021)."""

    world: Path
    """The scene file, absolute: ``world.file`` of the scenario, which the validator
    checked loads ``description`` (``mjcf.scene_without_description``)."""

    description: Path | None
    """The profile's MJCF, absolute."""

    gui: bool = True
    """Whether the world may open a window. A twin that cannot run without one
    refuses a scene asked to."""


@runtime_checkable
class MujocoTwin(Twin, Protocol):
    """A twin that is a MuJoCo world of its own, in the vocabulary of that world."""

    def processes(self, scene: MujocoScene) -> Sequence[TwinProcess]:
        """Processes starting this scene, in the order they must start in.

        A scene the twin cannot start is refused here with ``SimulationError``,
        before anything is started: the programs named must be among what the
        twin declares in ``requirements`` (ADR-0025).
        """
