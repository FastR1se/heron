"""What a twin contributes to a Gazebo scene (ADR-0021, ADR-0024).

The generators of this package know the world -- a description copied with its
meshes anchored, a robot state publisher, a spawn, a clock bridge, a
visualization -- and nothing about what drives a robot inside it. That is the
twin's: it adds elements to the copy of the description, files next to it,
nodes to the launch, and names the control plane an adapter will look for.

Everything a twin returns is a value, like every generated file (ADR-0010), so a
scene with a vendor's bridge is as checkable before a run as one without.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable
from xml.etree import ElementTree as ET

from ..core.models import RobotProfile
from ..core.protocols import Twin
from ..core.urdf import UrdfModel
from .common import GeneratedFile, GeneratorError, Instance

GAZEBO: Final = "gazebo"
"""The backend these generators produce a world for."""


@dataclass(frozen=True, slots=True)
class LaunchNode:
    """One process of a scene, started by the launch inside the robot's namespace."""

    package: str
    executable: str
    name: str
    arguments: tuple[str, ...] = ()


@runtime_checkable
class GazeboTwin(Twin, Protocol):
    """A twin living inside a Gazebo world, in the vocabulary of that world."""

    def description_elements(
        self, instance: Instance, description: UrdfModel, root: Path
    ) -> Sequence[ET.Element]:
        """Elements appended to the copy of the input description (ADR-0009)."""

    def files(self, instance: Instance, root: Path) -> Sequence[GeneratedFile]:
        """Files this robot's link needs next to its description."""

    def launch_nodes(self, instance: Instance) -> Sequence[LaunchNode]:
        """Processes started for this robot, in the order they must start in.

        Each from a ROS 2 package the twin needs in ``requirements``: a launch
        naming an absent package takes the whole scene down with it (ADR-0013),
        and a twin's package is the twin's to declare (ADR-0025).
        """

    def control_plane(self, namespace: str, profile: RobotProfile) -> Mapping[str, str | None]:
        """Names an adapter finds at run time, as written into the manifest (ADR-0014)."""


def require_twin(instance: Instance) -> GazeboTwin:
    """The twin of a driven robot. A driven robot without one has no link."""
    if instance.twin is None:
        raise GeneratorError(
            f"{instance.id}: управляемый робот без двойника — в сцене некому говорить "
            f"на языке его адаптера (ADR-0021)"
        )
    return instance.twin
