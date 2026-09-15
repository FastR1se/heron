"""The Gazebo twin of a G1: its drive firmware, as a ``ros2_control`` controller (ADR-0033).

On the robot every motor evaluates the law of the last ``LowCmd``. In a Gazebo world
the controller ``unitree_g1_bridge/LowLevelBridge`` of this package does, inside the
controller manager, in step with the physics, and publishes ``LowState`` back. The
DDS adapter on the other side is the one that drives the vendor's MuJoCo twin and
the robot, unchanged.

The controller is C++ and knows nothing about a G1 but its message types. What
joint sits in which slot, which body this is and what the topics are called is
written here into its parameters, from the tables the adapter reads (ADR-0027).

Everything else a Gazebo scene needs from a robot driven through ``ros2_control``
-- the hardware block, the manager's file, the broadcaster, spawners -- is the
platform's own vocabulary and is taken from it, not copied.

The world runs in the environment of the link (ADR-0034): CycloneDDS, the domain of
``params``, and the same CycloneDDS configuration the adapter's channel builds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree as ET

from heron.core.enums import TwinKind
from heron.core.environment import Requirement, RequirementKind, needs
from heron.core.models import RobotProfile, sim_target
from heron.core.urdf import UrdfModel
from heron.generators.common import GeneratedFile, GeneratorError, Instance
from heron.generators.ros2_control import (
    Controller,
    broadcaster,
    controllers_path,
    manager_config,
    ros2_control_block,
    simulator_plugin_block,
    spawner,
)
from heron.generators.twin import GAZEBO, LaunchNode

from .channel import COMMAND_TOPIC, STATE_TOPIC, LinkConfig, LinkConfigError, ros2_topic
from .slots import MACHINES, SlotError, machine_for, slots_of
from .twin import ADAPTER

BRIDGE_PACKAGE: Final = "unitree_g1_bridge"
"""The ROS 2 package under ``ros2/`` that holds the controller."""

CONTROLLER: Final = "low_level_bridge"
CONTROLLER_TYPE: Final = f"{BRIDGE_PACKAGE}/LowLevelBridge"

MESSAGES_PACKAGE: Final = "unitree_hg"
COMMAND_MESSAGE: Final = f"{MESSAGES_PACKAGE}/msg/LowCmd"
STATE_MESSAGE: Final = f"{MESSAGES_PACKAGE}/msg/LowState"

RMW: Final = "rmw_cyclonedds_cpp"
"""The ROS 2 middleware the world runs on: CycloneDDS 0.10, as the robot (ADR-0022, p. 6)."""

PARAMETERS: Final = ("joints", "slots", "mode_machine", "command_topic", "state_topic")
"""Parameters of the controller, in the order they are written. The C++ side declares
exactly these, and a test of the package reads its source to hold the two together."""

BRIDGE_PACKAGES: Final[dict[str, str]] = {
    BRIDGE_PACKAGE: "colcon build --base-paths examples/plugins/unitree_g1/ros2 "
    "<путь к unitree_hg> --packages-select unitree_hg unitree_g1_bridge; source install/setup.bash "
    "(scripts/phase4-stage5.sh env)",
    MESSAGES_PACKAGE: "colcon build из unitree_ros2, cyclonedds_ws/src/unitree/unitree_hg; "
    "source install/setup.bash",
    RMW: "sudo apt install ros-jazzy-rmw-cyclonedds-cpp",
    "controller_manager": "sudo apt install ros-jazzy-ros2-control (ставит scripts/phase0.sh)",
    "gz_ros2_control": "sudo apt install ros-jazzy-gz-ros2-control (ставит scripts/phase0.sh)",
    "joint_state_broadcaster": "sudo apt install ros-jazzy-ros2-controllers "
    "(ставит scripts/phase0.sh)",
}
"""Every ROS 2 package this twin names in what it writes, and how each gets onto a machine:
the controller and the broadcaster in the manager's file, the hardware component and the
simulator plugin in the description, the spawner in the launch, the middleware in the
environment, the messages on the link. A test derives the set from what is written."""

REQUIREMENTS: Final[tuple[Requirement, ...]] = tuple(
    needs(RequirementKind.ROS2, package, hint=hint)
    for package, hint in BRIDGE_PACKAGES.items()
)


class G1GazeboTwin:
    """``unitree_g1_bridge`` as the other side of a G1's link in a Gazebo world."""

    adapter = ADAPTER
    backend = GAZEBO
    kind = TwinKind.BRIDGE
    requirements: Sequence[Requirement] = REQUIREMENTS

    def description_elements(
        self, instance: Instance, description: UrdfModel, root: Path
    ) -> list[ET.Element]:
        _check_profile(instance.id, instance.profile)
        return [
            ros2_control_block(instance, description),
            simulator_plugin_block(instance, controllers_path(instance, root)),
        ]

    def files(self, instance: Instance, root: Path) -> list[GeneratedFile]:
        machine, slots = _check_profile(instance.id, instance.profile)
        controllers = [broadcaster(), bridge_controller(instance, machine, slots)]
        return [manager_config(instance, controllers)]

    def launch_nodes(self, instance: Instance) -> list[LaunchNode]:
        return [spawner(broadcaster().name), spawner(CONTROLLER)]

    def control_plane(self, namespace: str, profile: RobotProfile) -> Mapping[str, str | None]:
        # At the root, as on the robot: its topic names carry no namespace (ADR-0034, p. 5).
        return {
            "namespace": namespace,
            "controller": CONTROLLER,
            "state_topic": ros2_topic(STATE_TOPIC),
            "state_message": STATE_MESSAGE,
            "command_topic": ros2_topic(COMMAND_TOPIC),
            "command_message": COMMAND_MESSAGE,
        }

    def environment(self, instance: Instance) -> Mapping[str, str]:
        link = link_of(instance)
        return {
            "CYCLONEDDS_URI": link.cyclonedds_xml(),
            "RMW_IMPLEMENTATION": RMW,
            "ROS_DOMAIN_ID": str(link.domain),
        }


def link_of(instance: Instance) -> LinkConfig:
    """The link the adapter's ``params`` describe for this world: read, never declared again."""
    target = sim_target(GAZEBO)
    spec = instance.profile.adapters.get(target)
    if spec is None:
        raise GeneratorError(
            f"{instance.id}: у профиля {instance.profile.name!r} нет адаптера {target!r}"
        )
    try:
        return LinkConfig.from_params(spec.params)
    except LinkConfigError as exc:
        raise GeneratorError(f"{instance.id}: {target}: {exc}") from exc


def bridge_controller(instance: Instance, machine: int, slots: Sequence[int]) -> Controller:
    """The controller and everything about the robot it is told."""
    parameters: dict[str, Any] = {
        "joints": list(instance.profile.contract.joint_order),
        "slots": list(slots),
        "mode_machine": machine,
        "command_topic": ros2_topic(COMMAND_TOPIC),
        "state_topic": ros2_topic(STATE_TOPIC),
    }
    return Controller(CONTROLLER, CONTROLLER_TYPE, {name: parameters[name] for name in PARAMETERS})


def _check_profile(robot: str, profile: RobotProfile) -> tuple[int, tuple[int, ...]]:
    """The body and the slots of a contract the bridge can drive, or a refusal.

    The same refusals the adapter makes at ``connect``, made before a world exists:
    a scene the adapter would refuse to drive is not worth starting.
    """
    joints = profile.contract.joint_order
    if not profile.contract.action.has_gains:
        raise GeneratorError(
            f"{robot}: мост G1 исполняет закон привода из команды, а действие контракта "
            f"профиля {profile.name!r} без kp и kd (ADR-0019)"
        )
    try:
        slots = slots_of(joints)
    except SlotError as exc:
        raise GeneratorError(f"{robot}: {exc}") from exc
    machine = machine_for(joints)
    if machine is None:
        known = ", ".join(f"{len(body)} DoF" for body in MACHINES.values())
        raise GeneratorError(
            f"{robot}: контракт профиля {profile.name!r} ({len(joints)} суставов) не описывает "
            f"ни одно тело G1, которое знает пакет ({known})"
        )
    return machine, slots
