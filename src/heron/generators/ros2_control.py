"""The contract, materialized as control interfaces.

This is the generator that earns ADR-0009. Everything it writes is derived, and
each derivation is a sentence from the contract turned into a sentence the
control stack understands:

* ``action.type`` says what the robot is commanded with -- one command interface;
* ``observation`` says what is read back -- the state interfaces, and nothing
  beyond them, because an interface nobody declared is an interface nobody
  checked;
* ``joint_order`` says which joints and in what order, everywhere;
* ``limits`` bounds the command interface, so the safety envelope of the profile
  is present in the hardware description rather than only in a validator run;
* ``state_rate_hz`` sets the rate of the controller manager;
* ``control_mode`` chooses the class of controller, because the difference
  between a stream of commands and "perform this motion and report completion"
  is exactly the difference between a forwarding controller and a trajectory
  controller (ARCHITECTURE.md, section 3).

An action the target format has no interface for is refused, not approximated.
A Cartesian pose is not a joint command, and writing one as if it were would
produce a robot that moves and is wrong -- the failure this platform exists to
prevent.
"""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Any, Final
from xml.etree import ElementTree as ET

import yaml

from ..core.enums import ActionType, ControlMode, ObservationType
from ..core.models import RobotProfile
from ..core.units import to_radians
from ..core.urdf import UrdfModel, parse_urdf
from .common import (
    XML_COMMENT,
    GeneratedFile,
    GeneratorError,
    Instance,
    header,
    xml_text,
)

HARDWARE_PLUGIN: Final = "gz_ros2_control/GazeboSimSystem"
"""Hardware component of the simulator that owns the physics (ADR-0002)."""

SYSTEM_PLUGIN_FILE: Final = "gz_ros2_control-system"
SYSTEM_PLUGIN_NAME: Final = "gz_ros2_control::GazeboSimROS2ControlPlugin"

COMMAND_INTERFACES: Final[dict[ActionType, str]] = {
    ActionType.JOINT_POSITION: "position",
    ActionType.JOINT_VELOCITY: "velocity",
    ActionType.JOINT_TORQUE: "effort",
}
"""Action types that map onto a per-joint command interface."""

STATE_INTERFACES: Final[dict[ObservationType, str]] = {
    ObservationType.JOINT_POS: "position",
    ObservationType.JOINT_VEL: "velocity",
    ObservationType.JOINT_TORQUE: "effort",
}
"""Observation terms that map onto a per-joint state interface. Terms about the
base or about the previous action are not joint state: the first comes from
odometry, the second from whoever produced the action."""

STREAMING_CONTROLLERS: Final[dict[ActionType, str]] = {
    ActionType.JOINT_POSITION: "position_controllers/JointGroupPositionController",
    ActionType.JOINT_VELOCITY: "velocity_controllers/JointGroupVelocityController",
    ActionType.JOINT_TORQUE: "effort_controllers/JointGroupEffortController",
}
TRAJECTORY_CONTROLLER: Final = "joint_trajectory_controller/JointTrajectoryController"

BROADCASTER: Final = "joint_state_broadcaster"
BROADCASTER_TYPE: Final = "joint_state_broadcaster/JointStateBroadcaster"

CONTROLLERS_FILE: Final = "controllers.yaml"
DESCRIPTION_FILE: Final = "description.urdf"


def command_interface(profile: RobotProfile) -> str:
    """The one command interface this robot's action maps onto."""
    action = profile.contract.action.type
    try:
        return COMMAND_INTERFACES[action]
    except KeyError:
        raise GeneratorError(
            f"{profile.name}: действие {action} не отображается на интерфейс сустава. "
            f"Отображаются: {', '.join(sorted(COMMAND_INTERFACES))}. "
            f"Декартова поза и ширина захвата требуют собственного контроллера, "
            f"и подменять их позициями суставов платформа не станет (ADR-0009)"
        ) from None


def state_interfaces(profile: RobotProfile) -> list[str]:
    """State interfaces the contract's observation asks for, in a stable order.

    Position is always present: without it neither the broadcaster nor a
    visualization has anything to show, and a contract that observes only
    velocity still describes a robot whose joints have positions.
    """
    wanted = {"position"}
    for term in profile.contract.observation:
        interface = STATE_INTERFACES.get(term.type)
        if interface is not None:
            wanted.add(interface)
    order = ["position", "velocity", "effort"]
    return [name for name in order if name in wanted]


def controller_name(profile: RobotProfile) -> str:
    """Name of the controller this robot is driven by.

    Derived from what it is driven with, not from what kind of machine it is:
    the core knows capabilities, not types (invariant 7).
    """
    if profile.control_mode is ControlMode.COMMAND:
        return "joint_trajectory_controller"
    return f"{profile.contract.action.type}_controller"


def controller_type(profile: RobotProfile) -> str:
    """Class of controller, chosen by control mode and action (section 3)."""
    if profile.control_mode is ControlMode.COMMAND:
        return TRAJECTORY_CONTROLLER
    action = profile.contract.action.type
    if action not in STREAMING_CONTROLLERS:
        raise GeneratorError(
            f"{profile.name}: для действия {action} в потоковом режиме нет "
            f"группового контроллера"
        )
    return STREAMING_CONTROLLERS[action]


def _joint_bounds(profile: RobotProfile, index: int) -> tuple[float, float] | None:
    """Command bounds for one joint, expressed in the units URDF uses."""
    bounds = profile.limits.joint_position
    if bounds is None or profile.contract.action.type is not ActionType.JOINT_POSITION:
        return None
    unit = profile.contract.angle_unit
    return to_radians(bounds.min[index], unit), to_radians(bounds.max[index], unit)


def ros2_control_block(instance: Instance, description: UrdfModel) -> ET.Element:
    """The ``<ros2_control>`` element describing this robot's interfaces."""
    profile = instance.profile
    command = command_interface(profile)
    states = state_interfaces(profile)

    element = ET.Element("ros2_control", {"name": f"{instance.id}_system", "type": "system"})
    hardware = ET.SubElement(element, "hardware")
    ET.SubElement(hardware, "plugin").text = HARDWARE_PLUGIN

    for index, name in enumerate(profile.contract.joint_order):
        joint = ET.SubElement(element, "joint", {"name": name})
        command_element = ET.SubElement(joint, "command_interface", {"name": command})
        bounds = _joint_bounds(profile, index)
        described = description.joint(name)
        if bounds is not None and described is not None and described.angular:
            low, high = bounds
            ET.SubElement(command_element, "param", {"name": "min"}).text = f"{low:g}"
            ET.SubElement(command_element, "param", {"name": "max"}).text = f"{high:g}"
        for state in states:
            ET.SubElement(joint, "state_interface", {"name": state})
    return element


def simulator_plugin_block(instance: Instance, controllers: Path) -> ET.Element:
    """The ``<gazebo>`` element that starts a controller manager inside the world.

    The path to the controller configuration is absolute because the format
    accepts nothing else: a relative path would resolve against the working
    directory of whoever started the simulator. ADR-0010 allows exactly this
    one absolute path and derives it from the output directory.
    """
    element = ET.Element("gazebo")
    plugin = ET.SubElement(
        element, "plugin", {"filename": SYSTEM_PLUGIN_FILE, "name": SYSTEM_PLUGIN_NAME}
    )
    ET.SubElement(plugin, "parameters").text = controllers.as_posix()
    ros = ET.SubElement(plugin, "ros")
    ET.SubElement(ros, "namespace").text = instance.id
    return element


def merged_description(instance: Instance, root: Path) -> GeneratedFile:
    """The input description with the derived blocks merged into a copy.

    ADR-0009: not a line of kinematics is written here. The input tree is copied
    whole -- including everything the platform does not understand -- and two
    derived elements are appended.
    """
    description = parse_urdf(instance.description)
    tree = copy.deepcopy(description.tree)

    for existing in tree.findall("ros2_control"):
        tree.remove(existing)

    if instance.controlled:
        tree.append(ros2_control_block(instance, description))
        tree.append(
            simulator_plugin_block(
                instance, (root / Path(*instance.directory.parts) / CONTROLLERS_FILE).resolve()
            )
        )
    else:
        frozen = ET.Element("gazebo")
        ET.SubElement(frozen, "static").text = "true"
        tree.append(frozen)

    head = header(
        "Итоговое описание робота для запуска",
        f"{instance.description.name} + профиль {instance.profile.name}",
        contract_id=instance.profile.contract.contract_id,
        comment=XML_COMMENT,
    )
    return GeneratedFile(
        path=instance.directory / DESCRIPTION_FILE, content=xml_text(tree, head=head)
    )


def controllers_config(instance: Instance) -> GeneratedFile:
    """Configuration of the controller manager and of the robot's controllers."""
    profile = instance.profile
    name = controller_name(profile)
    joints = list(profile.contract.joint_order)

    manager: dict[str, Any] = {
        "update_rate": int(profile.contract.state_rate_hz),
        BROADCASTER: {"type": BROADCASTER_TYPE},
        name: {"type": controller_type(profile)},
    }
    controller: dict[str, Any] = {"joints": joints}
    if profile.control_mode is ControlMode.COMMAND:
        controller["command_interfaces"] = [command_interface(profile)]
        controller["state_interfaces"] = state_interfaces(profile)

    document: dict[str, Any] = {
        "controller_manager": {"ros__parameters": manager},
        name: {"ros__parameters": controller},
    }
    body = yaml.safe_dump(document, allow_unicode=True, sort_keys=False, default_flow_style=False)
    head = header(
        "Конфигурация контроллеров",
        f"профиль {profile.name}, режим {profile.control_mode}",
        contract_id=profile.contract.contract_id,
    )
    return GeneratedFile(path=instance.directory / CONTROLLERS_FILE, content=head + body)


def robot_files(instance: Instance, root: Path) -> list[GeneratedFile]:
    """Everything one robot instance contributes to a bundle."""
    files = [merged_description(instance, root)]
    if instance.controlled:
        files.append(controllers_config(instance))
    return files


def spawner_arguments(instance: Instance) -> list[str]:
    """Controllers to start, in the order they must be started in."""
    return [BROADCASTER, controller_name(instance.profile)]


def description_path(instance: Instance) -> PurePosixPath:
    return instance.directory / DESCRIPTION_FILE
