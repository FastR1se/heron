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

What things are *called* is not decided here. Controller names, interfaces and
topics live in ``ros2.names``, because the adapter has to find at run time
exactly what this module configures, and two modules spelling the same name
apart is how phase 2 lost a scene (ADR-0013, ADR-0014). This module decides how
they are written down; that one decides what they are.
"""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Any, Final
from xml.etree import ElementTree as ET

import yaml

from ..core.enums import ActionType, ControlMode
from ..core.models import RobotProfile
from ..core.units import to_radians
from ..core.urdf import UrdfModel, parse_urdf
from ..ros2 import names
from .common import (
    XML_COMMENT,
    GeneratedFile,
    GeneratorError,
    Instance,
    header,
    namespace,
    node_name,
    xml_text,
)

HARDWARE_PLUGIN: Final = "gz_ros2_control/GazeboSimSystem"
"""Hardware component of the simulator that owns the physics (ADR-0002)."""

SYSTEM_PLUGIN_FILE: Final = "gz_ros2_control-system"
SYSTEM_PLUGIN_NAME: Final = "gz_ros2_control::GazeboSimROS2ControlPlugin"

BROADCASTER: Final = names.BROADCASTER
BROADCASTER_TYPE: Final = names.BROADCASTER_TYPE

CONTROLLERS_FILE: Final = "controllers.yaml"
DESCRIPTION_FILE: Final = "description.urdf"


def command_interface(profile: RobotProfile) -> str:
    """The one command interface this robot's action maps onto."""
    interface = names.command_interface(profile)
    if interface is None:
        action = profile.contract.action.type
        raise GeneratorError(
            f"{profile.name}: действие {action} не отображается на интерфейс сустава. "
            f"Отображаются: {', '.join(sorted(names.COMMAND_INTERFACES))}. "
            f"Декартова поза и ширина захвата требуют собственного контроллера, "
            f"и подменять их позициями суставов платформа не станет (ADR-0009)"
        )
    return interface


def state_interfaces(profile: RobotProfile) -> list[str]:
    """State interfaces the contract's observation asks for (``ros2.names``)."""
    return names.state_interfaces(profile)


def controller_name(profile: RobotProfile) -> str:
    """Name of the controller this robot is driven by (``ros2.names``)."""
    return names.controller_name(profile)


def controller_type(profile: RobotProfile) -> str:
    """Class of controller, chosen by control mode and action (section 3)."""
    kind = names.controller_type(profile)
    if kind is None:
        raise GeneratorError(
            f"{profile.name}: для действия {profile.contract.action.type} в потоковом "
            f"режиме нет группового контроллера"
        )
    return kind


def _joint_bounds(profile: RobotProfile, index: int) -> tuple[float, float] | None:
    """Command bounds for one joint, expressed in the units URDF uses."""
    bounds = profile.limits.joint_position
    if bounds is None or profile.contract.action.type is not ActionType.JOINT_POSITION:
        return None
    unit = profile.contract.angle_unit
    return to_radians(bounds.min[index], unit), to_radians(bounds.max[index], unit)


def observed_joints(profile: RobotProfile, description: UrdfModel) -> list[str]:
    """Movable joints of the description the contract does not name (ADR-0012).

    The platform reads their state and cannot command them: they get state
    interfaces and no command interface. A joint nobody reads cannot be compared
    with anything, and a robot drawn without half its links looks healthy.
    """
    declared = set(profile.contract.joint_order)
    return [item.name for item in description.commandable_joints if item.name not in declared]


def ros2_control_block(instance: Instance, description: UrdfModel) -> ET.Element:
    """The ``<ros2_control>`` element describing this robot's interfaces.

    Two sources, each in its own role (ADR-0012): the contract says what is
    commanded, the description says what moves.
    """
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

    for name in observed_joints(profile, description):
        joint = ET.SubElement(element, "joint", {"name": name})
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
    ET.SubElement(ros, "namespace").text = namespace(instance)
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

    # Where the broadcaster publishes is stated, not inherited. Its default
    # today resolves to <namespace>/joint_states, which is what the adapter
    # subscribes to -- but a default is a fact about someone else's source, and
    # the adapter would find an empty topic on the day it changed (ADR-0014).
    broadcaster: dict[str, Any] = {names.LOCAL_TOPICS_PARAM: False}

    # Keys are fully qualified: the controller manager of this robot runs under
    # the robot's namespace, and a parameter file keyed on a bare node name never
    # reaches it. Nothing warns about that -- the parameters are simply absent,
    # and the first sign is a controller that "has no type".
    document: dict[str, Any] = {
        node_name(instance, "controller_manager"): {"ros__parameters": manager},
        node_name(instance, BROADCASTER): {"ros__parameters": broadcaster},
        node_name(instance, name): {"ros__parameters": controller},
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
