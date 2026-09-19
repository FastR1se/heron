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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree as ET

import yaml

from ..core.enums import AngleUnit, TwinKind
from ..core.environment import Requirement, RequirementKind, needs
from ..core.models import RobotProfile
from ..core.units import to_radians
from ..core.urdf import UrdfModel
from ..ros2 import names
from .common import (
    GeneratedFile,
    GeneratorError,
    Instance,
    header,
    namespace,
    node_name,
)
from .description import description_copy
from .twin import GAZEBO, LaunchNode

ADAPTER: Final = "ros2_control"
"""Plugin name of the adapter this twin answers for, as ``pyproject.toml``
registers it and a profile writes it."""

SPAWNER_TIMEOUT: Final = "60"
"""How long a spawner waits for the world to come up: for the controller manager to
appear, for the switch that activates its controller, and for the answer about that
switch.

All three are the same condition -- a world that has started -- and are given the same
deadline, which ends the moment the condition holds. The switch is performed in the
manager's update loop, which runs only while the simulation steps; the spawner's own
default for it is 5 s, "useful when switching cannot be performed immediately, e.g.,
paused simulations at startup" (``spawner.py``, jazzy). Left at that default, a slow
first start failed one controller and not the next (journal 2026-09-19).

The third is not a detail of the same wait but the one that decided the run of
2026-09-20. A client that gives up before the deadline it gave the server asks the
same question twice, and the second question is a different question: the switch it
requested has meanwhile been performed, so ``STRICT`` finds the controller already
active, refuses the switch -- "Aborting, no controller is switched!" -- and the
spawner dies reporting failure for a controller that is running. ``--switch-timeout``
is sent to the manager as the deadline of the request; ``--service-call-timeout``
bounds how long the spawner itself waits for the reply, and its default is 10 s
(``service_caller``, ros2_control 4.21+). Sixty against ten is what "failed getting a
result ... in 10.0" meant, three times, before the abort."""

HARDWARE_PLUGIN: Final = "gz_ros2_control/GazeboSimSystem"
"""Hardware component of the simulator that owns the physics (ADR-0002)."""

SYSTEM_PLUGIN_FILE: Final = "gz_ros2_control-system"
SYSTEM_PLUGIN_NAME: Final = "gz_ros2_control::GazeboSimROS2ControlPlugin"

BROADCASTER: Final = names.BROADCASTER
BROADCASTER_TYPE: Final = names.BROADCASTER_TYPE

CONTROLLERS_FILE: Final = "controllers.yaml"

INTERPOLATION_NONE: Final = "none"
"""``interpolation_method`` of the trajectory controller for an action closed through gains."""

TWIN_PACKAGES: Final[dict[str, str]] = {
    "controller_manager": "ros2-control",
    "gz_ros2_control": "gz-ros2-control",
    "joint_state_broadcaster": "ros2-controllers",
    "joint_trajectory_controller": "ros2-controllers",
    "position_controllers": "ros2-controllers",
    "velocity_controllers": "ros2-controllers",
    "effort_controllers": "ros2-controllers",
}
"""ROS 2 packages this twin names, and what installs each.

Named in three places, none of them the launch alone: the hardware component and
the simulator plugin in the description, the controller classes in their
configuration, the spawner in the launch. A missing one fails differently in each
-- a scene that does not parse, a controller that "has no type", a model that
never gets a controller manager -- and all of it after a world was started. The
rule of ADR-0013, which held only for the launch, now holds for everything the
twin writes: a test derives the set from what it writes, and phase 0 installs it."""

REQUIREMENTS: Final[tuple[Requirement, ...]] = tuple(
    needs(
        RequirementKind.ROS2,
        package,
        hint=f"пакет ros-<дистрибутив>-{installed_by} (ставит scripts/install.sh install ros2)",
    )
    for package, installed_by in TWIN_PACKAGES.items()
)


def check_link(profile: RobotProfile) -> None:
    """Refuse a robot the link to this scene could not drive.

    ROS 2 measures angles in radians (REP 103), and the adapter talking to this
    control stack does not convert (ADR-0060). The description gets its bounds in
    radians either way; a scene whose robot would be commanded 57 times off is not
    worth starting.
    """
    unit = profile.contract.angle_unit
    if unit is not AngleUnit.RAD:
        raise GeneratorError(
            f"{profile.name}: контракт в {unit}, а стек ros2_control и связь с ним — в "
            f"радианах; связь единиц не переводит (ADR-0060). Для сцены с ros2_control "
            f"контракт — angle_unit: rad"
        )


def command_interface(profile: RobotProfile) -> str:
    """The one command interface this robot's action maps onto."""
    check_link(profile)
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
    """Command bounds for one joint, expressed in the units URDF uses.

    Only a position command interface is bounded by positions. A position action
    closed through gains is commanded with effort (ADR-0019), and hanging angle
    limits on a torque would bound the wrong quantity.
    """
    bounds = profile.limits.joint_position
    if bounds is None or names.command_interface(profile) != names.POSITION_INTERFACE:
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


@dataclass(frozen=True, slots=True)
class Controller:
    """One controller a robot's manager loads: its name, its class, its own parameters."""

    name: str
    type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


def broadcaster() -> Controller:
    """The joint state broadcaster, which every driven robot's manager runs.

    Where it publishes is stated, not inherited. Its default today resolves to
    <namespace>/joint_states, which is what the adapter and the description
    publisher subscribe to -- but a default is a fact about someone else's source,
    and they would find an empty topic on the day it changed (ADR-0014).
    """
    return Controller(BROADCASTER, BROADCASTER_TYPE, {names.LOCAL_TOPICS_PARAM: False})


def controllers_path(instance: Instance, root: Path) -> Path:
    """Where the manager's configuration of this robot lands, as the simulator reads it."""
    return (root / Path(*instance.directory.parts) / CONTROLLERS_FILE).resolve()


def manager_config(instance: Instance, controllers: Sequence[Controller]) -> GeneratedFile:
    """Configuration of the controller manager and of the controllers it loads.

    The manager runs at ``state_rate_hz``: the rate the contract promises states at
    is the rate the stack reads the joints at.
    """
    profile = instance.profile
    manager: dict[str, Any] = {"update_rate": int(profile.contract.state_rate_hz)}
    for controller in controllers:
        manager[controller.name] = {"type": controller.type}

    # Keys are fully qualified: the controller manager of this robot runs under
    # the robot's namespace, and a parameter file keyed on a bare node name never
    # reaches it. Nothing warns about that -- the parameters are simply absent,
    # and the first sign is a controller that "has no type".
    document: dict[str, Any] = {
        node_name(instance, "controller_manager"): {"ros__parameters": manager},
    }
    for controller in controllers:
        document[node_name(instance, controller.name)] = {
            "ros__parameters": dict(controller.parameters)
        }
    body = yaml.safe_dump(document, allow_unicode=True, sort_keys=False, default_flow_style=False)
    head = header(
        "Конфигурация контроллеров",
        f"профиль {profile.name}, режим {profile.control_mode}",
        contract_id=profile.contract.contract_id,
    )
    return GeneratedFile(path=instance.directory / CONTROLLERS_FILE, content=head + body)


def spawner(controller: str) -> LaunchNode:
    """The node that loads and activates one controller of the robot's manager.

    A spawner waits for a controller manager that only exists once the simulator
    has loaded the model; it waits with its own timeout, not a generated sleep.

    One deadline, three flags: the manager has to appear, the switch has to be
    performed, and the answer about it has to come back. Asking for less patience
    on the last than on the middle one is what turns a slow start into a failed
    controller (``SPAWNER_TIMEOUT``).
    """
    return LaunchNode(
        package="controller_manager",
        executable="spawner",
        name=f"spawn_{controller}",
        arguments=(
            controller,
            "--controller-manager", "controller_manager",
            "--controller-manager-timeout", SPAWNER_TIMEOUT,
            "--switch-timeout", SPAWNER_TIMEOUT,
            "--service-call-timeout", SPAWNER_TIMEOUT,
        ),
    )


def controllers_config(instance: Instance) -> GeneratedFile:
    """Configuration of the controller manager and of the robot's controllers."""
    profile = instance.profile
    joints = list(profile.contract.joint_order)

    controller: dict[str, Any] = {"joints": joints}
    if names.uses_trajectory_controller(profile):
        controller["command_interfaces"] = [command_interface(profile)]
        controller["state_interfaces"] = names.trajectory_state_interfaces(profile)
    action = profile.contract.action
    if action.kp is not None and action.kd is not None:
        # The law of the joint drive, run by the controller at the manager's
        # rate: effort = p * position error + d * velocity error. The contract
        # states the coefficients; nothing here chooses them (ADR-0019).
        controller["gains"] = {
            joint: {"p": p, "d": d}
            for joint, p, d in zip(joints, action.kp, action.kd, strict=True)
        }
        # No interpolation between the state at receipt and the point: with splines
        # the desired position is a ramp and the damping term acts on its slope,
        # (q* - q) / T, which neither Isaac nor a drive has. The adapter sends the
        # point at time zero with zero velocity, so the law is the contract's (ADR-0045).
        controller["interpolation_method"] = INTERPOLATION_NONE

    return manager_config(
        instance,
        [broadcaster(), Controller(controller_name(profile), controller_type(profile), controller)],
    )


def spawner_arguments(instance: Instance) -> list[str]:
    """Controllers to start, in the order they must be started in."""
    return [BROADCASTER, controller_name(instance.profile)]


class Ros2ControlTwin:
    """The platform's own twin: ``ros2_control`` inside the Gazebo world.

    Registered through the same group as a vendor's twin (ADR-0021), for the
    same reason built-in adapters are: a private path for our own robots would
    hide whether the public one is good enough. It is also the second
    implementation of the twin protocol, and the reason that protocol is not
    shaped after one vendor.
    """

    adapter = ADAPTER
    backend = GAZEBO
    kind = TwinKind.BRIDGE
    requirements: Sequence[Requirement] = REQUIREMENTS

    def description_elements(
        self, instance: Instance, description: UrdfModel, root: Path
    ) -> list[ET.Element]:
        return [
            ros2_control_block(instance, description),
            simulator_plugin_block(instance, controllers_path(instance, root)),
        ]

    def files(self, instance: Instance, root: Path) -> list[GeneratedFile]:
        return [controllers_config(instance)]

    def launch_nodes(self, instance: Instance) -> list[LaunchNode]:
        return [spawner(controller) for controller in spawner_arguments(instance)]

    def control_plane(self, namespace: str, profile: RobotProfile) -> Mapping[str, str | None]:
        return names.control_plane(namespace, profile)

    def environment(self, instance: Instance) -> Mapping[str, str]:
        # The link is ROS 2 itself: it works in whatever environment the world has.
        return {}


def merged_description(instance: Instance, root: Path) -> GeneratedFile:
    """The copy of the description this twin's robot is started from."""
    return description_copy(replace(instance, twin=Ros2ControlTwin()), root)
