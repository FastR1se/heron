"""What things are called in the ROS 2 control plane.

Two components speak these names and never meet. The generator writes the
configuration that brings the controllers into existence; the adapter, in
another process and often on another machine, talks to the controllers that
configuration created. Nothing connects them but agreement on a handful of
strings.

Phase 2 failed four times in the same shape -- an artifact correct on its own,
read differently by whoever consumed it -- and the last of those (ADR-0013) was
exactly two lists of names kept in two places until they drifted. Controller
names and topics are the same kind of value, so they are given one home before
they are given a chance to drift.

Where the names come from is not a convention this platform invented. Three of
them are read off the sources of ``ros2_controllers``:

* a forward command controller subscribes to ``~/commands``;
* the trajectory controller subscribes to ``~/joint_trajectory``;
* the broadcaster publishes ``joint_states``, resolved against its namespace --
  but only while ``use_local_topics`` is false, so the generated configuration
  states that parameter instead of inheriting it (ADR-0014).

Nothing here imports rclpy or any other part of a ROS 2 distribution: these are
names, and a name is a string.

A value the control plane has no counterpart for is reported as ``None`` rather
than approximated. What to do about it differs by caller -- the generator
refuses to write a configuration, the adapter refuses to drive -- so the answer
is not given here.
"""

from __future__ import annotations

from typing import Final

from ..core.enums import ActionType, ControlMode, ObservationType
from ..core.models import RobotProfile

# -- interfaces ------------------------------------------------------------

COMMAND_INTERFACES: Final[dict[ActionType, str]] = {
    ActionType.JOINT_POSITION: "position",
    ActionType.JOINT_VELOCITY: "velocity",
    ActionType.JOINT_TORQUE: "effort",
}
"""Action types that map onto a per-joint command interface."""

EFFORT_INTERFACE: Final = "effort"
POSITION_INTERFACE: Final = "position"
VELOCITY_INTERFACE: Final = "velocity"

STATE_INTERFACES: Final[dict[ObservationType, str]] = {
    ObservationType.JOINT_POS: "position",
    ObservationType.JOINT_VEL: "velocity",
    ObservationType.JOINT_TORQUE: "effort",
}
"""Observation terms that map onto a per-joint state interface. Terms about the
base or about the previous action are not joint state: the first comes from
odometry, the second from whoever produced the action."""

INTERFACE_ORDER: Final = ("position", "velocity", "effort")
"""Stable order for interfaces, so two runs list them the same way."""

# -- controllers -----------------------------------------------------------

STREAMING_CONTROLLERS: Final[dict[ActionType, str]] = {
    ActionType.JOINT_POSITION: "position_controllers/JointGroupPositionController",
    ActionType.JOINT_VELOCITY: "velocity_controllers/JointGroupVelocityController",
    ActionType.JOINT_TORQUE: "effort_controllers/JointGroupEffortController",
}
TRAJECTORY_CONTROLLER: Final = "joint_trajectory_controller/JointTrajectoryController"
TRAJECTORY_CONTROLLER_NAME: Final = "joint_trajectory_controller"

BROADCASTER: Final = "joint_state_broadcaster"
BROADCASTER_TYPE: Final = "joint_state_broadcaster/JointStateBroadcaster"

LOCAL_TOPICS_PARAM: Final = "use_local_topics"
"""Parameter of the broadcaster that decides where states are published.

Stated in the generated configuration rather than left at its default, because
the adapter subscribes to the result: a default is something the platform reads
off someone else's source, and a value is something it declares (ADR-0014)."""

# -- topics and messages ---------------------------------------------------

STATE_TOPIC: Final = "joint_states"
"""Relative name the broadcaster publishes under while ``use_local_topics`` is
false. Resolved against the namespace of the controller manager, not against
the controller's own node."""

COMMANDS_TOPIC: Final = "commands"
TRAJECTORY_TOPIC: Final = "joint_trajectory"

STATE_MESSAGE: Final = "sensor_msgs/msg/JointState"
ARRAY_MESSAGE: Final = "std_msgs/msg/Float64MultiArray"
TRAJECTORY_MESSAGE: Final = "trajectory_msgs/msg/JointTrajectory"

MESSAGES: Final = frozenset({STATE_MESSAGE, ARRAY_MESSAGE, TRAJECTORY_MESSAGE})
"""Every message type the platform reads or writes. A transport that cannot
build one of these cannot drive a robot the generator can configure, and a test
holds those two sets together."""

ADAPTER_PACKAGES: Final[dict[str, str]] = {
    "rclpy": "desktop",
    "sensor_msgs": "desktop",
    "std_msgs": "desktop",
    "trajectory_msgs": "desktop",
    "builtin_interfaces": "desktop",
}
"""Python packages the ROS 2 transport imports, and what installs each.

The same rule as ``LAUNCH_PACKAGES`` and for the same reason (ADR-0013): an
import that fails at run time is an environment nobody declared. The set is
checked from both sides by a test -- nothing outside it may be imported by the
transport, and everything in it must be installed by the phase 0 script."""


# -- derivations -----------------------------------------------------------


def closes_through_gains(profile: RobotProfile) -> bool:
    """Whether a position target is turned into a torque by the contract's gains.

    Then the joint is commanded with effort, and the law that turns the target
    into that effort runs inside the control stack, at the rate of the controller
    manager -- where a joint drive runs it on hardware. Commanding position
    instead would hand the law to the simulator, which executes it as a stiff
    servo that no gain in the contract describes (ADR-0019).
    """
    return profile.contract.action.has_gains


def uses_trajectory_controller(profile: RobotProfile) -> bool:
    """Command mode, or a position action closed through gains.

    The trajectory controller is the controller of ``ros2_controllers`` that
    both accepts "reach this" and closes a position error through per-joint
    gains into an effort command.
    """
    return profile.control_mode is ControlMode.COMMAND or closes_through_gains(profile)


def command_interface(profile: RobotProfile) -> str | None:
    """The one command interface this robot's action maps onto, if any."""
    if closes_through_gains(profile):
        return EFFORT_INTERFACE
    return COMMAND_INTERFACES.get(profile.contract.action.type)


def state_interfaces(profile: RobotProfile) -> list[str]:
    """State interfaces the contract's observation asks for, in a stable order.

    Position is always present: without it neither the broadcaster nor a
    visualization has anything to show, and a contract that observes only
    velocity still describes a robot whose joints have positions.

    Velocity joins when the action closes through gains: the damping term acts on
    it, and a controller asked to damp a velocity nobody reads refuses to start.
    """
    wanted = {POSITION_INTERFACE}
    for term in profile.contract.observation:
        interface = STATE_INTERFACES.get(term.type)
        if interface is not None:
            wanted.add(interface)
    if closes_through_gains(profile):
        wanted.add(VELOCITY_INTERFACE)
    return [name for name in INTERFACE_ORDER if name in wanted]


def controller_name(profile: RobotProfile) -> str:
    """Name of the controller this robot is driven by.

    Derived from what it is driven with, not from what kind of machine it is:
    the core knows capabilities, not types (invariant 7).
    """
    if uses_trajectory_controller(profile):
        return TRAJECTORY_CONTROLLER_NAME
    return f"{profile.contract.action.type}_controller"


def controller_type(profile: RobotProfile) -> str | None:
    """Class of controller, chosen by control mode and action (section 3)."""
    if uses_trajectory_controller(profile):
        return TRAJECTORY_CONTROLLER
    return STREAMING_CONTROLLERS.get(profile.contract.action.type)


def qualified(namespace: str, *parts: str) -> str:
    """An absolute topic name under a namespace.

    An empty namespace is a real configuration, not a missing value: a robot
    brought up without one publishes at the root, and inventing a namespace for
    it would subscribe the adapter to a topic nobody writes.
    """
    segments = [segment.strip("/") for segment in (namespace, *parts)]
    return "/" + "/".join(segment for segment in segments if segment)


def state_topic(namespace: str) -> str:
    """Where this robot's joint states arrive."""
    return qualified(namespace, STATE_TOPIC)


def command_topic(namespace: str, profile: RobotProfile) -> str | None:
    """Where this robot's commands go, or ``None`` if its action has no home."""
    if uses_trajectory_controller(profile):
        return qualified(namespace, controller_name(profile), TRAJECTORY_TOPIC)
    if command_interface(profile) is None:
        return None
    return qualified(namespace, controller_name(profile), COMMANDS_TOPIC)


def command_message(profile: RobotProfile) -> str | None:
    """Type of message a command becomes, or ``None`` if it becomes none."""
    if uses_trajectory_controller(profile):
        return TRAJECTORY_MESSAGE
    if command_interface(profile) is None:
        return None
    return ARRAY_MESSAGE


def control_plane(namespace: str, profile: RobotProfile) -> dict[str, str | None]:
    """Everything an adapter needs to find this robot, as plain values.

    Written into the bundle manifest, so the names a run will use are on disk
    before the run, next to the configuration that creates them. Two components
    then read one description instead of agreeing out of habit.
    """
    return {
        "namespace": namespace,
        "controller": controller_name(profile),
        "state_topic": state_topic(namespace),
        "state_message": STATE_MESSAGE,
        "command_topic": command_topic(namespace, profile),
        "command_message": command_message(profile),
    }
