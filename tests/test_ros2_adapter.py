"""The ros2_control adapter, without ROS 2.

Everything that decides what goes on the wire is decided before anything
touches a ROS 2 distribution (ADR-0014), so all of it is checkable here: which
topic, which message, which values, in which order. What is not checkable here
is that a real controller accepts the result -- these tests say nothing about
that, and no comment in them should suggest otherwise.
"""

from __future__ import annotations

import time

import pytest

from conftest import make_contract, make_profile
from heron.adapters.ros2_control import (
    NotConnectedError,
    Ros2ControlAdapter,
    SafetyViolationError,
    StateMismatchError,
    UnsupportedActionError,
    UnsupportedControlModeError,
)
from heron.core.contract import ActionSpec
from heron.core.enums import ActionType, ControlMode
from heron.core.models import RobotProfile
from heron.core.types import Command
from heron.ros2 import names
from heron.ros2.transport import Publication, Reading, RecordingTransport, TransportError

JOINTS = ["joint0", "joint1", "joint2"]


def connected(
    profile: RobotProfile | None = None,
    *,
    namespace: str = "arm1",
    states: int = 8,
    **config: object,
) -> tuple[Ros2ControlAdapter, RecordingTransport]:
    """An adapter wired to a transport with a few states already queued."""
    profile = profile or make_profile()
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)
    # A short link deadline: this transport answers instantly, so waiting the
    # real one would spend ten seconds discovering nothing.
    adapter.connect(profile, {"namespace": namespace, "link_timeout": 0.05, **config})
    topic = names.state_topic(namespace)
    for index in range(states):
        transport.deliver_state(
            topic,
            stamp=index * 0.01,
            names_=list(profile.contract.joint_order),
            position=[0.0] * profile.contract.dof,
            velocity=[0.0] * profile.contract.dof,
        )
    return adapter, transport


# -- what it subscribes to and advertises ----------------------------------


def test_connect_wires_the_topics_the_generator_creates() -> None:
    adapter, transport = connected()
    assert transport.subscribed == {"/arm1/joint_states": names.STATE_MESSAGE}
    assert transport.advertised == {
        "/arm1/joint_position_controller/commands": names.ARRAY_MESSAGE
    }
    assert adapter.topics["state"] == "/arm1/joint_states"


def test_a_robot_with_drive_gains_is_driven_through_the_trajectory_controller() -> None:
    """ADR-0019: the adapter finds the controller the generator configured for a
    position action closed through gains, and sends it one-point trajectories."""
    profile = make_profile(
        contract=make_contract(
            3,
            action=ActionSpec(type=ActionType.JOINT_POSITION, kp=[60.0] * 3, kd=[1.0] * 3),
        )
    )
    adapter, transport = connected(profile)
    assert transport.advertised == {
        "/arm1/joint_trajectory_controller/joint_trajectory": names.TRAJECTORY_MESSAGE
    }

    adapter.read_state()
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.1, 0.2, 0.3)))
    sent = transport.published[-1]
    assert sent.message_type == names.TRAJECTORY_MESSAGE
    assert sent.fields["joint_names"] == JOINTS
    assert sent.fields["points"][0]["positions"] == [0.1, 0.2, 0.3]


def test_two_instances_of_one_profile_do_not_share_topics() -> None:
    first, one = connected(namespace="arm1")
    second, two = connected(namespace="arm2")
    assert set(one.subscribed) & set(two.subscribed) == set()
    assert first.topics != second.topics


def test_action_without_an_interface_is_refused_at_connect() -> None:
    """Before anything is subscribed, not halfway through a motion."""
    profile = make_profile(
        contract=make_contract(3, action=ActionSpec(type=ActionType.CARTESIAN_POSE))
    )
    with pytest.raises(UnsupportedActionError, match="не отображается"):
        Ros2ControlAdapter(RecordingTransport()).connect(profile, {})


# -- reading state ---------------------------------------------------------


def test_state_is_reordered_by_name_not_by_position() -> None:
    """The message decides nothing about order; the contract decides all of it."""
    profile = make_profile()
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)
    adapter.connect(profile, {"namespace": "arm1"})
    transport.deliver_state(
        "/arm1/joint_states",
        stamp=1.0,
        names_=["joint2", "joint0", "joint1"],
        position=[0.3, 0.1, 0.2],
        velocity=[3.0, 1.0, 2.0],
    )

    state = adapter.read_state()
    assert state.joint_position == (0.1, 0.2, 0.3)
    assert state.joint_velocity == (1.0, 2.0, 3.0)


def test_joint_the_contract_declares_but_the_robot_never_reports() -> None:
    profile = make_profile()
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)
    adapter.connect(profile, {"namespace": "arm1"})
    transport.deliver_state(
        "/arm1/joint_states", stamp=1.0, names_=["joint0", "joint1"], position=[0.0, 0.0]
    )

    with pytest.raises(StateMismatchError, match="joint2"):
        adapter.read_state()


def test_joints_outside_the_contract_are_read_and_not_dropped() -> None:
    """ADR-0012 at the other end of the link: the platform reads what moves."""
    profile = make_profile()
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)
    adapter.connect(profile, {"namespace": "arm1"})
    transport.deliver_state(
        "/arm1/joint_states",
        stamp=1.0,
        names_=[*JOINTS, "finger_left", "finger_right"],
        position=[0.0, 0.0, 0.0, 0.04, 0.04],
        velocity=[0.0] * 5,
    )

    state = adapter.read_state()
    assert len(state.joint_position) == 3
    assert state.extra["observed"] == {"finger_left": 0.04, "finger_right": 0.04}


def test_a_message_that_does_not_describe_what_it_names_is_refused() -> None:
    profile = make_profile()
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)
    adapter.connect(profile, {"namespace": "arm1"})
    transport.deliver(
        Reading(
            topic="/arm1/joint_states",
            stamp=1.0,
            fields={"name": JOINTS, "position": [0.0, 0.0]},
        )
    )
    with pytest.raises(StateMismatchError, match="позиций"):
        adapter.read_state()


def test_state_that_stops_arriving_keeps_its_timestamp() -> None:
    """Staleness stays visible instead of being papered over with a fresh stamp."""
    adapter, _ = connected(states=1)
    first = adapter.read_state()
    second = adapter.read_state()
    assert second.timestamp == first.timestamp


def test_a_link_nobody_publishes_on_says_so() -> None:
    """Two failures, two fixes: nobody publishes, or nothing arrives."""
    adapter, _ = connected(states=0)
    with pytest.raises(StateMismatchError, match="никто не публикует"):
        adapter.read_state()


def test_a_link_that_publishes_but_delivers_nothing_says_that_instead() -> None:
    adapter, transport = connected(states=0)
    transport.publishers[names.state_topic("arm1")] = 1

    with pytest.raises(StateMismatchError, match="не доходят"):
        adapter.read_state()


def test_the_first_read_waits_for_the_link_and_later_ones_do_not() -> None:
    """Establishing a link is not the same as using one (ADR-0016)."""
    profile = make_profile()
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)
    adapter.connect(profile, {"namespace": "arm1", "link_timeout": 5.0})
    transport.deliver_state(
        names.state_topic("arm1"),
        stamp=1.0,
        names_=JOINTS,
        position=[0.0] * 3,
        velocity=[0.0] * 3,
    )

    first = time.monotonic()
    adapter.read_state()
    adapter.read_state()  # nothing queued: must not wait the link deadline again
    assert time.monotonic() - first < 1.0


# -- sending commands ------------------------------------------------------


def test_position_command_becomes_a_float_array_on_the_controller_topic() -> None:
    adapter, transport = connected()
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.1, 0.2, 0.3)))

    assert transport.published == [
        Publication(
            topic="/arm1/joint_position_controller/commands",
            message_type=names.ARRAY_MESSAGE,
            fields={"data": [0.1, 0.2, 0.3]},
        )
    ]


def test_a_robot_of_the_command_class_is_refused_before_anything_is_wired() -> None:
    """ADR-0058: the link has no phase, no envelope-bound motion and no refusal during
    one (ADR-0049), so it does not drive a profile that promises them."""
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)

    with pytest.raises(UnsupportedControlModeError, match="control_mode: streaming"):
        adapter.connect(make_profile(control_mode=ControlMode.COMMAND), {"namespace": "arm1"})
    assert not transport.subscribed
    assert not transport.advertised


def test_the_link_reports_its_own_class_not_the_profiles() -> None:
    adapter, _ = connected()
    assert adapter.control_mode() is ControlMode.STREAMING


def test_gains_send_the_point_at_once_and_ask_for_no_speed() -> None:
    """ADR-0045: the desired state is ``(q*, 0)`` from the tick the point arrives.

    A later time makes the controller interpolate a ramp -- and the damping term
    then acts on the slope of that ramp, which is a law neither Isaac nor a drive
    computes. A whole period at zero torque is the other way to get it wrong.
    """
    profile = make_profile(
        contract=make_contract(
            action=ActionSpec(
                type=ActionType.JOINT_POSITION, kp=[60.0] * 3, kd=[1.0] * 3
            )
        )
    )
    adapter, transport = connected(profile)
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.1, 0.2, 0.3)))

    point = transport.published[0].fields["points"][0]
    assert point["positions"] == [0.1, 0.2, 0.3]
    assert point["velocities"] == [0.0, 0.0, 0.0]
    assert point["time_from_start"] == 0.0


def test_command_of_the_wrong_kind_is_refused_not_converted() -> None:
    adapter, transport = connected()
    with pytest.raises(UnsupportedActionError, match="joint_velocity"):
        adapter.send_command(
            Command(type=ActionType.JOINT_VELOCITY, values=(1.0, 0.0, 0.0))
        )
    assert not transport.published


def test_out_of_limit_command_is_refused_not_clipped() -> None:
    adapter, transport = connected()
    with pytest.raises(SafetyViolationError):
        adapter.send_command(
            Command(type=ActionType.JOINT_POSITION, values=(99.0, 0.0, 0.0))
        )
    assert not transport.published


def test_wrong_command_width_is_refused() -> None:
    adapter, transport = connected()
    with pytest.raises(ValueError, match="суставов"):
        adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.0, 0.0)))
    assert not transport.published


# -- stopping --------------------------------------------------------------


def test_emergency_stop_holds_the_measured_position() -> None:
    profile = make_profile()
    transport = RecordingTransport()
    adapter = Ros2ControlAdapter(transport)
    adapter.connect(profile, {"namespace": "arm1"})
    transport.deliver_state(
        "/arm1/joint_states", stamp=1.0, names_=JOINTS, position=[0.1, 0.2, 0.3]
    )
    adapter.read_state()

    adapter.emergency_stop()
    assert transport.published[-1].fields == {"data": [0.1, 0.2, 0.3]}
    assert adapter.stopped


def test_emergency_stop_on_a_velocity_robot_commands_zero() -> None:
    profile = make_profile(
        contract=make_contract(3, action=ActionSpec(type=ActionType.JOINT_VELOCITY))
    )
    adapter, transport = connected(profile)
    adapter.read_state()
    adapter.emergency_stop()
    assert transport.published[-1].fields == {"data": [0.0, 0.0, 0.0]}


def test_disconnect_makes_the_adapter_unusable() -> None:
    adapter, _ = connected()
    adapter.disconnect()
    with pytest.raises(NotConnectedError):
        adapter.read_state()


# -- the transport itself --------------------------------------------------


def test_publishing_to_an_unadvertised_topic_is_an_error() -> None:
    transport = RecordingTransport()
    with pytest.raises(TransportError, match="необъявленный"):
        transport.publish(Publication("/x", names.ARRAY_MESSAGE, {"data": []}))
