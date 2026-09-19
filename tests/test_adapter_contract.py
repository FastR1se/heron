"""The platform's own adapters, held to the public contract suite (ADR-0056).

The suite itself is ``testing.adapter_contract``, shipped with the platform so that
a robot package runs it from its own tests. Only the built-in adapters are listed
here: a vendor's package is not the platform's to enumerate, and the G1 package
holds itself to the suite the way any user's package would.

The entries are not alike, and that is the point of the set. One is a robot made
of numbers streaming, the same robot planning its own motions, and one talks to a
robot over ROS 2 topics behind a transport that carries nothing (ADR-0014).
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import make_command_profile, make_profile
from heron.adapters.dummy import (
    DummyAdapter,
    LinkConfigurationError,
    MotionInProgressError,
    NotConnectedError,
    SafetyViolationError,
)
from heron.adapters.motion import Trapezoid
from heron.adapters.ros2_control import Ros2ControlAdapter
from heron.core.capabilities import Capability
from heron.core.enums import ActionType, MotionPhase
from heron.core.models import JointMax, Limits, RobotProfile
from heron.core.protocols import ControlAdapter
from heron.core.types import Command
from heron.ros2 import names
from heron.ros2.transport import GoalStatus, RecordingTransport
from heron.testing.adapter_contract import AdapterContract, Harness

STATES = 32
"""Readings queued for a link-based adapter, enough for any check of the suite."""

MAX_READINGS = 10_000


def _over_topics() -> ControlAdapter:
    """A ros2_control adapter over a transport that carries nothing."""
    return Ros2ControlAdapter(RecordingTransport())


def _prime_topics(adapter: Any, profile: RobotProfile) -> None:
    transport = adapter.transport
    assert isinstance(transport, RecordingTransport)
    dof = profile.contract.dof
    for index in range(STATES):
        transport.deliver_state(
            names.state_topic(""),
            stamp=index / profile.contract.state_rate_hz,
            names_=list(profile.contract.joint_order),
            position=[0.0] * dof,
            velocity=[0.0] * dof,
        )


def _until_idle(adapter: Any, profile: RobotProfile) -> None:
    """A robot made of numbers ends its motion by itself; read until it has."""
    for _ in range(MAX_READINGS):
        if adapter.read_state().motion is MotionPhase.IDLE:
            return
    raise AssertionError(f"движение не закончилось за {MAX_READINGS} показаний")


class TestDummyAdapter(AdapterContract):
    harness = Harness(plugin="dummy", make=DummyAdapter, profile=make_profile)


class TestDummyCommandAdapter(AdapterContract):
    harness = Harness(
        plugin="dummy", make=DummyAdapter, profile=make_command_profile, finish=_until_idle
    )


class TestRos2ControlAdapter(AdapterContract):
    harness = Harness(
        plugin="ros2_control", make=_over_topics, profile=make_profile, prime=_prime_topics
    )


def _over_an_action() -> ControlAdapter:
    """A ros2_control adapter whose trajectory controller serves its action (ADR-0060)."""
    action = names.command_action("", make_command_profile())
    assert action is not None
    return Ros2ControlAdapter(RecordingTransport(servers={action}))


def _goal_over(adapter: Any, profile: RobotProfile) -> None:
    """The controller reports the goal reached, and a state follows."""
    transport = adapter.transport
    assert isinstance(transport, RecordingTransport)
    action = names.command_action("", profile)
    assert action is not None
    transport.settle(action, GoalStatus.SUCCEEDED)
    transport.deliver_state(
        names.state_topic(""),
        stamp=STATES / profile.contract.state_rate_hz,
        names_=list(profile.contract.joint_order),
        position=[0.0] * profile.contract.dof,
        velocity=[0.0] * profile.contract.dof,
    )


class TestRos2ControlCommandAdapter(AdapterContract):
    harness = Harness(
        plugin="ros2_control",
        make=_over_an_action,
        profile=make_command_profile,
        prime=_prime_topics,
        finish=_goal_over,
    )


# -- dummy-specific behaviour ----------------------------------------------


def test_position_command_moves_the_robot() -> None:
    adapter = DummyAdapter()
    profile = make_profile()
    adapter.connect(profile, {})
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.1, 0.2, 0.3)))
    assert adapter.read_state().joint_position == (0.1, 0.2, 0.3)


def test_velocity_command_integrates() -> None:
    adapter = DummyAdapter()
    adapter.connect(make_profile(), {})
    adapter.send_command(Command(type=ActionType.JOINT_VELOCITY, values=(1.0, 0.0, 0.0)))
    position = adapter.read_state().joint_position
    assert position[0] == pytest.approx(0.01)  # 1 rad/s over one 100 Hz period


def test_out_of_limit_command_is_refused_not_clipped() -> None:
    """Silent repair is worse than a stop: the platform's whole position."""
    adapter = DummyAdapter()
    adapter.connect(make_profile(), {})
    with pytest.raises(SafetyViolationError):
        adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(99.0, 0.0, 0.0)))
    assert adapter.read_state().joint_position == (0.0, 0.0, 0.0)


def test_wrong_command_width_is_refused() -> None:
    adapter = DummyAdapter()
    adapter.connect(make_profile(dof=3), {})
    with pytest.raises(ValueError, match="суставов"):
        adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.0, 0.0)))


def test_link_may_offer_less_than_the_profile() -> None:
    """A simulated twin is allowed to be poorer than the real machine."""
    adapter = DummyAdapter()
    profile = make_profile(
        capabilities=frozenset({Capability("arm.joint_position"), Capability("gripper.binary")})
    )
    adapter.connect(profile, {"capabilities": ["arm.joint_position"]})
    assert adapter.capabilities() == frozenset({Capability("arm.joint_position")})


def test_reset_clears_emergency_stop() -> None:
    adapter = DummyAdapter()
    adapter.connect(make_profile(), {})
    adapter.emergency_stop()
    adapter.reset()
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.0, 0.0, 0.0)))


def test_disconnect_makes_the_adapter_unusable() -> None:
    adapter = DummyAdapter()
    adapter.connect(make_profile(), {})
    adapter.disconnect()
    with pytest.raises(NotConnectedError):
        adapter.read_state()


def test_limits_wider_than_the_contract_are_refused() -> None:
    """The validator catches this; the adapter must not index past the end."""
    from heron.core.models import JointRange

    adapter = DummyAdapter()
    profile = make_profile(
        dof=3, limits=Limits(joint_position=JointRange(min=[-1.0], max=[1.0]))
    )
    with pytest.raises(ValueError, match="лимиты"):
        adapter.connect(profile, {})


# -- dummy in the command class (ADR-0049, ADR-0052) ------------------------


def _command_link(scale: float = 1.0, **overrides: Any) -> tuple[DummyAdapter, RobotProfile]:
    adapter = DummyAdapter()
    profile = make_command_profile(**overrides)
    adapter.connect(profile, {"speed_scale": scale})
    return adapter, profile


def _goal(*values: float) -> Command:
    return Command(type=ActionType.JOINT_POSITION, values=values)


def _motion_time(adapter: DummyAdapter) -> float:
    started = adapter.time
    _until_idle(adapter, make_command_profile())
    return adapter.time - started


@pytest.mark.parametrize(
    ("distance", "duration"),
    [
        (0.0, 0.0),
        (0.25, 2 * 0.125**0.5),  # a triangle: peak sqrt(0.5) below the speed
        (2.0, 2.5),  # a trapezoid: two ramps of 0.5 and 1.5 at the speed
        (-2.0, 2.5),
    ],
)
def test_a_trapezoid_takes_the_time_its_bounds_allow(distance: float, duration: float) -> None:
    joint = Trapezoid(distance, speed=1.0, acceleration=2.0)
    assert joint.duration == pytest.approx(duration)
    assert joint.at(joint.duration) == (pytest.approx(distance), 0.0)
    assert abs(joint.at(joint.duration / 2)[1]) <= 1.0 + 1e-12


def test_a_command_link_moves_along_its_bounds_and_arrives() -> None:
    adapter, _ = _command_link()
    adapter.send_command(_goal(2.0, 0.0, -1.0))
    speeds = []
    while (state := adapter.read_state()).motion is not MotionPhase.IDLE:
        speeds.append(max(abs(v) for v in state.joint_velocity))
    assert max(speeds) <= 1.0 + 1e-9
    assert state.joint_position == pytest.approx((2.0, 0.0, -1.0))
    assert state.timestamp == pytest.approx(2.5, abs=0.05 + 1e-9)


@pytest.mark.parametrize("distance", [2.0, 0.25], ids=["trapezoid", "triangle"])
def test_the_speed_scale_stretches_the_motion_by_its_inverse(distance: float) -> None:
    """Velocity times s, acceleration times s squared: 1/s as long, ramp or not (ADR-0057)."""
    full, _ = _command_link()
    full.send_command(_goal(distance, 0.0, 0.0))
    slow, _ = _command_link(scale=0.5)
    slow.send_command(_goal(distance, 0.0, 0.0))
    # Each time is read to one state period (20 Hz), the doubled one to two.
    assert _motion_time(slow) == pytest.approx(2 * _motion_time(full), abs=0.15)


def test_a_second_command_during_a_motion_is_refused_and_later_accepted() -> None:
    adapter, _ = _command_link()
    adapter.send_command(_goal(0.5, 0.0, 0.0))
    with pytest.raises(MotionInProgressError):
        adapter.send_command(_goal(0.0, 0.0, 0.0))
    _until_idle(adapter, make_command_profile())
    adapter.send_command(_goal(0.0, 0.0, 0.0))


def test_an_emergency_stop_halts_the_motion_where_it_is() -> None:
    adapter, _ = _command_link()
    adapter.send_command(_goal(2.0, 0.0, 0.0))
    for _ in range(10):
        adapter.read_state()
    adapter.emergency_stop()
    first, second = adapter.read_state(), adapter.read_state()
    assert first.motion is second.motion is MotionPhase.HALTED
    assert first.joint_position == second.joint_position != (2.0, 0.0, 0.0)
    adapter.reset()
    assert adapter.read_state().motion is MotionPhase.IDLE


def test_disconnect_revokes_the_motion() -> None:
    adapter, profile = _command_link()
    adapter.send_command(_goal(2.0, 0.0, 0.0))
    adapter.disconnect()
    adapter.connect(profile, {})
    assert adapter.read_state().motion is MotionPhase.IDLE


@pytest.mark.parametrize("missing", ["joint_velocity", "joint_acceleration"])
def test_a_command_link_without_motion_bounds_refuses_to_connect(missing: str) -> None:
    limits = make_command_profile().limits.model_copy(update={missing: None})
    with pytest.raises(LinkConfigurationError, match=missing):
        _command_link(limits=limits)


def test_a_stream_is_not_slowed() -> None:
    with pytest.raises(LinkConfigurationError, match="потоковую"):
        DummyAdapter().connect(make_profile(), {"speed_scale": 0.5})


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.5])
def test_a_speed_scale_outside_its_range_is_refused(scale: float) -> None:
    with pytest.raises(LinkConfigurationError, match="0 < s"):
        _command_link(scale=scale)


def test_a_stream_reports_no_motion_phase() -> None:
    adapter = DummyAdapter()
    adapter.connect(make_profile(), {})
    assert adapter.read_state().motion is None


def test_acceleration_limits_wider_than_the_contract_are_refused() -> None:
    limits = make_command_profile().limits.model_copy(
        update={"joint_acceleration": JointMax(max=[1.0])}
    )
    with pytest.raises(ValueError, match="лимиты"):
        _command_link(limits=limits)
