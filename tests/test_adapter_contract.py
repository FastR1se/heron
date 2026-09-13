"""The contract tests every adapter must pass (ARCHITECTURE.md, section 13).

"Each adapter is run through a shared set of checks" -- this is that set. It is
written against the protocol, not against ``DummyAdapter``, so a vendor adapter
can be added to ``ADAPTERS`` and be held to the same standard. Phase 1 left the
list with one entry and claimed the second would cost nothing; phase 3 added it,
and it did.

The two entries are not alike, and that is the point of the set. One is a robot
made of numbers with no outside at all; the other talks to a robot over ROS 2
topics, and appears here behind a transport that carries nothing (ADR-0014). The
checks below know about neither -- they ask only what the protocol promises.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from conftest import make_profile
from heron.adapters.dummy import DummyAdapter, NotConnectedError, SafetyViolationError
from heron.adapters.ros2_control import Ros2ControlAdapter
from heron.core.capabilities import Capability
from heron.core.enums import ActionType, ControlMode
from heron.core.models import RobotProfile
from heron.core.protocols import CommandRefusedError, ControlAdapter
from heron.core.types import Command
from heron.ros2 import names
from heron.ros2.transport import RecordingTransport

STATES = 32
"""Readings queued for a link-based adapter, enough for any check below."""


def _over_topics() -> ControlAdapter:
    """A ros2_control adapter over a transport that carries nothing.

    The queued readings are the part a simulator would supply. Nothing here
    integrates them or moves anything: the checks in this file are about what
    the protocol promises, and none of them is about motion.
    """
    return Ros2ControlAdapter(RecordingTransport())


ADAPTERS: list[Callable[[], ControlAdapter]] = [DummyAdapter, _over_topics]


@pytest.fixture(params=ADAPTERS)
def adapter(request: pytest.FixtureRequest) -> ControlAdapter:
    return request.param()


@pytest.fixture
def connected(adapter: ControlAdapter, profile: RobotProfile) -> ControlAdapter:
    adapter.connect(profile, {})
    _prime(adapter, profile)
    return adapter


def _prime(adapter: ControlAdapter, profile: RobotProfile) -> None:
    """Give a topic-driven adapter something to read, as a running world would."""
    transport = getattr(adapter, "transport", None)
    if not isinstance(transport, RecordingTransport):
        return
    dof = profile.contract.dof
    for index in range(STATES):
        transport.deliver_state(
            names.state_topic(""),
            stamp=index / profile.contract.state_rate_hz,
            names_=list(profile.contract.joint_order),
            position=[0.0] * dof,
            velocity=[0.0] * dof,
        )


# -- the shared contract ---------------------------------------------------


def test_satisfies_the_protocol(adapter: ControlAdapter) -> None:
    assert isinstance(adapter, ControlAdapter)


def test_declares_capabilities(connected: ControlAdapter) -> None:
    caps = connected.capabilities()
    assert caps
    assert all(isinstance(c, Capability) for c in caps)


def test_declares_control_mode(connected: ControlAdapter) -> None:
    assert isinstance(connected.control_mode(), ControlMode)


def test_state_width_matches_contract(connected: ControlAdapter, profile: RobotProfile) -> None:
    state = connected.read_state()
    assert len(state.joint_position) == profile.contract.dof
    assert len(state.joint_velocity) == profile.contract.dof


def test_clock_advances(connected: ControlAdapter) -> None:
    first = connected.read_state().timestamp
    second = connected.read_state().timestamp
    assert second > first


def test_emergency_stop_exists_and_stops(connected: ControlAdapter) -> None:
    """Section 12 makes it mandatory; an adapter without it is not deployable."""
    connected.emergency_stop()
    with pytest.raises(CommandRefusedError):
        connected.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.0, 0.0, 0.0)))


def test_a_command_outside_the_envelope_is_refused_by_name(connected: ControlAdapter) -> None:
    """The refusal the executor can tell from a defect, naming what it refused (ADR-0026)."""
    with pytest.raises(CommandRefusedError, match="joint0 = 99"):
        connected.send_command(Command(type=ActionType.JOINT_POSITION, values=(99.0, 0.0, 0.0)))


def test_refuses_use_before_connect(adapter: ControlAdapter) -> None:
    with pytest.raises(RuntimeError):
        adapter.read_state()


def test_safety_limits_come_from_the_profile(
    connected: ControlAdapter, profile: RobotProfile
) -> None:
    assert connected.safety_limits() == profile.limits


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
    from heron.core.models import JointRange, Limits

    adapter = DummyAdapter()
    profile = make_profile(
        dof=3, limits=Limits(joint_position=JointRange(min=[-1.0], max=[1.0]))
    )
    with pytest.raises(ValueError, match="лимиты"):
        adapter.connect(profile, {})
