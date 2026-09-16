"""The contract every adapter must pass (ARCHITECTURE.md, section 13; ADR-0056).

"Each adapter is run through a shared set of checks" -- this is that set, shipped
with the platform so that a robot package can hold its own adapter to it from its
own tests. It lived in the platform's tests until phase 5, and a vendor's package
joined it by an edit to those tests: a path nobody who installed the platform as a
package had.

A package subclasses ``AdapterContract`` and says, in a ``Harness``, what the
protocol cannot -- which robot the adapter drives, what link configuration it
needs, and how a running robot would give it something to read::

    class TestMyRobotAdapter(AdapterContract):
        harness = Harness(plugin="my_robot", make=..., profile=..., prime=...)

The checks know about no adapter. They ask what the protocol promises, and what
the class of link promises on top of it: a ``command`` link reports the phase of a
motion and holds it from the moment a command is accepted (ADR-0049).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from ..core.capabilities import Capability
from ..core.enums import ActionType, ControlMode, MotionPhase
from ..core.models import RobotProfile
from ..core.plugins import load
from ..core.protocols import CommandRefusedError, ControlAdapter
from ..core.types import Command, RobotState

LATCHED_READINGS = 3
"""Readings taken after an accepted command, before the robot reports its end."""


@dataclass(frozen=True)
class Harness:
    """One adapter, as the shared checks need it."""

    plugin: str
    """Entry-point name, exactly as a profile writes it."""

    make: Callable[[], ControlAdapter]
    profile: Callable[[], RobotProfile]
    config: dict[str, Any] = field(default_factory=dict)
    prime: Callable[[Any, RobotProfile], None] = lambda adapter, profile: None
    """Give the adapter something to read, before it connects, as a running robot would."""

    finish: Callable[[Any, RobotProfile], None] | None = None
    """Make the robot report that the accepted motion is over. Required of a
    ``command`` link: without it the latch of the motion phase is not checkable."""


class AdapterContract:
    """The shared checks. Subclass it as ``Test...`` and set ``harness``."""

    harness: ClassVar[Harness]

    # -- fixtures ------------------------------------------------------------

    @pytest.fixture
    def rig(self) -> Harness:
        rig = getattr(type(self), "harness", None)
        if not isinstance(rig, Harness):
            pytest.fail(
                f"{type(self).__name__}: объявите harness = Harness(...) — "
                f"как построить адаптер, какой профиль и что ему читать"
            )
        return rig

    @pytest.fixture
    def adapter(self, rig: Harness) -> ControlAdapter:
        return rig.make()

    @pytest.fixture
    def robot(self, rig: Harness) -> RobotProfile:
        return rig.profile()

    @pytest.fixture
    def connected(
        self, rig: Harness, adapter: ControlAdapter, robot: RobotProfile
    ) -> ControlAdapter:
        rig.prime(adapter, robot)
        adapter.connect(robot, dict(rig.config))
        return adapter

    @pytest.fixture
    def command_link(self, rig: Harness, connected: ControlAdapter) -> ControlAdapter:
        if connected.control_mode() is not ControlMode.COMMAND:
            pytest.skip("обязанность командного класса, связь потоковая")
        if rig.finish is None:
            pytest.fail(
                "командный адаптер: обвязке нужен finish — как робот сообщает, что движение "
                "закончено. Без него защёлка фазы движения не проверяема (ADR-0049)"
            )
        return connected

    # -- every adapter -------------------------------------------------------

    def test_satisfies_the_protocol(self, adapter: ControlAdapter) -> None:
        assert isinstance(adapter, ControlAdapter)

    def test_is_found_through_its_entry_point(
        self, rig: Harness, adapter: ControlAdapter
    ) -> None:
        """The platform has no private path to any adapter, its own or a vendor's."""
        assert type(adapter) is load("adapters", rig.plugin)

    def test_declares_whether_it_executes_a_speed_scale(self, adapter: ControlAdapter) -> None:
        """Read off the class before connecting, so declared there and a bool (ADR-0052)."""
        declared = getattr(type(adapter), "supports_speed_scale", None)
        assert isinstance(declared, bool), (
            f"{type(adapter).__name__}.supports_speed_scale: объявите True или False на классе; "
            f"получено {declared!r}"
        )

    def test_declares_capabilities(self, connected: ControlAdapter) -> None:
        caps = connected.capabilities()
        assert caps
        assert all(isinstance(c, Capability) for c in caps)

    def test_declares_control_mode(self, connected: ControlAdapter) -> None:
        assert isinstance(connected.control_mode(), ControlMode)

    def test_state_width_matches_contract(
        self, connected: ControlAdapter, robot: RobotProfile
    ) -> None:
        state = connected.read_state()
        assert len(state.joint_position) == robot.contract.dof
        assert len(state.joint_velocity) == robot.contract.dof

    def test_clock_advances(self, connected: ControlAdapter) -> None:
        first = connected.read_state().timestamp
        second = connected.read_state().timestamp
        assert second > first

    def test_emergency_stop_exists_and_stops(
        self, connected: ControlAdapter, robot: RobotProfile
    ) -> None:
        """Section 12 makes it mandatory; an adapter without it is not deployable."""
        connected.emergency_stop()
        with pytest.raises(CommandRefusedError):
            connected.send_command(_position(robot))

    def test_a_command_outside_the_envelope_is_refused_by_name(
        self, connected: ControlAdapter, robot: RobotProfile
    ) -> None:
        """The refusal the executor can tell from a defect, naming what it refused (ADR-0026)."""
        with pytest.raises(CommandRefusedError, match=f"{robot.contract.joint_order[0]} = 99"):
            connected.send_command(_position(robot, first=99.0))

    def test_refuses_use_before_connect(self, adapter: ControlAdapter) -> None:
        with pytest.raises(RuntimeError) as refusal:
            adapter.read_state()
        # NotImplementedError is a RuntimeError too: a method left as a TODO refuses
        # everything, before connect and after, and proves nothing about either.
        assert not isinstance(refusal.value, NotImplementedError), refusal.value

    def test_safety_limits_come_from_the_profile(
        self, connected: ControlAdapter, robot: RobotProfile
    ) -> None:
        assert connected.safety_limits() == robot.limits

    # -- a command link (ADR-0049) ---------------------------------------------

    def test_a_command_link_reports_the_motion_phase(self, command_link: ControlAdapter) -> None:
        assert isinstance(command_link.read_state().motion, MotionPhase)

    def test_the_motion_phase_holds_until_the_robot_reports_the_end(
        self, rig: Harness, command_link: ControlAdapter, robot: RobotProfile
    ) -> None:
        """A reading right after a command must not say "done" because the robot
        has not taken the motion yet."""
        command_link.send_command(_motion(command_link.read_state(), robot))
        phases = [command_link.read_state().motion for _ in range(LATCHED_READINGS)]
        assert MotionPhase.IDLE not in phases, phases

        assert rig.finish is not None
        rig.finish(command_link, robot)
        assert command_link.read_state().motion is MotionPhase.IDLE

    def test_a_command_during_a_motion_is_refused(
        self, command_link: ControlAdapter, robot: RobotProfile
    ) -> None:
        """One motion at a time: a second command would queue a point nobody planned."""
        command = _motion(command_link.read_state(), robot)
        command_link.send_command(command)
        with pytest.raises(CommandRefusedError):
            command_link.send_command(command)


def _position(robot: RobotProfile, first: float = 0.0) -> Command:
    values = (first, *(0.0,) * (robot.contract.dof - 1))
    return Command(type=ActionType.JOINT_POSITION, values=values)


def _motion(state: RobotState, robot: RobotProfile) -> Command:
    """A pose inside the envelope that differs from where the robot is.

    The first joint goes halfway towards its farther bound, so that the motion is
    not empty; the others stay where they are, taken onto the envelope.
    """
    bounds = robot.limits.joint_position
    if bounds is None:
        pytest.fail("профилю обвязки нужна рамка limits.joint_position: в ней строится цель")
    values = [
        min(max(value, low), high)
        for value, low, high in zip(state.joint_position, bounds.min, bounds.max, strict=True)
    ]
    first, low, high = values[0], bounds.min[0], bounds.max[0]
    farther = low if first - low > high - first else high
    values[0] = first + (farther - first) / 2
    return Command(type=ActionType.JOINT_POSITION, values=tuple(values))
