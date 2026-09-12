"""The executor: a period from the contract and a watchdog from section 12.

No sleeping happens here. The clock is injected precisely so that a test about
timing does not cost the time it is about (ADR-0015), and a watchdog that could
only be observed by waiting for it would be a watchdog nobody checks.
"""

from __future__ import annotations

import pytest

from conftest import make_profile
from heron.adapters.dummy import DummyAdapter
from heron.core.enums import ActionType, ControlMode, Status
from heron.core.models import Task
from heron.core.types import RobotState
from heron.sdk.executor import (
    WATCHDOG_FLOOR_S,
    WATCHDOG_PERIODS,
    Executor,
    ManualClock,
    watchdog_timeout,
)
from heron.sdk.runner import IncompatibleSkillError
from heron.skills.hold import HoldSkill
from heron.skills.move_joints import MoveJointsSkill


class FrozenAdapter:
    """A link that answers with one state forever: the failure a watchdog is for.

    Not a broken adapter -- a working one on the far side of a link that went
    quiet, which is what the state topic looks like when a controller dies.
    """

    def __init__(self, profile: object, stamp: float = 1.0) -> None:
        self.profile = profile
        self.stamp = stamp
        self.stopped = False
        self.commands: list[object] = []

    def connect(self, profile: object, config: dict[str, object]) -> None: ...

    def disconnect(self) -> None: ...

    def read_state(self) -> RobotState:
        return RobotState(timestamp=self.stamp, joint_position=(0.0, 0.0, 0.0))

    def send_command(self, cmd: object) -> None:
        self.commands.append(cmd)

    def capabilities(self) -> frozenset[object]:
        return frozenset(self.profile.capabilities)  # type: ignore[attr-defined]

    def control_mode(self) -> ControlMode:
        return ControlMode.STREAMING

    def safety_limits(self) -> object:
        return self.profile.limits  # type: ignore[attr-defined]

    def emergency_stop(self) -> None:
        self.stopped = True


# -- the period ------------------------------------------------------------


def test_loop_is_paced_by_the_contracts_command_rate() -> None:
    profile = make_profile()  # 50 Hz commands
    adapter = DummyAdapter()
    adapter.connect(profile, {})
    clock = ManualClock()

    result = Executor(adapter, profile, clock=clock).run(HoldSkill(), {"duration": 0.1})

    assert result.status is Status.SUCCESS
    # Every tick but the last one waited a full period: nothing else consumed time.
    assert clock.current == pytest.approx((result.ticks - 1) * 0.02)


def test_work_inside_a_cycle_comes_out_of_its_own_period() -> None:
    """The period counts from the start of the cycle, not from the end of the
    work, so a loop that spends half a period reading still commands at the
    contract's rate. Sleeping a full period after every cycle would run the
    robot slower than the contract says, by an amount nobody declared."""
    profile = make_profile()  # 50 Hz -> 20 ms period
    inner = DummyAdapter()
    inner.connect(profile, {})
    clock = ManualClock()

    class SlowLink:
        """The same adapter, but reading state off it costs 8 ms of wall time."""

        def __getattr__(self, item: str) -> object:
            return getattr(inner, item)

        def read_state(self) -> RobotState:
            clock.current += 0.008
            return inner.read_state()

    result = Executor(SlowLink(), profile, clock=clock).run(  # type: ignore[arg-type]
        HoldSkill(), {"duration": 0.1}
    )
    # Every completed cycle cost exactly one period, reading included; the last
    # one ended on a terminal status and never reached the wait.
    assert clock.current == pytest.approx((result.ticks - 1) * 0.02 + 0.008)


# -- the watchdog ----------------------------------------------------------


def test_watchdog_is_derived_from_the_state_rate() -> None:
    fast = make_profile()  # 100 Hz -> 0.1 s, below the floor
    assert watchdog_timeout(fast) == WATCHDOG_FLOOR_S

    slow = make_profile(contract=_rate(5.0))
    assert slow.contract.state_rate_hz == 5.0
    assert watchdog_timeout(slow) == pytest.approx(WATCHDOG_PERIODS / 5.0)


def _rate(state_rate_hz: float) -> object:
    from conftest import make_contract

    return make_contract(3, state_rate_hz=state_rate_hz, command_rate_hz=5.0)


def test_state_that_stops_arriving_stops_the_robot() -> None:
    profile = make_profile()
    adapter = FrozenAdapter(profile)
    clock = ManualClock()

    result = Executor(adapter, profile, clock=clock).run(  # type: ignore[arg-type]
        MoveJointsSkill(), {"target": [0.5, 0.0, 0.0], "duration": 1.0}
    )

    assert result.status is Status.FAILURE
    assert adapter.stopped, "робот должен быть остановлен раньше, чем сообщён отказ"
    assert result.reason is not None and "не обновлялось" in result.reason


def test_a_link_that_keeps_answering_never_trips_the_watchdog() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    result = Executor(adapter, profile, clock=ManualClock()).run(
        MoveJointsSkill(), {"target": [0.5, 0.0, 0.0], "duration": 0.2}
    )
    assert result.status is Status.SUCCESS
    assert result.reason is None


# -- tasks -----------------------------------------------------------------


def test_task_is_looked_up_by_name_in_the_library() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    result = Executor(adapter, profile, clock=ManualClock()).run_task(
        Task(skill="move_joints", target=[0.1, 0.1, 0.1], duration=0.1)
    )
    assert result.skill == "move_joints"
    assert result.ok


def test_a_sequence_stops_at_the_first_failure() -> None:
    """Carrying on would start the next motion from a pose nobody planned for."""
    profile = make_profile()
    adapter = FrozenAdapter(profile)  # its link is quiet: the first task will fail
    executor = Executor(adapter, profile, clock=ManualClock())  # type: ignore[arg-type]

    results = executor.run_tasks(
        [
            Task(skill="move_joints", target=[0.1, 0.1, 0.1], duration=1.0),
            Task(skill="move_joints", target=[0.2, 0.2, 0.2], duration=1.0),
        ]
    )
    assert len(results) == 1
    assert not results[0].ok


def test_a_sequence_that_succeeds_runs_every_task() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})
    executor = Executor(adapter, profile, clock=ManualClock())

    results = executor.run_tasks(
        [
            Task(skill="move_joints", target=[0.1, 0.1, 0.1], duration=0.1),
            Task(skill="move_joints", target=[0.2, 0.2, 0.2], duration=0.1),
        ]
    )
    assert [item.ok for item in results] == [True, True]
    assert adapter.read_state().joint_position == pytest.approx((0.2, 0.2, 0.2))


def test_commands_the_skill_sent_are_kept() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    result = Executor(adapter, profile, clock=ManualClock()).run(
        MoveJointsSkill(), {"target": [0.4, 0.0, 0.0], "duration": 0.1}
    )
    assert len(result.commands) == result.ticks
    assert all(cmd.type is ActionType.JOINT_POSITION for cmd in result.commands)


def test_an_incompatible_skill_never_reaches_the_first_tick() -> None:
    profile = make_profile(control_mode=ControlMode.COMMAND)
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    with pytest.raises(IncompatibleSkillError):
        Executor(adapter, profile, clock=ManualClock()).run(MoveJointsSkill(), {})
    assert not adapter.commands


def test_a_skill_that_never_terminates_ends_the_run() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    result = Executor(adapter, profile, clock=ManualClock()).run(
        HoldSkill(), {"duration": 1e9}, max_ticks=5
    )
    assert result.status is Status.FAILURE
    assert result.ticks == 5
    assert result.reason is not None and "тактов" in result.reason
