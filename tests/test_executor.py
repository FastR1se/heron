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


def test_waiting_for_the_link_is_not_silence() -> None:
    """Regression, found by the first humanoid run: the watchdog tripped after
    two ticks with "1.009 s of silence".

    The first read waited about a second for the link to come up (ADR-0016), the
    second found no new message within its cycle, and the silence was counted
    from the start of the first cycle -- that is, from before the link existed."""
    profile = make_profile()  # 100 Hz state -> 0.5 s watchdog
    inner = DummyAdapter()
    inner.connect(profile, {})
    clock = ManualClock()

    class LateLink:
        """Takes a second to come up, then misses one cycle."""

        def __init__(self) -> None:
            self.reads = 0
            self.last: RobotState | None = None

        def __getattr__(self, item: str) -> object:
            return getattr(inner, item)

        def read_state(self) -> RobotState:
            self.reads += 1
            if self.reads == 1:
                clock.current += 1.0
            if self.reads == 2 and self.last is not None:
                return self.last
            self.last = inner.read_state()
            return self.last

    result = Executor(LateLink(), profile, clock=clock).run(  # type: ignore[arg-type]
        MoveJointsSkill(), {"target": [0.5, 0.0, 0.0], "duration": 0.2}
    )
    assert result.status is Status.SUCCESS, result.reason


def test_a_link_that_keeps_answering_never_trips_the_watchdog() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    result = Executor(adapter, profile, clock=ManualClock()).run(
        MoveJointsSkill(), {"target": [0.5, 0.0, 0.0], "duration": 0.2}
    )
    assert result.status is Status.SUCCESS
    assert result.reason is None


# -- refusals --------------------------------------------------------------


def test_a_refused_command_stops_the_robot_and_ends_the_skill() -> None:
    """The second humanoid run: a fallen robot, a start pose outside the envelope,
    and a refusal that came out of the command as a traceback, with no stop."""
    profile = make_profile()  # limits ±2.0
    adapter = DummyAdapter()
    adapter.connect(profile, {"initial_position": [2.5, 0.0, 0.0]})
    stopped: list[bool] = []
    real_stop = adapter.emergency_stop
    adapter.emergency_stop = lambda: (stopped.append(True), real_stop())  # type: ignore[method-assign]

    result = Executor(adapter, profile, clock=ManualClock()).run(
        MoveJointsSkill(), {"target": [0.0, 0.0, 0.0], "duration": 1.0}
    )

    assert result.status is Status.FAILURE
    assert stopped == [True]
    assert result.reason is not None
    assert "joint0 = 2.5 при [-2, 2]" in result.reason
    assert not result.commands, "отказанная команда не отправлена"


def test_a_defect_during_a_tick_stops_the_robot_and_stays_a_defect() -> None:
    profile = make_profile()
    adapter = FrozenAdapter(profile)

    class Broken(MoveJointsSkill):
        def on_tick(self, state: RobotState) -> Status:
            raise KeyError("ошибка в навыке")

    with pytest.raises(KeyError, match="ошибка в навыке"):
        Executor(adapter, profile, clock=ManualClock()).run(  # type: ignore[arg-type]
            Broken(), {"target": [0.1, 0.0, 0.0], "duration": 1.0}
        )
    assert adapter.stopped


def test_a_stop_that_fails_does_not_hide_the_refusal() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {"initial_position": [2.5, 0.0, 0.0]})

    def failing_stop() -> None:
        raise RuntimeError("связь потеряна")

    adapter.emergency_stop = failing_stop  # type: ignore[method-assign]
    result = Executor(adapter, profile, clock=ManualClock()).run(
        MoveJointsSkill(), {"target": [0.0, 0.0, 0.0], "duration": 1.0}
    )
    assert result.reason is not None
    assert result.reason.index("отказал") < result.reason.index("останов не выполнен")


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
