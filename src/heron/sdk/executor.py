"""Running a task on one robot.

Phase 1 had a loop that read state, ticked a skill and stopped on a terminal
status; it ran as fast as the adapter answered, which is right for a robot made
of numbers and wrong for everything else. Two things are added here and nothing
else is (ADR-0015).

*A period.* The contract already says how often this robot is commanded, so the
loop is paced by ``command_rate_hz`` rather than by how quickly the link
replies. A skill that emitted commands as fast as the network allowed would be
tuned to the machine it was debugged on, and the same skill would behave
differently in simulation and on hardware -- which invariant 3 forbids in the
one place it matters most.

*A watchdog.* Section 12: state missing longer than a threshold stops the
robot. The threshold is derived from ``state_rate_hz`` and not declared
anywhere, because the profile already says how often state is supposed to
arrive; a field for "how late is too late" would describe the same quantity
twice.

Both clocks that matter are visible here and are not the same clock. The world
clock is whatever the adapter stamps its readings with -- simulated time, which
can run faster or slower than the wall. The wall clock is what the watchdog
measures against, because "the state stopped arriving" is a fact about the link
and not about the world.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from ..core.enums import Status
from ..core.models import RobotProfile, SkillSpec, Task
from ..core.protocols import CommandRefusedError, ControlAdapter, Skill
from ..core.types import Command, RobotState
from ..skills.library import SkillLibrary
from ..store.policies import PolicyStore
from .runner import AdapterContext, IncompatibleSkillError, check_runnable

WATCHDOG_PERIODS: Final = 10
"""Missed state periods after which the link counts as lost, not merely late.

Ten rather than one: a single late message is scheduling, and stopping a robot
on scheduling noise would make the watchdog the most common cause of a failed
run."""

WATCHDOG_FLOOR_S: Final = 0.5
"""Lower bound on the watchdog, whatever the state rate.

At 100 Hz ten periods is a tenth of a second, which a loaded machine loses to a
garbage collection. The floor says what the ratio cannot: below half a second,
silence is not yet evidence."""

MAX_TICKS: Final = 1_000_000
"""Upper bound on one skill, so that a skill which never terminates ends the
run instead of owning the process."""


@runtime_checkable
class Clock(Protocol):
    """Wall time and the ability to wait, injected so that tests do neither."""

    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class WallClock:
    """The real one."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


@dataclass(slots=True)
class ManualClock:
    """A clock that only moves when it is told to, or when it is slept on.

    Not a test double bolted onto production code: the executor needs *some*
    clock, and a run without a wall -- a replay, a faster-than-real-time
    rollout -- needs one that is not the wall's.
    """

    current: float = 0.0

    def now(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += max(seconds, 0.0)


@dataclass(slots=True)
class FrozenClock:
    """A clock that never advances: the loop then runs as fast as it can.

    What phase 1's runner did, named rather than implied, so that the two
    behaviours are one loop with two clocks instead of two loops.
    """

    def now(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        return None


@runtime_checkable
class Recorder(Protocol):
    """Where a run goes while it happens (ADR-0038).

    ``wall`` is the executor's clock. ``world`` of a command is the stamp of the
    state it was computed from.
    """

    def state(self, wall: float, state: RobotState) -> None: ...

    def command(self, wall: float, world: float, command: Command) -> None: ...


class Cancellation:
    """A request from outside the loop to end the skill that is running (ADR-P8E).

    Checked at the start of every tick and answered the way the watchdog is: the
    robot is stopped first, then the skill ends with the reason. Asking twice keeps
    the first reason -- it is the one a human has to see. Safe to request from
    another thread: a coordinator's channel is read by one.
    """

    def __init__(self) -> None:
        self._reason: str | None = None
        self._lock = threading.Lock()

    def request(self, reason: str) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason

    @property
    def reason(self) -> str | None:
        """Why the skill is to end, or ``None`` while nobody asked."""
        return self._reason


@dataclass(frozen=True, slots=True)
class RunResult:
    """What one task did."""

    skill: str
    status: Status
    ticks: int
    states: list[RobotState] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    reason: str | None = None
    """Why it ended, when the status alone does not say."""

    @property
    def ok(self) -> bool:
        return self.status is Status.SUCCESS


def _name_of(skill: Skill) -> str:
    """What to call a skill in a report when the caller did not say."""
    spec = getattr(skill, "spec", None)
    return spec.name if isinstance(spec, SkillSpec) else "<навык без объявления>"


def watchdog_timeout(profile: RobotProfile) -> float:
    """How long state may be missing before the robot is stopped."""
    return max(WATCHDOG_PERIODS / profile.contract.state_rate_hz, WATCHDOG_FLOOR_S)


class Executor:
    """Runs skills on one robot, through one link, at the contract's rate.

    One robot, because that is what a robot container holds (section 4).
    Distributing tasks across robots by capability is the coordinator's job and
    a separate deployment unit; putting it here would make every robot able to
    command every other one.
    """

    def __init__(
        self,
        adapter: ControlAdapter,
        profile: RobotProfile,
        *,
        clock: Clock | None = None,
        library: SkillLibrary | None = None,
        recorder: Recorder | None = None,
        store: PolicyStore | None = None,
    ) -> None:
        self.adapter = adapter
        self.profile = profile
        self.clock: Clock = clock if clock is not None else WallClock()
        self.library = library if library is not None else SkillLibrary()
        self.recorder = recorder
        self.store = store
        """Where a skill bound to a policy is looked for; the platform's store when unset."""
        self.period = 1.0 / profile.contract.command_rate_hz
        self.watchdog = watchdog_timeout(profile)

    # -- running ------------------------------------------------------------

    def run_task(self, task: Task, *, cancel: Cancellation | None = None) -> RunResult:
        """One task: the skill as this robot's profile binds it, run against its goal.

        Code or a policy -- the profile's ``skills`` says which, and nothing here asks
        (invariant 4, ADR-0046). A skill the profile does not declare is refused.
        """
        skill = self.library.create_for(self.profile, task.skill, store=self.store)
        return self.run(skill, task.goal, name=task.skill, cancel=cancel)  # type: ignore[arg-type]

    def run_tasks(self, tasks: Sequence[Task]) -> list[RunResult]:
        """Tasks in order, stopping at the first one that does not succeed.

        Carrying on past a failure would mean starting a motion from a pose
        nobody planned for, which is worse than stopping short of the goal.
        """
        results: list[RunResult] = []
        for task in tasks:
            result = self.run_task(task)
            results.append(result)
            if not result.ok:
                break
        return results

    def run(
        self,
        skill: Skill,
        goal: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        max_ticks: int = MAX_TICKS,
        cancel: Cancellation | None = None,
    ) -> RunResult:
        """Run one skill to a terminal status.

        The compatibility check happens before the first tick, not during it:
        a skill the robot cannot run is a validator error, not a failure
        discovered halfway through a motion (invariant 6).

        ``cancel`` is how someone outside the loop -- a coordinator whose mission
        failed elsewhere -- ends the skill: at the next tick, robot stopped first
        (ADR-P8E).
        """
        label = name if name is not None else _name_of(skill)
        report = check_runnable(skill, self.profile, self.adapter)
        if not report.ok:
            raise IncompatibleSkillError(report)

        ctx = AdapterContext(profile_=self.profile, adapter=self.adapter)
        recorder = self.recorder
        if recorder is not None:
            clock = self.clock
            ctx.on_sent = lambda cmd: recorder.command(clock.now(), ctx.current_time, cmd)
        skill.start(dict(goal or {}), ctx)

        seen: list[RobotState] = []
        fresh_at = self.clock.now()
        last_stamp: float | None = None

        for tick in range(1, max_ticks + 1):
            if cancel is not None and (why := cancel.reason) is not None:
                return self._halt(skill, label, tick, seen, ctx, f"отменено: {why}")
            started = self.clock.now()
            state = self.adapter.read_state()
            read_at = self.clock.now()
            seen.append(state)
            if recorder is not None:
                recorder.state(read_at, state)

            # Freshness counts from the moment a new reading was in hand, not
            # from the start of the cycle that fetched it. The first read may
            # spend a second waiting for the link to come up (ADR-0016), and
            # counting that wait as silence trips the watchdog on a link that has
            # only just started talking -- the first humanoid run did exactly that.
            if last_stamp is None or state.timestamp != last_stamp:
                last_stamp, fresh_at = state.timestamp, read_at
            elif read_at - fresh_at > self.watchdog:
                return self._trip(skill, label, tick, seen, ctx, read_at - fresh_at)

            ctx.current_time = state.timestamp
            try:
                status = skill.tick(state)
            except CommandRefusedError as exc:
                # The adapter did its job; the robot still has to be stopped
                # before anyone is told (ADR-0026).
                return self._halt(
                    skill, label, tick, seen, ctx, f"адаптер отказал в команде: {exc}"
                )
            except BaseException as exc:
                # A defect, or Ctrl+C. Either way the robot is stopped first, and
                # the exception goes on as it came.
                self._stop(skill, exc)
                raise
            if status is not Status.RUNNING:
                return RunResult(
                    skill=label, status=status, ticks=tick, states=seen, commands=ctx.sent
                )
            self._wait(started)

        skill.cancel()
        return RunResult(
            skill=label,
            status=Status.FAILURE,
            ticks=max_ticks,
            states=seen,
            commands=ctx.sent,
            reason=f"навык не завершился за {max_ticks} тактов",
        )

    # -- internals ----------------------------------------------------------

    def _wait(self, started: float) -> None:
        """Hold the command period, counting from the start of the cycle.

        From the start and not from here, so that time spent reading state and
        thinking is part of the period rather than added to it. A loop that
        sleeps a full period after every cycle runs slower than the contract
        says, and by an amount nobody declared.
        """
        self.clock.sleep(self.period - (self.clock.now() - started))

    def _trip(
        self,
        skill: Skill,
        label: str,
        tick: int,
        seen: list[RobotState],
        ctx: AdapterContext,
        silence: float,
    ) -> RunResult:
        """The watchdog: state stopped arriving (section 12)."""
        reason = (
            f"состояние не обновлялось {silence:.3f} с при пороге "
            f"{self.watchdog:.3f} с ({WATCHDOG_PERIODS} периодов при "
            f"{self.profile.contract.state_rate_hz:g} Гц)"
        )
        return self._halt(skill, label, tick, seen, ctx, reason)

    def _stop(self, skill: Skill, cause: BaseException | None = None) -> str | None:
        """Stop the robot and cancel the skill. Returns why stopping failed, if it did.

        A stop that fails does not replace what caused it: the cause is what a
        human has to see, and the failed stop is added to it, not put in its place.
        """
        try:
            self.adapter.emergency_stop()
        except Exception as exc:
            failure = f"аварийный останов не выполнен: {exc}"
            if cause is not None:
                cause.add_note(failure)
            return failure
        finally:
            skill.cancel()
        return None

    def _halt(
        self,
        skill: Skill,
        label: str,
        tick: int,
        seen: list[RobotState],
        ctx: AdapterContext,
        reason: str,
    ) -> RunResult:
        """Stop the robot first, report afterwards (section 12)."""
        failure = self._stop(skill)
        if failure is not None:
            reason = f"{reason}; {failure}"
        return RunResult(
            skill=label,
            status=Status.FAILURE,
            ticks=tick,
            states=seen,
            commands=ctx.sent,
            reason=reason,
        )


def run_skill(
    skill: Skill,
    adapter: ControlAdapter,
    profile: RobotProfile,
    goal: dict[str, Any] | None = None,
    *,
    max_ticks: int = 10_000,
) -> tuple[Status, list[RobotState]]:
    """Run one skill as fast as the link answers, with no pacing.

    The form phase 1 had, kept because it is what a robot made of numbers wants:
    a ``dummy`` twin advances its own clock on every read, so pacing it against
    a wall clock would only make a test slow. It is the same loop under a clock
    that does not move (``FrozenClock``), not a second implementation.
    """
    result = Executor(adapter, profile, clock=FrozenClock()).run(
        skill, goal, max_ticks=max_ticks
    )
    return result.status, result.states
