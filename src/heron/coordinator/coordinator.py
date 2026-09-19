"""The coordinator: hands tasks to robots and watches them end (ADR-P8C, ADR-P8E).

It decides nothing the plan has not decided. Which robot does a task was fixed
before the run (ADR-P8B); what is left is *when*: a task starts once the tasks it
waits for have succeeded, its robot is free and that robot's tasks written above
it are done. The first task that does not succeed stops the system: every running
task is cancelled -- its executor stops the robot first -- and nothing new starts.

It knows robots only as names it sends messages to and events it receives, so the
same loop runs over sockets to processes and over a list in a test.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol

from . import protocol as p
from .plan import MissionPlan, PlannedTask

POLL_S: Final = 0.2
"""How often a quiet wait looks at whether a robot's process is still there."""


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    """Stopped by the coordinator because another task failed (ADR-P8E)."""
    NOT_STARTED = "not_started"


class MissionStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class TaskOutcome:
    """How one task of the mission went."""

    task: str
    robot: str
    skill: str
    state: TaskState = TaskState.PENDING
    reason: str | None = None
    ticks: int = 0
    commands: int = 0
    started: float | None = None
    """Seconds of the coordinator's clock since the mission started."""
    finished: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "robot": self.robot,
            "skill": self.skill,
            "state": str(self.state),
            "reason": self.reason,
            "ticks": self.ticks,
            "commands": self.commands,
            "started": self.started,
            "finished": self.finished,
        }


@dataclass(frozen=True, slots=True)
class MissionResult:
    status: MissionStatus
    tasks: list[TaskOutcome]
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is MissionStatus.SUCCESS

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "reason": self.reason,
            "tasks": [item.as_dict() for item in self.tasks],
        }


@dataclass(frozen=True, slots=True)
class Event:
    """Something a robot said, or ``lost`` when its channel ended."""

    robot: str
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)


class EventSource(Protocol):
    def get(self, timeout: float) -> Event | None:
        """The next event, or ``None`` after ``timeout`` seconds of silence."""


Send = Callable[[str, Mapping[str, Any]], None]
Watch = Callable[[], Sequence[tuple[str, str]]]
"""Robots whose process is gone, with why; asked whenever the wait is quiet."""


class Coordinator:
    """Runs one plan to a result."""

    def __init__(
        self,
        plan: MissionPlan,
        send: Send,
        events: EventSource,
        *,
        clock: Callable[[], float] = time.monotonic,
        watch: Watch | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        poll_s: float = POLL_S,
    ) -> None:
        self.plan = plan
        self._send = send
        self._events = events
        self._clock = clock
        self._watch = watch
        self._on_event = on_event
        self._poll_s = poll_s
        self._t0 = clock()
        self._robots = [robot.id for robot in plan.robots]
        self._tasks: dict[str, PlannedTask] = {task.id: task for task in plan.tasks}
        self.outcomes: dict[str, TaskOutcome] = {
            task.id: TaskOutcome(task=task.id, robot=task.robot, skill=task.skill)
            for task in plan.tasks
        }
        self._ready: set[str] = set()
        self._gone: set[str] = set()
        self._cancelled: set[str] = set()
        self._stop: tuple[MissionStatus, str] | None = None

    # -- running ------------------------------------------------------------

    def run(self) -> MissionResult:
        """Wait for every robot, run the tasks, stop on the first failure.

        Ctrl+C is a stop like a failure: running tasks are cancelled and waited for.
        A second Ctrl+C during that wait goes on as it came -- the robot processes
        then lose their channel and stop themselves (ADR-P8D).
        """
        self._emit("mission_started", robots=list(self._robots), tasks=len(self._tasks))
        try:
            try:
                self._loop()
            except KeyboardInterrupt:
                self._halt(MissionStatus.INTERRUPTED, "прервано оператором (Ctrl+C)")
                self._loop()
        finally:
            for robot in self._robots:
                self._deliver(robot, p.message(p.END))
        result = self._result()
        self._emit("mission_finished", status=str(result.status), reason=result.reason)
        return result

    def _loop(self) -> None:
        while True:
            if self._stop is None and self._all_ready():
                self._start_ready()
                if not self._running() and not self._pending():
                    return
                if not self._running():
                    self._halt(MissionStatus.FAILURE, "задачи не могут начаться")
            if self._stop is not None and not self._running():
                return
            event = self._events.get(self._poll_s)
            if event is None:
                for robot, reason in self._watch() if self._watch is not None else ():
                    self._handle(Event(robot, p.LOST, {"reason": reason}))
                continue
            self._handle(event)

    # -- events -------------------------------------------------------------

    def _handle(self, event: Event) -> None:
        robot, kind, data = event.robot, event.kind, event.data
        if robot not in self._robots or robot in self._gone:
            return
        if kind == p.HELLO:
            self._emit("robot_connected", robot=robot)
        elif kind == p.READY:
            self._ready.add(robot)
            self._emit("robot_ready", robot=robot)
        elif kind == p.REFUSED:
            reason = str(data.get("reason", ""))
            self._emit("robot_refused", robot=robot, reason=reason)
            self._halt(MissionStatus.FAILURE, f"робот {robot} не готов: {reason}")
        elif kind == p.FINISHED:
            self._finished(robot, data)
        elif kind == p.LOST:
            self._lost(robot, str(data.get("reason", "связь закрыта")))

    def _finished(self, robot: str, data: Mapping[str, Any]) -> None:
        task = str(data.get("task", ""))
        outcome = self.outcomes.get(task)
        if outcome is None or outcome.robot != robot or outcome.state is not TaskState.RUNNING:
            return
        outcome.finished = self._now()
        outcome.ticks = int(data.get("ticks", 0))
        outcome.commands = int(data.get("commands", 0))
        reason = data.get("reason")
        outcome.reason = str(reason) if reason else None
        if data.get("status") == "success":
            outcome.state = TaskState.SUCCEEDED
        elif task in self._cancelled:
            outcome.state = TaskState.CANCELLED
        else:
            outcome.state = TaskState.FAILED
        self._emit(
            "task_finished",
            task=task,
            robot=robot,
            state=str(outcome.state),
            reason=outcome.reason,
            ticks=outcome.ticks,
            commands=outcome.commands,
        )
        if outcome.state is TaskState.FAILED:
            self._halt(
                MissionStatus.FAILURE,
                f"задача {task} ({robot}, {outcome.skill}): {outcome.reason or 'отказ'}",
            )

    def _lost(self, robot: str, reason: str) -> None:
        self._gone.add(robot)
        self._emit("robot_lost", robot=robot, reason=reason)
        running = [
            o for o in self.outcomes.values() if o.robot == robot and o.state is TaskState.RUNNING
        ]
        for outcome in running:
            outcome.state = TaskState.FAILED
            outcome.finished = self._now()
            outcome.reason = f"связь с роботом потеряна: {reason}"
            self._emit(
                "task_finished",
                task=outcome.task,
                robot=robot,
                state=str(outcome.state),
                reason=outcome.reason,
            )
        owes = running or any(
            o.robot == robot and o.state is TaskState.PENDING for o in self.outcomes.values()
        )
        if owes or robot not in self._ready:
            self._halt(MissionStatus.FAILURE, f"связь с роботом {robot} потеряна: {reason}")

    # -- decisions ----------------------------------------------------------

    def _start_ready(self) -> None:
        for task in self.plan.tasks:
            outcome = self.outcomes[task.id]
            if outcome.state is not TaskState.PENDING or not self._may_start(task):
                continue
            outcome.state = TaskState.RUNNING
            outcome.started = self._now()
            self._emit("task_started", task=task.id, robot=task.robot, skill=task.skill)
            if not self._deliver(task.robot, p.message(p.START, task=task.id)):
                self._lost(task.robot, "команда start не доставлена")
                return

    def _may_start(self, task: PlannedTask) -> bool:
        if any(self.outcomes[name].state is not TaskState.SUCCEEDED for name in task.after):
            return False
        for other in self.plan.tasks:
            if other.id == task.id:
                return True
            done = self.outcomes[other.id].state is TaskState.SUCCEEDED
            if other.robot == task.robot and not done:
                return False
        return True

    def _halt(self, status: MissionStatus, reason: str) -> None:
        """Stop the system once: cancel what runs, start nothing more (ADR-P8E)."""
        if self._stop is not None:
            return
        self._stop = (status, reason)
        self._emit("mission_stopping", status=str(status), reason=reason)
        for outcome in self.outcomes.values():
            if outcome.state is TaskState.RUNNING:
                self._cancelled.add(outcome.task)
                why = f"отменена: {reason}"
                if not self._deliver(outcome.robot, p.message(p.CANCEL, reason=why)):
                    self._lost(outcome.robot, "команда cancel не доставлена")

    # -- helpers ------------------------------------------------------------

    def _deliver(self, robot: str, message: Mapping[str, Any]) -> bool:
        if robot in self._gone:
            return False
        try:
            self._send(robot, message)
        except (OSError, KeyError):
            return False
        return True

    def _all_ready(self) -> bool:
        return all(robot in self._ready for robot in self._robots)

    def _running(self) -> bool:
        return any(o.state is TaskState.RUNNING for o in self.outcomes.values())

    def _pending(self) -> bool:
        return any(o.state is TaskState.PENDING for o in self.outcomes.values())

    def _now(self) -> float:
        return self._clock() - self._t0

    def _result(self) -> MissionResult:
        status, reason = self._stop if self._stop is not None else (MissionStatus.SUCCESS, None)
        for outcome in self.outcomes.values():
            if outcome.state is TaskState.PENDING:
                outcome.state = TaskState.NOT_STARTED
                outcome.reason = f"не начата: {reason}" if reason else None
        return MissionResult(status=status, tasks=list(self.outcomes.values()), reason=reason)

    def _emit(self, kind: str, **fields: Any) -> None:
        if self._on_event is not None:
            self._on_event(
                {"t": round(self._now(), 6), "utc_ns": time.time_ns(), "event": kind, **fields}
            )
