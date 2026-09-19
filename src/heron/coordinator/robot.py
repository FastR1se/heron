"""A robot's side of a mission: its executor, driven by the coordinator (ADR-P8D).

The link to the robot is already up when this starts, and every skill of the
robot's tasks is already built: what could be refused was refused before ``ready``
was sent. What remains is to do what the coordinator says, one task at a time, on
the one executor this link has (ADR-0016), and to say how each ended.

A ``cancel`` is read by a thread of its own while a task runs, and handed to the
executor as a ``Cancellation``: the robot is stopped at the next tick, then the
task reports (ADR-P8E). Losing the coordinator is the same cancel -- a robot does
not keep moving for a mission nobody is watching.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.enums import Status
from ..sdk.executor import Cancellation, Executor, RunResult
from . import protocol as p

LOST_COORDINATOR = "связь с координатором потеряна"


class Recording(Protocol):
    """What the executor records into for one task, and closes afterwards (ADR-P8F)."""

    def state(self, wall: float, state: Any) -> None: ...

    def command(self, wall: float, world: float, command: Any) -> None: ...

    def close(self, result: RunResult | None, *, error: BaseException | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedTask:
    """A task of this robot with its skill built before the link was (ADR-0047)."""

    id: str
    skill_name: str
    skill: Any
    goal: dict[str, Any] = field(default_factory=dict)


class RobotSide:
    """Serves one robot's tasks over one channel."""

    def __init__(
        self,
        robot: str,
        channel: p.Channel,
        executor: Executor,
        tasks: Mapping[str, PreparedTask],
        *,
        recording_for: Callable[[PreparedTask], Recording | None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.robot = robot
        self.channel = channel
        self.executor = executor
        self.tasks = dict(tasks)
        self.recording_for = recording_for
        self.log = log if log is not None else (lambda line: None)
        self.cancel = Cancellation()
        self.results: dict[str, RunResult | None] = {}
        self._inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def serve(self) -> bool:
        """Until ``end`` or a lost coordinator. True when every task run here succeeded."""
        threading.Thread(target=self._read, name="coordinator-channel", daemon=True).start()
        ok = True
        while True:
            message = self._inbox.get()
            if message is None:
                return False
            if message["type"] == p.END:
                return ok
            if message["type"] != p.START:
                continue
            outcome = self._run(str(message.get("task", "")))
            ok = ok and outcome["status"] == str(Status.SUCCESS)
            try:
                self.channel.send(p.message(p.FINISHED, **outcome))
            except OSError:
                self.cancel.request(LOST_COORDINATOR)
                return False

    def _read(self) -> None:
        while True:
            try:
                message = self.channel.receive()
            except p.ProtocolError as exc:
                self.log(f"{self.robot}: {exc}")
                message = None
            if message is None:
                self.cancel.request(LOST_COORDINATOR)
                self._inbox.put(None)
                return
            if message["type"] == p.CANCEL:
                self.cancel.request(str(message.get("reason", "отменена координатором")))
                continue
            self._inbox.put(message)

    def _run(self, task_id: str) -> dict[str, Any]:
        task = self.tasks.get(task_id)
        if task is None:
            return _outcome(
                task_id, Status.FAILURE, f"задачи {task_id!r} у робота {self.robot} нет"
            )
        if (why := self.cancel.reason) is not None:
            return _outcome(task_id, Status.FAILURE, f"отменено до начала: {why}")
        self.log(f"{self.robot}: задача {task_id} — навык {task.skill_name}")
        recording = self.recording_for(task) if self.recording_for is not None else None
        self.executor.recorder = recording
        result: RunResult | None = None
        error: BaseException | None = None
        try:
            result = self.executor.run(
                task.skill, task.goal, name=task.skill_name, cancel=self.cancel
            )
        except Exception as exc:
            # A skill refusing its goal on start, an incompatible link, a defect. The
            # executor has stopped the robot if a tick was under way (ADR-0026);
            # the coordinator is told either way and stops the rest (ADR-P8E).
            error = exc
        except BaseException as exc:  # Ctrl+C reaching this process: recorded, then on
            error = exc
            raise
        finally:
            self.results[task_id] = result
            self.executor.recorder = None
            if recording is not None:
                recording.close(result, error=error)
        if result is None:
            return _outcome(task_id, Status.FAILURE, f"{type(error).__name__}: {error}")
        self.log(
            f"{self.robot}: задача {task_id} — {result.status}"
            + (f": {result.reason}" if result.reason else "")
        )
        return _outcome(
            task_id, result.status, result.reason, ticks=result.ticks, commands=len(result.commands)
        )


def _outcome(
    task: str, status: Status, reason: str | None, *, ticks: int = 0, commands: int = 0
) -> dict[str, Any]:
    return {
        "task": task,
        "status": str(status),
        "reason": reason,
        "ticks": ticks,
        "commands": commands,
    }
