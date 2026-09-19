"""The coordinator: when tasks start, and what a failure does to the rest (ADR-0067, ADR-0069).

The scheduler is exercised over scripted robots first -- no threads, no sockets --
so that each rule is visible in the order of events. Then the real parts: the
executor's cancellation, and a mission over the line protocol to two robots made
of numbers, one streaming and one of the command class, each on a thread of its
own with the socket in between (ADR-0068).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping
from typing import Any

import pytest

from conftest import make_command_profile, make_profile
from heron.adapters.dummy import DummyAdapter
from heron.coordinator import protocol as p
from heron.coordinator.coordinator import (
    Coordinator,
    Event,
    MissionResult,
    MissionStatus,
    TaskState,
)
from heron.coordinator.plan import MissionPlan, PlannedRobot, PlannedTask
from heron.coordinator.robot import PreparedTask, RobotSide
from heron.coordinator.server import RobotServer
from heron.core.enums import Status
from heron.core.types import RobotState
from heron.sdk.executor import Cancellation, Executor, FrozenClock
from heron.skills.goto_joints import GotoJointsSkill
from heron.skills.hold import HoldSkill
from heron.skills.move_joints import MoveJointsSkill


def plan(*tasks: tuple[str, str, tuple[str, ...]]) -> MissionPlan:
    robots = sorted({robot for _, robot, _ in tasks})
    return MissionPlan(
        mission="m",
        scenario="s",
        scenario_path="s.yaml",
        registry=(),
        backend="dummy",
        target="sim.dummy",
        robots=tuple(
            PlannedRobot(id=r, profile=r, target="sim.dummy", plugin="dummy", contract_id="0" * 16)
            for r in robots
        ),
        tasks=tuple(
            PlannedTask(id=tid, skill="hold", robot=robot, after=after)
            for tid, robot, after in tasks
        ),
        started_utc="2026-09-19T00:00:00+00:00",
    )


class ScriptedRobots:
    """Robots whose every answer is written in advance.

    ``outcomes`` maps a task to ``"success"``, ``"failure"`` or ``"hang"`` -- a task
    that runs until it is cancelled. Unlisted tasks succeed.
    """

    def __init__(
        self,
        robots: list[str],
        outcomes: Mapping[str, str] | None = None,
        *,
        refuse: str | None = None,
        interrupt_when: str | None = None,
    ) -> None:
        self.outcomes = dict(outcomes or {})
        self.queue: deque[Event] = deque()
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.hanging: dict[str, str] = {}
        self.interrupt_when = interrupt_when
        for robot in robots:
            self.queue.append(Event(robot, p.HELLO))
            if robot == refuse:
                self.queue.append(Event(robot, p.REFUSED, {"reason": "адаптер не установлен"}))
            else:
                self.queue.append(Event(robot, p.READY))

    def send(self, robot: str, message: Mapping[str, Any]) -> None:
        self.sent.append((robot, dict(message)))
        kind = message["type"]
        if kind == p.START:
            task = message["task"]
            outcome = self.outcomes.get(task, "success")
            if outcome == "hang":
                self.hanging[robot] = task
                if task == self.interrupt_when:
                    self.queue.append(Event(robot, "interrupt"))
            else:
                self.queue.append(self._finished(robot, task, outcome, "навык не справился"))
        elif kind == p.CANCEL and robot in self.hanging:
            task = self.hanging.pop(robot)
            self.queue.append(self._finished(robot, task, "failure", message["reason"]))

    def get(self, timeout: float) -> Event | None:
        if not self.queue:
            return None
        event = self.queue.popleft()
        if event.kind == "interrupt":
            raise KeyboardInterrupt
        return event

    @staticmethod
    def _finished(robot: str, task: str, status: str, reason: str) -> Event:
        return Event(
            robot,
            p.FINISHED,
            {"task": task, "status": status, "reason": None if status == "success" else reason},
        )

    def started(self) -> list[str]:
        return [m["task"] for _, m in self.sent if m["type"] == p.START]

    def cancelled(self) -> list[str]:
        return [r for r, m in self.sent if m["type"] == p.CANCEL]


def run(
    the_plan: MissionPlan, robots: ScriptedRobots, **kwargs: Any
) -> tuple[MissionResult, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    result = Coordinator(
        the_plan, robots.send, robots, on_event=events.append, poll_s=0.0, **kwargs
    ).run()
    return result, events


def order(events: list[dict[str, Any]]) -> list[str]:
    """``+task`` when started, ``-task`` when finished."""
    marks = {"task_started": "+", "task_finished": "-"}
    return [marks[e["event"]] + e["task"] for e in events if e["event"] in marks]


def states(result: MissionResult) -> dict[str, TaskState]:
    return {item.task: item.state for item in result.tasks}


# -- when a task starts (ADR-0067) -------------------------------------------------------


def test_tasks_run_one_after_another_unless_written_otherwise() -> None:
    the_plan = plan(("a", "r1", ()), ("b", "r2", ("a",)))
    robots = ScriptedRobots(["r1", "r2"])
    result, events = run(the_plan, robots)
    assert result.ok
    assert order(events) == ["+a", "-a", "+b", "-b"]


def test_tasks_of_two_robots_that_wait_for_nobody_start_together() -> None:
    the_plan = plan(("a", "r1", ()), ("b", "r2", ()), ("c", "r1", ("a", "b")))
    result, events = run(the_plan, ScriptedRobots(["r1", "r2"]))
    assert result.ok
    assert order(events)[:2] == ["+a", "+b"]
    assert order(events)[-2:] == ["+c", "-c"]


def test_a_robot_takes_its_tasks_in_the_order_written() -> None:
    """``c`` waits for nobody, but its robot still owes ``a`` (ADR-0067, p. 3)."""
    the_plan = plan(("a", "r1", ()), ("b", "r2", ()), ("c", "r1", ()))
    _, events = run(the_plan, ScriptedRobots(["r1", "r2"]))
    sequence = order(events)
    assert sequence.index("-a") < sequence.index("+c")


def test_every_robot_is_told_the_mission_is_over() -> None:
    robots = ScriptedRobots(["r1", "r2"])
    run(plan(("a", "r1", ()), ("b", "r2", ())), robots)
    assert sorted(r for r, m in robots.sent if m["type"] == p.END) == ["r1", "r2"]


def test_nothing_starts_until_every_robot_is_ready() -> None:
    robots = ScriptedRobots(["r1", "r2"], refuse="r2")
    result, _ = run(plan(("a", "r1", ()), ("b", "r2", ())), robots)
    assert result.status is MissionStatus.FAILURE
    assert "r2 не готов" in (result.reason or "")
    assert robots.started() == []
    assert set(states(result).values()) == {TaskState.NOT_STARTED}


# -- what a failure does (ADR-0069) --------------------------------------------------------


def test_a_failure_cancels_the_running_and_starts_nothing_more() -> None:
    the_plan = plan(("a", "r1", ()), ("b", "r2", ()), ("c", "r1", ("a",)), ("d", "r3", ("b",)))
    robots = ScriptedRobots(["r1", "r2", "r3"], {"a": "hang", "b": "failure"})
    result, _ = run(the_plan, robots)

    assert result.status is MissionStatus.FAILURE
    assert result.reason is not None and result.reason.startswith("задача b (r2")
    assert states(result) == {
        "a": TaskState.CANCELLED,
        "b": TaskState.FAILED,
        "c": TaskState.NOT_STARTED,
        "d": TaskState.NOT_STARTED,
    }
    # Only the robot that was moving is sent anything; the idle one is not touched.
    assert robots.cancelled() == ["r1"]
    assert robots.started() == ["a", "b"]


def test_a_robot_lost_mid_task_fails_its_task_and_stops_the_rest() -> None:
    the_plan = plan(("a", "r1", ()), ("b", "r2", ()))
    robots = ScriptedRobots(["r1", "r2"], {"a": "hang", "b": "hang"})
    robots.queue.append(Event("r2", p.LOST, {"reason": "процесс робота закрыл связь"}))
    result, _ = run(the_plan, robots)
    assert states(result) == {"a": TaskState.CANCELLED, "b": TaskState.FAILED}
    assert "связь" in (result.reason or "")


def test_a_process_that_dies_before_connecting_fails_the_mission() -> None:
    robots = ScriptedRobots(["r1"])
    robots.queue.clear()
    reported = [("r1", "процесс робота вышел с кодом 1")]
    result, _ = run(plan(("a", "r1", ()),), robots, watch=lambda: reported)
    assert result.status is MissionStatus.FAILURE
    assert "кодом 1" in (result.reason or "")


def test_ctrl_c_stops_the_system_like_a_failure() -> None:
    the_plan = plan(("a", "r1", ()), ("b", "r1", ()))
    robots = ScriptedRobots(["r1"], {"a": "hang"}, interrupt_when="a")
    result, _ = run(the_plan, robots)
    assert result.status is MissionStatus.INTERRUPTED
    assert states(result) == {"a": TaskState.CANCELLED, "b": TaskState.NOT_STARTED}


# -- the executor's cancellation (ADR-0069) --------------------------------------------------


class CancelledOnTick(HoldSkill):
    """Hold, with someone outside asking to stop on the fifth tick."""

    def __init__(self, cancel: Cancellation) -> None:
        super().__init__()
        self.outside = cancel
        self.ticks = 0

    def on_tick(self, state: RobotState) -> Status:
        self.ticks += 1
        if self.ticks == 5:
            self.outside.request("задача другого робота не выполнена")
        return super().on_tick(state)


def test_a_cancelled_skill_stops_the_robot_first_then_reports() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})
    cancel = Cancellation()
    result = Executor(adapter, profile, clock=FrozenClock()).run(
        CancelledOnTick(cancel), {"duration": 100.0}, cancel=cancel
    )
    assert result.status is Status.FAILURE
    assert result.ticks == 6
    assert result.reason == "отменено: задача другого робота не выполнена"
    assert adapter.stopped


def test_a_cancellation_keeps_the_first_reason() -> None:
    cancel = Cancellation()
    cancel.request("первая")
    cancel.request("вторая")
    assert cancel.reason == "первая"


# -- over the line protocol, two robots of two classes (ADR-0068) ---------------------------


def serve_robot(
    server: RobotServer,
    plan_id: str,
    robot: str,
    adapter: DummyAdapter,
    executor: Executor,
    tasks: dict[str, PreparedTask],
    results: dict[str, bool],
) -> None:
    channel = p.Channel.connect(server.address.host, server.address.port)
    channel.send(p.message(p.HELLO, robot=robot, plan=plan_id))
    assert (channel.receive() or {})["type"] == p.WELCOME
    channel.send(p.message(p.READY))
    try:
        results[robot] = RobotSide(robot, channel, executor, tasks).serve()
    finally:
        adapter.disconnect()
        channel.close()


def mission_over_sockets(
    stream_tasks: dict[str, PreparedTask],
    command_tasks: dict[str, PreparedTask],
    the_plan: MissionPlan,
) -> tuple[MissionResult, dict[str, DummyAdapter], dict[str, bool]]:
    stream_profile, command_profile = make_profile(), make_command_profile()
    adapters = {"arm": DummyAdapter(), "cmd": DummyAdapter()}
    adapters["arm"].connect(stream_profile, {})
    adapters["cmd"].connect(command_profile, {})
    executors = {
        "arm": Executor(adapters["arm"], stream_profile, clock=FrozenClock()),
        "cmd": Executor(adapters["cmd"], command_profile, clock=FrozenClock()),
    }
    server = RobotServer(["arm", "cmd"])
    the_plan = the_plan.model_copy(update={"address": server.address})
    server.open(the_plan.plan_id)
    results: dict[str, bool] = {}
    threads = [
        threading.Thread(
            target=serve_robot,
            args=(
                server, the_plan.plan_id, robot, adapters[robot], executors[robot], tasks,
                results,
            ),
            daemon=True,
        )
        for robot, tasks in (("arm", stream_tasks), ("cmd", command_tasks))
    ]
    for thread in threads:
        thread.start()
    try:
        result = Coordinator(the_plan, server.send, server, poll_s=0.05).run()
    finally:
        for thread in threads:
            thread.join(timeout=10)
        server.close()
    return result, adapters, results


def socket_plan(*tasks: tuple[str, str, str, tuple[str, ...]]) -> MissionPlan:
    base = plan(*((tid, robot, after) for tid, robot, _, after in tasks))
    return base.model_copy(
        update={
            "tasks": tuple(
                PlannedTask(id=tid, skill=skill, robot=robot, after=after)
                for tid, robot, skill, after in tasks
            )
        }
    )


def test_a_mission_runs_on_a_streaming_and_a_command_robot() -> None:
    result, adapters, results = mission_over_sockets(
        {
            "reach": PreparedTask("reach", "move_joints", MoveJointsSkill(),
                                  {"target": [0.3, -0.2, 0.1], "duration": 0.2}),
            "keep": PreparedTask("keep", "hold", HoldSkill(), {"duration": 0.1}),
        },
        {"go": PreparedTask("go", "goto_joints", GotoJointsSkill(),
                            {"target": [0.2, 0.0, -0.1], "tolerance": 0.01})},
        socket_plan(
            ("reach", "arm", "move_joints", ()),
            ("go", "cmd", "goto_joints", ()),
            ("keep", "arm", "hold", ("reach", "go")),
        ),
    )
    assert result.ok, result.reason
    assert results == {"arm": True, "cmd": True}
    assert [item.state for item in result.tasks] == [TaskState.SUCCEEDED] * 3
    go = next(item for item in result.tasks if item.task == "go")
    assert go.commands == 1  # the command class sends the goal once (ADR-0049)
    assert not adapters["arm"].stopped and not adapters["cmd"].stopped


def test_a_goal_refused_on_one_robot_stops_the_other_mid_motion() -> None:
    result, adapters, results = mission_over_sockets(
        {"long": PreparedTask("long", "hold", HoldSkill(), {"duration": 1e4})},
        # Four joints for a robot of three: the skill refuses its goal on start.
        {"bad": PreparedTask("bad", "goto_joints", GotoJointsSkill(),
                             {"target": [0.1, 0.1, 0.1, 0.1], "tolerance": 0.01})},
        socket_plan(("long", "arm", "hold", ()), ("bad", "cmd", "goto_joints", ())),
    )
    assert result.status is MissionStatus.FAILURE
    outcome = {item.task: item for item in result.tasks}
    assert outcome["bad"].state is TaskState.FAILED
    assert "ValueError" in (outcome["bad"].reason or "")
    assert outcome["long"].state is TaskState.CANCELLED
    if outcome["long"].ticks:
        assert adapters["arm"].stopped, "a cancelled motion is stopped first (ADR-0026)"
    assert results == {"arm": False, "cmd": False}


def test_a_process_of_another_run_is_turned_away() -> None:
    server = RobotServer(["arm"])
    server.open("a" * 16)
    try:
        channel = p.Channel.connect(server.address.host, server.address.port)
        channel.send(p.message(p.HELLO, robot="arm", plan="b" * 16))
        answer = channel.receive() or {}
        assert answer["type"] == p.REJECTED
        assert "чужой план" in answer["reason"]
        channel.close()
    finally:
        server.close()


@pytest.mark.parametrize("line", [b"not json\n", b'{"no": "type"}\n'])
def test_a_line_outside_the_protocol_is_named(line: bytes) -> None:
    import socket

    left, right = socket.socketpair()
    try:
        left.sendall(line)
        with pytest.raises(p.ProtocolError):
            p.Channel(right).receive()
    finally:
        left.close()
        right.close()
