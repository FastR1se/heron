"""Carrying messages to and from a running robot.

ADR-0011 put process launching behind an injected interface so that what a run
will execute is checkable before executing it. This module applies the same
position one layer down, to what a run will *publish*: an adapter hands over a
``Publication`` -- a topic, a message type and plain fields -- and never builds
a ROS 2 message itself.

Two things follow. The adapter's whole behaviour (which topic, which type, what
values, in what order) is checkable on a machine with no ROS 2 on it, which is
where this platform is written. And the part that genuinely needs a ROS 2
distribution shrinks to conversion: fields in, message out. Phase 2 spent four
sessions on untested layers stacked on untested layers, so the layer that
cannot be tested here is made as thin as it can be, and the table that drives
it is checked against ``names.MESSAGES`` from both sides.

Two implementations ship, as with the launcher. One talks to ROS 2. One writes
down what it was asked to publish and answers with whatever it was given, which
is what a test and a machine without a simulator both need.

A goal sent through an action is held the same way (ADR-0060): a ``Goal`` of plain
fields goes in, a ``GoalOutcome`` -- a status and the controller's own words -- comes
out. One goal per action at a time, because one motion at a time is the command
class (ADR-0049).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

from . import names

QUEUE_DEPTH: Final = 10
"""Messages a subscription keeps before dropping the oldest, and so the most a
single read may have to drain to reach the newest one."""


class TransportError(RuntimeError):
    """The link to the robot cannot be established or used."""


class UnknownMessageError(TransportError):
    """A message type this transport cannot build or read."""


@dataclass(frozen=True, slots=True)
class Publication:
    """One outgoing message, as a value.

    Held as plain fields rather than as a ROS 2 object so that it can be
    compared, printed and asserted on without a ROS 2 distribution present.
    """

    topic: str
    message_type: str
    fields: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Reading:
    """One incoming message, flattened to plain values."""

    topic: str
    stamp: float
    """Seconds. The clock of whoever published, not of whoever reads."""

    fields: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Goal:
    """One goal sent through an action, as a value -- like ``Publication``."""

    action: str
    action_type: str
    fields: dict[str, Any]


class GoalStatus(StrEnum):
    """Where the last goal sent through an action is, as the link last heard."""

    NONE = "none"
    """No goal was sent through this action."""
    PENDING = "pending"
    """Sent; the server has not answered whether it takes it."""
    ACCEPTED = "accepted"
    """Taken and not over: executing, or being cancelled."""
    REJECTED = "rejected"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"
    CANCELED = "canceled"


OVER: Final = frozenset(
    {GoalStatus.REJECTED, GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.CANCELED}
)
"""Statuses after which the server does nothing more with the goal."""


@dataclass(frozen=True, slots=True)
class GoalOutcome:
    """The status of a goal and what the server said about it, if anything."""

    status: GoalStatus
    detail: str = ""

    @property
    def over(self) -> bool:
        return self.status in OVER


@runtime_checkable
class Transport(Protocol):
    """Publish and subscribe, as much of it as an adapter is allowed to touch."""

    def advertise(self, topic: str, message_type: str) -> None: ...

    def subscribe(self, topic: str, message_type: str) -> None: ...

    def publish(self, message: Publication) -> None: ...

    def poll(self, topic: str, timeout: float) -> Reading | None:
        """The next message on this topic, or ``None`` if none arrived in time.

        **Blocks for up to ``timeout`` seconds.** Part of the contract, not an
        implementation detail: a caller waiting for a link to come up polls
        against a deadline, and a transport that answers instantly turns that
        wait into a busy loop.
        """

    def publisher_count(self, topic: str) -> int:
        """How many publishers this link can currently see on a topic.

        Not a convenience: it is what separates "nobody is publishing" from
        "somebody is publishing and the messages do not reach us". Those two
        have different causes and different fixes, and an adapter that reported
        only silence would make the reader guess between them (ADR-0016).
        """

    def connect_action(self, action: str, action_type: str) -> None: ...

    def action_ready(self, action: str, timeout: float = 0.0) -> bool:
        """Whether the server of this action is there to take a goal, waiting **up to**
        ``timeout`` for it to appear -- a deadline on a condition, as ``poll`` has."""

    def send_goal(self, goal: Goal) -> None:
        """Send a goal; the previous one through the same action is no longer followed."""

    def goal_status(self, action: str, timeout: float = 0.0) -> GoalOutcome:
        """The last goal's outcome, waiting **up to** ``timeout`` for it to leave ``pending``."""

    def cancel_goal(self, action: str) -> None:
        """Ask the server to stop the last goal, if it took it and it is not over."""

    def close(self) -> None: ...


# --------------------------------------------------------------------------
# Conversion. The only knowledge here that belongs to ROS 2 is the shape of
# three messages, and it is written as data so that a test can hold it against
# what the control plane can ask for.
# --------------------------------------------------------------------------

MESSAGE_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    names.STATE_MESSAGE: ("name", "position", "velocity", "effort"),
    names.ARRAY_MESSAGE: ("data",),
    names.TRAJECTORY_MESSAGE: ("joint_names", "points"),
}
"""Fields of each message this platform exchanges.

Not a description of the message type -- the type has more fields than these --
but of the part the platform fills in or reads out. Anything omitted is left at
the default the message class itself defines, which is the honest treatment of
a field the platform has nothing to say about."""


GOAL_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    names.TRAJECTORY_ACTION_TYPE: ("trajectory",),
}
"""Fields of each goal this platform sends, the part it fills in.

The tolerances of ``FollowJointTrajectory`` are not among them: left empty, they are
the controller's own configuration, which is where the generator put what the
platform has to say about the end of a motion."""


def message_class_path(message_type: str) -> tuple[str, str]:
    """``sensor_msgs/msg/JointState`` -> module and class name to import."""
    parts = message_type.split("/")
    if len(parts) != 3:
        raise UnknownMessageError(
            f"{message_type!r}: тип сообщения записывается как "
            f"<пакет>/msg/<Класс>, например {names.STATE_MESSAGE!r}"
        )
    package, middle, name = parts
    return f"{package}.{middle}", name


# --------------------------------------------------------------------------


@dataclass(slots=True)
class RecordingTransport:
    """Writes down publications and answers from a queue of prepared readings.

    What a dry run prints and what a test asserts on are the same list, so the
    thing shown to a human is the thing that was checked -- the property
    ``PlanningLauncher`` holds for processes, held here for messages.

    It is not a simulator and does not pretend to be one: it moves no robot and
    integrates nothing. Feeding a reading is the caller's job, and a test that
    wants a robot that responds says so in the test.

    The one part of the ``Transport`` contract it cannot honour is blocking:
    everything here is in-process and answers at once. A caller that waits
    against a deadline should therefore shorten that deadline in a test, rather
    than spend it a hundred thousand times over.
    """

    published: list[Publication] = field(default_factory=list)
    advertised: dict[str, str] = field(default_factory=dict)
    subscribed: dict[str, str] = field(default_factory=dict)
    queues: dict[str, deque[Reading]] = field(default_factory=dict)
    publishers: dict[str, int] = field(default_factory=dict)
    """Publishers this link pretends to see, per topic. Set by ``deliver``, and
    settable by hand for the case a test needs: a topic somebody publishes on
    whose messages never arrive."""

    closed: bool = False

    actions: dict[str, str] = field(default_factory=dict)
    servers: set[str] = field(default_factory=set)
    """Action servers this link pretends to see."""

    goals: list[Goal] = field(default_factory=list)
    outcomes: dict[str, GoalOutcome] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)
    answer: GoalStatus = GoalStatus.ACCEPTED
    """How a server this link sees answers a goal: it takes it, unless a test says
    otherwise. A goal to a server it does not see stays pending."""

    def advertise(self, topic: str, message_type: str) -> None:
        self.advertised[topic] = message_type

    def subscribe(self, topic: str, message_type: str) -> None:
        self.subscribed[topic] = message_type
        self.queues.setdefault(topic, deque())

    def publish(self, message: Publication) -> None:
        if message.topic not in self.advertised:
            raise TransportError(
                f"{message.topic}: публикация в необъявленный топик; "
                f"сначала advertise()"
            )
        self.published.append(message)

    def poll(self, topic: str, timeout: float) -> Reading | None:
        queue = self.queues.get(topic)
        if not queue:
            return None
        return queue.popleft()

    def publisher_count(self, topic: str) -> int:
        return self.publishers.get(topic, 0)

    def connect_action(self, action: str, action_type: str) -> None:
        self.actions[action] = action_type

    def action_ready(self, action: str, timeout: float = 0.0) -> bool:
        return action in self.servers

    def send_goal(self, goal: Goal) -> None:
        if goal.action not in self.actions:
            raise TransportError(
                f"{goal.action}: цель в необъявленное действие; сначала connect_action()"
            )
        self.goals.append(goal)
        answer = self.answer if goal.action in self.servers else GoalStatus.PENDING
        self.outcomes[goal.action] = GoalOutcome(answer)

    def goal_status(self, action: str, timeout: float = 0.0) -> GoalOutcome:
        return self.outcomes.get(action, GoalOutcome(GoalStatus.NONE))

    def cancel_goal(self, action: str) -> None:
        self.cancelled.append(action)
        if self.goal_status(action).status is GoalStatus.ACCEPTED:
            self.outcomes[action] = GoalOutcome(GoalStatus.CANCELED)

    def close(self) -> None:
        self.closed = True

    # -- feeding it, which only a caller in the same process can do ---------

    def settle(self, action: str, status: GoalStatus, detail: str = "") -> None:
        """Make the server say what became of the last goal, as a controller would."""
        self.outcomes[action] = GoalOutcome(status, detail)

    def deliver(self, reading: Reading) -> None:
        self.queues.setdefault(reading.topic, deque()).append(reading)
        self.publishers.setdefault(reading.topic, 1)

    def deliver_state(
        self,
        topic: str,
        stamp: float,
        names_: Sequence[str],
        position: Sequence[float],
        velocity: Sequence[float] = (),
        effort: Sequence[float] = (),
    ) -> None:
        """Queue one joint state message, in the shape the broadcaster sends."""
        self.deliver(
            Reading(
                topic=topic,
                stamp=stamp,
                fields={
                    "name": list(names_),
                    "position": list(position),
                    "velocity": list(velocity),
                    "effort": list(effort),
                },
            )
        )

    def __iter__(self) -> Iterator[Publication]:
        return iter(self.published)


# --------------------------------------------------------------------------


class RclpyTransport:
    """The real link: one ROS 2 node, publishers and subscriptions on it.

    Deliberately the smallest class in this package. Everything decidable
    without a ROS 2 distribution has been decided before a message reaches it --
    which topic, which type, which values, in which order -- so what is left is
    import, convert, hand over. That is the part no test in this repository can
    reach, and it is kept small for exactly that reason.
    """

    def __init__(self, node_name: str = "control_adapter", *, use_sim_time: bool = True) -> None:
        self._rclpy = _import_ros2("rclpy")
        self._node_name = node_name
        self._use_sim_time = use_sim_time
        self._node: Any = None
        self._executor: Any = None
        self._owns_context = False
        self._publishers: dict[str, Any] = {}
        self._latest: dict[str, Reading] = {}
        self._actions: dict[str, _ActionLink] = {}
        self._start()

    # -- Transport ----------------------------------------------------------

    def advertise(self, topic: str, message_type: str) -> None:
        self._publishers[topic] = self._node.create_publisher(
            _message_class(message_type), topic, QUEUE_DEPTH
        )

    def subscribe(self, topic: str, message_type: str) -> None:
        def receive(message: Any) -> None:
            self._latest[topic] = _read(topic, message_type, message)

        self._node.create_subscription(
            _message_class(message_type), topic, receive, QUEUE_DEPTH
        )

    def publish(self, message: Publication) -> None:
        publisher = self._publishers.get(message.topic)
        if publisher is None:
            raise TransportError(
                f"{message.topic}: публикация в необъявленный топик; сначала advertise()"
            )
        publisher.publish(_build(message))

    def poll(self, topic: str, timeout: float) -> Reading | None:
        """The newest message that arrived within ``timeout``, or ``None``.

        One ``spin_once`` runs one callback, and a subscription takes one message
        per callback. A robot publishing state faster than the loop reads it
        fills the queue, and the message at its head is up to a full queue old:
        fresh by the motion of its stamp, stale by its content. So once a message
        is in hand the queue is drained, and the newest reading is returned.
        """
        self._latest.pop(topic, None)
        deadline = time.monotonic() + timeout
        while topic not in self._latest:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._executor.spin_once(timeout_sec=remaining)
        for _ in range(QUEUE_DEPTH):
            before = self._latest[topic]
            self._executor.spin_once(timeout_sec=0.0)
            if self._latest[topic] is before:
                break
        return self._latest[topic]

    def publisher_count(self, topic: str) -> int:
        return int(self._node.count_publishers(topic))

    def connect_action(self, action: str, action_type: str) -> None:
        client = _import_ros2("rclpy.action").ActionClient
        self._actions[action] = _ActionLink(
            client=client(self._node, _message_class(action_type), action)
        )

    def action_ready(self, action: str, timeout: float = 0.0) -> bool:
        client = self._action(action).client
        deadline = time.monotonic() + timeout
        while True:
            # Spinning keeps the state subscription served while the server is looked
            # for; it returns as soon as something arrives, so this is not a pause.
            self._executor.spin_once(timeout_sec=min(max(deadline - time.monotonic(), 0.0), 0.1))
            if client.server_is_ready():
                return True
            if time.monotonic() >= deadline:
                return False

    def send_goal(self, goal: Goal) -> None:
        link = self._action(goal.action)
        link.follow(link.client.send_goal_async(_build_goal(goal)))

    def goal_status(self, action: str, timeout: float = 0.0) -> GoalOutcome:
        """The last goal's outcome, spinning up to ``timeout`` while it is pending.

        Answers and results arrive as callbacks of the same executor the state
        subscription runs on, so reading the state moves them along as well.
        """
        link = self._action(action)
        deadline = time.monotonic() + timeout
        self._executor.spin_once(timeout_sec=0.0)
        while link.outcome.status is GoalStatus.PENDING:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._executor.spin_once(timeout_sec=remaining)
        return link.outcome

    def cancel_goal(self, action: str) -> None:
        link = self._action(action)
        if link.handle is not None and link.outcome.status is GoalStatus.ACCEPTED:
            # The request leaves when it is made; spinning only brings the answer,
            # which the result of the goal repeats.
            link.handle.cancel_goal_async()
            self._executor.spin_once(timeout_sec=0.0)

    def close(self) -> None:
        for link in self._actions.values():
            link.client.destroy()
        self._actions.clear()
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_context:
            self._rclpy.shutdown()
            self._owns_context = False

    # -- internals ----------------------------------------------------------

    def _action(self, action: str) -> _ActionLink:
        link = self._actions.get(action)
        if link is None:
            raise TransportError(f"{action}: действие не объявлено; сначала connect_action()")
        return link

    def _start(self) -> None:
        if not self._rclpy.ok():
            self._rclpy.init()
            self._owns_context = True
        self._node = self._rclpy.create_node(self._node_name)
        parameter = _import_ros2("rclpy.parameter").Parameter
        self._node.set_parameters(
            [parameter("use_sim_time", parameter.Type.BOOL, self._use_sim_time)]
        )
        # One executor for the life of the link, not one per poll. A fresh
        # executor per call rebuilds its wait set every time, and a subscription
        # whose wait set is rebuilt between two messages can miss the one that
        # arrived in between -- which looks exactly like a robot that publishes
        # nothing (ADR-0016).
        self._executor = _import_ros2("rclpy.executors").SingleThreadedExecutor()
        self._executor.add_node(self._node)


@dataclass(slots=True)
class _ActionLink:
    """One action client and what became of the last goal sent through it."""

    client: Any
    outcome: GoalOutcome = GoalOutcome(GoalStatus.NONE)
    handle: Any = None
    generation: int = 0

    def follow(self, sent: Any) -> None:
        """Follow a goal from its sending to its result. A goal sent later replaces it:
        a callback of the earlier one that arrives after that changes nothing."""
        self.generation += 1
        generation = self.generation
        self.outcome = GoalOutcome(GoalStatus.PENDING)
        self.handle = None

        def finished(future: Any) -> None:
            if generation == self.generation:
                self.outcome = _outcome_of(future.result())

        def answered(future: Any) -> None:
            if generation != self.generation:
                return
            handle = future.result()
            if handle is None or not handle.accepted:
                self.outcome = GoalOutcome(GoalStatus.REJECTED, "сервер не принял цель")
                return
            self.handle = handle
            self.outcome = GoalOutcome(GoalStatus.ACCEPTED)
            handle.get_result_async().add_done_callback(finished)

        sent.add_done_callback(answered)


def _outcome_of(response: Any) -> GoalOutcome:
    """The end of a goal: the terminal status of ``action_msgs`` and the result's words."""
    codes = _import_ros2("action_msgs.msg").GoalStatus
    status = {
        codes.STATUS_SUCCEEDED: GoalStatus.SUCCEEDED,
        codes.STATUS_ABORTED: GoalStatus.ABORTED,
        codes.STATUS_CANCELED: GoalStatus.CANCELED,
    }.get(response.status)
    result = response.result
    words = " ".join(
        str(part)
        for part in (getattr(result, "error_code", ""), getattr(result, "error_string", ""))
        if part not in ("", None)
    )
    if status is None:
        # A result with no terminal status is over without having succeeded.
        return GoalOutcome(GoalStatus.ABORTED, f"статус действия {response.status}; {words}")
    return GoalOutcome(status, words)


def _import_ros2(module: str) -> Any:
    """Import a ROS 2 module, or say which environment is missing.

    Lazy on purpose: this package is imported by the CLI on every command, and
    the core installs without vendor dependencies (section 10). A machine
    without ROS 2 must be able to validate, generate and run the test suite.
    """
    import importlib

    root = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        install = names.ADAPTER_PACKAGES.get(root, "desktop")
        raise TransportError(
            f"{module} не импортируется: окружение ROS 2 не подключено. "
            f"Нужен пакет ros-<дистрибутив>-{install}; его ставит scripts/install.sh install ros2. "
            f"Проверьте, что окружение подключено: source /opt/ros/<дистрибутив>/setup.bash"
        ) from exc


def _message_class(message_type: str) -> Any:
    module_path, class_name = message_class_path(message_type)
    return getattr(_import_ros2(module_path), class_name)


def _build(message: Publication) -> Any:
    """Turn a publication into an instance of its message class."""
    known = MESSAGE_FIELDS.get(message.message_type)
    if known is None:
        raise UnknownMessageError(
            f"{message.message_type}: транспорт не умеет собирать это сообщение. "
            f"Умеет: {', '.join(sorted(MESSAGE_FIELDS))}"
        )
    unknown = sorted(set(message.fields) - set(known))
    if unknown:
        raise UnknownMessageError(
            f"{message.message_type}: неизвестные поля {', '.join(unknown)}"
        )

    built = _message_class(message.message_type)()
    if message.message_type == names.TRAJECTORY_MESSAGE:
        return _build_trajectory(built, message.fields)
    for key, value in message.fields.items():
        setattr(built, key, list(value) if isinstance(value, list | tuple) else value)
    return built


def _build_goal(goal: Goal) -> Any:
    """Turn a goal into an instance of its action's ``Goal`` class."""
    known = GOAL_FIELDS.get(goal.action_type)
    if known is None:
        raise UnknownMessageError(
            f"{goal.action_type}: транспорт не умеет собирать цель этого действия. "
            f"Умеет: {', '.join(sorted(GOAL_FIELDS))}"
        )
    unknown = sorted(set(goal.fields) - set(known))
    if unknown:
        raise UnknownMessageError(f"{goal.action_type}: неизвестные поля цели {', '.join(unknown)}")

    built = _message_class(goal.action_type).Goal()
    if goal.action_type == names.TRAJECTORY_ACTION_TYPE:
        built.trajectory = _build_trajectory(
            _message_class(names.TRAJECTORY_MESSAGE)(), goal.fields["trajectory"]
        )
    return built


def _build_trajectory(built: Any, fields: Mapping[str, Any]) -> Any:
    """A trajectory carries nested points, so it is filled in by hand."""
    point_class = _message_class("trajectory_msgs/msg/JointTrajectoryPoint")
    duration_class = _message_class("builtin_interfaces/msg/Duration")

    built.joint_names = list(fields.get("joint_names", []))
    points = []
    for item in fields.get("points", []):
        point = point_class()
        point.positions = [float(v) for v in item.get("positions", [])]
        if "velocities" in item:
            point.velocities = [float(v) for v in item["velocities"]]
        seconds = float(item.get("time_from_start", 0.0))
        whole = int(seconds)
        point.time_from_start = duration_class(
            sec=whole, nanosec=round((seconds - whole) * 1e9)
        )
        points.append(point)
    built.points = points
    return built


def _read(topic: str, message_type: str, message: Any) -> Reading:
    """Flatten an incoming message to plain values."""
    known = MESSAGE_FIELDS.get(message_type)
    if known is None:  # pragma: no cover - only reachable with a new message type
        raise UnknownMessageError(f"{message_type}: транспорт не умеет читать это сообщение")
    fields = {name: list(getattr(message, name, [])) for name in known}
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    seconds = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) / 1e9
    return Reading(topic=topic, stamp=seconds, fields=fields)
