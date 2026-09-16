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
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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

    def close(self) -> None:
        self.closed = True

    # -- feeding it, which only a caller in the same process can do ---------

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

    def close(self) -> None:
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
            f"Нужен пакет ros-<дистрибутив>-{install}; его ставит scripts/phase0.sh. "
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
