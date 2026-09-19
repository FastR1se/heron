"""The coordinator's end of every robot's channel (ADR-0068).

Listens on the loopback, lets in the robots the plan names -- each once, and only
with the plan's identifier -- and turns whatever they say into events on one
queue. A channel that ends before the coordinator said ``end`` becomes a ``lost``
event: a robot process that died is a failure of its task, not a silence.
"""

from __future__ import annotations

import contextlib
import queue
import socket
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from . import protocol as p
from .coordinator import Event
from .plan import LOOPBACK, Address


class RobotServer:
    """Accepts robot processes and relays their messages."""

    def __init__(self, robots: Iterable[str], *, host: str = LOOPBACK) -> None:
        self._robots = frozenset(robots)
        self._listener = socket.create_server((host, 0))
        self.address = Address(host=host, port=self._listener.getsockname()[1])
        self._events: queue.Queue[Event] = queue.Queue()
        self._channels: dict[str, p.Channel] = {}
        self._lock = threading.Lock()
        self._closing = False
        self._plan_id: str | None = None

    def open(self, plan_id: str) -> None:
        """Start letting robots in. The plan is written by now: its id names the run."""
        self._plan_id = plan_id
        threading.Thread(target=self._accept, name="robot-server", daemon=True).start()

    # -- EventSource and Send -------------------------------------------------

    def get(self, timeout: float) -> Event | None:
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def send(self, robot: str, message: Mapping[str, Any]) -> None:
        with self._lock:
            channel = self._channels[robot]
        channel.send(message)

    def close(self) -> None:
        self._closing = True
        with contextlib.suppress(OSError):
            self._listener.close()
        with self._lock:
            channels = list(self._channels.values())
        for channel in channels:
            channel.close()

    # -- internals -----------------------------------------------------------

    def _accept(self) -> None:
        while not self._closing:
            try:
                sock, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(p.Channel(sock),), name="robot-channel", daemon=True
            ).start()

    def _serve(self, channel: p.Channel) -> None:
        try:
            hello = channel.receive()
        except p.ProtocolError:
            hello = None
        robot = self._admit(channel, hello)
        if robot is None:
            channel.close()
            return
        self._events.put(Event(robot, p.HELLO, dict(hello or {})))
        reason = "процесс робота закрыл связь"
        while True:
            try:
                message = channel.receive()
            except p.ProtocolError as exc:
                reason = str(exc)
                break
            if message is None:
                break
            self._events.put(Event(robot, str(message["type"]), message))
        if not self._closing:
            self._events.put(Event(robot, p.LOST, {"reason": reason}))

    def _admit(self, channel: p.Channel, hello: Mapping[str, Any] | None) -> str | None:
        if hello is None or hello.get("type") != p.HELLO:
            return None
        robot = str(hello.get("robot", ""))
        why = None
        if hello.get("plan") != self._plan_id:
            why = f"чужой план {hello.get('plan')!r}, у этого прогона {self._plan_id!r}"
        elif robot not in self._robots:
            why = f"робота {robot!r} в плане нет"
        with self._lock:
            if why is None and robot in self._channels:
                why = f"робот {robot!r} уже подключён"
            if why is None:
                self._channels[robot] = channel
        if why is None:
            try:
                channel.send(p.message(p.WELCOME))
            except OSError:
                return None
            return robot
        with contextlib.suppress(OSError):
            channel.send(p.message(p.REJECTED, reason=why))
        return None
