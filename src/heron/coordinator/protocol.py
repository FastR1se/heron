"""The line protocol between the coordinator and a robot's process (ADR-0068).

One JSON object per line, ASCII only, over a TCP connection. What travels is
decisions and outcomes -- start this task, cancel, it ended so -- never telemetry:
a robot's states stay with its executor and its record (ADR-0070).

From a robot: ``hello`` (who it is and which plan it read), then ``ready`` or
``refused``, then one ``finished`` per ``start``. From the coordinator: ``welcome``
or ``rejected`` in answer to ``hello`` -- before the robot connects its link, so a
process that does not belong to the run touches no robot -- then ``start``,
``cancel`` and ``end``.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
from collections.abc import Mapping
from typing import Any, Final

HELLO: Final = "hello"
WELCOME: Final = "welcome"
READY: Final = "ready"
REFUSED: Final = "refused"
FINISHED: Final = "finished"
START: Final = "start"
CANCEL: Final = "cancel"
END: Final = "end"
REJECTED: Final = "rejected"
LOST: Final = "lost"
"""Not a message: what the coordinator's side reports when a channel ends early."""

CONNECT_TIMEOUT_S: Final = 10.0
"""How long a robot process tries to reach the coordinator named by the plan."""


class ProtocolError(RuntimeError):
    """A line that is not a message of this protocol."""


class Channel:
    """One end of a connection, sending and receiving whole messages.

    Sending is safe from several threads; receiving belongs to one reader.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = b""
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def connect(cls, host: str, port: int, *, timeout: float = CONNECT_TIMEOUT_S) -> Channel:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(None)
        return cls(sock)

    def send(self, message: Mapping[str, Any]) -> None:
        line = json.dumps(dict(message), ensure_ascii=True, separators=(",", ":")) + "\n"
        with self._lock:
            self._sock.sendall(line.encode("ascii"))

    def receive(self) -> dict[str, Any] | None:
        """The next message, or ``None`` when the other side has gone.

        Read from the socket itself rather than through a file over it: closing the
        socket must wake a reader blocked in another thread, and a file object keeps
        the descriptor open under it -- on Windows the reader then waits forever.
        """
        while b"\n" not in self._buffer:
            try:
                chunk = self._sock.recv(65536)
            except OSError:
                return None
            if not chunk:
                return None
            self._buffer += chunk
        raw, _, self._buffer = self._buffer.partition(b"\n")
        line = raw.decode("ascii", errors="replace")
        try:
            message = json.loads(line)
        except ValueError as exc:
            raise ProtocolError(f"строка не JSON: {line[:80]!r}") from exc
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ProtocolError(f"сообщение без типа: {line[:80]!r}")
        return message

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


def message(kind: str, **fields: Any) -> dict[str, Any]:
    return {"type": kind, **fields}
