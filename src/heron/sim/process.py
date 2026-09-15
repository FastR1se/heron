"""Starting and stopping external processes.

ADR-0011: a simulation backend orchestrates processes, and it does so through
this indirection rather than through ``subprocess`` directly. The reason is not
testing convenience but the platform's own position: what a run will execute is
checkable before executing it, and a backend that can only be observed by
running it makes that claim false.

Two implementations ship. One starts processes. One writes down what it was
asked to start and starts nothing, which is what a dry run and a machine without
a simulator both need.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, runtime_checkable


class StopRequest(StrEnum):
    """How a process is asked to stop before it is killed (ADR-0035)."""

    TERMINATE = "terminate"
    """SIGTERM: what a program of its own is asked to stop with."""

    INTERRUPT = "interrupt"
    """SIGINT, as Ctrl+C at a terminal. The one signal ``ros2 launch`` stops its
    children on: on SIGTERM it cancels itself and leaves them running, and its own
    log says so."""


@runtime_checkable
class Process(Protocol):
    """A started process, as much of one as a caller is allowed to touch."""

    @property
    def command(self) -> tuple[str, ...]: ...

    def poll(self) -> int | None:
        """Exit code, or ``None`` while it is still running."""

    def interrupt(self) -> None:
        """Ask it to stop the way Ctrl+C at a terminal would."""

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int:
        """Exit code, raising ``subprocess.TimeoutExpired`` if it outlives the wait."""


@runtime_checkable
class Launcher(Protocol):
    """Whatever turns a command into a process."""

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Process: ...


GROUP_POLL_S: Final = 0.1
"""How often a stopping world is looked at for processes still in it."""


@dataclass(slots=True)
class SubprocessProcess:
    """A real process, and on POSIX the process group it leads (ADR-0037).

    A world is rarely one process: ``ros2 launch`` starts nodes, and a vendor's
    launch file starts ``gz sim`` behind ``/bin/sh -c``, which passes no signal on.
    So the group is what is interrupted, waited for and killed -- as a terminal's
    Ctrl+C reaches every process of the foreground group, whatever stands between
    them. ``terminate`` stays addressed to the one process: it asks a program that
    stops itself.
    """

    command: tuple[str, ...]
    handle: subprocess.Popen[bytes]

    def poll(self) -> int | None:
        return self.handle.poll()

    def interrupt(self) -> None:
        if sys.platform == "win32":  # pragma: no cover - worlds are started on Linux
            # Windows delivers Ctrl+C to a console group, never to one process.
            self.handle.terminate()
        else:
            self._signal_group(signal.SIGINT)

    def terminate(self) -> None:
        self.handle.terminate()

    def kill(self) -> None:
        if sys.platform == "win32":  # pragma: no cover - worlds are started on Linux
            self.handle.kill()
        else:
            self._signal_group(signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int:
        """Exit code of the process, once nothing of its group is left.

        A leader that exits before its children has not stopped a world, and a
        child still closing a graphics context is given the rest of the deadline.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        code = self.handle.wait(timeout=timeout)
        if sys.platform != "win32":
            while self._group_alive():
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(list(self.command), timeout or 0.0)
                time.sleep(GROUP_POLL_S)
        return code

    if sys.platform != "win32":

        def _signal_group(self, signum: signal.Signals) -> None:
            try:
                os.killpg(self.handle.pid, signum)
            except ProcessLookupError:
                return

        def _group_alive(self) -> bool:
            try:
                os.killpg(self.handle.pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:  # pragma: no cover - a member changed its user
                return True
            return True


def _default_sigint() -> None:  # pragma: no cover - runs in the child, before exec
    """SIGINT back to its default action in a child about to run (ADR-0036).

    An ignored signal survives exec, and a process started from a script's
    background job inherits SIGINT ignored. CPython then installs no handler of
    its own, ``ros2 launch`` -- which listens only through Python's handlers --
    never hears the interrupt it would stop its children on, and a stop becomes a
    kill. A launcher that promises to interrupt makes the interrupt deliverable.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)


@dataclass(slots=True)
class SubprocessLauncher:
    """Starts processes for real, and remembers what it started.

    The command that started a world waits on these, so it has to know them
    without knowing which backend asked for them (ADR-0021).
    """

    processes: list[Process] = field(default_factory=list)

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Process:
        handle = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            preexec_fn=None if sys.platform == "win32" else _default_sigint,
            # Its own session, so its own group: what it starts is stopped with it (ADR-0037).
            start_new_session=sys.platform != "win32",
        )
        process = SubprocessProcess(command=tuple(command), handle=handle)
        self.processes.append(process)
        return process

    @property
    def commands(self) -> list[tuple[str, ...]]:
        return [item.command for item in self.processes]


def stop_all(
    processes: list[Process],
    timeout: float,
    *,
    request: StopRequest = StopRequest.TERMINATE,
) -> None:
    """Stop processes newest first, and wait for each to actually be gone.

    The list is emptied. Each process is asked once, the way ``request`` says --
    the backend knows what it started. A process that outlives ``timeout`` is
    killed, not asked again another way: a second request can undo the first, and
    SIGTERM to a launch stopping on SIGINT orphans what it has not stopped yet
    (ADR-0035). A simulator killed while holding the graphics context leaves the
    next run without one, so the wait is generous, but it ends.
    """
    while processes:
        process = processes.pop()
        if process.poll() is not None:
            continue
        if request is StopRequest.INTERRUPT:
            process.interrupt()
        else:
            process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@dataclass(slots=True)
class PlannedProcess:
    """A process that was planned and never started."""

    command: tuple[str, ...]
    cwd: Path | None = None

    def poll(self) -> int | None:
        return 0

    def interrupt(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0


@dataclass(slots=True)
class PlanningLauncher:
    """Writes down commands instead of running them.

    What a dry run prints and what a test asserts on are the same list, so the
    thing shown to a human is the thing that was checked.
    """

    planned: list[PlannedProcess] = field(default_factory=list)

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Process:
        process = PlannedProcess(command=tuple(command), cwd=cwd)
        self.planned.append(process)
        return process

    @property
    def commands(self) -> list[tuple[str, ...]]:
        return [item.command for item in self.planned]
