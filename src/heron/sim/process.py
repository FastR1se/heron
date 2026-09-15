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

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Process(Protocol):
    """A started process, as much of one as a caller is allowed to touch."""

    @property
    def command(self) -> tuple[str, ...]: ...

    def poll(self) -> int | None:
        """Exit code, or ``None`` while it is still running."""

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


@dataclass(slots=True)
class SubprocessProcess:
    """A real process."""

    command: tuple[str, ...]
    handle: subprocess.Popen[bytes]

    def poll(self) -> int | None:
        return self.handle.poll()

    def terminate(self) -> None:
        self.handle.terminate()

    def kill(self) -> None:
        self.handle.kill()

    def wait(self, timeout: float | None = None) -> int:
        return self.handle.wait(timeout=timeout)


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
        )
        process = SubprocessProcess(command=tuple(command), handle=handle)
        self.processes.append(process)
        return process

    @property
    def commands(self) -> list[tuple[str, ...]]:
        return [item.command for item in self.processes]


def stop_all(processes: list[Process], timeout: float) -> None:
    """Stop processes newest first, and wait for each to actually be gone.

    The list is emptied. A process that outlives ``timeout`` after being asked
    to terminate is killed: a simulator killed while holding the graphics context
    leaves the next run without one, so the wait is generous, but it ends.
    """
    while processes:
        process = processes.pop()
        if process.poll() is not None:
            continue
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
