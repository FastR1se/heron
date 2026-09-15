"""``sim run`` ends the processes it started, however it is ended itself.

Found on the Linux run of stage 4: a world is stopped by Ctrl+C, and nothing
else stopped it. SIGTERM -- ``kill``, a service manager, a container runtime --
ended the command and left the vendor's simulator running.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from heron.cli import main as cli
from heron.core.models import RobotProfile, Scenario
from heron.sim.process import StopRequest, stop_all

runner = CliRunner()


class TerminatedWhileRunning:
    """A world process during whose wait the command receives SIGTERM."""

    command = ("elsewhere-sim",)

    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def interrupt(self) -> None:
        self.terminated = True

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if not self.terminated:
            signal.raise_signal(signal.SIGTERM)
        return 0


class InterruptedWhileRunning(TerminatedWhileRunning):
    """A world process during whose wait the command receives SIGINT."""

    def wait(self, timeout: float | None = None) -> int:
        if not self.terminated:
            signal.raise_signal(signal.SIGINT)
        return 0


class Launcher:
    started: ClassVar[list[TerminatedWhileRunning]] = []
    kind: ClassVar[type[TerminatedWhileRunning]] = TerminatedWhileRunning

    def __init__(self) -> None:
        self.processes: list[TerminatedWhileRunning] = []

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> TerminatedWhileRunning:
        process = self.kind()
        self.processes.append(process)
        Launcher.started.append(process)
        return process

    @property
    def commands(self) -> list[tuple[str, ...]]:
        return [item.command for item in self.processes]


class Backend:
    def __init__(self, registry: Any, **kwargs: Any) -> None:
        self.launcher = kwargs["launcher"]
        self.processes: list[Any] = []

    def build_world(self, scenario: Scenario) -> object:
        return object()

    def spawn(self, profile: RobotProfile, pose: Any) -> object:
        return object()

    def start(self) -> None:
        self.processes.append(self.launcher.start(["elsewhere-sim"]))

    def stop(self) -> None:
        stop_all(self.processes, 1.0)


def test_sigterm_stops_the_world_it_started(tmp_path: Path, monkeypatch: Any) -> None:
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        "schema_version: 1\nname: scene\nworld: {backend: elsewhere}\n"
        "robots:\n  - {id: arm1, profile: arm, role: scripted}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "load", lambda kind, name: Backend)
    monkeypatch.setattr(cli, "SubprocessLauncher", Launcher)
    Launcher.started = []
    before = signal.getsignal(signal.SIGTERM)

    result = runner.invoke(cli.app, ["sim", "run", str(scene), "--into", str(tmp_path / "out")])

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "останавливаю" in result.output
    assert [process.terminated for process in Launcher.started] == [True]
    assert signal.getsignal(signal.SIGTERM) == before


# -- how each world is asked to stop (ADR-0035) ------------------------------------


class Recorder:
    """A process that writes down what it was asked, and can outlive its wait."""

    command = ("ros2", "launch", "bringup.launch.xml")

    def __init__(self, *, outlives: bool = False) -> None:
        self.calls: list[str] = []
        self.outlives = outlives

    def poll(self) -> int | None:
        return None

    def interrupt(self) -> None:
        self.calls.append("interrupt")

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append("wait")
        if self.outlives and "kill" not in self.calls:
            raise subprocess.TimeoutExpired(list(self.command), timeout or 0.0)
        return 0


def test_a_gazebo_world_is_interrupted_as_at_a_terminal(tmp_path: Path) -> None:
    """Stage 5 on Linux: SIGTERM to ``ros2 launch`` left Gazebo and every node running."""
    from heron.sim.gazebo import GazeboBackend

    backend = GazeboBackend(into=tmp_path)
    launch = Recorder()
    backend.processes = [launch]
    backend.stop()

    assert launch.calls == ["interrupt", "wait"]
    assert backend.processes == []


def test_a_launch_that_outlives_the_interrupt_is_killed_not_terminated(tmp_path: Path) -> None:
    """A second, different request would cut the launch's own shutdown short."""
    from heron.sim.gazebo import GazeboBackend

    backend = GazeboBackend(into=tmp_path)
    launch = Recorder(outlives=True)
    backend.processes = [launch]
    backend.stop()

    assert launch.calls == ["interrupt", "wait", "kill", "wait"]


def test_a_vendors_program_is_still_terminated() -> None:
    program = Recorder()
    stop_all([program], 1.0)
    assert program.calls == ["terminate", "wait"]


@pytest.mark.skipif(os.name == "nt", reason="SIGINT одному процессу есть только на POSIX")
def test_a_real_process_is_interrupted_with_sigint() -> None:
    from heron.sim.process import SubprocessLauncher

    code = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(7))\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    launcher = SubprocessLauncher()
    process = launcher.start([sys.executable, "-c", code])
    time.sleep(1.0)
    stop_all(list(launcher.processes), 10.0, request=StopRequest.INTERRUPT)
    assert process.poll() == 7


# -- the interrupt reaches its process (ADR-0036) ---------------------------------


@pytest.mark.skipif(os.name == "nt", reason="маска сигналов наследуется только на POSIX")
def test_a_child_of_a_parent_ignoring_sigint_can_still_be_interrupted() -> None:
    """Stage 5, second run: the script's background job ignored SIGINT, ``ros2 launch``
    inherited that, never heard the interrupt, and was killed after the deadline.

    The child here has no handler of its own, as ``ros2 launch`` has none: it stops on
    SIGINT only if the signal is not ignored when it starts."""
    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "from heron.sim.process import StopRequest, SubprocessLauncher, stop_all\n"
        "launcher = SubprocessLauncher()\n"
        "child = launcher.start([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "time.sleep(1.0)\n"
        "stop_all(list(launcher.processes), 10.0, request=StopRequest.INTERRUPT)\n"
        "print(child.poll())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    # KeyboardInterrupt (1) or the default action of SIGINT (-2); -9 is the kill that
    # follows an interrupt nobody heard.
    assert result.stdout.strip() in {"1", "-2"}, result.stdout


def test_sigint_stops_the_world_even_when_it_was_inherited_ignored(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Stage 4: ``kill -INT`` to a ``sim run`` started from a script reached nothing."""
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        "schema_version: 1\nname: scene\nworld: {backend: elsewhere}\n"
        "robots:\n  - {id: arm1, profile: arm, role: scripted}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "load", lambda kind, name: Backend)
    monkeypatch.setattr(cli, "SubprocessLauncher", Launcher)
    monkeypatch.setattr(Launcher, "kind", InterruptedWhileRunning)
    Launcher.started = []

    inherited = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        result = runner.invoke(
            cli.app, ["sim", "run", str(scene), "--into", str(tmp_path / "out")]
        )
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, inherited)

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "останавливаю" in result.output
    assert [process.terminated for process in Launcher.started] == [True]
