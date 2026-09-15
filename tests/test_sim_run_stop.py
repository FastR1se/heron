"""``sim run`` ends the processes it started, however it is ended itself.

Found on the Linux run of stage 4: a world is stopped by Ctrl+C, and nothing
else stopped it. SIGTERM -- ``kill``, a service manager, a container runtime --
ended the command and left the vendor's simulator running.
"""

from __future__ import annotations

import signal
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from typer.testing import CliRunner

from heron.cli import main as cli
from heron.core.models import RobotProfile, Scenario
from heron.sim.process import stop_all

runner = CliRunner()


class TerminatedWhileRunning:
    """A world process during whose wait the command receives SIGTERM."""

    command = ("elsewhere-sim",)

    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if not self.terminated:
            signal.raise_signal(signal.SIGTERM)
        return 0


class Launcher:
    started: ClassVar[list[TerminatedWhileRunning]] = []

    def __init__(self) -> None:
        self.processes: list[TerminatedWhileRunning] = []

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> TerminatedWhileRunning:
        process = TerminatedWhileRunning()
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
