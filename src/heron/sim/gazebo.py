"""Simulation backend for the world of the system.

ADR-0011 settles what this class is: an orchestrator of processes, not a handle
on bodies in a scene. It generates the artifacts (ADR-0009, ADR-0010), then
starts the simulator, the clock bridge, the description publishers and the
controller spawners -- which is one launch file, because that is what a launch
file is for -- and owns the processes until it is asked to stop.

The order of the two responsibilities matters and is the platform's whole
position in one method: ``build_world`` validates the scenario and refuses to
build one the validator rejects. A world that cannot start is better than a
world that starts wrong, and invariant 6 says so.

Nothing here imports a simulator. The core installs without vendor dependencies
(section 10), and this module talks to the outside world in argument vectors.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.models import RobotProfile, Scenario, sim_target
from ..core.types import Pose
from ..generators.bundle import build
from ..generators.common import Bundle, Instance, resolve_description
from ..generators.description import description_copy
from ..generators.launch import LAUNCH_FILE
from ..generators.twin import GAZEBO
from ..registry.registry import Registry
from ..validator.rules import check_scenario
from .base import SimulationError
from .process import Launcher, Process, SubprocessLauncher

BACKEND: Final = GAZEBO
"""Name this backend answers to in ``world.backend`` and adapter keys."""

LAUNCH_TOOL: Final = ("ros2", "launch")
SPAWN_TOOL: Final = ("ros2", "run", "ros_gz_sim", "create")
STOP_TIMEOUT: Final = 15.0
"""Seconds a process gets to exit on its own before it is killed.

Generous on purpose: a simulator that is killed while holding the graphics
context leaves the next run without one, and phase 2 is a phase of repeated
runs."""


@dataclass(frozen=True, slots=True)
class GazeboRobot:
    """A robot placed in a running world."""

    id: str
    profile: str
    description: Path


@dataclass(slots=True)
class GazeboWorld:
    """A built world: where its artifacts are and what it is called."""

    name: str
    root: Path
    bundle: Bundle
    robots: tuple[GazeboRobot, ...] = ()
    closer: GazeboBackend | None = field(default=None, repr=False, compare=False)

    @property
    def launch_file(self) -> Path:
        return self.root / LAUNCH_FILE

    def close(self) -> None:
        if self.closer is not None:
            self.closer.stop()


class GazeboBackend:
    """Simulation backend driving the world through its own command line tools."""

    name = BACKEND

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        into: Path,
        launcher: Launcher | None = None,
        scenario_path: Path | None = None,
        gui: bool = True,
        rviz: bool = True,
    ) -> None:
        self.registry = registry if registry is not None else Registry()
        self.into = Path(into)
        self.launcher: Launcher = launcher if launcher is not None else SubprocessLauncher()
        self.scenario_path = scenario_path
        self.gui = gui
        self.rviz = rviz
        self.world: GazeboWorld | None = None
        self.processes: list[Process] = []

    # -- SimBackend ---------------------------------------------------------

    def build_world(self, scenario: Scenario) -> GazeboWorld:
        """Validate the scenario, generate its artifacts, write them out."""
        if scenario.world.backend != BACKEND:
            raise SimulationError(
                f"сцена объявляет владельцем физики {scenario.world.backend!r}, "
                f"а это бэкенд {BACKEND!r}; в сцене владелец один (ADR-0002)"
            )
        report = check_scenario(scenario, self.registry)
        if not report.ok:
            raise SimulationError(
                f"сцена {scenario.name!r} не проходит проверку, мир не строится "
                f"(инвариант 6):",
                report,
            )

        bundle = build(
            scenario, self.registry, into=self.into, scenario_path=self.scenario_path
        )
        bundle.write()
        world = GazeboWorld(
            name=scenario.name,
            root=bundle.root,
            bundle=bundle,
            robots=tuple(
                GazeboRobot(
                    id=robot.id,
                    profile=robot.profile,
                    description=bundle.root / robot.id,
                )
                for robot in scenario.robots
            ),
            closer=self,
        )
        self.world = world
        return world

    def spawn(self, profile: RobotProfile, pose: Pose) -> GazeboRobot:
        """Place one more robot into a world that is already running."""
        world = self._require_world()
        if sim_target(BACKEND) not in profile.adapters:
            raise SimulationError(
                f"у профиля {profile.name!r} нет адаптера {sim_target(BACKEND)!r} "
                f"(ADR-0002)"
            )
        source = self.registry.source(profile.name)
        instance = Instance(
            id=profile.name,
            profile=profile,
            description=resolve_description(profile, source.path),
            pose=pose.position,
        )
        # Placed as an obstacle: a robot added to a running world has no
        # controller manager started for it, so no twin speaks for it.
        generated = description_copy(instance, world.root)
        target = generated.resolve(world.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(generated.content, encoding="utf-8", newline="\n")

        x, y, z = pose.position
        self._run(
            [
                *SPAWN_TOOL,
                "-world", world.name,
                "-file", str(target),
                "-name", instance.id,
                "-x", f"{x:g}",
                "-y", f"{y:g}",
                "-z", f"{z:g}",
            ]
        )
        robot = GazeboRobot(id=instance.id, profile=profile.name, description=target)
        world.robots = (*world.robots, robot)
        return robot

    def start(self) -> None:
        """Start the world and everything that talks to it."""
        world = self._require_world()
        if self.processes:
            raise SimulationError("мир уже запущен; остановите его перед новым запуском")
        self._run(self.launch_command(world))

    def stop(self) -> None:
        """Stop everything, newest first, and wait for it to actually be gone."""
        while self.processes:
            process = self.processes.pop()
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    # -- specifics ----------------------------------------------------------

    def launch_command(self, world: GazeboWorld) -> list[str]:
        """The command ``start`` will run. Checkable without running it."""
        return [
            *LAUNCH_TOOL,
            str(world.launch_file),
            f"headless:={str(not self.gui).lower()}",
            f"rviz:={str(self.rviz).lower()}",
        ]

    def _require_world(self) -> GazeboWorld:
        if self.world is None:
            raise SimulationError("мир не построен: сначала build_world()")
        return self.world

    def _run(self, command: list[str]) -> Process:
        process = self.launcher.start(command, cwd=self.into)
        self.processes.append(process)
        return process
