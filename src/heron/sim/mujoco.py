"""Simulation backend for a MuJoCo world (ADR-0030).

Shared by every robot and knowing none (ADR-0021). The platform starts no MuJoCo
of its own: a MuJoCo world here is the robot's twin, a simulator that loads the
scene, steps the physics and answers in the robot's protocol. So this backend is
an orchestrator of processes in the plainest sense of ADR-0011 -- it validates
the scenario, finds the twin of its one robot, tells the twin the scene and the
link, and starts what the twin returns through the launcher it was handed.

What it deliberately does not do:

* generate anything. The scene is an input like the description (ADR-0009), and
  the validator checks it loads the description that was compared with the
  profile;
* set the physics step. The step is the scene's -- a vendor's simulator takes it
  from the file it loads -- and the validator holds it against the contract's
  state rate instead (``mjcf.timestep_slower_than_state``);
* import MuJoCo. The core installs without a simulator (section 10), and a
  backend that starts somebody else's simulator has no use for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..core.enums import Role, TwinKind
from ..core.mjcf import MUJOCO
from ..core.models import RobotProfile, Scenario, ScenarioRobot, sim_target
from ..core.plugins import available
from ..core.twins import TwinError, find_twin, why_missing
from ..core.types import Pose
from ..registry.registry import Registry
from ..validator.rules import check_scenario
from .base import SimulationError
from .mujoco_twin import MujocoScene, MujocoTwin, TwinProcess
from .process import Launcher, Process, SubprocessLauncher, stop_all

BACKEND: Final = MUJOCO
"""Name this backend answers to in ``world.backend`` and adapter keys."""

STOP_TIMEOUT: Final = 15.0
"""Seconds a twin gets to exit on its own before it is killed: it holds a window."""


@dataclass(frozen=True, slots=True)
class MujocoRobot:
    """The one robot of a MuJoCo world."""

    id: str
    profile: str


@dataclass(slots=True)
class MujocoWorld:
    """A built world: the scene a twin was told, and the processes it answered with."""

    name: str
    scene: MujocoScene
    processes: tuple[TwinProcess, ...]
    closer: MujocoBackend | None = field(default=None, repr=False, compare=False)

    @property
    def robots(self) -> tuple[MujocoRobot, ...]:
        return (MujocoRobot(id=self.scene.robot, profile=self.scene.profile.name),)

    def close(self) -> None:
        if self.closer is not None:
            self.closer.stop()


class MujocoBackend:
    """Starts the twin that is the MuJoCo world of a one-robot scene."""

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
        # Accepted because every backend is constructed alike (sim.base); a twin's
        # world is drawn in the twin's own window, and nothing here starts RViz.
        self.rviz = rviz
        self.world: MujocoWorld | None = None
        self.processes: list[Process] = []

    # -- SimBackend ---------------------------------------------------------

    def build_world(self, scenario: Scenario) -> MujocoWorld:
        """Validate the scenario, ask the robot's twin how to start it, keep the answer."""
        if scenario.world.backend != BACKEND:
            raise SimulationError(
                f"сцена объявляет владельцем физики {scenario.world.backend!r}, "
                f"а это бэкенд {BACKEND!r}; в сцене владелец один (ADR-0002)"
            )
        report = check_scenario(scenario, self.registry, path=self.scenario_path)
        if not report.ok:
            raise SimulationError(
                f"сцена {scenario.name!r} не проходит проверку, мир не строится "
                f"(инвариант 6):",
                report,
            )

        robot = _the_robot(scenario)
        source = self.registry.source(robot.profile)
        profile = source.profile
        spec = profile.adapters[sim_target(BACKEND)]
        twin = _twin(robot.id, spec.plugin)
        scene = MujocoScene(
            robot=robot.id,
            profile=profile,
            params=dict(spec.params),
            world=self._scene_file(scenario),
            description=_description(profile, source.path),
            gui=self.gui,
        )
        try:
            processes = tuple(twin.processes(scene))
        except SimulationError:
            raise
        except (ValueError, RuntimeError) as exc:
            raise SimulationError(
                f"{robot.id}: двойник {type(twin).__name__} не собирает запуск сцены — {exc}"
            ) from exc
        if not processes:
            raise SimulationError(
                f"{robot.id}: двойник {type(twin).__name__} владеет физикой, но не запускает ни "
                f"одного процесса — мира не будет"
            )

        self.into.mkdir(parents=True, exist_ok=True)
        world = MujocoWorld(name=scenario.name, scene=scene, processes=processes, closer=self)
        self.world = world
        return world

    def spawn(self, profile: RobotProfile, pose: Pose) -> MujocoRobot:
        """Refused: the twin owns the physics, and its scene holds one robot (ADR-0002)."""
        raise SimulationError(
            f"{profile.name}: в мир MuJoCo робота не добавить — физикой владеет двойник, и "
            f"участник в его сцене один (ADR-0002)"
        )

    def start(self) -> None:
        """Start the twin's processes, in the order it gave them."""
        world = self._require_world()
        if self.processes:
            raise SimulationError("мир уже запущен; остановите его перед новым запуском")
        for process in world.processes:
            self.processes.append(self.launcher.start(process.command, cwd=self.into))

    def stop(self) -> None:
        stop_all(self.processes, STOP_TIMEOUT)

    # -- specifics ----------------------------------------------------------

    def _require_world(self) -> MujocoWorld:
        if self.world is None:
            raise SimulationError("мир не построен: сначала build_world()")
        return self.world

    def _scene_file(self, scenario: Scenario) -> Path:
        stated = scenario.world.file
        if stated is None:  # pragma: no cover - mjcf.scene_missing stops it earlier
            raise SimulationError(f"сцена {scenario.name!r} не называет world.file")
        path = Path(stated)
        if not path.is_absolute():
            if self.scenario_path is None:
                raise SimulationError(
                    f"сцена {stated!r} задана относительно файла сценария, а он не передан"
                )
            path = self.scenario_path.parent / path
        return path.resolve()


def _the_robot(scenario: Scenario) -> ScenarioRobot:
    """The one participant. The validator says the same once a twin is found."""
    if len(scenario.robots) != 1:
        raise SimulationError(
            f"сцена {scenario.name!r}: участников {len(scenario.robots)}, а мир MuJoCo поднимает "
            f"двойник одного робота — второму в нём места нет (ADR-0002)"
        )
    robot = scenario.robots[0]
    if robot.role is Role.STATIC:
        raise SimulationError(
            f"сцена {scenario.name!r}: единственный участник {robot.id} — препятствие. Мир MuJoCo "
            f"поднимает двойник связи, а у препятствия связи нет"
        )
    return robot


def _twin(robot: str, plugin: str) -> MujocoTwin:
    try:
        twin = find_twin(plugin, BACKEND)
    except TwinError as exc:
        raise SimulationError(f"{robot}: {exc}") from exc
    if twin is None:
        if ":" not in plugin and plugin not in available("adapters"):
            reason = "адаптер не установлен, а двойник поставляется тем же пакетом робота"
        else:
            reason = why_missing() or "пакет робота не поставляет двойника под этот бэкенд"
        raise SimulationError(
            f"{robot}: у адаптера {plugin!r} нет двойника под бэкенд {BACKEND!r} — мир MuJoCo "
            f"поднимает двойник, и поднять его некому: {reason} (ADR-0021)"
        )
    if not isinstance(twin, MujocoTwin):
        raise SimulationError(
            f"{robot}: двойник {type(twin).__name__} не называет процессы сцены MuJoCo: нужен "
            f"processes (ADR-0024, ADR-0030)"
        )
    if twin.kind != TwinKind.OWNS_PHYSICS:
        raise SimulationError(
            f"{robot}: двойник {type(twin).__name__} — мост, а своего мира MuJoCo платформа не "
            f"поднимает: встраивать мост некуда (ADR-0030)"
        )
    return twin


def _description(profile: RobotProfile, profile_path: Path) -> Path | None:
    if profile.description.mjcf is None:
        return None
    stated = Path(profile.description.mjcf)
    return (stated if stated.is_absolute() else profile_path.parent / stated).resolve()
