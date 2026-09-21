"""The training backend: an orchestrator of one process (ADR-0044).

Nothing of the learning framework is imported here, and nothing of it is a
dependency of the platform. This module derives an environment from data,
writes that derivation to disk, starts one process with it, and turns what the
process left behind into an artifact of the store. The simulator lives in that
process and nowhere else.

The reason is the one ADR-0011 gives for simulation, plus one of its own:
``AppLauncher`` takes over the process it is raised in -- signals, the Kit
application, and an exit that a probe once watched hang. A command that raised it
inside itself could neither be shown before it runs nor stopped as a group, and
the platform's own claim -- that what a run will execute is checkable before
executing it -- would stop being true exactly where a run costs the most.

So the boundary is a directory:

* ``spec.json`` -- the environment, derived and complete, written before the
  process starts;
* ``description.urdf`` -- the copy the converter reads (ADR-0048);
* ``policy.onnx`` and ``export.json`` -- what the process leaves;
* ``conditions.json`` -- what the run cost and what versions produced it.

``export`` reads that directory and writes an artifact. The manifest is the
platform's to write, in the training environment, by the same code that computes
a ``contract_id`` on the side that executes -- which is the whole point of the
core installing there (ADR-0046).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.branding import data_home
from ..core.environment import Requirement, RequirementKind, forbids, needs
from ..core.loading import LoadError, parse_model
from ..core.models import (
    Completion,
    OnnxSignature,
    PolicyManifest,
    RobotProfile,
    Scenario,
    ScenarioRobot,
    SkillSpec,
)
from ..registry.registry import ProfileNotFoundError, Registry
from ..sim.process import Launcher, SubprocessLauncher
from ..store.policies import PolicyArtifact, PolicyStore, file_sha256
from ..validator.report import Report
from ..validator.rules import check_scenario
from .spec import DESCRIPTION_FILE, SPEC_FILE, TrainSpec, derive, description_text
from .tasks import TaskError, find_task

BACKEND: Final = "isaaclab"
"""Name this backend answers to in ``world.backend``."""

_PROCESS_MODULE: Final = "isaaclab_process"
PROCESS_MODULE: Final = f"{__package__}.{_PROCESS_MODULE}"
"""What ``python -m`` is given. Built from the package, never spelled out: the
platform's name appears in exactly one file (section 17)."""

ONNX_FILE: Final = "policy.onnx"
EXPORT_FILE: Final = "export.json"
CONDITIONS_FILE: Final = "conditions.json"
RUNS_DIR: Final = "train"
"""Runs live under the platform's data, not in the store: a run is not an
artifact, and only what crosses the boundary is (ADR-0046)."""

REQUIREMENTS: Final[tuple[Requirement, ...]] = (
    needs(
        RequirementKind.PYTHON,
        "isaaclab",
        hint="окружение обучения: scripts/install.sh install train --accept-nvidia-eula",
    ),
    needs(
        RequirementKind.PYTHON,
        "isaacsim",
        hint="Isaac Sim — часть окружения обучения; проверьте активированное окружение",
    ),
    needs(
        RequirementKind.PYTHON,
        "rsl_rl",
        hint="pip install rsl-rl-lib в окружении обучения",
    ),
    forbids(
        RequirementKind.PYTHON,
        "rclpy",
        because=(
            "ROS 2 в окружение обучения не попадает: граница миров — файлы, а не общий "
            "процесс"
        ),
        hint="обучайте в окружении без source /opt/ros/<дистрибутив>/setup.bash",
    ),
)
"""What the process that trains needs, and what it must not find (ADR-0003, ADR-0025)."""


class TrainError(RuntimeError):
    """Training cannot be planned, started or exported."""

    def __init__(self, message: str, report: Report | None = None) -> None:
        self.report = report
        if report is not None:
            message = "\n".join([message, *(str(f) for f in report.errors)])
        super().__init__(message)


class RunConditions(BaseModel):
    """The knobs of one run. Not part of the environment, and not of the artifact's identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    envs: int = Field(default=512, gt=0)
    """Parallel environments. The threshold of ADR-0005 is measured in these."""

    iterations: int = Field(default=1000, gt=0)
    seed: int = 1
    device: str = "cuda:0"


class TrainRun(BaseModel):
    """What ``spec.json`` holds: an environment and the conditions it is run under."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    spec: TrainSpec
    conditions: RunConditions


class ExportedPolicy(BaseModel):
    """What the training process says about the graph it wrote (``export.json``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    file: str = ONNX_FILE
    input: str
    output: str
    observation_size: int = Field(gt=0)
    action_size: int = Field(gt=0)
    opset: int = Field(gt=0)


@dataclass(frozen=True, slots=True)
class TrainPlan:
    """A training run before it is a run: the environment, the files, the command."""

    spec: TrainSpec
    description: str
    """The copy of the input description, as text."""

    run: Path
    command: tuple[str, ...]
    files: dict[str, str] = field(default_factory=dict)

    def lines(self) -> list[str]:
        """What a dry run prints: the derivation, then what would be executed."""
        return [
            *self.spec.lines(),
            f"каталог     {self.run}",
            "",
            "будет выполнено:",
            "  " + " ".join(self.command),
        ]


class IsaacLabBackend:
    """Derives a training environment, runs one process, exports the artifact."""

    name = BACKEND
    requirements = REQUIREMENTS

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        into: Path | None = None,
        launcher: Launcher | None = None,
        scenario_path: Path | None = None,
        store: PolicyStore | None = None,
    ) -> None:
        self.registry = registry if registry is not None else Registry()
        self.into = Path(into) if into is not None else None
        self.launcher: Launcher = launcher if launcher is not None else SubprocessLauncher()
        self.scenario_path = scenario_path
        self.store = store if store is not None else PolicyStore()

    # -- TrainBackend -------------------------------------------------------

    def build_env(self, scenario: Scenario, learner: str | None = None) -> TrainPlan:
        """The environment this scenario describes, as data, with nothing started.

        The scenario is validated first, for the reason ``build_world`` validates
        one: a mismatch between environments is an error before a run, not strange
        behaviour during it (invariant 6). Here it is also the only cheap moment --
        the next one costs a GPU for hours.
        """
        if scenario.world.backend != BACKEND:
            raise TrainError(
                f"сцена объявляет владельцем физики {scenario.world.backend!r}, "
                f"а это бэкенд обучения {BACKEND!r}; в сцене владелец один (ADR-0002)"
            )
        report = check_scenario(scenario, self.registry, path=self.scenario_path)
        if not report.ok:
            raise TrainError(
                f"сцена {scenario.name!r} не проходит проверку, среда обучения не строится "
                f"(инвариант 6):",
                report,
            )

        robot = _the_learner(scenario, learner)
        profile, description = self._profile_of(robot)
        task = self._task(robot)
        spec = derive(
            scenario=scenario.name,
            robot=robot.id,
            backend=BACKEND,
            task=str(robot.task),
            profile=profile,
            description=description,
            pose=robot.pose,
            skill=_skill_spec(task, robot),
            completion=_completion(task),
        )
        run = self.into if self.into is not None else default_run_dir(spec)
        return TrainPlan(
            spec=spec,
            description=description_text(profile, description),
            run=run,
            command=(sys.executable, "-m", PROCESS_MODULE, str(run)),
        )

    def train(self, plan: TrainPlan, conditions: RunConditions | None = None) -> Path:
        """Write the run down, start the one process, wait for it. Returns the directory."""
        run = write_run(plan, conditions or RunConditions())
        process = self.launcher.start(plan.command, cwd=run)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.interrupt()
            raise
        if code != 0:
            raise TrainError(
                f"обучение остановлено с кодом {code}; каталог прогона {run} "
                f"остаётся как есть — в нём журнал процесса"
            )
        return run

    def export(self, run: Path) -> PolicyArtifact:
        """The artifact of a finished run: ONNX, manifest, conditions (ADR-0046)."""
        run = Path(run)
        record = read_run(run)
        spec = record.spec
        onnx = run / EXPORT_FILE
        try:
            exported = parse_model(
                json.loads(onnx.read_text(encoding="utf-8")), ExportedPolicy, source=onnx
            )
        except OSError as exc:
            raise TrainError(
                f"{onnx}: прогон не оставил описания экспорта — обучение не дошло до конца "
                f"или упало при экспорте"
            ) from exc
        except (json.JSONDecodeError, LoadError) as exc:
            raise TrainError(f"{onnx}: описание экспорта не читается — {exc}") from exc

        network = run / exported.file
        if not network.is_file():
            raise TrainError(f"{network}: экспорт объявлен, а файла сети нет")
        if exported.observation_size != spec.observation_size:
            raise TrainError(
                f"сеть на {exported.observation_size} входов, а контракт выводит "
                f"{spec.observation_size}: среда собралась не из этого контракта"
            )
        if exported.action_size != spec.action_size:
            raise TrainError(
                f"сеть на {exported.action_size} выходов, а контракт выводит "
                f"{spec.action_size}: среда собралась не из этого контракта"
            )

        manifest = PolicyManifest(
            schema_version=1,
            name=spec.task,
            contract=spec.contract,
            contract_id=spec.contract.contract_id,
            capabilities=spec.skill.required_capabilities,
            control_mode=spec.skill.required_control_mode,
            requires=spec.skill.requires,
            completion=spec.completion,
            onnx=OnnxSignature(
                sha256=file_sha256(network),
                input=exported.input,
                output=exported.output,
                observation_size=exported.observation_size,
                action_size=exported.action_size,
                opset=exported.opset,
            ),
        )
        return self.store.put(manifest, network, conditions_of(run, record))

    # -- specifics ----------------------------------------------------------

    def _profile_of(self, robot: ScenarioRobot) -> tuple[RobotProfile, Path]:
        try:
            source = self.registry.source(robot.profile)
        except ProfileNotFoundError as exc:  # pragma: no cover - check_scenario saw it first
            raise TrainError(str(exc)) from exc
        stated = Path(source.profile.description.urdf)
        description = stated if stated.is_absolute() else source.path.parent / stated
        return source.profile, description.resolve()

    def _task(self, robot: ScenarioRobot) -> Any:
        try:
            return find_task(str(robot.task), BACKEND)
        except TaskError as exc:  # pragma: no cover - check_scenario saw it first
            raise TrainError(f"{robot.id}: {exc}") from exc


# -- the run on disk --------------------------------------------------------


def default_run_dir(spec: TrainSpec, *, now: datetime | None = None) -> Path:
    """``<data>/train/<task>/<time>``: a run is named by when it happened."""
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d_%H-%M-%S")
    return data_home() / RUNS_DIR / spec.task / stamp


def write_run(plan: TrainPlan, conditions: RunConditions) -> Path:
    """Put the plan on disk, so that what the process reads is what was checked."""
    run = plan.run
    run.mkdir(parents=True, exist_ok=True)
    record = TrainRun(spec=plan.spec, conditions=conditions)
    (run / SPEC_FILE).write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (run / DESCRIPTION_FILE).write_text(plan.description, encoding="utf-8")
    for name, content in plan.files.items():
        (run / name).write_text(content, encoding="utf-8")
    return run


def read_run(run: Path) -> TrainRun:
    """The environment and conditions a run was started with."""
    path = Path(run) / SPEC_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrainError(f"{path}: каталог прогона не читается — {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TrainError(f"{path}: не разбирается — {exc}") from exc
    try:
        return parse_model(data, TrainRun, source=path)
    except LoadError as exc:
        raise TrainError(str(exc)) from exc


def conditions_of(run: Path, record: TrainRun) -> dict[str, Any]:
    """What goes into ``train.json`` next to the artifact.

    The conditions the run was asked for, plus whatever the process measured and
    left behind. Outside the artifact's identity (ADR-0046): the same policy
    trained twice on two machines is the same policy.
    """
    conditions: dict[str, Any] = {
        "backend": record.spec.backend,
        "scenario": record.spec.scenario,
        "profile": record.spec.profile.name,
        "task": record.spec.task,
        "contract_id": record.spec.contract_id,
        **record.conditions.model_dump(mode="json", exclude={"schema_version"}),
        "run": str(run),
    }
    measured = Path(run) / CONDITIONS_FILE
    if measured.is_file():
        try:
            extra = json.loads(measured.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            extra = {}
        if isinstance(extra, Mapping):
            conditions.update(extra)
    return conditions


def _the_learner(scenario: Scenario, learner: str | None) -> ScenarioRobot:
    learners = scenario.learners
    if learner is not None:
        found = next((item for item in learners if item.id == learner), None)
        if found is None:
            raise TrainError(
                f"{learner!r}: в сцене {scenario.name!r} такого ученика нет; "
                f"есть: {', '.join(item.id for item in learners) or '—'}"
            )
        return found
    if len(learners) != 1:
        raise TrainError(
            f"сцена {scenario.name!r}: ролей learner {len(learners)}, а учится один "
            f"(ADR-0044)"
        )
    return learners[0]


def _skill_spec(task: Any, robot: ScenarioRobot) -> SkillSpec:
    spec = task.spec
    if not isinstance(spec, SkillSpec):  # pragma: no cover - find_task checked the protocol
        raise TrainError(f"{robot.id}: задача объявляет spec не SkillSpec")
    return spec


def _completion(task: Any) -> Completion:
    completion = task.completion
    if not isinstance(completion, Completion):  # pragma: no cover - same
        raise TrainError("задача объявляет completion не Completion")
    return completion
