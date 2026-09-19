"""``run`` -- execute a scenario's task on one of its robots.

Kept apart from ``sim run`` deliberately, and the split is the shape of the
system rather than a convenience. ``sim run`` brings a world up and owns its
processes; this command connects to a world that is already up and drives one
robot in it. They are two deployment units in the architecture -- the world and
a robot container (section 4) -- so they are two commands, run in two terminals,
and either can be restarted without the other.

Everything decidable before a robot moves is decided before it moves: the
scenario is validated, the robot chosen, the profile loaded, the adapter plugin
resolved, and the skill checked against both the profile and the link. Only then
is anything connected. ``--dry-run`` stops exactly there and prints what it
found, which is the same property ADR-0011 gave the launch commands, applied to
a run.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..core.enums import Role, Status
from ..core.loading import LoadError
from ..core.models import (
    REAL_TARGET,
    SPEED_SCALE_PARAM,
    RobotProfile,
    Scenario,
    ScenarioRobot,
    Task,
    sim_backend_of,
    sim_target,
)
from ..core.plugins import PluginError, load
from ..core.protocols import ControlAdapter
from ..core.twins import TwinError, find_twin
from ..generators.bundle import world_environment
from ..generators.common import GeneratorError
from ..generators.twin import GazeboTwin
from ..registry.registry import Registry
from ..sdk.executor import Executor, RunResult
from ..sdk.record import McapRecorder, RecordError, require_writer
from ..skills.learned import PolicyRejectedError, PolicyRuntimeError
from ..skills.library import SkillLibrary, SkillNotFoundError
from ..store.policies import ArtifactNotFoundError, PolicyStore
from ..validator.environment import Machine, check_adapter
from ..validator.report import Report
from ..validator.rules import check_real_target, check_scenario

NAMESPACE_PARAM = "namespace"
SHOWN_VALUE_WIDTH = 40
"""Longer values of the world's environment are named in the plan, not printed: a
CycloneDDS configuration is a screen of XML."""


class RunError(RuntimeError):
    """The run cannot be set up. Never raised once a robot is moving."""

    def __init__(self, message: str, report: Report | None = None) -> None:
        self.report = report
        if report is not None:
            message = "\n".join([message, *(str(item) for item in report.errors)])
        super().__init__(message)


def choose_robot(scenario: Scenario, wanted: str | None) -> ScenarioRobot:
    """Which robot of the scenario the task is for.

    A scenario with one driven robot answers this by itself. A scenario with
    several does not, and the platform does not pick: distributing tasks among
    robots by capability is the coordinator's job (section 4), and guessing here
    would quietly do a fraction of it.
    """
    if wanted is not None:
        for robot in scenario.robots:
            if robot.id == wanted:
                return robot
        known = ", ".join(item.id for item in scenario.robots)
        raise RunError(f"{wanted!r}: в сцене {scenario.name!r} нет такого робота; есть: {known}")

    driven = [robot for robot in scenario.robots if robot.role is not Role.STATIC]
    if not driven:
        raise RunError(
            f"сцена {scenario.name!r}: управляемых роботов нет — все участники "
            f"объявлены препятствиями (роль static)"
        )
    if len(driven) > 1:
        known = ", ".join(item.id for item in driven)
        raise RunError(
            f"сцена {scenario.name!r}: управляемых роботов {len(driven)} ({known}). "
            f"Укажите робота явно: --robot <id>. Раздача задач по роботам — дело "
            f"координатора, и угадывать за него платформа не станет"
        )
    return driven[0]


def choose_target(scenario: Scenario, profile: RobotProfile, wanted: str | None) -> str:
    """Which adapter of the profile links to this world.

    Derived from the scenario's physics owner, not chosen: a scenario has
    exactly one backend (ADR-0002), and the adapter that talks to it is the one
    keyed by that backend's name.
    """
    target = wanted if wanted is not None else sim_target(scenario.world.backend)
    if target not in profile.adapters:
        known = ", ".join(sorted(profile.adapters)) or "—"
        raise RunError(
            f"{profile.name}: нет адаптера {target!r}; есть: {known}. "
            f"Сцена объявляет владельцем физики {scenario.world.backend!r}, "
            f"и связь с ним нужна каждому участнику (ADR-0002)"
        )
    return target


def adapter_config(
    profile: RobotProfile,
    robot: ScenarioRobot,
    target: str,
    speed_scale: float | None = None,
) -> dict[str, Any]:
    """Configuration handed to the adapter on connect.

    The namespace is the scenario identifier and comes from the scenario, not
    from the profile: two robots of one kind share a profile and must not share
    topics. The profile's own parameters are merged underneath, so a profile can
    state what belongs to the link and nothing it cannot know.

    The speed scale is the run's, and is handed only to an adapter that declared it
    executes one (ADR-0052).
    """
    config = dict(profile.adapters[target].params)
    config.setdefault(NAMESPACE_PARAM, robot.id)
    if speed_scale is not None:
        config[SPEED_SCALE_PARAM] = speed_scale
    return config


def adapter_class(profile: RobotProfile, target: str) -> Any:
    """The adapter plugin the profile names for this target, not yet constructed."""
    plugin = profile.adapters[target].plugin
    try:
        factory = load("adapters", plugin)
    except PluginError as exc:
        raise RunError(
            f"{profile.name}: адаптер {plugin!r} для {target!r} не загружается — {exc}"
        ) from exc
    if not callable(factory):
        raise RunError(f"{plugin!r}: точка входа не вызываема")
    return factory


def build_adapter(profile: RobotProfile, target: str) -> ControlAdapter:
    """Load the adapter plugin the profile names for this target."""
    return adapter_class(profile, target)()  # type: ignore[no-any-return]


def choose_speed_scale(target: str, wanted: float | None) -> float:
    """The run's speed scale: stated on hardware every time, 1 in a simulator (ADR-0052)."""
    if wanted is None:
        if target == REAL_TARGET:
            raise RunError(
                f"цель {REAL_TARGET!r}: шкала скорости обязательна, умолчания нет — "
                f"--speed-scale s, 0 < s ≤ 1; первый прогон на железе замедлен (раздел 12, "
                f"ADR-0052). Полная скорость пишется явно: --speed-scale 1"
            )
        return 1.0
    if not 0 < wanted <= 1:
        raise RunError(f"--speed-scale {wanted:g}: нужно 0 < s ≤ 1 (ADR-0052)")
    return wanted


def refuse_undeclared_scale(plugin: str, factory: Any, scale: float) -> bool:
    """Whether the adapter executes the scale; refused when it is not 1 and it does not."""
    declared = getattr(factory, "supports_speed_scale", None)
    if scale != 1 and declared is not True:
        raise RunError(
            f"адаптер {plugin!r} не исполняет шкалу скорости (supports_speed_scale = "
            f"{declared!r}), а прогон просит {scale:g}. Связь, не умеющая замедлить робота, "
            f"шкалу не получает — прогон отменён (ADR-0052)"
        )
    return declared is True


def task_of(scenario: Scenario, skill: str | None, goal: dict[str, Any]) -> Task:
    """The task to run: the scenario's, or one named on the command line."""
    if skill is not None:
        return Task(skill=skill, goal=goal)
    if scenario.task is None:
        raise RunError(
            f"сцена {scenario.name!r} не объявляет задачу и навык не задан; "
            f"добавьте блок task или укажите --skill"
        )
    if goal:
        return Task(skill=scenario.task.skill, goal={**scenario.task.goal, **goal})
    return scenario.task


def control_plane(
    robot: ScenarioRobot, profile: RobotProfile, target: str
) -> Mapping[str, str | None] | None:
    """Names of the link as the twin of this adapter states them, if one does.

    A lookup that fails says nothing here: the scenario was validated before a
    plan is made, and a twin that cannot be loaded is a finding there
    (``twin.unloadable``), not a line on a plan.
    """
    backend = sim_backend_of(target)
    if backend is None:
        return None
    try:
        twin = find_twin(profile.adapters[target].plugin, backend)
    except TwinError:
        return None
    if not isinstance(twin, GazeboTwin):
        return None
    return twin.control_plane(robot.id, profile)


def plan(
    scenario: Scenario,
    robot: ScenarioRobot,
    profile: RobotProfile,
    target: str,
    task: Task,
    speed_scale: float = 1.0,
) -> list[str]:
    """What the run will do, in lines, before it does any of it.

    The control plane is printed from the twin of the link -- the same
    derivation the generator wrote into the bundle manifest and the adapter
    subscribes to -- so these lines can be held against ``ros2 topic list`` on
    the running machine.

    Printed only when a twin names it, because that is exactly when those
    topics exist. A robot made of numbers has no topics, and naming some for it
    would put a plausible falsehood on the screen at the moment a human is
    looking for the truth.
    """
    binding = profile.skill(task.skill)
    made_of = f" -> политика {binding.policy}" if binding is not None and binding.policy else ""
    world = (
        "мир       действительность, окружение мира не применяется"
        if target == REAL_TARGET
        else f"мир       {scenario.world.backend}  (владелец физики)"
    )
    lines = [
        f"сцена     {scenario.name}",
        world,
        f"робот     {robot.id}  профиль {profile.name}  роль {robot.role}",
        f"связь     {target} -> {profile.adapters[target].plugin}",
        f"контракт  {profile.contract.contract_id}  revision {profile.contract.revision}",
        f"навык     {task.skill}{made_of}"
        + (f"  цель {task.goal}" if task.goal else ""),
        f"темп      команды {profile.contract.command_rate_hz:g} Гц, "
        f"состояние {profile.contract.state_rate_hz:g} Гц",
        f"шкала     скорости {speed_scale:g}",
    ]
    plane = control_plane(robot, profile, target)
    if plane is not None and plane.get("command_topic") is not None:
        lines.append(f"состояние {plane['state_topic']}  ({plane['state_message']})")
        lines.append(f"команды   {plane['command_topic']}  ({plane['command_message']})")
    return lines


def environment_conflicts(world: Mapping[str, str], process: Mapping[str, str]) -> list[str]:
    """Variables this process already has with a value the world does not (ADR-0039)."""
    return [
        f"{name}: у процесса {process[name]!r}, у мира {value!r}"
        for name, value in world.items()
        if name in process and process[name] != value
    ]


def describe_environment(world: Mapping[str, str]) -> str:
    return ", ".join(
        f"{name}={value}" if len(value) <= SHOWN_VALUE_WIDTH else f"{name}=<{len(value)} симв.>"
        for name, value in world.items()
    )


def record_metadata(
    scenario: Scenario,
    robot: ScenarioRobot,
    profile: RobotProfile,
    target: str,
    task: Task,
    speed_scale: float = 1.0,
) -> dict[str, str]:
    """What a record says about its run before the first tick (ADR-0038)."""
    contract = profile.contract
    return {
        "scenario": scenario.name,
        "backend": scenario.world.backend,
        "robot": robot.id,
        "profile": profile.name,
        "target": target,
        "speed_scale": f"{speed_scale:g}",
        "adapter": profile.adapters[target].plugin,
        "contract_id": contract.contract_id,
        "revision": str(contract.revision),
        "contract": contract.canonical_json(),
        "skill": task.skill,
        "policy": getattr(profile.skill(task.skill), "policy", None) or "",
        "goal": json.dumps(task.goal, ensure_ascii=False, default=str),
        "platform_version": __version__,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def execute(
    scenario: Scenario,
    registry: Registry,
    *,
    robot_id: str | None = None,
    target: str | None = None,
    skill: str | None = None,
    goal: dict[str, Any] | None = None,
    library: SkillLibrary | None = None,
    store: PolicyStore | None = None,
    dry_run: bool = False,
    machine: Machine | None = None,
    scenario_path: Path | None = None,
    record: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
    speed_scale: float | None = None,
) -> tuple[list[str], RunResult | None]:
    """Set the run up, then run it. Returns the plan it printed and the result.

    The scenario is validated first and a failing one is not run at all. The
    world is already up by the time this command is called, which makes the
    check look redundant -- it is not: ``sim run`` refuses to *build* a bad
    scene, and nothing stops a hand-started world or an edited profile from
    putting one there anyway (invariant 6).

    ``environ`` is the environment of this process, the one the world's is applied
    to (ADR-0039); ``record`` is where the run is written as it happens (ADR-0038).

    The target chooses who owns the physics (ADR-0051): ``sim.<backend>`` is the
    scenario's world, ``real`` is reality -- the scenario is still checked whole, the
    world's environment is not applied, and the envelope must be declared.
    ``speed_scale`` is the run's (ADR-0052): required on ``real``, 1 elsewhere.
    """
    process = os.environ if environ is None else environ
    skills = library if library is not None else SkillLibrary()
    policies = store if store is not None else PolicyStore()
    report = check_scenario(scenario, registry, path=scenario_path, store=policies)
    if not report.ok:
        raise RunError(
            f"сцена {scenario.name!r} не проходит проверку, запуск отменён "
            f"(инвариант 6):",
            report,
        )
    robot = choose_robot(scenario, robot_id)
    profile = registry.load(robot.profile)
    chosen = choose_target(scenario, profile, target)
    on_hardware = chosen == REAL_TARGET
    if on_hardware:
        envelope = check_real_target(profile)
        if not envelope.ok:
            raise RunError(f"{robot.id}: на железо не идёт (раздел 12):", envelope)
    scale = choose_speed_scale(chosen, speed_scale)
    task = task_of(scenario, skill, dict(goal or {}))
    lines = plan(scenario, robot, profile, chosen, task, scale)

    # The declaration of the skill as this robot runs it: from the library, or from the
    # manifest of the policy the profile binds it to (ADR-0046). An artifact this machine
    # does not have is a line on a dry run's plan and a refusal of a real one.
    try:
        skills.spec_for(profile, task.skill, store=policies)
    except ArtifactNotFoundError as exc:
        if not dry_run:
            raise RunError(f"{robot.id}: {exc}") from exc
        lines.append(f"политика  не найдена: {exc}")
    except (SkillNotFoundError, LoadError) as exc:
        raise RunError(str(exc)) from exc
    try:
        # Reality is not the scenario's world: its environment would lead the link to
        # a real robot into the simulator's domain (ADR-0051).
        world = {} if on_hardware else world_environment(scenario, registry)
    except GeneratorError as exc:
        if not dry_run:
            raise RunError(
                f"{robot.id}: окружение мира {scenario.name!r} на этой машине не выводится, "
                f"а без него связь ждала бы робота в чужом домене (ADR-0039): {exc}"
            ) from exc
        world = {}
        lines.append(f"окружение не выводится: {exc}")
    conflicts = environment_conflicts(world, process)
    if conflicts:
        raise RunError(
            f"{robot.id}: окружение процесса расходится с окружением мира {scenario.name!r}, "
            f"в котором работает робот (ADR-0039):\n  "
            + "\n  ".join(conflicts)
            + "\nуберите эти переменные из окружения процесса или поднимите мир в нём"
        )
    if world:
        lines.append(f"окружение {describe_environment(world)}")
    if record is not None:
        lines.append(f"запись    {record}")
    if dry_run:
        # Nothing is connected and nothing is constructed: a dry run must be possible
        # on a machine that has neither the robot nor the plugin. Whether the link
        # executes the scale is a declaration of its class (ADR-0052), so a plugin that
        # is here answers it, and a plan that says "go" is not refused by the run.
        try:
            factory = adapter_class(profile, chosen)
        except RunError as exc:
            lines.append(f"шкала     не сверена с адаптером: {exc}")
        else:
            refuse_undeclared_scale(profile.adapters[chosen].plugin, factory, scale)
        return lines, None

    if record is not None:
        try:
            require_writer()
        except RecordError as exc:
            raise RunError(f"{robot.id}: {exc}") from exc

    # The skill is built before the link is: a policy that cannot drive this robot is a
    # report with codes here, not a robot going mad on the first tick (ADR-0047).
    try:
        runnable = skills.create_for(profile, task.skill, store=policies)
    except PolicyRejectedError as exc:
        raise RunError(
            f"{robot.id}: политика навыка {task.skill!r} не может вести этого робота, "
            f"запуск отменён (инвариант 6):",
            exc.report,
        ) from exc
    except (PolicyRuntimeError, ArtifactNotFoundError, SkillNotFoundError, LoadError) as exc:
        raise RunError(f"{robot.id}: {exc}") from exc

    process.update(world)

    # What the adapter declares it needs is checked before it is constructed:
    # an unsourced environment is a named finding, not an import error from
    # somewhere inside connect (ADR-0025).
    plugin = profile.adapters[chosen].plugin
    environment = check_adapter(plugin, machine=machine)
    if not environment.report.ok:
        raise RunError(
            f"{robot.id}: окружению этой машины не хватает того, что нужно адаптеру "
            f"{plugin!r}, запуск отменён:",
            environment.report,
        )

    factory = adapter_class(profile, chosen)
    handed = scale if refuse_undeclared_scale(plugin, factory, scale) else None
    adapter: ControlAdapter = factory()
    try:
        adapter.connect(profile, adapter_config(profile, robot, chosen, handed))
    except (RuntimeError, ImportError) as exc:
        # A link that cannot be established is a setup failure by definition,
        # and this command reports setup failures instead of unwinding: the
        # commonest one is an environment that was never sourced, and a
        # traceback is the wrong way to say so. An import that fails inside a
        # vendor's library on first use is the same failure arriving late.
        raise RunError(
            f"{robot.id}: связь {chosen!r} не установлена — {exc}"
        ) from exc
    try:
        recorder = _open_record(
            record, record_metadata(scenario, robot, profile, chosen, task, scale)
        )
        result: RunResult | None = None
        try:
            result = Executor(
                adapter, profile, library=skills, recorder=recorder, store=policies
            ).run(runnable, task.goal, name=task.skill)  # type: ignore[arg-type]
        finally:
            if recorder is not None:
                recorder.close(result, error=sys.exc_info()[1])
    finally:
        adapter.disconnect()
    return lines, result


def _open_record(path: Path | None, run: Mapping[str, str]) -> McapRecorder | None:
    if path is None:
        return None
    try:
        return McapRecorder(path, run)
    except RecordError as exc:
        raise RunError(str(exc)) from exc


def describe(result: RunResult) -> list[str]:
    """The run, afterwards."""
    mark = {
        Status.SUCCESS: "выполнено",
        Status.FAILURE: "отказ    ",
        Status.RUNNING: "прервано ",
    }[result.status]
    lines = [f"{mark}  навык {result.skill}, тактов {result.ticks}, "
             f"команд {len(result.commands)}"]
    if result.reason:
        lines.append(f"причина   {result.reason}")
    return lines
