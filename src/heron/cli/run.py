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

from collections.abc import Mapping
from typing import Any

from ..core.enums import Role, Status
from ..core.models import RobotProfile, Scenario, ScenarioRobot, Task, sim_backend_of, sim_target
from ..core.plugins import PluginError, load
from ..core.protocols import ControlAdapter
from ..core.twins import TwinError, find_twin
from ..generators.twin import GazeboTwin
from ..registry.registry import Registry
from ..sdk.executor import Executor, RunResult
from ..skills.library import SkillLibrary, SkillNotFoundError
from ..validator.environment import Machine, check_adapter
from ..validator.report import Report
from ..validator.rules import check_scenario

NAMESPACE_PARAM = "namespace"


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


def adapter_config(profile: RobotProfile, robot: ScenarioRobot, target: str) -> dict[str, Any]:
    """Configuration handed to the adapter on connect.

    The namespace is the scenario identifier and comes from the scenario, not
    from the profile: two robots of one kind share a profile and must not share
    topics. The profile's own parameters are merged underneath, so a profile can
    state what belongs to the link and nothing it cannot know.
    """
    config = dict(profile.adapters[target].params)
    config.setdefault(NAMESPACE_PARAM, robot.id)
    return config


def build_adapter(profile: RobotProfile, target: str) -> ControlAdapter:
    """Load the adapter plugin the profile names for this target."""
    plugin = profile.adapters[target].plugin
    try:
        factory = load("adapters", plugin)
    except PluginError as exc:
        raise RunError(
            f"{profile.name}: адаптер {plugin!r} для {target!r} не загружается — {exc}"
        ) from exc
    if not callable(factory):
        raise RunError(f"{plugin!r}: точка входа не вызываема")
    return factory()  # type: ignore[no-any-return]


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
    lines = [
        f"сцена     {scenario.name}  (владелец физики {scenario.world.backend})",
        f"робот     {robot.id}  профиль {profile.name}  роль {robot.role}",
        f"связь     {target} -> {profile.adapters[target].plugin}",
        f"контракт  {profile.contract.contract_id}  revision {profile.contract.revision}",
        f"навык     {task.skill}"
        + (f"  цель {task.goal}" if task.goal else ""),
        f"темп      команды {profile.contract.command_rate_hz:g} Гц, "
        f"состояние {profile.contract.state_rate_hz:g} Гц",
    ]
    plane = control_plane(robot, profile, target)
    if plane is not None and plane.get("command_topic") is not None:
        lines.append(f"состояние {plane['state_topic']}  ({plane['state_message']})")
        lines.append(f"команды   {plane['command_topic']}  ({plane['command_message']})")
    return lines


def execute(
    scenario: Scenario,
    registry: Registry,
    *,
    robot_id: str | None = None,
    target: str | None = None,
    skill: str | None = None,
    goal: dict[str, Any] | None = None,
    library: SkillLibrary | None = None,
    dry_run: bool = False,
    machine: Machine | None = None,
) -> tuple[list[str], RunResult | None]:
    """Set the run up, then run it. Returns the plan it printed and the result.

    The scenario is validated first and a failing one is not run at all. The
    world is already up by the time this command is called, which makes the
    check look redundant -- it is not: ``sim run`` refuses to *build* a bad
    scene, and nothing stops a hand-started world or an edited profile from
    putting one there anyway (invariant 6).
    """
    report = check_scenario(scenario, registry)
    if not report.ok:
        raise RunError(
            f"сцена {scenario.name!r} не проходит проверку, запуск отменён "
            f"(инвариант 6):",
            report,
        )
    robot = choose_robot(scenario, robot_id)
    profile = registry.load(robot.profile)
    chosen = choose_target(scenario, profile, target)
    try:
        task = task_of(scenario, skill, dict(goal or {}))
        (library or SkillLibrary()).spec(task.skill)
    except SkillNotFoundError as exc:
        raise RunError(str(exc)) from exc

    lines = plan(scenario, robot, profile, chosen, task)
    if dry_run:
        # Nothing is connected and nothing is loaded: a dry run must be
        # possible on a machine that has neither the robot nor the plugin.
        return lines, None

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

    adapter = build_adapter(profile, chosen)
    try:
        adapter.connect(profile, adapter_config(profile, robot, chosen))
    except RuntimeError as exc:
        # A link that cannot be established is a setup failure by definition,
        # and this command reports setup failures instead of unwinding: the
        # commonest one is an environment that was never sourced, and a
        # traceback is the wrong way to say so.
        raise RunError(
            f"{robot.id}: связь {chosen!r} не установлена — {exc}"
        ) from exc
    try:
        result = Executor(adapter, profile, library=library).run_task(task)
    finally:
        adapter.disconnect()
    return lines, result


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
