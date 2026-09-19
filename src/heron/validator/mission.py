"""A mission against a scenario: who does each task, decided before anything starts.

ADR-0066. A robot is a candidate for a task when the comparison ``skill check``
makes holds for it -- the skill declared by its profile, capabilities by equality
of keys, the control class by equality, the quantities of ``requires`` -- and when
it also meets what the task itself asks for. Exactly one candidate is an
assignment; none, or more than one, is an error with the reasons named. The
assignment is static: it is printed before a run and does not change during one.

ADR-0067. ``after`` names only tasks written above, which is what keeps the order
free of cycles by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..coordinator.mission import Mission, MissionTask
from ..core import capabilities as caps
from ..core import requires as req
from ..core.enums import Role
from ..core.loading import LoadError
from ..core.models import RobotProfile, Scenario, ScenarioRobot
from ..registry.registry import ProfileNotFoundError, Registry
from ..skills.library import SkillLibrary, SkillNotFoundError
from ..store.policies import ArtifactNotFoundError, PolicyStore
from .report import Report, error, warning
from .rules import check_skill_against_profile

ADR = "ADR-0066"


@dataclass(frozen=True, slots=True)
class Reason:
    """Why a robot cannot take a task: the code of the rule that failed, and its words."""

    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(slots=True)
class MissionCheck:
    """What the validator found, and the assignment it could make."""

    report: Report = field(default_factory=Report)
    assignment: dict[str, str] = field(default_factory=dict)
    """Task id -> robot id, for every task with exactly one candidate."""

    reasons: dict[str, dict[str, list[Reason]]] = field(default_factory=dict)
    """Task id -> robot id -> why that robot is not a candidate."""

    @property
    def ok(self) -> bool:
        return self.report.ok


@dataclass(frozen=True, slots=True)
class _Participant:
    robot: ScenarioRobot
    profile: RobotProfile | None


def check_mission(
    mission: Mission,
    scenario: Scenario,
    registry: Registry,
    *,
    library: SkillLibrary | None = None,
    store: PolicyStore | None = None,
) -> MissionCheck:
    """Every task assigned to exactly one capable robot, and every wait written above."""
    skills = library if library is not None else SkillLibrary()
    policies = store if store is not None else PolicyStore()
    participants = {robot.id: _participant(robot, registry) for robot in scenario.robots}
    result = MissionCheck()
    position = {task.id: index for index, task in enumerate(mission.tasks)}

    for index, task in enumerate(mission.tasks):
        spot = f"mission:{mission.name}.tasks.{task.id}"
        _check_waits(mission, index, position, spot, result.report)

        if task.robot is not None and task.robot not in participants:
            result.report.add(
                error(
                    "mission.robot_unknown",
                    spot,
                    f"задача привязана к роботу {task.robot!r}, а в сцене {scenario.name!r} "
                    f"его нет; есть: {', '.join(participants)}",
                    hint="привязка сужает круг кандидатов сцены, а не добавляет робота",
                    adr=ADR,
                    detail={"robot": task.robot, "scenario": scenario.name},
                )
            )
            continue

        pool = [participants[task.robot]] if task.robot is not None else list(participants.values())
        candidates: list[str] = []
        unverified: dict[str, str] = {}
        refusals: dict[str, list[Reason]] = {}
        for participant in pool:
            reasons, doubt = _qualify(task, participant, skills, policies)
            if reasons:
                refusals[participant.robot.id] = reasons
                continue
            candidates.append(participant.robot.id)
            if doubt is not None:
                unverified[participant.robot.id] = doubt
        result.reasons[task.id] = refusals

        if len(candidates) == 1:
            robot = candidates[0]
            result.assignment[task.id] = robot
            if robot in unverified:
                result.report.add(
                    warning(
                        "mission.skill_unverified",
                        spot,
                        f"навык {task.skill!r} робота {robot!r} не сверен: {unverified[robot]}",
                        hint=(
                            "робот назначен по объявлению в профиле; процесс робота откажет "
                            "до подключения, если политики не будет и там"
                        ),
                        adr="ADR-0046",
                        detail={"robot": robot, "skill": task.skill},
                    )
                )
        elif not candidates:
            result.report.add(
                error(
                    "mission.task_unassigned",
                    spot,
                    f"навык {task.skill!r}: ни один робот сцены {scenario.name!r} не может "
                    f"выполнить задачу — "
                    + "; ".join(
                        f"{robot}: {', '.join(r.message for r in reasons)}"
                        for robot, reasons in refusals.items()
                    ),
                    hint=(
                        "сопоставление — то же, что skill check: навык в профиле, возможности "
                        "равенством ключей, класс связи, requires; плюс требования задачи"
                    ),
                    adr=ADR,
                    detail={
                        "skill": task.skill,
                        "reasons": {
                            robot: [r.as_dict() for r in reasons]
                            for robot, reasons in refusals.items()
                        },
                    },
                )
            )
        else:
            result.report.add(
                error(
                    "mission.task_ambiguous",
                    spot,
                    f"навык {task.skill!r}: подходят роботы {', '.join(candidates)} — "
                    f"координатор не выбирает наугад",
                    hint=(
                        "сузьте задачу: required_capabilities или requires, которым отвечает "
                        "только один робот, или привяжите её полем robot"
                    ),
                    adr=ADR,
                    detail={"skill": task.skill, "candidates": candidates},
                )
            )
    return result


def _participant(robot: ScenarioRobot, registry: Registry) -> _Participant:
    """A robot and its profile; a profile that cannot be found is the scenario's finding."""
    try:
        return _Participant(robot, registry.load(robot.profile))
    except (ProfileNotFoundError, LoadError):
        return _Participant(robot, None)


def _check_waits(
    mission: Mission, index: int, position: dict[str, int], spot: str, report: Report
) -> None:
    for name in mission.tasks[index].after or ():
        if name not in position:
            report.add(
                error(
                    "mission.dependency_unknown",
                    f"{spot}.after",
                    f"задача ждёт {name!r}, а такой задачи в миссии нет",
                    hint="after называет задачи миссии, записанные выше",
                    adr="ADR-0067",
                    detail={"after": name},
                )
            )
        elif position[name] >= index:
            report.add(
                error(
                    "mission.dependency_forward",
                    f"{spot}.after",
                    f"задача ждёт {name!r}, записанную "
                    + ("ею же" if position[name] == index else "ниже"),
                    hint=(
                        "ждать можно только задачи выше: так порядок не может замкнуться в "
                        "цикл; переставьте задачи"
                    ),
                    adr="ADR-0067",
                    detail={"after": name},
                )
            )


def _qualify(
    task: MissionTask,
    participant: _Participant,
    skills: SkillLibrary,
    policies: PolicyStore,
) -> tuple[list[Reason], str | None]:
    """Why this robot cannot take the task (empty when it can), and a doubt, if any."""
    robot, profile = participant.robot, participant.profile
    if profile is None:
        return [Reason("scenario.profile_not_found", f"профиль {robot.profile!r} не найден")], None
    if robot.role is Role.STATIC:
        return [Reason("mission.robot_static", "препятствие (роль static) задач не получает")], None
    if profile.skill(task.skill) is None:
        return [
            Reason("scenario.skill_not_in_profile", f"профиль {profile.name!r} не объявляет навык")
        ], None

    reasons: list[Reason] = []
    doubt: str | None = None
    try:
        spec = skills.spec_for(profile, task.skill, store=policies)
    except ArtifactNotFoundError as exc:
        spec, doubt = None, str(exc)
    except (SkillNotFoundError, LoadError) as exc:
        return [Reason("skill.unresolved", str(exc))], None
    if spec is not None:
        reasons.extend(
            Reason(finding.code, finding.message)
            for finding in check_skill_against_profile(spec, profile).errors
        )

    absent = caps.missing(task.required_capabilities, profile.capabilities)
    if absent:
        reasons.append(
            Reason("task.capabilities_missing", f"не объявляет {', '.join(sorted(absent))}")
        )
    reasons.extend(
        Reason("task.requirement_unmet", unmet.describe())
        for unmet in req.check(task.requires, profile)
    )
    return reasons, doubt


def describe_reasons(reasons: dict[str, list[Reason]]) -> list[str]:
    """One line per robot that is not a candidate, for a human."""
    return [
        f"{robot}: " + "; ".join(f"[{r.code}] {r.message}" for r in items)
        for robot, items in reasons.items()
    ]
