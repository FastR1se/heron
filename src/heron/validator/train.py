"""Rules over a scenario whose world is a training environment (ADR-0044).

A scenario with ``world.backend`` naming a training backend is a different kind
of scene, and most of what the validator says about a simulated one does not
apply to it: there is no adapter to require, because nobody drives the learner --
the algorithm does -- and there is no twin, because there is no link. What does
apply is everything that decides whether an environment can be derived from the
contract at all, and that is checked here, before a process that costs GPU hours
is started.

As with the ``twin.*`` rules (ADR-0024), these run only in the world of an
installed backend: a scenario naming a training backend nobody has installed is
refused by name when somebody tries to train it, and validating it on a machine
without one would report the absence of a plugin as a defect of the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from ..core.contract import JOINT_ACTIONS, Contract
from ..core.enums import Role
from ..core.models import RobotProfile, Scenario, ScenarioRobot
from ..core.plugins import available
from ..train.spec import RATE_TOLERANCE, SUPPORTED_TERMS, decimation_of
from ..train.tasks import TaskError, find_task
from .report import Report, error

ADR: Final = "ADR-0044"


def is_training_world(scenario: Scenario) -> bool:
    """Whether this scenario's world is a training backend installed on this machine."""
    return scenario.world.backend in available("train")


def check_training(
    scenario: Scenario,
    participants: list[tuple[ScenarioRobot, RobotProfile]],
    report: Report,
) -> None:
    """Everything a training scenario has to satisfy before an environment is built."""
    where = f"scenario:{scenario.name}"
    backend = scenario.world.backend

    if backend in available("sim"):
        report.add(
            error(
                "train.backend_ambiguous",
                f"{where}.world.backend",
                f"{backend!r} зарегистрирован и как бэкенд симуляции, и как бэкенд обучения",
                hint=(
                    "одно имя — один владелец физики; переименуйте один из плагинов "
                    "или снимите лишний"
                ),
                adr="ADR-0002",
                detail={"backend": backend},
            )
        )

    learners = scenario.learners
    if not learners:
        report.add(
            error(
                "train.learner_missing",
                where,
                f"мир {backend!r} — среда обучения, а роли learner в сцене нет",
                hint="учиться должен кто-то один: объявьте роль learner и её task",
                adr=ADR,
                detail={"roles": sorted({str(item.role) for item in scenario.robots})},
            )
        )

    neighbours = [item.id for item in scenario.robots if item.role is not Role.LEARNER]
    if neighbours:
        report.add(
            error(
                "train.neighbours_unsupported",
                where,
                f"кроме ученика в сцене обучения есть участники: {', '.join(neighbours)}",
                hint=(
                    "сосед в среде обучения — отложенная задача, а не забытая: сначала "
                    "один ученик, затем роли frozen, replay и scripted рядом с ним"
                ),
                adr=ADR,
                detail={"neighbours": neighbours},
            )
        )

    for robot, profile in participants:
        if robot.role is not Role.LEARNER or robot.task is None:
            continue
        spot = f"{where}.robots.{robot.id}"
        report.extend(check_contract(profile.contract, where=spot).findings)
        try:
            find_task(robot.task, backend)
        except TaskError as exc:
            report.add(
                error(
                    exc.code,
                    f"{spot}.task",
                    str(exc),
                    hint="задача обучения ставится пакетом и объявляется точкой входа",
                    adr=ADR,
                    detail={"task": robot.task, "backend": backend},
                )
            )


def check_contract(contract: Contract, *, where: str) -> Report:
    """Everything about a contract that decides whether an environment is derivable.

    Separate from building one so that the validator can report it over a
    scenario, before anything asks for a plan (invariant 6).
    """
    report = Report()
    action = contract.action

    if decimation_of(contract) is None:
        report.add(
            error(
                "train.rates_not_integer",
                f"{where}.contract",
                f"состояние {contract.state_rate_hz:g} Гц не делится нацело на команды "
                f"{contract.command_rate_hz:g} Гц: шагов физики на такт политики "
                f"{contract.state_rate_hz / contract.command_rate_hz:g}",
                hint=(
                    "шаг физики — 1/state_rate_hz, такт политики — 1/command_rate_hz; "
                    "среда не умеет делить шаг"
                ),
                adr=ADR,
                detail={
                    "state_rate_hz": contract.state_rate_hz,
                    "command_rate_hz": contract.command_rate_hz,
                },
            )
        )

    if action.type not in JOINT_ACTIONS:
        report.add(
            error(
                "train.action_unsupported",
                f"{where}.contract.action",
                f"действие {action.type} среда обучения не строит; "
                f"есть: {', '.join(sorted(JOINT_ACTIONS))}",
                hint="действие политики — значения по суставам contract.joint_order",
                adr="ADR-0041",
                detail={"type": str(action.type)},
            )
        )
    elif not action.has_gains:
        report.add(
            error(
                "train.gains_missing",
                f"{where}.contract.action",
                "в действии нет kp и kd, а политика учится через конкретный закон привода",
                hint=(
                    "объявите kp и kd в contract.action: без них жёсткость в обучении "
                    "задаёт импортёр, а в исполнении — симулятор, и это разные законы"
                ),
                adr="ADR-0045",
            )
        )

    if action.clip is not None and abs(action.clip[0] + action.clip[1]) > RATE_TOLERANCE:
        report.add(
            error(
                "train.clip_not_symmetric",
                f"{where}.contract.action.clip",
                f"clip {action.clip} несимметричен, а обучение ограничивает сырой выход "
                f"одним числом",
                hint=(
                    "клип над сырым выходом здесь один на все суставы и симметричен; "
                    "несимметричную рамку выразите смещением и масштабом действия"
                ),
                adr="ADR-0041",
                detail={"clip": list(action.clip)},
            )
        )

    unsupported = sorted(
        {str(term.type) for term in contract.observation if term.type not in SUPPORTED_TERMS}
    )
    if unsupported:
        report.add(
            error(
                "train.observation_term_unsupported",
                f"{where}.contract.observation",
                f"термы, которых среда обучения не строит: {', '.join(unsupported)}",
                hint="терм реестра, которого не умеет бэкенд, — правка бэкенда, а не контракта",
                adr="ADR-0042",
                detail={"terms": unsupported},
            )
        )
    return report


def check_goal_range(
    minimum: Sequence[float],
    maximum: Sequence[float],
    profile: RobotProfile,
    *,
    where: str,
) -> Report:
    """Whether a task may draw its goal from this box (ADR-0043, item 5).

    Two conditions, and they are different questions. A goal outside the profile's
    envelope is one the robot may not be asked for at all. A goal outside the set
    of targets the mapped action can command is one the policy cannot reach however
    well it learns -- it would be trained against an unreachable goal and score the
    failure as its own.
    """
    contract = profile.contract
    report = Report()
    dof = contract.dof
    if len(minimum) != dof or len(maximum) != dof:
        report.add(
            error(
                "train.goal_range_width",
                where,
                f"диапазон цели на {len(minimum)}/{len(maximum)} суставов при {dof} "
                f"в contract.joint_order",
                hint="диапазон упорядочен как joint_order, по значению на сустав",
                adr="ADR-0043",
            )
        )
        return report

    wrong = [
        contract.joint_order[i] for i, (lo, hi) in enumerate(zip(minimum, maximum, strict=True))
        if lo > hi
    ]
    if wrong:
        report.add(
            error(
                "train.goal_range_width",
                where,
                f"нижняя граница выше верхней в суставах {', '.join(wrong)}",
                hint="диапазон — прямоугольник: min по суставу не больше max",
                adr="ADR-0043",
            )
        )
        return report

    bounds = profile.limits.joint_position
    if bounds is not None and len(bounds.min) == dof:
        outside = {
            contract.joint_order[i]: {"range": [minimum[i], maximum[i]],
                                      "limits": [bounds.min[i], bounds.max[i]]}
            for i in range(dof)
            if minimum[i] < bounds.min[i] or maximum[i] > bounds.max[i]
        }
        if outside:
            report.add(
                error(
                    "train.goal_outside_limits",
                    where,
                    f"диапазон цели выходит за limits.joint_position в суставах "
                    f"{', '.join(outside)}",
                    hint="цель обучения — поза, которую роботу можно приказать",
                    adr="ADR-0043",
                    detail={"joints": outside},
                )
            )

    targets = contract.action_targets()
    if targets is None:
        return report
    unreachable = {
        contract.joint_order[i]: {"range": [minimum[i], maximum[i]], "targets": list(targets[i])}
        for i in range(dof)
        if minimum[i] < targets[i][0] or maximum[i] > targets[i][1]
    }
    if unreachable:
        report.add(
            error(
                "train.goal_unreachable",
                where,
                f"диапазон цели шире множества целей действия в суставах "
                f"{', '.join(unreachable)}",
                hint=(
                    "политика приказывает clip(a)·scale + origin; цель вне этого множества "
                    "недостижима при любом обучении"
                ),
                adr="ADR-0041",
                detail={"joints": unreachable},
            )
        )
    return report
