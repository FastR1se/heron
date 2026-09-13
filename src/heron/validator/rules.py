"""Validation rules.

Invariant 6: a mismatch between environments is a validator error, not strange
behaviour of a robot. Everything checkable before a run is checked here.

Rules are grouped by what they need to see. A rule over one profile cannot know
about a scenario, and a rule over a scenario needs the registry; keeping that
split explicit is what stops the validator from quietly becoming a loader.
"""

from __future__ import annotations

from pathlib import Path

from ..core import capabilities as caps
from ..core import requires as req
from ..core.branding import DOCKER_NAMESPACE
from ..core.contract import Contract, diff
from ..core.enums import ControlMode, Role, TwinKind
from ..core.models import (
    PolicyManifest,
    RobotProfile,
    Scenario,
    ScenarioRobot,
    SkillSpec,
    sim_target,
)
from ..core.plugins import PluginError, available
from ..core.twins import TwinError, find_twin
from ..registry.lock import ContractLock
from ..registry.registry import ProfileNotFoundError, Registry
from .description import check_description
from .report import Report, error, warning

# --------------------------------------------------------------------------
# One profile on its own.
# --------------------------------------------------------------------------


def check_profile(
    profile: RobotProfile,
    *,
    lock: ContractLock | None = None,
    path: Path | None = None,
) -> Report:
    """Everything decidable from a single profile.

    ``path`` is where the profile was read from. With it the description it
    points at becomes reachable, and the rules of ``description`` join in; a
    profile validated as a bare object is validated without them, and says so by
    simply not reporting them.
    """
    report = Report()
    where = f"profile:{profile.name}"
    contract = profile.contract

    _check_limit_lengths(profile, report)
    _check_rates(profile, report)
    _check_observation_terms(profile, report)
    _check_image(profile, report)
    _check_adapters(profile, report)
    if path is not None:
        report.extend(check_description(profile, path).findings)

    if lock is not None:
        previous = lock.undeclared_change(profile.name, contract)
        if previous is not None:
            report.add(
                error(
                    "contract.undeclared_change",
                    f"{where}.contract",
                    f"нормативное содержимое контракта изменилось, "
                    f"а revision остался {contract.revision}",
                    hint=(
                        f"было {previous.contract_id}, стало {contract.contract_id}. "
                        f"Поднимите revision и выполните: contract lock {profile.name}"
                    ),
                    adr="ADR-0004",
                    detail={
                        "was": previous.contract_id,
                        "now": contract.contract_id,
                        "revision": contract.revision,
                    },
                )
            )
    return report


def _check_limit_lengths(profile: RobotProfile, report: Report) -> None:
    """Per-joint limits are ordered like ``joint_order``, so lengths must match."""
    dof = profile.contract.dof
    where = f"profile:{profile.name}.limits"
    checks = (
        ("joint_position.min", getattr(profile.limits.joint_position, "min", None)),
        ("joint_position.max", getattr(profile.limits.joint_position, "max", None)),
        ("joint_velocity.max", getattr(profile.limits.joint_velocity, "max", None)),
        ("joint_torque.max", getattr(profile.limits.joint_torque, "max", None)),
    )
    for field_name, values in checks:
        if values is not None and len(values) != dof:
            report.add(
                error(
                    "limits.length_mismatch",
                    f"{where}.{field_name}",
                    f"{len(values)} значений при {dof} суставах в contract.joint_order",
                    hint="лимиты упорядочены как joint_order; длина обязана совпадать",
                )
            )


def _check_rates(profile: RobotProfile, report: Report) -> None:
    contract = profile.contract
    where = f"profile:{profile.name}.contract"
    if contract.command_rate_hz > contract.state_rate_hz:
        report.add(
            warning(
                "contract.command_faster_than_state",
                where,
                f"команды {contract.command_rate_hz:g} Гц чаще состояния "
                f"{contract.state_rate_hz:g} Гц",
                hint="политика будет считать действие по несвежему состоянию",
            )
        )
    if profile.control_mode is ControlMode.COMMAND and contract.command_rate_hz > 20:
        report.add(
            warning(
                "contract.command_rate_suspicious",
                where,
                f"режим command при command_rate_hz = {contract.command_rate_hz:g}",
                hint=(
                    "командный режим — «выполни движение и сообщи о завершении»; "
                    "высокая частота здесь обычно означает ошибку в профиле"
                ),
            )
        )


def _check_observation_terms(profile: RobotProfile, report: Report) -> None:
    terms = [term.type for term in profile.contract.observation]
    duplicates = sorted({t for t in terms if terms.count(t) > 1})
    if duplicates:
        report.add(
            warning(
                "contract.observation_duplicate",
                f"profile:{profile.name}.contract.observation",
                f"терм повторяется: {', '.join(duplicates)}",
                hint="повтор с разными scale читается как ошибка копирования",
            )
        )


def _check_image(profile: RobotProfile, report: Report) -> None:
    """The platform name is not part of user data (section 17)."""
    if profile.image and "/" in profile.image:
        report.add(
            error(
                "profile.image_has_namespace",
                f"profile:{profile.name}.image",
                f"{profile.image!r} содержит пространство имён",
                hint=(
                    f"в профиле хранится голое имя образа; пространство имён "
                    f"({DOCKER_NAMESPACE!r}) подставляет платформа"
                ),
                adr="раздел 17",
            )
        )


def _check_adapters(profile: RobotProfile, report: Report) -> None:
    """An adapter plugin that is not installed is a warning, not an error.

    CI runs without vendor packages by design (section 10), so an absent plugin
    must not fail validation of the profile itself. It fails at deployment,
    where it actually matters.
    """
    try:
        installed = set(available("adapters"))
    except PluginError:  # pragma: no cover - group names come from branding
        installed = set()
    for target, spec in profile.adapters.items():
        if ":" in spec.plugin or spec.plugin in installed:
            continue
        report.add(
            warning(
                "adapter.plugin_not_installed",
                f"profile:{profile.name}.adapters.{target}",
                f"плагин {spec.plugin!r} не установлен",
                hint="профиль валиден, но запуск с этим адаптером невозможен",
                detail={"plugin": spec.plugin, "target": target},
            )
        )


# --------------------------------------------------------------------------
# A skill against a robot.
# --------------------------------------------------------------------------


def check_skill_against_profile(skill: SkillSpec, profile: RobotProfile) -> Report:
    """Whether this robot can run this skill at all."""
    report = Report()
    where = f"skill:{skill.name} -> profile:{profile.name}"

    absent = caps.missing(skill.required_capabilities, profile.capabilities)
    if absent:
        report.add(
            error(
                "skill.capabilities_missing",
                where,
                f"робот не объявляет: {', '.join(sorted(absent))}",
                hint="сопоставление — равенство ключей, следствия не выводятся",
                adr="ADR-0001",
                detail={"missing": sorted(absent)},
            )
        )

    if (
        skill.required_control_mode is ControlMode.STREAMING
        and profile.control_mode is ControlMode.COMMAND
    ):
        report.add(
            error(
                "skill.control_mode_incompatible",
                where,
                "навык требует потокового управления, робот принимает команды целиком",
                hint="политика с циклом на фиксированной частоте здесь неисполнима",
                adr="раздел 3",
                detail={"required": "streaming", "actual": "command"},
            )
        )

    for unmet in req.check(skill.requires, profile):
        report.add(
            error(
                "skill.requirement_unmet",
                where,
                unmet.describe(),
                hint="количественные требования сверяются с contract и limits",
                adr="ADR-0001",
                detail={
                    "quantity": unmet.quantity,
                    "clause": str(unmet.clause),
                    "actual": unmet.actual,
                },
            )
        )
    return report


# --------------------------------------------------------------------------
# A policy artifact against a robot.
# --------------------------------------------------------------------------


def check_policy_against_profile(manifest: PolicyManifest, profile: RobotProfile) -> Report:
    """Whether an exported policy may drive this robot (invariant 2, ADR-0004)."""
    report = Report()
    where = f"policy:{manifest.name} -> profile:{profile.name}"

    if not manifest.self_consistent:
        report.add(
            error(
                "policy.manifest_inconsistent",
                where,
                f"записанный contract_id {manifest.contract_id} не соответствует "
                f"приложенному контракту ({manifest.contract.contract_id})",
                hint=(
                    "идентификатор вычисляется, а не пишется; "
                    "артефакт повреждён или отредактирован"
                ),
                adr="ADR-0004",
            )
        )

    report.extend(check_contracts(manifest.contract, profile.contract, where=where).findings)

    absent = caps.missing(manifest.capabilities, profile.capabilities)
    if absent:
        report.add(
            error(
                "policy.capabilities_missing",
                where,
                f"робот не объявляет: {', '.join(sorted(absent))}",
                adr="ADR-0001",
                detail={"missing": sorted(absent)},
            )
        )
    return report


def check_contracts(left: Contract, right: Contract, *, where: str) -> Report:
    """Compare two contracts, reporting differing fields rather than hashes."""
    report = Report()
    if left.contract_id == right.contract_id:
        return report
    differences = diff(left, right)
    for item in differences:
        report.add(
            error(
                "contract.field_mismatch",
                f"{where}.{item.field}",
                item.describe("у артефакта", "в профиле"),
                hint="переобучать политику или править профиль — решает человек",
                adr="ADR-0004",
                detail={"field": item.field, "artifact": item.left, "profile": item.right},
            )
        )
    if not differences:  # pragma: no cover - would mean a broken canonical form
        report.add(
            error(
                "contract.identity_mismatch",
                where,
                f"идентификаторы различаются ({left.contract_id} и {right.contract_id}) "
                f"при совпадающих нормативных полях",
                hint="сломана каноническая сериализация — это ошибка ядра",
                adr="ADR-0004",
            )
        )
    return report


# --------------------------------------------------------------------------
# A scenario against the registry.
# --------------------------------------------------------------------------


def check_scenario(scenario: Scenario, registry: Registry) -> Report:
    """Everything decidable about a scenario before the world is built."""
    report = Report()
    where = f"scenario:{scenario.name}"
    required_target = sim_target(scenario.world.backend)
    # Two robots may share a profile; its own findings are reported once.
    checked: set[str] = set()
    participants: list[tuple[ScenarioRobot, RobotProfile]] = []

    for robot in scenario.robots:
        spot = f"{where}.robots.{robot.id}"
        try:
            source = registry.source(robot.profile)
            profile = source.profile
        except ProfileNotFoundError as exc:
            report.add(
                error(
                    "scenario.profile_not_found",
                    spot,
                    str(exc),
                    hint="профиль — данные; добавьте файл в реестр, ядро править не нужно",
                    adr="инвариант 5",
                    detail={"profile": robot.profile},
                )
            )
            continue

        participants.append((robot, profile))
        if required_target not in profile.adapters:
            report.add(
                error(
                    "scenario.backend_adapter_missing",
                    spot,
                    f"у профиля {profile.name!r} нет адаптера {required_target!r}; "
                    f"есть: {', '.join(sorted(profile.adapters)) or '—'}",
                    hint=(
                        "в сцене один владелец физики: робот без адаптера под этот "
                        "бэкенд участвовать не может"
                    ),
                    adr="ADR-0002",
                    detail={"backend": scenario.world.backend, "required": required_target},
                )
            )

        if profile.name not in checked:
            checked.add(profile.name)
            report.extend(check_profile(profile, path=source.path).findings)

    _check_twins(scenario, participants, report)

    learners = scenario.learners
    if len(learners) > 1:
        report.add(
            error(
                "scenario.multiple_learners",
                where,
                f"ролей learner больше одной: {', '.join(r.id for r in learners)}",
                hint="совместное мультиагентное обучение сознательно отложено",
                adr="раздел 15",
                detail={"learners": [r.id for r in learners]},
            )
        )

    for robot in scenario.robots:
        if robot.role is Role.REPLAY:
            report.add(
                warning(
                    "scenario.replay_trajectory_unchecked",
                    f"{where}.robots.{robot.id}",
                    f"траектория {robot.trajectory!r} не проверена: хранилища ещё нет",
                    hint="контракт траектории сверяется тем же contract_id, что и политика",
                    adr="ADR-0003",
                )
            )
    return report


def _check_twins(
    scenario: Scenario,
    participants: list[tuple[ScenarioRobot, RobotProfile]],
    report: Report,
) -> None:
    """In a world the platform starts, every driven robot has a twin (ADR-0021).

    Only for a backend installed as a plugin. A backend that is not installed
    brings no world up -- ``sim run`` refuses it by name -- and the adapter of a
    scene without one, like the dummy on a bench, is both sides of its own link.
    An obstacle has no link and needs no twin. An adapter that is not installed
    is already a warning of its own, and its twin ships in the same package.
    """
    backend = scenario.world.backend
    if backend not in available("sim"):
        return
    where = f"scenario:{scenario.name}"
    target = sim_target(backend)
    adapters = set(available("adapters"))
    owners: list[str] = []

    for robot, profile in participants:
        spec = profile.adapters.get(target)
        if robot.role is Role.STATIC or spec is None:
            continue
        if ":" not in spec.plugin and spec.plugin not in adapters:
            continue
        spot = f"{where}.robots.{robot.id}"
        detail = {"adapter": spec.plugin, "backend": backend}
        try:
            twin = find_twin(spec.plugin, backend)
        except TwinError as exc:
            report.add(
                error(
                    "twin.unloadable",
                    spot,
                    str(exc),
                    hint="двойник устанавливается пакетом робота вместе с адаптером",
                    adr="ADR-0021",
                    detail=detail,
                )
            )
            continue
        if twin is None:
            report.add(
                error(
                    "twin.missing",
                    spot,
                    f"у адаптера {spec.plugin!r} нет двойника под бэкенд {backend!r}",
                    hint=(
                        "в симуляции другую сторону связи держит двойник, говорящий "
                        "протоколом робота; его поставляет пакет робота вместе с адаптером"
                    ),
                    adr="ADR-0021",
                    detail=detail,
                )
            )
        elif twin.kind == TwinKind.OWNS_PHYSICS:
            owners.append(robot.id)

    if owners and len(scenario.robots) > 1:
        report.add(
            error(
                "twin.owns_physics_not_alone",
                where,
                f"двойник робота {', '.join(owners)} сам владеет физикой, "
                f"а участников в сцене {len(scenario.robots)}",
                hint=(
                    "такой двойник — целый симулятор производителя; второй участник в нём "
                    "невозможен, как невозможна сцена из двух движков"
                ),
                adr="ADR-0002",
                detail={"owners": owners, "robots": [item.id for item in scenario.robots]},
            )
        )
