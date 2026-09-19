"""Validation rules.

Invariant 6: a mismatch between environments is a validator error, not strange
behaviour of a robot. Everything checkable before a run is checked here.

Rules are grouped by what they need to see. A rule over one profile cannot know
about a scenario, and a rule over a scenario needs the registry; keeping that
split explicit is what stops the validator from quietly becoming a loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from ..core import capabilities as caps
from ..core import requires as req
from ..core.branding import DOCKER_NAMESPACE
from ..core.enums import ActionType, ControlMode, Role, TwinKind
from ..core.loading import LoadError
from ..core.models import (
    RobotProfile,
    Scenario,
    ScenarioRobot,
    SkillSpec,
    sim_target,
)
from ..core.plugins import PluginError, available
from ..core.twins import TwinError, find_twin, why_missing
from ..registry.lock import ContractLock
from ..registry.registry import ProfileNotFoundError, Registry
from ..store.policies import PolicyStore
from .description import check_description
from .mjcf import check_mjcf, check_mujoco_world
from .policy import check_policy_against_profile
from .report import Report, error, warning
from .train import check_training, is_training_world

# --------------------------------------------------------------------------
# One profile on its own.
# --------------------------------------------------------------------------


def check_profile(
    profile: RobotProfile,
    *,
    lock: ContractLock | None = None,
    path: Path | None = None,
    store: PolicyStore | None = None,
) -> Report:
    """Everything decidable from a single profile.

    ``path`` is where the profile was read from. With it the description it
    points at becomes reachable, and the rules of ``description`` join in; a
    profile validated as a bare object is validated without them, and says so by
    simply not reporting them.

    ``store`` is where a policy a skill is bound to is looked for; the platform's
    own store when not given, and only opened when a skill is bound to one.
    """
    report = Report()
    where = f"profile:{profile.name}"
    contract = profile.contract

    _check_limit_lengths(profile, report)
    _check_rates(profile, report)
    _check_observation_terms(profile, report)
    _check_action_envelope(profile, report)
    _check_image(profile, report)
    _check_adapters(profile, report)
    _check_skill_bindings(profile, report, store)
    _check_declared_skills(profile, report)
    if path is not None:
        report.extend(check_description(profile, path).findings)
        report.extend(check_mjcf(profile, path).findings)

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
        ("joint_acceleration.max", getattr(profile.limits.joint_acceleration, "max", None)),
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


def _check_action_envelope(profile: RobotProfile, report: Report) -> None:
    """What a policy can command lies inside the envelope, checked before it moves (ADR-0041).

    A target outside the envelope is a refusal of the adapter halfway through a motion
    (ADR-0026), and the set of targets a mapped action can reach is known from the
    contract alone: ``origin + [clip_lo, clip_hi] * scale``, joint by joint.
    """
    contract = profile.contract
    action = contract.action
    where = f"profile:{profile.name}.contract"
    bounds = profile.limits.joint_position
    if bounds is not None and len(bounds.min) != contract.dof:
        return  # limits.length_mismatch already says so, and nothing lines up to compare

    if contract.default_pose is not None and bounds is not None:
        outside = {
            name: {"value": value, "min": low, "max": high}
            for name, value, low, high in zip(
                contract.joint_order, contract.default_pose, bounds.min, bounds.max, strict=True
            )
            if not low <= value <= high
        }
        if outside:
            report.add(
                error(
                    "contract.default_pose_outside_limits",
                    f"{where}.default_pose",
                    f"поза отсчёта вне рамки limits.joint_position в суставах "
                    f"{', '.join(outside)}",
                    hint="поза, от которой считаются действие и наблюдение, обязана быть в рамке",
                    adr="ADR-0041",
                    detail={"joints": outside},
                )
            )

    if action.type is not ActionType.JOINT_POSITION:
        return
    if action.clip is None and not action.maps_output:
        return
    if bounds is None:
        report.add(
            warning(
                "contract.action_envelope_unknown",
                f"{where}.action",
                "цели политики не с чем сверить: в профиле нет limits.joint_position",
                hint="без рамки адаптер не откажет, а валидатор не предупредит",
                adr="ADR-0041",
            )
        )
        return
    if action.clip is None:
        report.add(
            error(
                "contract.action_outside_limits",
                f"{where}.action",
                "масштаб или смещение заданы без clip: цели политики ничем не ограничены",
                hint="выход сети не ограничен; clip над сырым выходом задаёт множество целей",
                adr="ADR-0041",
                detail={"joints": {}, "clip": None},
            )
        )
        return

    clip_low, clip_high = action.clip
    outside_set: dict[str, dict[str, list[float]]] = {}
    for name, reach, low, high in zip(
        contract.joint_order,
        contract.action_targets() or (),
        bounds.min,
        bounds.max,
        strict=True,
    ):
        if reach[0] < low or reach[1] > high:
            outside_set[name] = {"targets": list(reach), "limits": [low, high]}
    if outside_set:
        report.add(
            error(
                "contract.action_outside_limits",
                f"{where}.action",
                f"политика может приказать цель вне рамки limits.joint_position в суставах "
                f"{', '.join(outside_set)}",
                hint=(
                    "уменьшите scale, сдвиньте default_pose или сузьте clip: "
                    "origin + clip · scale обязано лежать в рамке"
                ),
                adr="ADR-0041",
                detail={"joints": outside_set, "clip": [clip_low, clip_high]},
            )
        )


def _check_skill_bindings(
    profile: RobotProfile, report: Report, store: PolicyStore | None
) -> None:
    """A skill bound to a policy is checked against the artifact, when the artifact is here.

    Absent from this machine's store it is a warning: the data is right and the
    machine is not the one it was trained for, the way an environment is (ADR-0025).
    ``run`` makes the same absence an error (ADR-0046).
    """
    bound = [(binding, ref) for binding in profile.skills if (ref := binding.ref) is not None]
    if not bound:
        return
    policies = store if store is not None else PolicyStore()
    for binding, ref in bound:
        spot = f"profile:{profile.name}.skills.{binding.name}"
        try:
            artifact = policies.find(ref)
        except LoadError as exc:
            report.add(
                error(
                    "policy.artifact_inconsistent",
                    spot,
                    f"манифест политики {ref} не читается: {exc}",
                    adr="ADR-0046",
                    detail={"policy": str(ref)},
                )
            )
            continue
        if artifact is None:
            report.add(
                warning(
                    "policy.not_in_store",
                    spot,
                    f"политика {ref} не найдена в хранилище {policies.root}",
                    hint=(
                        "профиль верен, но на этой машине навык не исполнить; "
                        "run откажет с этой же причиной"
                    ),
                    adr="ADR-0046",
                    detail={"policy": str(ref), "store": str(policies.root)},
                )
            )
            continue
        report.extend(
            check_policy_against_profile(artifact.manifest, profile, artifact=artifact).findings
        )


def _check_declared_skills(profile: RobotProfile, report: Report) -> None:
    """A skill the profile declares as code is one this robot can run (ADR-0049).

    The same rules as ``skill check`` and the executor's ``check_runnable``, so a
    streaming skill on a command robot is refused by ``validate``, not after
    ``connect``. A skill bound to a policy is checked against its artifact above.
    A skill that is not installed here is a warning, as an adapter is: CI runs
    without vendor packages, and the absence fails where it matters -- ``run``.
    """
    # The library imports the learned skill, which imports the validator: resolving
    # names at call time keeps the two packages importable in either order.
    from ..skills.library import SkillLibrary, SkillNotFoundError

    library = SkillLibrary()
    for binding in profile.skills:
        if binding.ref is not None:
            continue
        try:
            spec = library.spec(binding.name)
        except SkillNotFoundError as exc:
            report.add(
                warning(
                    "skill.not_installed",
                    f"profile:{profile.name}.skills.{binding.name}",
                    str(exc),
                    hint="навык не встроен и не установлен пакетом на этой машине; run откажет",
                    adr="ADR-0046",
                    detail={"skill": binding.name},
                )
            )
            continue
        report.extend(check_skill_against_profile(spec, profile).findings)


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
# A robot against hardware.
# --------------------------------------------------------------------------

REAL_ENVELOPE: Final = ("joint_position", "joint_velocity")
"""What a profile must bound before a run on hardware (section 12, ADR-0051)."""


def check_real_target(profile: RobotProfile) -> Report:
    """Whether this robot may be run on hardware at all: the envelope is declared."""
    report = Report()
    missing = [name for name in REAL_ENVELOPE if getattr(profile.limits, name) is None]
    if missing:
        report.add(
            error(
                "real.limits_missing",
                f"profile:{profile.name}.limits",
                f"прогон на железе без рамки: нет {', '.join(missing)}",
                hint="валидатор сверяет пределы суставов и скоростей перед любым запуском на "
                "железе (раздел 12); объявите их в limits профиля",
                adr="ADR-0051",
                detail={"missing": missing},
            )
        )
    return report


# --------------------------------------------------------------------------
# A skill against a robot.
# --------------------------------------------------------------------------

CONTROL_MODE_MISMATCH: Final[dict[ControlMode, tuple[str, str]]] = {
    ControlMode.STREAMING: (
        "навык требует потокового управления",
        "политика с циклом на фиксированной частоте здесь неисполнима",
    ),
    ControlMode.COMMAND: (
        "навык отдаёт движение целиком",
        "одна цель без разгона на потоковой связи — ступень, "
        "а фазы движения потоковая связь не сообщает",
    ),
}
"""Why a skill of one class cannot run on a robot of the other: modes compare by
equality, in both directions (ADR-0049)."""

ROBOT_ACCEPTS: Final[dict[ControlMode, str]] = {
    ControlMode.STREAMING: "принимает поток команд",
    ControlMode.COMMAND: "принимает команды целиком",
}


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

    if skill.required_control_mode is not profile.control_mode:
        message, hint = CONTROL_MODE_MISMATCH[skill.required_control_mode]
        report.add(
            error(
                "skill.control_mode_incompatible",
                where,
                f"{message}, робот {ROBOT_ACCEPTS[profile.control_mode]}",
                hint=hint,
                adr="раздел 3, ADR-0049",
                detail={
                    "required": str(skill.required_control_mode),
                    "actual": str(profile.control_mode),
                },
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
# A scenario against the registry.
# --------------------------------------------------------------------------


def check_scenario(
    scenario: Scenario,
    registry: Registry,
    *,
    path: Path | None = None,
    store: PolicyStore | None = None,
) -> Report:
    """Everything decidable about a scenario before the world is built.

    ``path`` is where the scenario was read from. With it the files the scenario
    names relative to itself become reachable -- a MuJoCo scene -- as a profile's
    description does for ``check_profile``.
    """
    report = Report()
    where = f"scenario:{scenario.name}"
    required_target = sim_target(scenario.world.backend)
    # A world that trains has no adapter to require and no twin to find: nobody
    # drives the learner, the algorithm does (ADR-0044).
    training = is_training_world(scenario)
    # Two robots may share a profile; its own findings are reported once.
    checked: set[str] = set()
    participants: list[tuple[ScenarioRobot, RobotProfile]] = []
    located: list[tuple[ScenarioRobot, RobotProfile, Path]] = []

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
        located.append((robot, profile, source.path))
        if not training and required_target not in profile.adapters:
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
            report.extend(check_profile(profile, path=source.path, store=store).findings)

    _check_task_skill(scenario, participants, report)
    if training:
        check_training(scenario, participants, report)
    else:
        _check_twins(scenario, participants, report)
        report.extend(check_mujoco_world(scenario, located, path).findings)

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


def _check_task_skill(
    scenario: Scenario,
    participants: list[tuple[ScenarioRobot, RobotProfile]],
    report: Report,
) -> None:
    """The skill a task asks for is looked up through the profile of the robot (ADR-0046).

    Whether a robot runs it as code or as a policy is the profile's line; a robot
    whose profile does not declare it has no such line to read. An obstacle is
    asked for nothing.
    """
    if scenario.task is None:
        return
    skill = scenario.task.skill
    for robot, profile in participants:
        if robot.role is Role.STATIC or profile.skill(skill) is not None:
            continue
        declared = [binding.name for binding in profile.skills]
        report.add(
            error(
                "scenario.skill_not_in_profile",
                f"scenario:{scenario.name}.robots.{robot.id}",
                f"задача просит навык {skill!r}, а профиль {profile.name!r} его не объявляет; "
                f"объявлены: {', '.join(declared) or '—'}",
                hint=(
                    "навык задачи ищется через профиль: добавьте его в skills — именем из "
                    "библиотеки или привязкой к политике"
                ),
                adr="ADR-0046",
                detail={"skill": skill, "profile": profile.name, "declared": declared},
            )
        )


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
            stale = why_missing()
            report.add(
                error(
                    "twin.missing",
                    spot,
                    f"у адаптера {spec.plugin!r} нет двойника под бэкенд {backend!r}",
                    hint=stale
                    or (
                        "в симуляции другую сторону связи держит двойник, говорящий "
                        "протоколом робота; его поставляет пакет робота вместе с адаптером"
                    ),
                    adr="ADR-0021",
                    detail={**detail, "twins_installed": stale is None},
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
