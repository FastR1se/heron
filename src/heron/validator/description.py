"""Rules over a profile and the description it points at.

This is what the platform gains by not authoring kinematics (ADR-0009). The
profile and the description describe one robot in two languages, written by
different people at different times, and every disagreement between them is the
failure this platform exists to catch: a joint renamed in the model and not in
the profile, a limit relaxed in the profile and not in the model, a joint the
profile intends to command that the model welds shut.

Units are converted before comparison and shown in both forms in the message.
URDF fixes radians by specification and a contract may declare degrees; a
comparison without conversion would report a mismatch on a robot that is fine,
and a validator that cries wolf is worse than no validator -- it teaches people
to skip the check.

Limits are compared one way only. A profile narrower than the model is correct
and common: ``limits`` is a safety envelope, and tightening it is how a robot is
run carefully (section 12). A profile wider than the model is a promise the
machine cannot keep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from ..core.enums import AngleUnit
from ..core.loading import LoadError
from ..core.models import RobotProfile
from ..core.units import format_angle, to_radians
from ..core.urdf import UrdfJoint, UrdfModel, is_template, parse_urdf
from .report import Report, error, warning

TOLERANCE: Final = 1e-6
"""Slack in a limit comparison, in radians or metres. Float noise, not policy."""

UNRESOLVABLE_SCHEMES: Final = ("package://", "model://")
"""Mesh references only a ROS 2 or simulator resource path can resolve."""


def check_description(profile: RobotProfile, profile_path: Path) -> Report:
    """Compare a profile with the description it names."""
    report = Report()
    where = f"profile:{profile.name}.description"
    stated = Path(profile.description.urdf)
    path = stated if stated.is_absolute() else (profile_path.parent / stated)

    if is_template(path):
        report.add(
            warning(
                "urdf.not_rendered",
                where,
                f"{profile.description.urdf} — шаблон, а не описание",
                hint=(
                    "сверка с профилем пропущена; разверните шаблон в окружении "
                    "ROS 2, тогда правила urdf.* начнут работать"
                ),
                adr="ADR-0009",
                detail={"urdf": profile.description.urdf},
            )
        )
        return report

    if not path.is_file():
        report.add(
            error(
                "urdf.file_not_found",
                where,
                f"описание не найдено: {path}",
                hint="путь читается относительно файла профиля, а не рабочего каталога",
                adr="ADR-0009",
                detail={"urdf": profile.description.urdf, "resolved": str(path)},
            )
        )
        return report

    try:
        model = parse_urdf(path)
    except LoadError as exc:
        report.add(
            error(
                "urdf.unreadable",
                where,
                str(exc),
                hint="описание — вход платформы; исправьте файл, а не профиль",
                adr="ADR-0009",
            )
        )
        return report

    _check_joints(profile, model, report)
    _check_limits(profile, model, report)
    _check_structure(profile, model, report)
    _check_meshes(profile, model, path, report)
    return report


def _check_joints(profile: RobotProfile, model: UrdfModel, report: Report) -> None:
    where = f"profile:{profile.name}.contract.joint_order"
    declared = list(profile.contract.joint_order)

    for name in declared:
        joint = model.joint(name)
        if joint is None:
            report.add(
                error(
                    "urdf.joint_missing",
                    where,
                    f"сустава {name!r} нет в описании {model.path.name}",
                    hint=(
                        f"в описании есть: {', '.join(model.joint_names) or '—'}. "
                        f"Порядок суставов — нормативное поле контракта, и "
                        f"переименование в модели обязано доехать до профиля"
                    ),
                    adr="ADR-0004",
                    detail={"joint": name, "described": list(model.joint_names)},
                )
            )
            continue
        if not joint.commandable:
            report.add(
                error(
                    "urdf.joint_not_commandable",
                    where,
                    f"сустав {name!r} объявлен как {joint.type}, им нельзя командовать",
                    hint=(
                        "контракт называет сустав среди управляемых, а модель его "
                        "закрепляет; расходятся представления, а не мнения"
                    ),
                    adr="ADR-0009",
                    detail={"joint": name, "type": joint.type},
                )
            )

    extra = [item.name for item in model.commandable_joints if item.name not in declared]
    if extra:
        report.add(
            warning(
                "urdf.joint_not_in_contract",
                where,
                f"подвижные суставы описания вне контракта: {', '.join(extra)}",
                hint=(
                    "платформа ими не управляет, но состояние их читает "
                    "(ADR-0012); если это намеренно — встроенный контроллер "
                    "или пассивное звено — предупреждение ожидаемо"
                ),
                adr="ADR-0009",
                detail={"joints": extra},
            )
        )


def _bounds_in_model_units(
    value: float, joint: UrdfJoint, unit: AngleUnit
) -> float:
    """A profile limit expressed the way the description expresses it."""
    return to_radians(value, unit) if joint.angular else value


def _describe(value: float, joint: UrdfJoint, unit: AngleUnit) -> str:
    """A profile limit, in the profile's units and in the model's when they differ."""
    if not joint.angular or unit is AngleUnit.RAD:
        return f"{value:g}"
    return f"{format_angle(value, unit)} ({to_radians(value, unit):g} rad)"


def _check_limits(profile: RobotProfile, model: UrdfModel, report: Report) -> None:
    where = f"profile:{profile.name}.limits"
    unit = profile.contract.angle_unit
    positions = profile.limits.joint_position
    velocities = profile.limits.joint_velocity

    for index, name in enumerate(profile.contract.joint_order):
        joint = model.joint(name)
        if joint is None or not joint.commandable:
            continue

        if positions is not None and joint.limit.bounded:
            assert joint.limit.lower is not None and joint.limit.upper is not None
            wanted = (positions.min[index], positions.max[index])
            low = _bounds_in_model_units(wanted[0], joint, unit)
            high = _bounds_in_model_units(wanted[1], joint, unit)
            if low < joint.limit.lower - TOLERANCE or high > joint.limit.upper + TOLERANCE:
                report.add(
                    error(
                        "urdf.position_limit_wider",
                        f"{where}.joint_position",
                        f"{name}: профиль разрешает "
                        f"[{_describe(wanted[0], joint, unit)}, "
                        f"{_describe(wanted[1], joint, unit)}], "
                        f"модель — [{joint.limit.lower:g}, {joint.limit.upper:g}]",
                        hint=(
                            "лимиты профиля — предохранительная рамка: уже модели "
                            "можно, шире нельзя"
                        ),
                        adr="раздел 12",
                        detail={
                            "joint": name,
                            "profile": [low, high],
                            "described": [joint.limit.lower, joint.limit.upper],
                            "unit": str(unit),
                        },
                    )
                )

        if velocities is not None and joint.limit.velocity is not None:
            wanted_velocity = velocities.max[index]
            allowed = _bounds_in_model_units(wanted_velocity, joint, unit)
            if allowed > joint.limit.velocity + TOLERANCE:
                report.add(
                    error(
                        "urdf.velocity_limit_wider",
                        f"{where}.joint_velocity",
                        f"{name}: профиль разрешает "
                        f"{_describe(wanted_velocity, joint, unit)}/с, "
                        f"модель — {joint.limit.velocity:g}/с",
                        hint="скорость выше модельной физика не отработает",
                        adr="раздел 12",
                        detail={
                            "joint": name,
                            "profile": allowed,
                            "described": joint.limit.velocity,
                        },
                    )
                )


def _check_structure(profile: RobotProfile, model: UrdfModel, report: Report) -> None:
    where = f"profile:{profile.name}.description"
    if model.root_link is None:
        report.add(
            error(
                "urdf.no_root_link",
                where,
                "не видно единственного корневого звена",
                hint=(
                    "робот — дерево звеньев с одним корнем; без него не определить "
                    "систему отсчёта для визуализации и размещения в мире"
                ),
                detail={"links": list(model.links)},
            )
        )
    if model.name and model.name != profile.name:
        report.add(
            warning(
                "urdf.name_differs",
                where,
                f"описание называет робота {model.name!r}, профиль — {profile.name!r}",
                hint=(
                    "имена топиков и пространств имён строятся от идентификатора в "
                    "сценарии, так что запуску это не мешает; расхождение обычно "
                    "означает, что описание взято от другого робота"
                ),
                detail={"described": model.name, "profile": profile.name},
            )
        )


def _check_meshes(
    profile: RobotProfile, model: UrdfModel, path: Path, report: Report
) -> None:
    where = f"profile:{profile.name}.description"
    unresolved: list[str] = []
    external: list[str] = []

    for reference in model.meshes:
        if reference.startswith(UNRESOLVABLE_SCHEMES):
            external.append(reference)
            continue
        target = Path(reference.removeprefix("file://"))
        if not target.is_absolute():
            base = path.parent
            if profile.description.mesh_dir:
                base = path.parent / profile.description.mesh_dir
            target = base / target
        if not target.is_file():
            unresolved.append(reference)

    if unresolved:
        report.add(
            error(
                "urdf.mesh_not_found",
                where,
                f"меши не найдены: {', '.join(sorted(set(unresolved)))}",
                hint=(
                    "путь читается от файла описания, а при заданном mesh_dir — "
                    "от него; без меша модель не загрузится в мир"
                ),
                detail={"meshes": sorted(set(unresolved))},
            )
        )
    if external:
        report.add(
            warning(
                "urdf.mesh_unresolvable",
                where,
                f"ссылки разрешимы только средствами ROS 2 или симулятора: "
                f"{', '.join(sorted(set(external)))}",
                hint="платформа их не проверяет; проверит запуск",
                detail={"meshes": sorted(set(external))},
            )
        )
