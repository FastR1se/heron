"""Rules over a profile's MJCF description and over a MuJoCo scene (ADR-0031).

The same position as ``validator.description``, in a second language. A robot
described twice -- URDF for Gazebo, MJCF for the vendor's twin -- is described
by two documents written by different people for different simulators, and each
can disagree with the profile the way a URDF can. The profile is compared with
the MJCF by the rules ``urdf.*`` apply to the URDF: joints, whether each can be
commanded, position limits one way only, meshes.

What is new is what only two descriptions can show: that they disagree with each
other. Neither is the platform's to correct, and neither is known to be right,
so a disagreement is a warning with both numbers -- it is a condition of every
comparison between the two simulators, stated before one is made.

A MuJoCo world adds two checks of its own, because in it the scene is an input
too (ADR-0009): the scene has to load the description that was checked, and its
physics step cannot be slower than the state rate the contract promises.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from ..core.loading import LoadError
from ..core.mjcf import MUJOCO, MjcfModel, parse_mjcf
from ..core.models import RobotProfile, Scenario, ScenarioRobot
from ..core.units import format_angle, to_radians
from ..core.urdf import UrdfModel, is_template, parse_urdf
from .report import Report, error, warning

ADR: Final = "ADR-0031"

TOLERANCE: Final = 1e-5
"""Slack in a comparison with an MJCF number: radians, metres, N·m.

Wider than the ``urdf.*`` tolerance, and by a fact rather than a preference: the
first vendor MJCF met prints angles to five decimals where the same vendor's URDF
prints nine (1.97222 against 1.972222054, journal 2026-09-15). A difference below
the printed precision is not one either document states."""


def _resolve(stated: str, base: Path) -> Path:
    path = Path(stated)
    return path if path.is_absolute() else base / path


def _load(path: Path, where: str, report: Report) -> MjcfModel | None:
    if not path.is_file():
        report.add(
            error(
                "mjcf.file_not_found",
                where,
                f"описание MuJoCo не найдено: {path}",
                hint="путь читается относительно файла, который его называет, а не рабочего "
                "каталога; вендорское описание загружается скриптом рядом с профилем",
                adr=ADR,
                detail={"resolved": str(path)},
            )
        )
        return None
    try:
        return parse_mjcf(path)
    except LoadError as exc:
        report.add(
            error(
                "mjcf.unreadable",
                where,
                str(exc),
                hint="описание — вход платформы; исправьте файл, а не профиль",
                adr=ADR,
            )
        )
        return None


# --------------------------------------------------------------------------
# A profile and its MJCF.
# --------------------------------------------------------------------------


def check_mjcf(profile: RobotProfile, profile_path: Path) -> Report:
    """Compare a profile with the MJCF it names, and that MJCF with the URDF."""
    report = Report()
    if profile.description.mjcf is None:
        return report
    path = _resolve(profile.description.mjcf, profile_path.parent)
    model = _load(path, f"profile:{profile.name}.description.mjcf", report)
    if model is None:
        return report

    _check_joints(profile, model, report)
    _check_limits(profile, model, report)
    _check_meshes(profile, model, report)
    urdf = _urdf(profile, profile_path)
    if urdf is not None:
        _check_against_urdf(profile, model, urdf, report)
    return report


def _check_joints(profile: RobotProfile, model: MjcfModel, report: Report) -> None:
    where = f"profile:{profile.name}.contract.joint_order"
    for name in profile.contract.joint_order:
        joint = model.joint(name)
        if joint is None:
            report.add(
                error(
                    "mjcf.joint_missing",
                    where,
                    f"сустава {name!r} нет в описании MuJoCo {model.path.name}",
                    hint=f"в описании есть: {', '.join(model.joint_names) or '—'}",
                    adr=ADR,
                    detail={"joint": name, "described": list(model.joint_names)},
                )
            )
            continue
        if not joint.commandable:
            report.add(
                error(
                    "mjcf.joint_not_commandable",
                    where,
                    f"сустав {name!r} в описании MuJoCo — {joint.type}, одним числом им не "
                    f"командовать",
                    hint="контракт называет сустав среди управляемых, а модель MuJoCo "
                    "описывает его иначе",
                    adr=ADR,
                    detail={"joint": name, "type": joint.type},
                )
            )
            continue
        if not model.actuators_of(name):
            report.add(
                error(
                    "mjcf.joint_not_actuated",
                    where,
                    f"у сустава {name!r} в описании MuJoCo нет привода",
                    hint="в MuJoCo сустав движет только привод из блока <actuator>: двойнику "
                    "нечем приложить к нему закон привода, и сустав будет висеть",
                    adr=ADR,
                    detail={"joint": name},
                )
            )


def _check_limits(profile: RobotProfile, model: MjcfModel, report: Report) -> None:
    bounds = profile.limits.joint_position
    if bounds is None:
        return
    unit = profile.contract.angle_unit
    for index, name in enumerate(profile.contract.joint_order):
        joint = model.joint(name)
        if joint is None or not joint.commandable or joint.range is None:
            continue
        wanted = (bounds.min[index], bounds.max[index])
        low, high = (to_radians(v, unit) if joint.angular else v for v in wanted)
        if low >= joint.range[0] - TOLERANCE and high <= joint.range[1] + TOLERANCE:
            continue
        shown = (
            f"[{format_angle(wanted[0], unit)}, {format_angle(wanted[1], unit)}]"
            if joint.angular
            else f"[{wanted[0]:g}, {wanted[1]:g}]"
        )
        report.add(
            error(
                "mjcf.position_limit_wider",
                f"profile:{profile.name}.limits.joint_position",
                f"{name}: профиль разрешает {shown}, модель MuJoCo — "
                f"[{joint.range[0]:g}, {joint.range[1]:g}]",
                hint="лимиты профиля — предохранительная рамка: уже модели можно, шире нельзя",
                adr="раздел 12",
                detail={
                    "joint": name,
                    "profile": [low, high],
                    "described": list(joint.range),
                    "unit": str(unit),
                },
            )
        )


def _check_meshes(profile: RobotProfile, model: MjcfModel, report: Report) -> None:
    missing = sorted({stated for stated, path in model.meshes if not path.is_file()})
    if missing:
        report.add(
            error(
                "mjcf.mesh_not_found",
                f"profile:{profile.name}.description.mjcf",
                f"меши описания MuJoCo не найдены: {', '.join(missing)}",
                hint="путь меша читается от meshdir компилятора, а он — от главного файла; "
                "без меша симулятор не загрузит модель",
                adr=ADR,
                detail={"meshes": missing},
            )
        )


def _urdf(profile: RobotProfile, profile_path: Path) -> UrdfModel | None:
    """The URDF, when there is one to compare with. Its own problems are ``urdf.*``'s."""
    path = _resolve(profile.description.urdf, profile_path.parent)
    if is_template(path) or not path.is_file():
        return None
    try:
        return parse_urdf(path)
    except LoadError:
        return None


def _bound(value: float | None) -> str:
    return "без предела" if value is None else f"{value:g}"


def _check_against_urdf(
    profile: RobotProfile, model: MjcfModel, urdf: UrdfModel, report: Report
) -> None:
    """Where the two descriptions of one robot disagree, for the joints the contract commands."""
    ranges: dict[str, dict[str, list[float] | None]] = {}
    efforts: dict[str, dict[str, float | None]] = {}

    for name in profile.contract.joint_order:
        described, other = model.joint(name), urdf.joint(name)
        if described is None or other is None:
            continue
        urdf_range = (
            (other.limit.lower, other.limit.upper)
            if other.limit.lower is not None and other.limit.upper is not None
            else None
        )
        if (urdf_range is None) != (described.range is None) or (
            urdf_range is not None
            and described.range is not None
            and any(
                abs(a - b) > TOLERANCE for a, b in zip(urdf_range, described.range, strict=True)
            )
        ):
            ranges[name] = {
                "urdf": list(urdf_range) if urdf_range else None,
                "mjcf": list(described.range) if described.range else None,
            }
        effort = model.effort(name)
        if (effort is None) != (other.limit.effort is None) or (
            effort is not None
            and other.limit.effort is not None
            and abs(effort - other.limit.effort) > TOLERANCE
        ):
            efforts[name] = {"urdf": other.limit.effort, "mjcf": effort}

    where = f"profile:{profile.name}.description"
    hint = (
        "платформа не выбирает, какое описание право: Gazebo и MuJoCo поведут робота по разным "
        "пределам. Это условие любой сверки двух симуляторов — записывайте его рядом с "
        "результатом"
    )
    if efforts:
        listed = "; ".join(
            f"{name} URDF {_bound(values['urdf'])}, MJCF {_bound(values['mjcf'])}"
            for name, values in efforts.items()
        )
        report.add(
            warning(
                "mjcf.differs_from_urdf",
                where,
                f"описания расходятся в пределе момента: {listed}",
                hint=hint,
                adr=ADR,
                detail={"quantity": "effort", "joints": efforts},
            )
        )
    if ranges:
        listed = "; ".join(
            f"{name} URDF {values['urdf'] or 'без предела'}, MJCF {values['mjcf'] or 'без предела'}"
            for name, values in ranges.items()
        )
        report.add(
            warning(
                "mjcf.differs_from_urdf",
                where,
                f"описания расходятся в пределах позиции: {listed}",
                hint=hint,
                adr=ADR,
                detail={"quantity": "range", "joints": ranges},
            )
        )


# --------------------------------------------------------------------------
# A MuJoCo world.
# --------------------------------------------------------------------------


def check_mujoco_world(
    scenario: Scenario,
    participants: Sequence[tuple[ScenarioRobot, RobotProfile, Path]],
    scenario_path: Path | None,
) -> Report:
    """The scene a MuJoCo world loads, against the robots and contracts in it.

    ``scenario_path`` is where the scenario was read from; a relative scene is
    unreachable without it, and a scenario validated as a bare object is
    validated without the file checks, as a bare profile is without ``urdf.*``.
    """
    report = Report()
    if scenario.world.backend != MUJOCO:
        return report
    where = f"scenario:{scenario.name}.world"
    if scenario.world.file is None:
        report.add(
            error(
                "mjcf.scene_missing",
                where,
                "мир MuJoCo не называет сцену: world.file не задан",
                hint="сцена MuJoCo — вход, как описание: платформа её не порождает (ADR-0009). "
                "Укажите файл сцены, загружающий описание робота: world: {backend: mujoco, "
                "file: <путь к сцене>}",
                adr=ADR,
            )
        )
        return report
    stated = Path(scenario.world.file)
    if not stated.is_absolute():
        if scenario_path is None:
            return report
        stated = scenario_path.parent / stated
    scene = _load(stated, f"{where}.file", report)
    if scene is None:
        return report

    for robot, profile, profile_path in participants:
        spot = f"scenario:{scenario.name}.robots.{robot.id}"
        rate = profile.contract.state_rate_hz
        physics = 1.0 / scene.timestep
        if physics + 1e-9 < rate:
            report.add(
                error(
                    "mjcf.timestep_slower_than_state",
                    spot,
                    f"шаг физики сцены {scene.timestep:g} с — {physics:g} Гц, а контракт "
                    f"{profile.name} обещает состояние {rate:g} Гц",
                    hint="мир не меняется чаще своего шага: состояния между шагами несут ту же "
                    "метку времени, и сторожевой таймер примет их за молчание (ADR-0015). Шаг "
                    "задаёт сцена (<option timestep>), частоту — контракт",
                    adr=ADR,
                    detail={"timestep": scene.timestep, "state_rate_hz": rate},
                )
            )

        if profile.description.mjcf is None:
            report.add(
                error(
                    "mjcf.scene_without_description",
                    spot,
                    f"у профиля {profile.name!r} нет описания MuJoCo (description.mjcf): сцену "
                    f"{scene.path.name} не с чем сверить",
                    hint="робот в мире MuJoCo сверяется с профилем по MJCF, как в Gazebo — по URDF",
                    adr=ADR,
                    detail={"scene": str(scene.path)},
                )
            )
            continue
        described = _resolve(profile.description.mjcf, profile_path.parent)
        if not scene.loads(described):
            report.add(
                error(
                    "mjcf.scene_without_description",
                    spot,
                    f"сцена {scene.path.name} не загружает описание профиля {profile.name!r} "
                    f"({described.name}); загружает: "
                    f"{', '.join(path.name for path in scene.files)}",
                    hint="симулятор загрузит сцену, а с профилем сверено другое описание: робот "
                    "в мире — не тот, что проверен",
                    adr=ADR,
                    detail={
                        "scene": str(scene.path),
                        "description": str(described.resolve()),
                        "loads": [str(path) for path in scene.files],
                    },
                )
            )
    return report
