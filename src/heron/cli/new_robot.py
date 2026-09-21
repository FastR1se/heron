"""Generating a stub for a new robot (ARCHITECTURE.md, section 9; ADR-0056).

The standard procedure for adding a robot is a profile plus, if needed, an
adapter -- and no edit to the core (invariant 5). This module produces both, with
the TODOs a human has to fill and a contract that is deliberately minimal rather
than plausible: a stub that looks finished is a stub that ships unread.

The adapter is written as a package from the start, because that is the only way
the platform finds an adapter it did not ship (ADR-0021): a ``pyproject.toml``
with its entry point, the adapter, a recording harness and a test that holds the
adapter to the platform's contract suite. The suite is red until the TODOs are
done, and each failure names the one it hit. A stub that was not a package made
the first step of the path a step aside.

The generated adapter skeleton implements the whole ``ControlAdapter`` protocol,
``emergency_stop`` included, because section 12 makes it mandatory and an
omission here would propagate into every robot added this way. A robot of the
``command`` class gets the obligations of its class spelled out as TODOs, and a
profile with the envelope such a controller plans its motion in (ADR-0049).

A skeleton description is written too. It is a starting **input**, not an output
of the generators: the platform never rewrites it again and from then on only
compares it with the profile (ADR-0009). Without one the stub would not start in
any world, and its owner would be left facing an empty ``description.urdf``,
which is where a stub stops helping.

Whoever already has a description -- a CAD export, a vendor's model -- starts
from it instead (ADR-0074): the joints and their limits are read from the file,
which stays where it is and is referenced, not copied. That is a starting point
too, read once: from then on profile and description are two files the
validator compares, and nothing rebuilds the one from the other.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..core.branding import APP_NAME, APP_TITLE, ENTRY_POINT_GROUPS
from ..core.enums import ControlMode
from ..core.models import NAME_RE
from ..core.urdf import UrdfJoint, UrdfModel, parse_urdf

PROFILE_HEAD = """\
# Профиль робота {name}. Заготовка: заполните TODO и проверьте валидатором.
# Профиль — данные, не код: добавление робота не требует правок в ядре.
schema_version: 1
name: {name}

description:
  urdf: {urdf}   {urdf_note}

# Контракт — соглашение о представлении данных между средами.
# Его идентификатор вычисляется из содержимого; revision объявляется вручную.
contract:
  revision: 1
  joint_order: [{joints}]   {order_note}
  angle_unit: rad
"""

PROFILE_STREAMING = """\
{default_pose}  state_rate_hz: 100
  command_rate_hz: 50
  observation:
    - {{type: joint_pos, scale: 1.0}}
    - {{type: joint_vel, scale: 0.05}}
  action:
    type: joint_position
    clip: [-1.0, 1.0]
{action_map}
# TODO: оставьте только то, что робот действительно умеет.
capabilities: [arm.joint_position]

control_mode: streaming    # streaming — поток команд; command — «выполни движение»
access_levels: []          # для роботов со встроенным контроллером: high, partial, low

{limits_note}limits:
  joint_position:
    min: [{mins}]   {limits_todo}
    max: [{maxs}]   {limits_todo}
  joint_velocity:
    max: [{vels}]   {limits_todo}
{torque}
adapters:
  sim.dummy:
    plugin: dummy            # робот без железа: годится для проверки цепочки
  sim.gazebo:
    plugin: ros2_control     # мир комплекса; из этого адаптера порождаются среды
  real:
    plugin: {name}           # адаптер пакета, точка входа из pyproject.toml

skills: []
# image: {name}:latest       # голое имя; пространство имён подставляет платформа
"""

PROFILE_COMMAND = """\
  state_rate_hz: 20        # TODO: как часто робот публикует состояние (ADR-0050)
  command_rate_hz: 2       # как часто навык решает; выше 20 Гц — предупреждение
  observation:
    - {{type: joint_pos, scale: 1.0}}
  action:
    type: joint_position   # цель движения, а не уставка: разгон делает контроллер

# TODO: оставьте только то, что робот действительно умеет.
capabilities: [arm.joint_position]

control_mode: command      # «выполни движение и сообщи о завершении» (ADR-0049)
access_levels: []

# Рамка обязательна для прогона на железе (ADR-0051), а скорость и ускорение —
# то, по чему контроллер строит движение, умноженные на шкалу прогона (ADR-0052).
{limits_note}limits:
  joint_position:
    min: [{mins}]   {limits_todo}
    max: [{maxs}]   {limits_todo}
  joint_velocity:
    max: [{vels}]   {limits_todo}
{torque}  joint_acceleration:
    max: [{accs}]   {accs_todo}

adapters:
  sim.dummy:
    plugin: dummy            # мир из чисел в командном режиме: миссия проверяется до железа
  real:
    plugin: {name}           # адаптер пакета, точка входа из pyproject.toml

skills: [goto_joints]
"""

PYPROJECT_TEMPLATE = """\
[project]
name = "{dist}"
version = "0.1.0"
description = "{name} for {title}: an adapter for its API"
requires-python = ">=3.11"

# From the platform this package imports its public API and nothing else (ADR-0021).
# The robot's SDK is not listed here: the adapter declares it as an environment
# requirement, checked before connecting (ADR-0025).
dependencies = ["{pkg}"]

[project.optional-dependencies]
# The platform's contract suite, run from this package's tests (ADR-0056).
test = ["{pkg}[testing]"]

# How the platform finds the adapter: the group its built-in adapters use.
[project.entry-points."{group}"]
{name} = "{name}.adapter:{klass}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
"""

INIT_TEMPLATE = '''\
"""{name}: a robot joining {title} through its adapter."""
'''

ADAPTER_TEMPLATE = '''\
"""Control adapter for {name}.

Generated stub. Implement every method: the platform checks the protocol, not
the inheritance. ``emergency_stop`` is mandatory (ARCHITECTURE.md, section 12).

Talk to the robot through a channel handed to the constructor, and import the
vendor's SDK only when the channel is opened: then the adapter's decisions are
checkable by the contract suite over a recording channel, on a machine with
neither the SDK nor the robot (tests/).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from {pkg}.core.capabilities import Capability
from {pkg}.core.enums import ControlMode
from {pkg}.core.environment import Requirement
from {pkg}.core.models import Limits, RobotProfile
from {pkg}.core.types import Command, RobotState


class {klass}:
    """Link to {name} or to its twin in a simulator."""

    # Что процессу, ведущему робота, нужно от окружения: модули Python, пакеты
    # ROS 2, программы, переменные — и что в нём запрещено (ADR-0025). Умолчания
    # нет: пустой кортеж — это объявление «ничего», а не забытое поле.
    requirements: Sequence[Requirement] = ()  # TODO: needs(...), forbids(...)

    # Исполняет ли связь шкалу скорости прогона (ADR-0052). False — честное
    # умолчание для адаптера, который ещё ничего не умеет: такой адаптер шкалу
    # не получит, и run откажет в замедленном прогоне, а не проведёт его быстро.
    supports_speed_scale: bool = False  # TODO: True, когда шкала применяется и проверяется
{obligations}
    def __init__(self) -> None:
        self._profile: RobotProfile | None = None

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        raise NotImplementedError("TODO: открыть соединение с роботом")

    def disconnect(self) -> None:
        raise NotImplementedError("TODO: закрыть соединение")

    def read_state(self) -> RobotState:
        # Порядок величин — contract.joint_order, единицы — contract.angle_unit.
        # До connect — RuntimeError, не NotImplementedError.
        raise NotImplementedError("TODO: прочитать состояние")

    def send_command(self, cmd: Command) -> None:
        # Отказ — CommandRefusedError до отправки, с суставом, значением и пределами (ADR-0026).
        raise NotImplementedError("TODO: отправить команду")

    def capabilities(self) -> frozenset[Capability]:
        # Что умеет именно эта связь. Объявляет адаптер, а не ядро.
        raise NotImplementedError("TODO: перечислить возможности")

    def control_mode(self) -> ControlMode:
        raise NotImplementedError("TODO: вернуть ControlMode.{mode_member}")

    def safety_limits(self) -> Limits:
        raise NotImplementedError("TODO: пределы, которые связь гарантирует")

    def emergency_stop(self) -> None:
        # Обязателен. Без него адаптер нельзя выпускать на железо.
        raise NotImplementedError("TODO: аварийный останов")
'''

COMMAND_OBLIGATIONS = '''
    # Обязанности командного класса (ADR-0049), их проверяет общий набор:
    # TODO: RobotState.motion — idle, moving или halted, никогда не None;
    # TODO: с момента принятой команды до сообщения робота о конце движения — не idle;
    # TODO: команда при moving или halted — CommandRefusedError, одно движение за раз;
    # TODO: скорость и ускорение — из limits.joint_velocity и limits.joint_acceleration,
    #       умноженные на шкалу; без любого из двух — отказ в connect;
    # TODO: disconnect отзывает незавершённое движение, если связь ещё есть.
'''

TESTING_TEMPLATE = '''\
"""What the platform's contract suite needs to hold this adapter (ADR-0056).

A robot made of readings: the adapter over a recording channel, and the readings a
running robot would give it. Shipped with the package, so that its own tests and
anything else building the adapter for a test build it one way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from {pkg}.core.models import RobotProfile

from .adapter import {klass}

PLUGIN: Final = "{name}"
"""The adapter's entry-point name, as a profile writes it."""

PROFILE: Final = Path(__file__).resolve().parents[2] / "profile.yaml"
"""The profile beside this package's sources, as the stub laid them out."""


def profile() -> RobotProfile:
    return RobotProfile.model_validate(yaml.safe_load(PROFILE.read_text(encoding="utf-8")))


def recording_adapter() -> {klass}:
    # TODO: адаптер со внедрённым записывающим каналом — без SDK и без робота.
    return {klass}()


def prime(adapter: {klass}, profile: RobotProfile) -> None:
    """Queue readings on the recording channel before connecting, as a running robot would."""
    # TODO: подать показания в записывающий канал.
{finish}'''

FINISH_TEMPLATE = '''

def finish(adapter: {klass}, profile: RobotProfile) -> None:
    """Make the robot report that the accepted motion is over (ADR-0049)."""
    raise NotImplementedError("TODO: подать показание «движение закончено» в записывающий канал")
'''

TEST_TEMPLATE = '''\
"""{name}: the adapter held to the platform's contract suite (ADR-0056).

Red until every TODO of the adapter and the harness is done; each failure names
the one it hit.
"""

from __future__ import annotations

from {name} import testing

from {pkg}.testing.adapter_contract import AdapterContract, Harness


class Test{klass}(AdapterContract):
    harness = Harness(
        plugin=testing.PLUGIN,
        make=testing.recording_adapter,
        profile=testing.profile,
        prime=testing.prime,{finish}
    )
'''


URDF_TEMPLATE = """<?xml version="1.0"?>
<!--
  Заготовка описания кинематики {name}: {dof} суставов, примитивы вместо мешей.

  Это ВХОД платформы, а не её выход (ADR-0009). Порождена один раз, чтобы робот
  из заготовки можно было поставить в мир; дальше файл ваш — замените звенья,
  массы и пределы на настоящие или на модель производителя. Платформа больше
  никогда его не перезапишет, она будет только сверять его с профилем.

  Пределы здесь шире, чем в заготовке профиля, и так и должно быть: лимиты
  профиля — предохранительная рамка, она обязана быть уже модели (раздел 12).
-->
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.05" length="0.08"/></geometry>
    </visual>
    <collision>
      <geometry><cylinder radius="0.05" length="0.08"/></geometry>
    </collision>
  </link>
{links}</robot>
"""

URDF_LINK = """  <link name="link{index}">
    <inertial>
      <origin xyz="0 0 0.05"/>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry><cylinder radius="0.03" length="0.10"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry><cylinder radius="0.03" length="0.10"/></geometry>
    </collision>
  </link>
  <joint name="joint{index}" type="revolute">
    <parent link="{parent}"/>
    <child link="link{index}"/>
    <origin xyz="0 0 {offset}"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.20" upper="3.20" effort="50" velocity="2.0"/>
  </joint>
"""


@dataclass(frozen=True)
class Stub:
    """What ``write_stub`` created, by role."""

    profile: Path
    pyproject: Path
    package: Path
    adapter: Path
    testing: Path
    test: Path
    description: Path | None
    """The skeleton description written, or ``None`` when the stub references one (ADR-0074)."""

    @property
    def paths(self) -> list[Path]:
        created = [
            self.profile,
            self.pyproject,
            self.package,
            self.adapter,
            self.testing,
            self.test,
            self.description,
        ]
        return [path for path in created if path is not None]


def _skeleton_urdf(name: str, dof: int) -> str:
    """A chain of ``dof`` revolute joints named like the stub profile's."""
    links = []
    parent = "base_link"
    for index in range(dof):
        links.append(
            URDF_LINK.format(index=index, parent=parent, offset="0.08" if index == 0 else "0.10")
        )
        parent = f"link{index}"
    return URDF_TEMPLATE.format(name=name, dof=dof, links="".join(links))


DEFAULT_DOF: Final = 6

PLAIN_YAML_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
"""A joint name YAML reads as a string without quotes."""

UNSTATED_ANGLE: Final = math.pi
UNSTATED_LENGTH: Final = 1.0
UNSTATED_VELOCITY: Final = 1.0
"""What a stub writes where the description states no bound, always under a TODO."""


def _scalar(value: str) -> str:
    """A string as a YAML flow scalar: bare if it can be, JSON-quoted otherwise."""
    return value if PLAIN_YAML_NAME.match(value) else json.dumps(value, ensure_ascii=False)


def _number(value: float) -> str:
    """The shortest form that reads back as the same float, as the canonical form writes."""
    return repr(float(value))


def _listed(values: list[float]) -> str:
    return ", ".join(_number(value) for value in values)


def _todo(what: str, joints: list[str]) -> str:
    return f"# TODO: {what}: {', '.join(joints)}\n" if joints else ""


def _reference(description: Path, root: Path) -> str:
    """The description as the profile names it: relative to the profile where possible."""
    target = description.resolve()
    try:
        stated = Path(os.path.relpath(target, root.resolve())).as_posix()
    except ValueError:  # another drive: no relative path exists
        stated = target.as_posix()
    return json.dumps(stated, ensure_ascii=False)


def _fitted(low: float, high: float) -> tuple[float, float]:
    """Origin and scale that map ``clip [-1, 1]`` onto ``[low, high]`` and no further.

    The half-range is stepped down by an ulp while rounding would carry a target
    past a bound: the validator compares the same sums exactly (ADR-0041).
    """
    origin = (low + high) / 2
    scale = (high - low) / 2
    while origin + scale > high or origin - scale < low:
        scale = math.nextafter(scale, 0.0)
    return origin, scale


def _stub_joints(model: UrdfModel) -> tuple[UrdfJoint, ...]:
    """Joints a controller drives: commandable and not following another (ADR-0074)."""
    return tuple(joint for joint in model.commandable_joints if joint.mimic is None)


def _from_description(description: Path, root: Path) -> dict[str, str]:
    """Template values read from an existing description (ADR-0074).

    Only what the description states: joint order and the limits of each joint.
    A bound it does not state is written as a placeholder under a TODO naming
    the joint, never filled in silently.
    """
    model = parse_urdf(description)
    joints = _stub_joints(model)
    if not joints:
        raise ValueError(
            f"{description}: в описании нет управляемых суставов "
            f"(revolute, continuous, prismatic без <mimic>)"
        )

    mins: list[float] = []
    maxs: list[float] = []
    vels: list[float] = []
    unbounded: list[str] = []
    no_velocity: list[str] = []
    for joint in joints:
        lower, upper = joint.limit.lower, joint.limit.upper
        if lower is not None and upper is not None:
            mins.append(lower)
            maxs.append(upper)
        else:
            span = UNSTATED_ANGLE if joint.angular else UNSTATED_LENGTH
            mins.append(-span)
            maxs.append(span)
            unbounded.append(joint.name)
        velocity = joint.limit.velocity
        if velocity is not None and velocity > 0:
            vels.append(velocity)
        else:
            vels.append(UNSTATED_VELOCITY)
            no_velocity.append(joint.name)

    narrow = [joint.name for joint, lo, hi in zip(joints, mins, maxs, strict=True) if lo >= hi]
    if narrow:
        raise ValueError(
            f"{description}: у суставов {', '.join(narrow)} lower >= upper — такой сустав "
            f"заперт; в описании он fixed, а не управляемый"
        )
    fitted = [_fitted(lo, hi) for lo, hi in zip(mins, maxs, strict=True)]

    efforts = [joint.limit.effort for joint in joints]
    torque = ""
    if all(effort is not None and effort > 0 for effort in efforts):
        stated = [effort for effort in efforts if effort is not None]
        torque = f"  joint_torque:\n    max: [{_listed(stated)}]   # из описания\n"

    source = description.name
    return {
        "urdf": _reference(description, root),
        "urdf_note": "# описание — вход: платформа его не меняет, только сверяет (ADR-0009)",
        "joints": ", ".join(_scalar(joint.name) for joint in joints),
        "order_note": f"# из {source}, порядок документа; порядок — часть контракта, подтвердите",
        "mins": _listed(mins),
        "maxs": _listed(maxs),
        "vels": _listed(vels),
        "accs": ", ".join(["2.0"] * len(joints)),
        "limits_note": (
            f"# Пределы взяты из {source} (ADR-0074): это механика модели. Рамка профиля —\n"
            f"# предохранительная, не шире модели (раздел 12); сузьте её до рабочей.\n"
            + _todo("пределов позиции в описании нет, записано ±π (линейный — ±1.0 м)", unbounded)
            + _todo(
                f"скорость в описании не задана, записано {_number(UNSTATED_VELOCITY)}",
                no_velocity,
            )
        ),
        "limits_todo": "# TODO: сузить",
        "accs_todo": "# TODO: в описании ускорения нет",
        "torque": torque,
        "default_pose": (
            f"  # TODO: поза отсчёта — середина рамки описания; замените на рабочую позу,\n"
            f"  # а scale — на то, что нужно навыку: clip · scale + поза обязано лечь в рамку\n"
            f"  default_pose: [{_listed([origin for origin, _ in fitted])}]\n"
        ),
        "action_map": (
            f"    scale: [{_listed([scale for _, scale in fitted])}]\n"
            f"    offset: default_pose\n"
        ),
    }


def _placeholders(name: str, dof: int) -> dict[str, str]:
    """Template values of a stub with its own skeleton description."""
    return {
        "urdf": f"urdf/{name}.urdf",
        "urdf_note": "# заготовка описания; замените на настоящую модель",
        "joints": ", ".join(f"joint{i}" for i in range(dof)),
        "order_note": "# TODO: реальный порядок суставов",
        "mins": ", ".join(["-3.14"] * dof),
        "maxs": ", ".join(["3.14"] * dof),
        "vels": ", ".join(["1.0"] * dof),
        "accs": ", ".join(["2.0"] * dof),
        "limits_note": "",
        "limits_todo": "# TODO",
        "accs_todo": "# TODO",
        "torque": "",
        "default_pose": "",
        "action_map": "",
    }


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_")) + "Adapter"


def write_stub(
    name: str,
    into: Path,
    dof: int | None = None,
    control_mode: ControlMode = ControlMode.STREAMING,
    urdf: Path | None = None,
) -> Stub:
    """Write a profile and an adapter package, with a skeleton description or from ``urdf``.

    Given ``urdf``, the joints and limits come from that description and no
    skeleton is written (ADR-0074); the number of joints is then the
    description's, and stating ``dof`` as well is refused.
    """
    if not NAME_RE.match(name):
        raise ValueError(f"{name!r}: имя — нижний регистр, [a-z][a-z0-9_]*")
    if urdf is not None and dof is not None:
        raise ValueError("--dof и --from-urdf вместе не задаются: число суставов берёт описание")
    dof = DEFAULT_DOF if dof is None else dof
    if dof < 1:
        raise ValueError(f"число суставов должно быть положительным, получено {dof}")

    root = Path(into) / name
    sources = root / "src" / name
    stub = Stub(
        profile=root / "profile.yaml",
        pyproject=root / "pyproject.toml",
        package=sources / "__init__.py",
        adapter=sources / "adapter.py",
        testing=sources / "testing.py",
        test=root / "tests" / f"test_{name}_adapter.py",
        description=None if urdf is not None else root / "urdf" / f"{name}.urdf",
    )
    for path in stub.paths:
        if path.exists():
            raise FileExistsError(f"{path} уже существует; заготовка не перезаписывает файлы")

    shape = _from_description(Path(urdf), root) if urdf is not None else _placeholders(name, dof)
    command = control_mode is ControlMode.COMMAND
    klass = _class_name(name)
    values = {
        "name": name,
        "klass": klass,
        "pkg": APP_NAME,
        "title": APP_TITLE,
        "group": ENTRY_POINT_GROUPS["adapters"],
        "dist": name.replace("_", "-"),
        "mode_member": control_mode.name,
        **shape,
    }
    finish = FINISH_TEMPLATE.format(**values) if command else ""
    contents = {
        stub.profile: PROFILE_HEAD.format(**values)
        + (PROFILE_COMMAND if command else PROFILE_STREAMING).format(**values),
        stub.pyproject: PYPROJECT_TEMPLATE.format(**values),
        stub.package: INIT_TEMPLATE.format(**values),
        stub.adapter: ADAPTER_TEMPLATE.format(
            **values, obligations=COMMAND_OBLIGATIONS if command else ""
        ),
        stub.testing: TESTING_TEMPLATE.format(**values, finish=finish),
        stub.test: TEST_TEMPLATE.format(
            **values, finish="\n        finish=testing.finish," if command else ""
        ),
    }
    if stub.description is not None:
        contents[stub.description] = _skeleton_urdf(name, dof)
    for path, text in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return stub
