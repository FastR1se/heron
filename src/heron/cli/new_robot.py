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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.branding import APP_NAME, APP_TITLE, ENTRY_POINT_GROUPS
from ..core.enums import ControlMode
from ..core.models import NAME_RE

PROFILE_HEAD = """\
# Профиль робота {name}. Заготовка: заполните TODO и проверьте валидатором.
# Профиль — данные, не код: добавление робота не требует правок в ядре.
schema_version: 1
name: {name}

description:
  urdf: urdf/{name}.urdf   # заготовка описания; замените на настоящую модель

# Контракт — соглашение о представлении данных между средами.
# Его идентификатор вычисляется из содержимого; revision объявляется вручную.
contract:
  revision: 1
  joint_order: [{joints}]   # TODO: реальный порядок суставов
  angle_unit: rad
"""

PROFILE_STREAMING = """\
  state_rate_hz: 100
  command_rate_hz: 50
  observation:
    - {{type: joint_pos, scale: 1.0}}
    - {{type: joint_vel, scale: 0.05}}
  action:
    type: joint_position
    clip: [-1.0, 1.0]

# TODO: оставьте только то, что робот действительно умеет.
capabilities: [arm.joint_position]

control_mode: streaming    # streaming — поток команд; command — «выполни движение»
access_levels: []          # для роботов со встроенным контроллером: high, partial, low

limits:
  joint_position:
    min: [{mins}]   # TODO
    max: [{maxs}]   # TODO
  joint_velocity:
    max: [{vels}]   # TODO

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
limits:
  joint_position:
    min: [{mins}]   # TODO
    max: [{maxs}]   # TODO
  joint_velocity:
    max: [{vels}]   # TODO
  joint_acceleration:
    max: [{accs}]   # TODO

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
    description: Path

    @property
    def paths(self) -> list[Path]:
        return [
            self.profile,
            self.pyproject,
            self.package,
            self.adapter,
            self.testing,
            self.test,
            self.description,
        ]


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


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_")) + "Adapter"


def write_stub(
    name: str,
    into: Path,
    dof: int = 6,
    control_mode: ControlMode = ControlMode.STREAMING,
) -> Stub:
    """Write a profile, an adapter package and a skeleton description."""
    if not NAME_RE.match(name):
        raise ValueError(f"{name!r}: имя — нижний регистр, [a-z][a-z0-9_]*")
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
        description=root / "urdf" / f"{name}.urdf",
    )
    for path in stub.paths:
        if path.exists():
            raise FileExistsError(f"{path} уже существует; заготовка не перезаписывает файлы")

    command = control_mode is ControlMode.COMMAND
    klass = _class_name(name)
    values = {
        "name": name,
        "klass": klass,
        "pkg": APP_NAME,
        "title": APP_TITLE,
        "group": ENTRY_POINT_GROUPS["adapters"],
        "dist": name.replace("_", "-"),
        "joints": ", ".join(f"joint{i}" for i in range(dof)),
        "mins": ", ".join(["-3.14"] * dof),
        "maxs": ", ".join(["3.14"] * dof),
        "vels": ", ".join(["1.0"] * dof),
        "accs": ", ".join(["2.0"] * dof),
        "mode_member": control_mode.name,
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
        stub.description: _skeleton_urdf(name, dof),
    }
    for path, text in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return stub
