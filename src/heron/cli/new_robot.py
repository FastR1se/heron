"""Generating a stub for a new robot (ARCHITECTURE.md, section 9).

The standard procedure for adding a robot is a profile plus, if needed, an
adapter -- and no edit to the core (invariant 5). This module produces both, with
the TODOs a human has to fill and a contract that is deliberately minimal rather
than plausible: a stub that looks finished is a stub that ships unread.

The generated adapter skeleton implements the whole ``ControlAdapter`` protocol,
``emergency_stop`` included, because section 12 makes it mandatory and an
omission here would propagate into every robot added this way.

A skeleton description is written too. It is a starting **input**, not an output
of the generators: the platform never rewrites it again and from then on only
compares it with the profile (ADR-0009). Without one the stub would not start in
any world, and its owner would be left facing an empty ``description.urdf``,
which is where a stub stops helping.
"""

from __future__ import annotations

from pathlib import Path

from ..core.branding import APP_NAME, ENTRY_POINT_GROUPS
from ..core.models import NAME_RE

PROFILE_TEMPLATE = """\
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
  # TODO: при наличии железа — адаптер под реальный робот
  # real:
  #   plugin: {module}:{klass}

skills: []
# image: {name}:latest       # голое имя; пространство имён подставляет платформа
"""

ADAPTER_TEMPLATE = '''\
"""Control adapter for {name}.

Generated stub. Implement every method: the platform checks the protocol, not
the inheritance. ``emergency_stop`` is mandatory (ARCHITECTURE.md, section 12).

To make this adapter discoverable, declare an entry point in your pyproject:

    [project.entry-points."{group}"]
    {name} = "{module}:{klass}"

Nothing in this file needs to import the platform except the types it speaks in,
so it can live in a package of yours, outside the core.
"""

from __future__ import annotations

from typing import Any

from {pkg}.core.capabilities import Capability
from {pkg}.core.enums import ControlMode
from {pkg}.core.models import Limits, RobotProfile
from {pkg}.core.types import Command, RobotState


class {klass}:
    """Link to {name} or to its twin in a simulator."""

    def __init__(self) -> None:
        self._profile: RobotProfile | None = None

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        self._profile = profile
        raise NotImplementedError("TODO: открыть соединение с роботом")

    def disconnect(self) -> None:
        raise NotImplementedError("TODO: закрыть соединение")

    def read_state(self) -> RobotState:
        # Порядок величин — contract.joint_order, единицы — contract.angle_unit.
        raise NotImplementedError("TODO: прочитать состояние")

    def send_command(self, cmd: Command) -> None:
        raise NotImplementedError("TODO: отправить команду")

    def capabilities(self) -> frozenset[Capability]:
        # Что умеет именно эта связь. Объявляет адаптер, а не ядро.
        raise NotImplementedError("TODO: перечислить возможности")

    def control_mode(self) -> ControlMode:
        raise NotImplementedError("TODO: streaming или command")

    def safety_limits(self) -> Limits:
        raise NotImplementedError("TODO: пределы, которые связь гарантирует")

    def emergency_stop(self) -> None:
        # Обязателен. Без него адаптер нельзя выпускать на железо.
        raise NotImplementedError("TODO: аварийный останов")
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


def write_stub(name: str, into: Path, dof: int = 6) -> list[Path]:
    """Write a profile and an adapter skeleton, returning what was created."""
    if not NAME_RE.match(name):
        raise ValueError(f"{name!r}: имя — нижний регистр, [a-z][a-z0-9_]*")
    if dof < 1:
        raise ValueError(f"число суставов должно быть положительным, получено {dof}")

    target = Path(into) / name
    profile_path = target / "profile.yaml"
    adapter_path = target / "adapter.py"
    urdf_path = target / "urdf" / f"{name}.urdf"
    for path in (profile_path, adapter_path, urdf_path):
        if path.exists():
            raise FileExistsError(f"{path} уже существует; заготовка не перезаписывает файлы")

    klass = _class_name(name)
    module = f"{name}.adapter"
    target.mkdir(parents=True, exist_ok=True)

    profile_path.write_text(
        PROFILE_TEMPLATE.format(
            name=name,
            joints=", ".join(f"joint{i}" for i in range(dof)),
            mins=", ".join(["-3.14"] * dof),
            maxs=", ".join(["3.14"] * dof),
            vels=", ".join(["1.0"] * dof),
            module=module,
            klass=klass,
        ),
        encoding="utf-8",
    )
    adapter_path.write_text(
        ADAPTER_TEMPLATE.format(
            name=name,
            klass=klass,
            module=module,
            group=ENTRY_POINT_GROUPS["adapters"],
            pkg=APP_NAME,
        ),
        encoding="utf-8",
    )
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path.write_text(_skeleton_urdf(name, dof), encoding="utf-8")
    return [profile_path, adapter_path, urdf_path]
