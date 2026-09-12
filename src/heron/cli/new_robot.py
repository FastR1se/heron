"""Generating a stub for a new robot (ARCHITECTURE.md, section 9).

The standard procedure for adding a robot is a profile plus, if needed, an
adapter -- and no edit to the core (invariant 5). This module produces both, with
the TODOs a human has to fill and a contract that is deliberately minimal rather
than plausible: a stub that looks finished is a stub that ships unread.

The generated adapter skeleton implements the whole ``ControlAdapter`` protocol,
``emergency_stop`` included, because section 12 makes it mandatory and an
omission here would propagate into every robot added this way.
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
  urdf: urdf/{name}.urdf.xacro   # TODO: путь к описанию кинематики
  mesh_dir: meshes/

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
  # TODO: адаптер под бэкенд сцены и, при наличии железа, под реальный робот
  # sim.gazebo:
  #   plugin: ros2_control
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
    for path in (profile_path, adapter_path):
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
    return [profile_path, adapter_path]
