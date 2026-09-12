"""``dummy`` -- a robot that exists only as numbers.

Its purpose is CI and first contact with the platform (section 9): the whole
chain profile -> validator -> skill -> adapter runs with no simulator, no ROS 2
and no hardware. It is also the reference against which the adapter contract
tests are written, so anything a real adapter must do, this one does too --
including ``emergency_stop``, which section 12 makes mandatory.

Two design points worth keeping:

* Capabilities come from the profile by default but can be overridden through
  ``params``. Without the override the adapter would always agree with the
  robot, and a test could never show the validator catching a link that offers
  less than the profile claims. Knowledge of what the link can do belongs to the
  adapter (ADR-0001), so the override is the honest case, not a testing hack.
* Limits are enforced by refusing the command, not by clipping it. Clipping
  would repair a bad command silently, and the platform's whole position is that
  silent repair is worse than a stop.
"""

from __future__ import annotations

from typing import Any

from ..core.capabilities import Capability, parse_set
from ..core.enums import ActionType, ControlMode
from ..core.models import Limits, RobotProfile
from ..core.types import Command, Pose, RobotState


class SafetyViolationError(RuntimeError):
    """A command would leave the envelope the profile declares."""


class NotConnectedError(RuntimeError):
    """The adapter was used before ``connect`` or after ``disconnect``."""


class DummyAdapter:
    """An in-memory robot integrating commands on a virtual clock."""

    def __init__(self) -> None:
        self._profile: RobotProfile | None = None
        self._capabilities: frozenset[Capability] = frozenset()
        self._control_mode: ControlMode = ControlMode.STREAMING
        self._limits: Limits = Limits()
        self._enforce_limits = True
        self._position: list[float] = []
        self._velocity: list[float] = []
        self._time = 0.0
        self._dt = 0.01
        self._stopped = False
        self.commands: list[Command] = []
        """Everything sent, in order. Tests read it; nothing else should."""

    # -- ControlAdapter -----------------------------------------------------

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        self._profile = profile
        self._capabilities = (
            parse_set(config["capabilities"])
            if "capabilities" in config
            else frozenset(profile.capabilities)
        )
        self._control_mode = ControlMode(config.get("control_mode", profile.control_mode))
        self._limits = profile.limits
        self._enforce_limits = bool(config.get("enforce_limits", True))

        dof = profile.contract.dof
        start = config.get("initial_position")
        self._position = [float(v) for v in start] if start is not None else [0.0] * dof
        if len(self._position) != dof:
            raise ValueError(
                f"initial_position: {len(self._position)} значений при {dof} суставах"
            )
        self._velocity = [0.0] * dof
        self._check_limit_widths(dof)
        self._dt = 1.0 / profile.contract.state_rate_hz
        self._time = 0.0
        self._stopped = False
        self.commands.clear()

    def disconnect(self) -> None:
        self._profile = None

    def read_state(self) -> RobotState:
        """Read the state and advance the virtual clock by one state period."""
        self._require_connected()
        state = RobotState(
            timestamp=self._time,
            joint_position=tuple(self._position),
            joint_velocity=tuple(self._velocity),
            base=Pose(),
        )
        self._time += self._dt
        return state

    def send_command(self, cmd: Command) -> None:
        self._require_connected()
        if self._stopped:
            raise SafetyViolationError("аварийный останов: адаптер не принимает команды до reset()")

        if cmd.type is ActionType.JOINT_POSITION:
            self._apply_position(cmd.values)
        elif cmd.type is ActionType.JOINT_VELOCITY:
            self._apply_velocity(cmd.values)
        else:
            raise ValueError(
                f"{cmd.type}: dummy принимает joint_position и joint_velocity; "
                f"остальные типы действий требуют настоящего двойника"
            )
        self.commands.append(cmd)

    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def control_mode(self) -> ControlMode:
        return self._control_mode

    def safety_limits(self) -> Limits:
        return self._limits

    def emergency_stop(self) -> None:
        """Zero the velocities and refuse further commands until ``reset``."""
        self._velocity = [0.0] * len(self._velocity)
        self._stopped = True

    # -- dummy-specific -----------------------------------------------------

    def reset(self) -> None:
        """Clear an emergency stop. Real hardware needs a human here."""
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def time(self) -> float:
        return self._time

    # -- internals ----------------------------------------------------------

    def _require_connected(self) -> RobotProfile:
        if self._profile is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._profile

    def _check_limit_widths(self, dof: int) -> None:
        """Refuse limits that do not line up with the contract.

        The validator catches this too, but an adapter that trusts its input and
        then indexes past the end would turn a caught error into a crash.
        """
        widths = {
            "joint_position.min": getattr(self._limits.joint_position, "min", None),
            "joint_position.max": getattr(self._limits.joint_position, "max", None),
            "joint_velocity.max": getattr(self._limits.joint_velocity, "max", None),
        }
        wrong = {name: len(v) for name, v in widths.items() if v is not None and len(v) != dof}
        if wrong:
            raise ValueError(
                f"лимиты не совпадают с числом суставов ({dof}): {wrong}. "
                f"Профиль не прошёл бы валидатор (limits.length_mismatch)"
            )

    def _check_width(self, values: tuple[float, ...]) -> None:
        expected = len(self._position)
        if len(values) != expected:
            raise ValueError(f"команда на {len(values)} суставов при {expected} в контракте")

    def _apply_position(self, values: tuple[float, ...]) -> None:
        self._check_width(values)
        target = [float(v) for v in values]
        if self._enforce_limits and (bounds := self._limits.joint_position) is not None:
            outside = [
                i
                for i, v in enumerate(target)
                if not (bounds.min[i] <= v <= bounds.max[i])
            ]
            if outside:
                raise SafetyViolationError(
                    f"позиции вне лимитов в суставах {outside}; "
                    f"адаптер не подрезает команду, а отказывает"
                )
        self._velocity = [(t - p) / self._dt for t, p in zip(target, self._position, strict=True)]
        self._position = target

    def _apply_velocity(self, values: tuple[float, ...]) -> None:
        self._check_width(values)
        target = [float(v) for v in values]
        if self._enforce_limits and (bounds := self._limits.joint_velocity) is not None:
            outside = [i for i, v in enumerate(target) if abs(v) > bounds.max[i]]
            if outside:
                raise SafetyViolationError(f"скорости вне лимитов в суставах {outside}")
        self._velocity = target
        self._position = [p + v * self._dt for p, v in zip(self._position, target, strict=True)]
