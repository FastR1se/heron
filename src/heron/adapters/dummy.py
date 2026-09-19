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

In ``control_mode: command`` it is a controller that plans the motion itself
(ADR-0049): a position command is a goal, reached joint by joint along a
trapezoid bounded by ``limits.joint_velocity`` times the run's speed scale and
``limits.joint_acceleration`` times its square, so that the motion takes ``1/s``
as long and keeps its shape (ADR-0052, ADR-0057). The phase is latched from the
accepted command to the reading after the motion's end, and a second command
before then is refused. This is what a robot of the command class is held to
before any such robot exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.capabilities import Capability, parse_set
from ..core.enums import ActionType, ControlMode, MotionPhase
from ..core.environment import Requirement
from ..core.models import SPEED_SCALE_PARAM, Limits, RobotProfile
from ..core.protocols import CommandRefusedError
from ..core.types import Command, Pose, RobotState
from .common import outside_bounds
from .motion import Motion, plan


class SafetyViolationError(CommandRefusedError):
    """A command would leave the envelope the profile declares."""


class MotionInProgressError(CommandRefusedError):
    """A command arrived while the accepted motion is unfinished (ADR-0049)."""


class NotConnectedError(RuntimeError):
    """The adapter was used before ``connect`` or after ``disconnect``."""


class LinkConfigurationError(RuntimeError):
    """The profile or the run asks for something this link cannot do. Raised by ``connect``."""


class DummyAdapter:
    """An in-memory robot integrating commands on a virtual clock."""

    requirements: Sequence[Requirement] = ()
    """A robot made of numbers needs nothing from a machine (ADR-0025)."""

    supports_speed_scale: bool = True
    """In the command class the controller plans the motion, so it can slow it
    (ADR-0052). A stream it cannot slow, and says so on ``connect``."""

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
        self._speed_scale = 1.0
        self._motion: Motion | None = None
        self.commands: list[Command] = []
        """Everything sent, in order. Tests read it; nothing else should."""

    # -- ControlAdapter -----------------------------------------------------

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        self._profile = None
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
        self._speed_scale = self._scale_of(config)
        if self._control_mode is ControlMode.COMMAND:
            self._check_motion_limits()
        self._dt = 1.0 / profile.contract.state_rate_hz
        self._time = 0.0
        self._stopped = False
        self._motion = None
        self.commands.clear()
        self._profile = profile

    def disconnect(self) -> None:
        """Revoke an unfinished motion: nothing is left for the robot to do (ADR-0049)."""
        self._motion = None
        self._profile = None

    def read_state(self) -> RobotState:
        """Read the state and advance the virtual clock by one state period."""
        self._require_connected()
        motion: MotionPhase | None = None
        if self._control_mode is ControlMode.COMMAND:
            motion = self._advance_motion()
        state = RobotState(
            timestamp=self._time,
            joint_position=tuple(self._position),
            joint_velocity=tuple(self._velocity),
            base=Pose(),
            motion=motion,
        )
        self._time += self._dt
        return state

    def send_command(self, cmd: Command) -> None:
        self._require_connected()
        if self._stopped:
            raise SafetyViolationError("аварийный останов: адаптер не принимает команды до reset()")

        if self._control_mode is ControlMode.COMMAND:
            self._accept_motion(cmd)
        elif cmd.type is ActionType.JOINT_POSITION:
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
        """Zero the velocities and refuse further commands until ``reset``.

        A motion in progress stops where it is and stays ``halted`` -- accepted,
        neither executing nor finished -- until ``reset`` drops it.
        """
        if self._motion is not None and self._control_mode is ControlMode.COMMAND:
            self._advance_motion()
        self._velocity = [0.0] * len(self._velocity)
        self._stopped = True

    # -- dummy-specific -----------------------------------------------------

    def reset(self) -> None:
        """Clear an emergency stop. Real hardware needs a human here."""
        self._stopped = False
        self._motion = None

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def time(self) -> float:
        return self._time

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    # -- internals ----------------------------------------------------------

    def _require_connected(self) -> RobotProfile:
        if self._profile is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._profile

    def _scale_of(self, config: dict[str, Any]) -> float:
        scale = float(config.get(SPEED_SCALE_PARAM, 1.0))
        if not 0 < scale <= 1:
            raise LinkConfigurationError(f"{SPEED_SCALE_PARAM} = {scale:g}: нужно 0 < s ≤ 1")
        if scale != 1 and self._control_mode is ControlMode.STREAMING:
            raise LinkConfigurationError(
                f"{SPEED_SCALE_PARAM} = {scale:g}: потоковую связь адаптер не замедляет — "
                f"движение планирует навык (ADR-0052)"
            )
        return scale

    def _check_motion_limits(self) -> None:
        """A controller planning the motion needs both bounds of it (ADR-0049)."""
        missing = [
            name
            for name in ("joint_velocity", "joint_acceleration")
            if getattr(self._limits, name) is None
        ]
        if missing:
            raise LinkConfigurationError(
                f"командный режим: движение строится по рамке профиля, а в limits нет "
                f"{', '.join(missing)} (ADR-0049)"
            )

    def _check_limit_widths(self, dof: int) -> None:
        """Refuse limits that do not line up with the contract.

        The validator catches this too, but an adapter that trusts its input and
        then indexes past the end would turn a caught error into a crash.
        """
        widths = {
            "joint_position.min": getattr(self._limits.joint_position, "min", None),
            "joint_position.max": getattr(self._limits.joint_position, "max", None),
            "joint_velocity.max": getattr(self._limits.joint_velocity, "max", None),
            "joint_acceleration.max": getattr(self._limits.joint_acceleration, "max", None),
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

    def _check_envelope(self, target: list[float]) -> None:
        if self._enforce_limits and (bounds := self._limits.joint_position) is not None:
            outside = outside_bounds(self._require_connected(), target, bounds.min, bounds.max)
            if outside:
                raise SafetyViolationError(
                    f"позиции вне лимитов: {outside}; "
                    f"адаптер не подрезает команду, а отказывает"
                )

    def _apply_position(self, values: tuple[float, ...]) -> None:
        self._check_width(values)
        target = [float(v) for v in values]
        self._check_envelope(target)
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

    def _accept_motion(self, cmd: Command) -> None:
        if cmd.type is not ActionType.JOINT_POSITION:
            raise ValueError(
                f"{cmd.type}: в командном режиме dummy принимает цель joint_position"
            )
        if self._motion is not None and not self._motion.over(self._time):
            raise MotionInProgressError(
                "робот выполняет принятое движение; одно движение за раз (ADR-0049)"
            )
        self._check_width(cmd.values)
        target = [float(v) for v in cmd.values]
        self._check_envelope(target)
        speeds = self._limits.joint_velocity
        accelerations = self._limits.joint_acceleration
        assert speeds is not None and accelerations is not None  # checked on connect
        self._motion = plan(
            self._time, self._position, target, speeds.max, accelerations.max, self._speed_scale
        )

    def _advance_motion(self) -> MotionPhase:
        """Where the accepted motion is at the current time, applied to the pose."""
        motion = self._motion
        if motion is None:
            return MotionPhase.IDLE
        if self._stopped:
            return MotionPhase.HALTED
        self._position, self._velocity = motion.at(self._time)
        if motion.over(self._time):
            self._motion = None
            return MotionPhase.IDLE
        return MotionPhase.MOVING
