"""The adapter of an RC-series arm: a controller of the command class (ADR-0049).

Started from the stub ``profile new robopro_rc --control-mode command`` writes, the way
a user starts (ADR-0056), and filled in against the source of the vendor's API 1.4.1.

The controller plans the motion itself. A command is one waypoint in its buffer, the
buffer is executed with ``move``, and the motion is over when the buffer is empty and
the mode has fallen back to ``hold`` (facts 0.9, 0.32). So the adapter does not move
anything; it decides:

* whether a command may go at all -- inside the envelope, safety status ``normal`` or
  ``reduced``, controller in ``run``, no motion in progress, the applied speed scale
  not above the run's -- and refuses before anything is sent (ADR-0026, ADR-0053);
* how fast: the profile's envelope, with the run's scale applied by the controller,
  ``s`` on velocity and ``s²`` on acceleration, read back and checked (ADR-0052,
  ADR-0057);
* what phase the accepted motion is in, from one reading of the robot (ADR-0050);
* whether the robot on the link is the body the profile names (ADR-0054).

Everything that goes to the SDK is handed to a channel as values (``channel``), so
all of it runs on a recording channel without the SDK or the robot.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final

from heron.core.capabilities import Capability, parse_set
from heron.core.enums import ActionType, AngleUnit, ControlMode, MotionPhase
from heron.core.environment import Requirement, RequirementKind, needs
from heron.core.models import SPEED_SCALE_PARAM, Limits, RobotProfile
from heron.core.protocols import CommandRefusedError
from heron.core.types import Command, RobotState
from heron.core.units import from_radians, to_radians

from .channel import (
    WAYPOINT_COUNTER_MASK,
    Channel,
    ChannelError,
    ControllerState,
    LinkConfig,
    MotionMode,
    MotionWarning,
    RtdReading,
    SafetyStatus,
    Waypoint,
    code_name,
)

API_VERSION: Final = "1.4.1"
"""The version whose source this package is written against (fact 0.28 onwards)."""

REQUIREMENTS: Final[tuple[Requirement, ...]] = (
    needs(
        RequirementKind.PYTHON,
        "API",
        version=f"=={API_VERSION}",
        because="адаптер написан по исходнику этой версии и читает её закрытый приёмник RTD "
        "(ADR-0062); другая версия — другой протокол",
        hint="архив API «Робопро» версии 1.4.1 (Python 3.10–3.13): pip install <каталог "
        "архива с pyproject.toml>. Имя дистрибутива и модуля — API",
    ),
    needs(
        RequirementKind.PYTHON,
        "numpy",
        because="SDK импортирует его при загрузке RobotApi",
        hint="pip install numpy — ставится зависимостью архива API",
    ),
    needs(
        RequirementKind.PYTHON,
        "scipy",
        because="SDK импортирует его при загрузке RobotApi",
        hint="pip install scipy — ставится зависимостью архива API",
    ),
)
"""What the process driving an RC arm has to find (ADR-0025). The interpreter range
of the SDK (3.10–3.13) is held by ``requires-python`` of this package and of the
archive; ROS 2 is not needed and not forbidden."""

SPEED_LIMIT_RAD_S: Final = 3.14
ACCELERATION_LIMIT_RAD_S2: Final = 26.18
"""What ``add_new_waypoint`` accepts in radians (``JOINT_SPEED_LIMITS_RAD_SEC``,
``JOINT_ACCEL_LIMITS_RAD_SEC`` of API 1.4.1; fact 0.43). Note 3.14, not pi: a
profile at the passport's 180 °/s is refused by the SDK in radians."""

DEFAULT_POLL_S: Final = 0.1
"""How long one read waits for a new packet before returning the previous state."""

DEFAULT_LINK_S: Final = 5.0
"""How long ``connect`` waits for the first packet after the SDK connected (ADR-0016)."""

ACCEPTED_SAFETY: Final = frozenset({SafetyStatus.NORMAL, SafetyStatus.REDUCED})
EXECUTING: Final = frozenset({MotionMode.MOVE, MotionMode.MOVE_ADV})

log = logging.getLogger(__name__)


class NotConnectedError(RuntimeError):
    """The adapter was used before ``connect`` or after ``disconnect``."""


class LinkConfigurationError(RuntimeError):
    """The profile or the run asks for something this link cannot do. Raised by ``connect``."""


class BodyMismatchError(RuntimeError):
    """The robot on the link is not the body the profile names (ADR-0054)."""


class NotReadyError(RuntimeError):
    """The controller is not in ``run``: brakes are released by a human (ADR-0054, p. 3)."""


class ScaleNotAppliedError(RuntimeError):
    """The controller applies a faster scale than the run asked for (ADR-0052, p. 4)."""


class LinkError(RuntimeError):
    """The link never delivered a reading."""


class StopNotDeliveredError(RuntimeError):
    """The emergency stop could not be sent (ADR-0061)."""


class SafetyViolationError(CommandRefusedError):
    """A command the adapter refuses to send."""


class MotionInProgressError(SafetyViolationError):
    """A command arrived while the accepted motion is unfinished (ADR-0049)."""


def open_sdk(link: LinkConfig) -> Channel:
    """The real channel, imported here so importing the adapter imports no SDK."""
    from .sdk import SdkChannel

    return SdkChannel.open(link)


class RoboproRcAdapter:
    """Drives an RC arm through its vendor's ``RobotApi``, one waypoint per motion."""

    requirements: Sequence[Requirement] = REQUIREMENTS

    supports_speed_scale: bool = True
    """The controller scales every motion (``scale_setup``, fact 0.12), and reports the
    scale it applies in every packet (ADR-0052)."""

    def __init__(self, channel: Channel | None = None) -> None:
        self._channel = channel
        self._owns_channel = channel is None
        self._profile: RobotProfile | None = None
        self._link: LinkConfig | None = None
        self._unit = AngleUnit.RAD
        self._capabilities: frozenset[Capability] = frozenset()
        self._limits = Limits()
        self._scale = 1.0
        self._speed = 0.0
        self._acceleration = 0.0
        self._poll_s = DEFAULT_POLL_S
        self._link_s = DEFAULT_LINK_S
        self._reading: RtdReading | None = None
        self._last: RobotState | None = None
        self._latched: int | None = None
        """The waypoint counter the accepted motion is known by, until it is over."""
        self._stopped = False

    # -- ControlAdapter -------------------------------------------------------

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        """Check the profile and the run, then the robot on the link, then set the scale.

        In that order: a scale is a setting of the controller, and nothing is written
        to a robot before it has been shown to be the one the profile names.
        """
        self._profile = None
        self._check_profile(profile, config)
        link = LinkConfig.from_params(config)
        scale = _scale_of(config)
        speed, acceleration = _motion_bounds(profile)

        self._link = link
        self._unit = profile.contract.angle_unit
        self._capabilities = (
            parse_set(config["capabilities"])
            if "capabilities" in config
            else frozenset(profile.capabilities)
        )
        self._limits = profile.limits
        self._scale = scale
        self._speed, self._acceleration = speed, acceleration
        self._poll_s = float(config.get("poll_timeout", DEFAULT_POLL_S))
        self._link_s = float(config.get("link_timeout", DEFAULT_LINK_S))
        self._reading = None
        self._last = None
        self._latched = None
        self._stopped = False

        if self._channel is None:
            self._channel = open_sdk(link)
            self._owns_channel = True
        try:
            self._check_body(profile, link)
            first = self._await_link()
            self._check_reading(profile, first)
            self._apply_scale()
        except BaseException:
            self._close()
            raise
        self._profile = profile
        self._take(first)

    def disconnect(self) -> None:
        """Revoke an unfinished motion if the link is still there, then close it (ADR-0049).

        Never raises: a revocation that fails is logged, and the SDK is disconnected
        on every path, or its threads keep the process alive (ADR-0053, p. 2).
        """
        channel = self._channel
        if channel is not None and self._latched is not None and channel.lost() is None:
            try:
                channel.hold()
            except Exception as exc:
                log.error("незавершённое движение не отозвано при отключении: %s", exc)
        self._close()
        self._profile = None
        self._latched = None

    def read_state(self) -> RobotState:
        """The newest reading in the contract's units, or the previous one if none arrived.

        A previous reading comes back with its stamp, which is how staleness shows to
        the watchdog (ADR-0050, p. 3) -- and with its phase derived again, so that a
        command accepted since never reads as ``idle`` (ADR-0049).
        """
        self._require_connected()
        reading = self._link_channel().poll(self._poll_s)
        if reading is not None:
            return self._take(reading)
        if self._last is None or self._reading is None:  # pragma: no cover - connect reads one
            raise LinkError("нет ни одного показания, хотя связь установлена")
        return replace(self._last, motion=self._phase(self._reading))

    def send_command(self, cmd: Command) -> None:
        """One waypoint and ``move``, or ``CommandRefusedError`` with nothing sent."""
        profile = self._require_connected()
        if self._stopped:
            raise SafetyViolationError(
                "аварийный останов: адаптер не принимает команды до reset()"
            )
        if cmd.type is not ActionType.JOINT_POSITION:
            raise ValueError(f"контракт объявляет цель joint_position, команда несёт {cmd.type}")
        dof = profile.contract.dof
        if len(cmd.values) != dof:
            raise ValueError(f"команда на {len(cmd.values)} суставов при {dof} в контракте")
        bounds = self._limits.joint_position
        assert bounds is not None  # checked on connect
        outside = "; ".join(
            f"{profile.contract.joint_order[i]} = {value:.4g} при "
            f"[{bounds.min[i]:.4g}, {bounds.max[i]:.4g}]"
            for i, value in enumerate(cmd.values)
            if not bounds.min[i] <= value <= bounds.max[i]
        )
        if outside:
            raise SafetyViolationError(
                f"позиции вне лимитов: {outside}; адаптер не подрезает команду, а отказывает"
            )
        reading = self._reading
        assert reading is not None  # connect reads one
        self._refuse_unless_ready(reading)

        expected = (reading.waypoint_counter + 1) & WAYPOINT_COUNTER_MASK
        channel = self._link_channel()
        channel.add_waypoint(
            Waypoint(
                target=tuple(to_radians(float(v), self._unit) for v in cmd.values),
                speed=self._speed,
                acceleration=self._acceleration,
            )
        )
        # In the buffer now: whatever happens next, the robot has a motion to finish.
        self._latched = expected
        try:
            channel.start()
        except BaseException:
            try:
                channel.hold()
            except Exception as exc:
                log.error("точка осталась в буфере, hold не прошёл: %s", exc)
            raise

    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def control_mode(self) -> ControlMode:
        return ControlMode.COMMAND

    def safety_limits(self) -> Limits:
        return self._limits

    def emergency_stop(self) -> None:
        """``hold``: stop where the robot is and drop the buffer, power on (ADR-0061).

        The adapter counts as stopped before anything is sent, so a stop that cannot
        be sent still refuses what comes after it. A motion in progress stays
        ``halted`` until ``reset``. When the SDK has already closed the link there is
        nothing to send the stop through; that is said, not hidden, and the motion is
        the controller's (fact 0.44).
        """
        self._require_connected()
        self._stopped = True
        channel = self._link_channel()
        try:
            channel.hold()
        except ChannelError as exc:
            lost = channel.lost()
            raise StopNotDeliveredError(
                f"hold не доставлен: {exc}"
                + (
                    f". Связи нет ({lost}): движение ведёт контроллер, адаптеру "
                    f"останавливать нечем — кнопка у робота (ADR-0061)"
                    if lost is not None
                    else ""
                )
            ) from exc

    # -- adapter-specific -----------------------------------------------------

    def reset(self) -> None:
        """Clear an emergency stop. Real hardware needs a human here."""
        self._stopped = False
        self._latched = None

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def channel(self) -> Channel | None:
        """The link this adapter uses: the injected one, or the one ``connect`` opened."""
        return self._channel

    @property
    def speed_scale(self) -> float:
        return self._scale

    @property
    def motion_bounds(self) -> tuple[float, float]:
        """Speed (rad/s) and acceleration (rad/s²) every waypoint carries."""
        return self._speed, self._acceleration

    # -- internals ------------------------------------------------------------

    def _require_connected(self) -> RobotProfile:
        if self._profile is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._profile

    def _link_channel(self) -> Channel:
        if self._channel is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._channel

    def _close(self) -> None:
        if self._channel is not None and self._owns_channel:
            channel, self._channel = self._channel, None
            channel.close()

    @staticmethod
    def _check_profile(profile: RobotProfile, config: dict[str, Any]) -> None:
        mode = ControlMode(config.get("control_mode", profile.control_mode))
        if mode is not ControlMode.COMMAND:
            raise LinkConfigurationError(
                f"{profile.name}: контроллер RC принимает точки движения, а не поток — "
                f"control_mode: command (ADR-0049); профиль просит {mode}"
            )
        if profile.contract.action.type is not ActionType.JOINT_POSITION:
            raise LinkConfigurationError(
                f"{profile.name}: RC принимает цель по суставам (joint_position), действие "
                f"контракта — {profile.contract.action.type}"
            )
        limits = profile.limits
        missing = [
            name
            for name in ("joint_position", "joint_velocity", "joint_acceleration")
            if getattr(limits, name) is None
        ]
        if missing:
            raise LinkConfigurationError(
                f"{profile.name}: командный режим строит движение по рамке профиля, а в "
                f"limits нет {', '.join(missing)} (ADR-0049)"
            )
        dof = profile.contract.dof
        widths = {
            "joint_position.min": len(limits.joint_position.min),  # type: ignore[union-attr]
            "joint_position.max": len(limits.joint_position.max),  # type: ignore[union-attr]
            "joint_velocity.max": len(limits.joint_velocity.max),  # type: ignore[union-attr]
            "joint_acceleration.max": len(limits.joint_acceleration.max),  # type: ignore[union-attr]
        }
        wrong = {name: width for name, width in widths.items() if width != dof}
        if wrong:
            raise LinkConfigurationError(
                f"лимиты не совпадают с числом суставов ({dof}): {wrong}; профиль не прошёл бы "
                f"валидатор (limits.length_mismatch)"
            )

    def _check_body(self, profile: RobotProfile, link: LinkConfig) -> None:
        """``robot_model`` against ``params.model``, by equality (ADR-0054, p. 1)."""
        told = self._link_channel().identity().model
        if not told.strip():
            raise BodyMismatchError(
                f"{link.host}: контроллер не назвал модель робота — тело не удостоверено"
            )
        if told != link.model:
            raise BodyMismatchError(
                f"{link.host}: контроллер называет модель {told!r}, профиль {profile.name} — "
                f"{link.model!r}. Подключение к другому телу отвергнуто (ADR-0054)"
            )

    def _check_reading(self, profile: RobotProfile, first: RtdReading) -> None:
        dof = profile.contract.dof
        if len(first.q) != dof:
            raise BodyMismatchError(
                f"SDK сообщает {len(first.q)} суставов, контракт {profile.name} — {dof}"
            )
        if first.controller_state != ControllerState.RUN:
            raise NotReadyError(
                f"контроллер в состоянии {code_name(ControllerState, first.controller_state)}, "
                f"а не run: адаптер тормоза не снимает. Переведите робота в run в «Пульсе» "
                f"человеком у кнопки и отключите «Пульс» от управления (ADR-0054)"
            )

    def _apply_scale(self) -> None:
        """``scale_setup.set(s, s²)`` and read back: each applied scale not above its request."""
        channel = self._link_channel()
        wanted = (self._scale, self._scale**2)
        channel.set_scale(*wanted)
        applied = channel.scale()
        faster = [
            f"{name} {got:g} при запрошенной {asked:g}"
            for name, got, asked in zip(("скорость", "ускорение"), applied, wanted, strict=True)
            if got > asked
        ]
        if faster:
            raise ScaleNotAppliedError(
                f"контроллер применяет шкалу быстрее запрошенной: {'; '.join(faster)} "
                f"(ADR-0052, ADR-0057)"
            )

    def _await_link(self) -> RtdReading:
        channel = self._link_channel()
        deadline = time.monotonic() + self._link_s
        while True:
            reading = channel.poll(self._poll_s)
            if reading is not None:
                return reading
            lost = channel.lost()
            if lost is not None or time.monotonic() >= deadline:
                where = self._link.host if self._link is not None else "?"
                raise LinkError(
                    f"{where}: пакет RTD не пришёл за {self._link_s:g} с"
                    + (f" — {lost}" if lost is not None else "")
                )

    def _refuse_unless_ready(self, reading: RtdReading) -> None:
        """The refusals that depend on the robot, from its newest reading (ADR-0053)."""
        phase = self._phase(reading)
        if phase is not MotionPhase.IDLE:
            raise MotionInProgressError(
                f"робот {'выполняет' if phase is MotionPhase.MOVING else 'не завершил'} "
                f"движение (режим {code_name(MotionMode, reading.motion_mode)}, в буфере "
                f"{reading.buffer_fill}); одно движение за раз (ADR-0049)"
            )
        if reading.safety not in ACCEPTED_SAFETY:
            raise SafetyViolationError(
                f"статус безопасности {code_name(SafetyStatus, reading.safety)}: команды "
                f"принимаются только в normal и reduced (ADR-0053)"
            )
        if reading.controller_state != ControllerState.RUN:
            raise SafetyViolationError(
                f"контроллер в состоянии {code_name(ControllerState, reading.controller_state)},"
                f" а не run (ADR-0054)"
            )
        if reading.velocity_scale > self._scale:
            raise SafetyViolationError(
                f"контроллер применяет шкалу скорости {reading.velocity_scale:g}, прогон "
                f"просил {self._scale:g}: замедление не доказано (ADR-0052)"
            )

    def _phase(self, reading: RtdReading) -> MotionPhase:
        """What the accepted motion is doing, from this one reading (ADR-0049, ADR-0050).

        Over means: the controller took our point (its counter reached it), the buffer
        is empty and the mode has fallen back to ``hold``, in a state that could
        execute a motion. Until then the phase is latched: ``moving`` while the
        controller executes or has not yet shown our point, ``halted`` when it cannot
        execute -- paused, stopped, not ``run``, a manual mode -- or when the
        platform has stopped it.
        """
        healthy = (
            reading.safety in ACCEPTED_SAFETY and reading.controller_state == ControllerState.RUN
        )
        mode = reading.motion_mode
        if self._latched is None:
            if mode in EXECUTING:
                return MotionPhase.MOVING
            if mode == MotionMode.HOLD and reading.buffer_fill == 0:
                return MotionPhase.IDLE
            return MotionPhase.HALTED
        if self._stopped or not healthy or mode not in (*EXECUTING, MotionMode.HOLD):
            return MotionPhase.HALTED
        taken = reading.waypoint_counter == self._latched
        if taken and reading.buffer_fill == 0 and mode == MotionMode.HOLD:
            self._latched = None
            return MotionPhase.IDLE
        return MotionPhase.MOVING

    def _take(self, reading: RtdReading) -> RobotState:
        self._reading = reading
        unit = self._unit
        state = RobotState(
            timestamp=reading.stamp,
            joint_position=tuple(from_radians(v, unit) for v in reading.q),
            joint_velocity=tuple(from_radians(v, unit) for v in reading.qd),
            joint_torque=reading.torque,
            motion=self._phase(reading),
            extra={
                "safety": code_name(SafetyStatus, reading.safety),
                "controller_state": code_name(ControllerState, reading.controller_state),
                "motion_mode": code_name(MotionMode, reading.motion_mode),
                "warning": code_name(MotionWarning, reading.state_flags),
                "buffer_fill": reading.buffer_fill,
                "waypoint_counter": reading.waypoint_counter,
                "velocity_scale": reading.velocity_scale,
                "velocity_limit_reason": reading.velocity_limit_reason,
                "brakes_released": reading.brakes_released,
                "flange": reading.flange,
                "cycle_time": reading.cycle_time,
            },
        )
        self._last = state
        return state


def _scale_of(config: dict[str, Any]) -> float:
    scale = float(config.get(SPEED_SCALE_PARAM, 1.0))
    if not 0 < scale <= 1:
        raise LinkConfigurationError(f"{SPEED_SCALE_PARAM} = {scale:g}: нужно 0 < s ≤ 1")
    return scale


def _motion_bounds(profile: RobotProfile) -> tuple[float, float]:
    """One speed and one acceleration for a waypoint, in rad/s and rad/s².

    The SDK takes a single value of each for all joints of a point. The slowest
    joint's bound is the one every joint can keep; the run's scale is applied by the
    controller, not here (ADR-0057).
    """
    unit = profile.contract.angle_unit
    velocity = profile.limits.joint_velocity
    acceleration = profile.limits.joint_acceleration
    assert velocity is not None and acceleration is not None  # checked before
    speed = min(to_radians(v, unit) for v in velocity.max)
    accel = min(to_radians(a, unit) for a in acceleration.max)
    beyond = [
        f"{name} {value:.4g} при пределе SDK {limit}"
        for name, value, limit in (
            ("скорость, рад/с", speed, SPEED_LIMIT_RAD_S),
            ("ускорение, рад/с²", accel, ACCELERATION_LIMIT_RAD_S2),
        )
        if value > limit
    ]
    if beyond:
        raise LinkConfigurationError(
            f"{profile.name}: рамка шире того, что SDK принимает в точке: {'; '.join(beyond)} "
            f"(API 1.4.1, факт 0.43)"
        )
    return speed, accel
