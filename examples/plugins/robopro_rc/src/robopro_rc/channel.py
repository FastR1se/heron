"""The link to an RC controller, as values (ADR-0053, ADR-P5B).

The adapter decides everything -- which pose, how fast, whether a command may go at
all, what phase a motion is in -- and hands a channel a ``Waypoint``; the channel
reads the controller and hands back an ``RtdReading``. Neither is an object of the
vendor's SDK, so every decision of the adapter is checkable on a machine that has
neither the SDK nor the robot. What cannot be checked there is the translation to
the SDK, and it lives alone in ``sdk``.

Everything here is in wire units: angles in radians, lengths in metres, codes as
the controller sends them. Converting to the contract's unit is the adapter's.

Two channels ship: the real one (``sdk``), and one that writes commands down and
answers with readings it was given.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Final, Protocol, runtime_checkable

from heron.core.protocols import CommandRefusedError

WAYPOINT_COUNTER_MASK: Final = 0xFFFF
"""``wp_cntr`` is an unsigned 16-bit counter of accepted waypoints (RTD ``H``;
``WP_COUNTER_MAX_VALUE`` of API 1.4.1). The SDK expects the next one as
``(wp_cntr + 1) & 0xFFFF``, and so does the adapter."""


class SafetyStatus(IntEnum):
    """RTD ``safety``. Codes of ``InComingSafetyStatus``, API 1.4.1 (fact 0.14)."""

    DEINIT = 0
    RECOVERY = 1
    NORMAL = 2
    REDUCED = 3
    SAFEGUARD_STOP = 4
    EMERGENCY_STOP = 5
    FAULT = 6
    VIOLATION = 7


class ControllerState(IntEnum):
    """RTD ``state``. Codes of ``InComingControllerState``, API 1.4.1 (fact 0.15)."""

    IDLE = 0
    OFF = 1
    STBY = 2
    ON = 3
    RUN = 4
    CALIBRATION = 5
    FAILURE = 6
    FORCE_EXIT = 7


class MotionMode(IntEnum):
    """RTD ``motion_mode``. Codes of ``InComingMotionMode``, API 1.4.1.

    Incoming codes differ from the ones a command sends (``OutComingMotionMode``);
    only the incoming ones are read here.
    """

    HOLD = 0
    ZERO_GRAVITY = 1
    JOG = 2
    JOINT_JOG = 3
    MOVE = 4
    PAUSE = 5
    MOVE_ADV = 6


class MotionWarning(IntEnum):
    """RTD ``state_flags`` in ``pause``. Codes of ``MotionWarning``, API 1.4.1 (fact 0.16)."""

    NO_WARNING = 0
    PROTECTIVE_STOP = 1
    SELF_COLLISION = 2


def code_name(kind: type[IntEnum], code: int) -> str:
    """The name of a code, or the number itself when this version does not know it."""
    try:
        return kind(code).name.lower()
    except ValueError:
        return f"unknown({code})"


class ChannelError(RuntimeError):
    """The link cannot be opened or used."""


class LinkConfigError(ChannelError):
    """The adapter's ``params`` do not describe a link."""


class WaypointRefusedError(CommandRefusedError):
    """The controller did not take the waypoint: its counter did not move, nothing is
    in the buffer and nothing moves (fact 0.40). The words are the controller's."""


@dataclass(frozen=True, slots=True)
class LinkConfig:
    """Where the robot is, and which robot it has to be (ADR-0054)."""

    host: str
    model: str
    """``robot_model`` exactly as ``get_robot_info`` returns it. No default and no
    guess: the first real value comes from probe 0.22."""

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> LinkConfig:
        """Read the link from an adapter's ``params``, naming any key that is wrong.

        Neither key has a default. The address is a property of the stand, and a
        model the profile did not state is a body nobody checked.
        """
        host = params.get("host")
        if not isinstance(host, str) or not host.strip():
            raise LinkConfigError(
                f"params.host: нужен IPv4-адрес контроллера без порта, получено {host!r}; "
                f"по руководству ввода — 192.168.10.10 (факт 0.19), на стенде — свой"
            )
        model = params.get("model")
        if not isinstance(model, str) or not model.strip():
            raise LinkConfigError(
                f"params.model: нужна модель робота ровно так, как её возвращает "
                f"get_robot_info().robot_model, получено {model!r}. Значение даёт проба "
                f"0.22: python scripts/phase5-probe.py --host <адрес> --api-dir <архив> --full "
                f"(ADR-0054)"
            )
        return cls(host=host.strip(), model=model)


@dataclass(frozen=True, slots=True)
class Identity:
    """What the controller says it is, read once when the link opens."""

    model: str
    client_version: str = ""


@dataclass(frozen=True, slots=True)
class RtdReading:
    """The part of one RTD packet the adapter reads, copied in one pass (ADR-0053, p. 3)."""

    sec: int
    nsec: int
    """The controller's timestamp (fact 0.30): world time of the reading (ADR-0050)."""

    q: tuple[float, ...]
    """``act_q``, radians. Its width is the SDK's joint count (fact 0.43)."""

    qd: tuple[float, ...]
    torque: tuple[float, ...]
    flange: tuple[float, ...]
    """``act_flange_x``: x, y, z in metres, then three angles in radians -- the pose
    of the flange the controller computed, which the description is checked against
    (ADR-0055)."""

    controller_state: int
    safety: int
    motion_mode: int
    buffer_fill: int
    waypoint_counter: int
    velocity_scale: float
    """``act_vel_scale``: the scale the controller applies (ADR-0052, p. 4)."""

    velocity_limit_reason: int
    state_flags: int
    brakes_released: tuple[int, ...]
    """``joint_state``: 1 per joint whose brake is released."""

    cycle_time: float
    """``cycle_time`` as sent. Its unit is named neither in the SDK nor in the manual;
    probe 0.31 and 0.47 establish it, and until then it is only reported."""

    @property
    def stamp(self) -> float:
        return self.sec + self.nsec * 1e-9


@dataclass(frozen=True, slots=True)
class Waypoint:
    """One joint-space point: where, and within what speed and acceleration.

    The SDK takes one speed and one acceleration for all joints of a point, and a
    blend radius, which is always zero here: one motion at a time (ADR-0049)."""

    target: tuple[float, ...]
    """Radians, in the controller's joint order."""

    speed: float
    """rad/s."""

    acceleration: float
    """rad/s²."""


@runtime_checkable
class Channel(Protocol):
    """What the adapter may do with a link."""

    def identity(self) -> Identity: ...

    def poll(self, timeout: float) -> RtdReading | None:
        """The newest reading not returned before, waiting **up to** ``timeout`` for one.

        ``None``: nothing new -- the controller is silent, or the link is gone and
        the reason is in ``lost``. Blocking is part of the contract (ADR-0016).
        """

    def lost(self) -> str | None:
        """Why the link is gone, if the SDK closed it; ``None`` while it is up."""

    def set_scale(self, velocity: float, acceleration: float) -> None: ...

    def scale(self) -> tuple[float, float]:
        """The velocity and acceleration scales the controller reports it applies."""

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Put one point in the buffer, or raise ``WaypointRefusedError``: not taken."""

    def start(self) -> None:
        """Execute the buffer (``move``)."""

    def hold(self) -> None:
        """Hold the pose and drop the buffer (``hold``, fact 0.11)."""

    def close(self) -> None: ...


def standing(joints: int = 6) -> RtdReading:
    """A controller in ``run``, ``normal``, holding still with an empty buffer."""
    zeros = (0.0,) * joints
    return RtdReading(
        sec=1_000,
        nsec=0,
        q=zeros,
        qd=zeros,
        torque=zeros,
        flange=(0.0,) * 6,
        controller_state=ControllerState.RUN,
        safety=SafetyStatus.NORMAL,
        motion_mode=MotionMode.HOLD,
        buffer_fill=0,
        waypoint_counter=0,
        velocity_scale=1.0,
        velocity_limit_reason=0,
        state_flags=0,
        brakes_released=(1,) * joints,
        cycle_time=0.0,
    )


@dataclass(slots=True)
class RecordingChannel:
    """Writes commands down and answers from readings it was given.

    Not a simulator: it moves nothing. It keeps the controller's bookkeeping --
    a waypoint taken raises the counter and fills the buffer, ``move`` and ``hold``
    change the mode -- in ``robot``, the template the next ``deliver`` starts from.
    What a poll returns is only what was delivered, one reading per poll, in order.
    """

    robot: RtdReading = field(default_factory=standing)
    who: Identity = field(default_factory=lambda: Identity(model="rc_recording"))
    period_ns: int = 2_000_000
    """How far apart delivered readings are stamped."""

    readings: deque[RtdReading] = field(default_factory=deque)
    waypoints: list[Waypoint] = field(default_factory=list)
    started: int = 0
    holds: int = 0
    scales_set: list[tuple[float, float]] = field(default_factory=list)
    applied: tuple[float, float] | None = None
    """What ``scale`` answers; ``None`` echoes the last one set, as a controller that
    applies exactly what it is told."""

    refusal: str | None = None
    """Set to have the controller refuse the next waypoint with these words."""

    gone: str | None = None
    """Set to have the SDK close the link: nothing more arrives, nothing is sent."""

    failing_start: bool = False
    closed: bool = False

    # -- Channel ---------------------------------------------------------------

    def identity(self) -> Identity:
        return self.who

    def poll(self, timeout: float) -> RtdReading | None:
        if self.gone is not None:
            return None
        return self.readings.popleft() if self.readings else None

    def lost(self) -> str | None:
        return self.gone

    def set_scale(self, velocity: float, acceleration: float) -> None:
        self._require_link()
        self.scales_set.append((velocity, acceleration))

    def scale(self) -> tuple[float, float]:
        self._require_link()
        if self.applied is not None:
            return self.applied
        if not self.scales_set:
            return (1.0, 1.0)
        return self.scales_set[-1]

    def add_waypoint(self, waypoint: Waypoint) -> None:
        self._require_link()
        if self.refusal is not None:
            words, self.refusal = self.refusal, None
            raise WaypointRefusedError(f"контроллер не принял точку: {words}")
        self.waypoints.append(waypoint)
        self.robot = replace(
            self.robot,
            waypoint_counter=(self.robot.waypoint_counter + 1) & WAYPOINT_COUNTER_MASK,
            buffer_fill=self.robot.buffer_fill + 1,
        )

    def start(self) -> None:
        self._require_link()
        if self.failing_start:
            raise ChannelError("режим move не установлен")
        self.started += 1
        self.robot = replace(self.robot, motion_mode=MotionMode.MOVE)

    def hold(self) -> None:
        self._require_link()
        self.holds += 1
        self.robot = replace(self.robot, motion_mode=MotionMode.HOLD, buffer_fill=0)

    def close(self) -> None:
        self.closed = True

    # -- feeding it --------------------------------------------------------------

    def deliver(self, **changes: Any) -> RtdReading:
        """Queue the robot as it is now, one period later, with ``changes`` applied.

        A stamp given in ``changes`` (``sec``, ``nsec``) replaces the period's step."""
        stamp = self.robot.sec * 1_000_000_000 + self.robot.nsec + self.period_ns
        changes = {"sec": stamp // 1_000_000_000, "nsec": stamp % 1_000_000_000, **changes}
        self.robot = replace(self.robot, **changes)
        self.readings.append(self.robot)
        return self.robot

    def finish_motion(self, **changes: Any) -> RtdReading:
        """Forget what was queued and report the motion over: buffer empty, ``hold``."""
        self.readings.clear()
        return self.deliver(**{"motion_mode": MotionMode.HOLD, "buffer_fill": 0, **changes})

    def _require_link(self) -> None:
        if self.closed:
            raise ChannelError("канал закрыт")
        if self.gone is not None:
            raise ChannelError(f"связь потеряна: {self.gone}")
