"""The DDS adapter: one link to a G1, whoever answers on the other end.

The same class talks to the vendor's MuJoCo twin, to the bridge in Gazebo and to
the robot. What differs between them is the link in the adapter's ``params`` --
domain, interface, peers -- and nothing else (ADR-0020).

A G1 is not commanded a position. It is sent the parameters of a drive law,
``tau = tau_ff + kp * (q* - q) + kd * (dq* - dq)``, which the motor (or the
twin's bridge) evaluates against the last command it received. So a command here
carries the contract's gains in every slot (ADR-0019), and a stop is a law too
(ADR-0028).

Everything that goes on the wire is decided in this module and handed to a
channel as values (ADR-0029): which slot each joint occupies and whether the
robot on the link is the body the contract describes (ADR-0027), which gains,
what a stop sends, when a command is refused. All of it runs on a recording
channel, without CycloneDDS or the vendor's SDK.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Final

from heron.core.capabilities import Capability, parse_set
from heron.core.enums import ActionType, ControlMode
from heron.core.environment import Requirement, RequirementKind, forbids, needs
from heron.core.models import Limits, RobotProfile
from heron.core.protocols import CommandRefusedError
from heron.core.types import Command, RobotState

from .channel import (
    STATE_TOPIC,
    Channel,
    LinkConfig,
    LowCommand,
    LowStateReading,
    MotorCommand,
)
from .slots import SLOT_COUNT, BodyMismatchError, check_body, is_upper_body, slots_of

REQUIREMENTS: Final[tuple[Requirement, ...]] = (
    needs(
        RequirementKind.PYTHON,
        "cyclonedds",
        version="==0.10.2",
        because="на этой версии работают робот и двойник производителя",
        hint="C-библиотека CycloneDDS 0.10.2 из исходника в свой префикс, затем обвязка против "
        "неё: CYCLONEDDS_HOME=<префикс> pip install cyclonedds==0.10.2 "
        "(колёс для Python 3.11+ нет)",
    ),
    forbids(
        RequirementKind.PYTHON,
        "cyclonedds",
        version=">=0.11",
        because="участник CycloneDDS 0.11+ при обнаружении роняет процессы робота и двойника "
        "на 0.10.2",
        hint="pip uninstall cyclonedds, затем обвязка 0.10.2 из исходника",
    ),
    needs(
        RequirementKind.PYTHON,
        "unitree_sdk2py",
        because="IDL-типы unitree_hg и CRC команды",
        hint="git clone https://github.com/unitreerobotics/unitree_sdk2_python; "
        "pip install --no-deps -e <клон>",
    ),
)
"""What the process driving a G1 has to find, and must not (ADR-0025).

The SDK imports all of its subpackages when any is imported, but the only
third-party module they need at that point is ``cyclonedds``."""

DEFAULT_POLL_S: Final = 0.1
"""How long one read waits for a state before returning the previous one."""

DEFAULT_LINK_S: Final = 10.0
"""How long ``connect`` waits for the first state (ADR-0016): discovery takes as
long as it takes, and the first ``LowState`` is also where the body on the link
is learned."""

TICK_S: Final = 1e-3
"""``LowState.tick`` is milliseconds of world time (ADR-0023). A ``uint32``: it
wraps after 49.7 days of one world, which the executor, comparing stamps only for
change, does not mistake for silence."""

MODE_PR_SERIES: Final = 0
"""``mode_pr`` 0: ankle joints commanded as pitch and roll, the joints the
description and the contract name. 1 would reinterpret the same two slots as the
parallel actuators A and B."""

MOTOR_ENABLED: Final = 1


class NotConnectedError(RuntimeError):
    """The adapter was used before ``connect`` or after ``disconnect``."""


class SafetyViolationError(CommandRefusedError):
    """A command the adapter refuses to send."""


class UnsupportedActionError(RuntimeError):
    """The contract's action has no counterpart in ``LowCmd``."""


class StateMismatchError(RuntimeError):
    """What arrived on the link does not match what was connected to."""


class LinkError(RuntimeError):
    """The link never delivered a state."""


def open_cyclone(link: LinkConfig) -> Channel:
    """The real channel, imported here so importing the adapter imports no DDS."""
    from .cyclone import CycloneChannel

    return CycloneChannel(link)


class G1DdsAdapter:
    """Drives a G1 through ``rt/lowcmd`` and reads it from ``rt/lowstate``."""

    requirements: Sequence[Requirement] = REQUIREMENTS

    supports_speed_scale: bool = False
    """A stream is planned by the skill or the policy; the link cannot slow it (ADR-0052)."""

    def __init__(self, channel: Channel | None = None) -> None:
        self._channel = channel
        self._owns_channel = channel is None
        self._profile: RobotProfile | None = None
        self._link: LinkConfig | None = None
        self._slots: tuple[int, ...] = ()
        self._kp: tuple[float, ...] = ()
        self._kd: tuple[float, ...] = ()
        self._machine = 0
        self._capabilities: frozenset[Capability] = frozenset()
        self._control_mode = ControlMode.STREAMING
        self._limits = Limits()
        self._enforce_limits = True
        self._poll_s = DEFAULT_POLL_S
        self._link_s = DEFAULT_LINK_S
        self._last: RobotState | None = None
        self._stopped = False

    # -- ControlAdapter -------------------------------------------------------

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        """Check everything the profile and ``params`` decide, then wait for the robot.

        Returns once a state has arrived and the body behind it is the body the
        contract describes: without its ``mode_machine`` no command can be
        built, and without the check a command could reach the wrong motors.
        """
        action = profile.contract.action
        if action.type is not ActionType.JOINT_POSITION or action.kp is None or action.kd is None:
            raise UnsupportedActionError(
                f"{profile.name}: G1 командуется законом привода, а действие контракта — "
                f"{action.type} {'без kp и kd' if action.kp is None else ''}. Нужны "
                f"joint_position с kp и kd по суставам (ADR-0019): без коэффициентов команда "
                f"LowCmd — это мотор без момента"
            )
        slots = slots_of(profile.contract.joint_order)
        upper = is_upper_body(profile.contract.joint_order)
        if upper:
            _check_upper_body_profile(profile)
        link = LinkConfig.from_params(config)
        _check_limit_widths(profile)

        self._profile = profile
        self._link = link
        self._slots = slots
        self._kp = tuple(float(v) for v in action.kp)
        self._kd = tuple(float(v) for v in action.kd)
        self._capabilities = (
            parse_set(config["capabilities"])
            if "capabilities" in config
            else frozenset(profile.capabilities)
        )
        self._control_mode = ControlMode(config.get("control_mode", profile.control_mode))
        self._limits = profile.limits
        self._enforce_limits = bool(config.get("enforce_limits", True))
        self._poll_s = float(config.get("poll_timeout", DEFAULT_POLL_S))
        self._link_s = float(config.get("link_timeout", DEFAULT_LINK_S))
        self._last = None
        self._stopped = False

        if self._channel is None:
            self._channel = open_cyclone(link)
            self._owns_channel = True
        try:
            first = self._await_link()
            check_body(first.mode_machine, profile.contract.joint_order, upper_body=upper)
        except BaseException:
            self.disconnect()
            raise
        self._machine = first.mode_machine
        self._last = self._state(first)

    def disconnect(self) -> None:
        if self._channel is not None and self._owns_channel:
            self._channel.close()
            self._channel = None
        self._profile = None
        self._last = None

    def read_state(self) -> RobotState:
        """The newest state in the contract's order, or the previous one if none arrived.

        A previous reading returned unchanged is how staleness shows: its stamp
        stops moving, and the executor's watchdog watches exactly that.
        """
        profile = self._require_connected()
        reading = self._link_channel().poll(self._poll_s)
        if reading is None:
            return self._last if self._last is not None else self._state_missing()
        if reading.mode_machine != self._machine:
            raise StateMismatchError(
                f"{profile.name}: mode_machine на связи сменился с {self._machine} на "
                f"{reading.mode_machine} посреди прогона — отвечает другое тело"
            )
        state = self._state(reading)
        self._last = state
        return state

    def send_command(self, cmd: Command) -> None:
        profile = self._require_connected()
        if self._stopped:
            raise SafetyViolationError(
                "аварийный останов: адаптер не принимает команды до reset()"
            )
        if cmd.type is not ActionType.JOINT_POSITION:
            raise UnsupportedActionError(
                f"контракт объявляет действие joint_position, команда несёт {cmd.type}"
            )
        dof = profile.contract.dof
        if len(cmd.values) != dof:
            raise ValueError(f"команда на {len(cmd.values)} суставов при {dof} в контракте")
        bounds = self._limits.joint_position
        if self._enforce_limits and bounds is not None:
            outside = _outside_bounds(profile, cmd.values, bounds.min, bounds.max)
            if outside:
                raise SafetyViolationError(
                    f"позиции вне лимитов: {outside}; адаптер не подрезает команду, а отказывает"
                )
        self._link_channel().publish(self._command(cmd.values, self._kp, self._kd))

    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def control_mode(self) -> ControlMode:
        return self._control_mode

    def safety_limits(self) -> Limits:
        return self._limits

    def emergency_stop(self) -> None:
        """Damp every joint of the contract, then refuse further commands (ADR-0028).

        ``kp = 0``: nothing is pulled anywhere, so the stop does not depend on
        how fresh the last measured pose is -- the watchdog calls this exactly
        when it is not. ``kd`` is the contract's. The adapter counts as stopped
        before the command is sent, so a stop that fails to send still refuses
        what comes after it.
        """
        profile = self._require_connected()
        self._stopped = True
        dof = profile.contract.dof
        held = self._last.joint_position if self._last is not None else (0.0,) * dof
        self._link_channel().publish(self._command(held, (0.0,) * dof, self._kd))

    # -- adapter-specific -----------------------------------------------------

    def reset(self) -> None:
        """Clear an emergency stop. Real hardware needs a human here."""
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def channel(self) -> Channel | None:
        """The link this adapter uses: the injected one, or the one ``connect`` opened."""
        return self._channel

    @property
    def link(self) -> LinkConfig | None:
        return self._link

    @property
    def mode_machine(self) -> int:
        """The body on the link, as it reported itself on connect."""
        return self._machine

    # -- internals ------------------------------------------------------------

    def _require_connected(self) -> RobotProfile:
        if self._profile is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._profile

    def _link_channel(self) -> Channel:
        if self._channel is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._channel

    def _await_link(self) -> LowStateReading:
        channel = self._link_channel()
        deadline = time.monotonic() + self._link_s
        while True:
            reading = channel.poll(self._poll_s)
            if reading is not None:
                return reading
            if time.monotonic() >= deadline:
                raise self._silence(channel)

    def _silence(self, channel: Channel) -> LinkError:
        """Say which half of the link is missing (ADR-0016)."""
        link = self._link
        where = (
            f"домен {link.domain}, интерфейс {link.interface}"
            + (f", пиры {', '.join(link.peers)}" if link.peers else ", обнаружение мультикастом")
            if link is not None
            else "связь не настроена"
        )
        publishers = channel.publisher_count()
        if publishers == 0:
            detail = (
                "никто не публикует. Проверьте, что двойник или робот запущен в том же домене "
                "(params.domain) и виден через params.interface"
            )
        else:
            detail = (
                f"публикующих: {publishers}, но сообщения не доходят. Обычно это QoS или "
                f"CycloneDDS другой ветки у одной из сторон"
            )
        return LinkError(
            f"{STATE_TOPIC}: состояние не пришло за {self._link_s:g} с ({where}) — {detail}"
        )

    def _state_missing(self) -> RobotState:  # pragma: no cover - connect guarantees a state
        raise StateMismatchError("нет ни одного показания, хотя связь установлена")

    def _state(self, reading: LowStateReading) -> RobotState:
        return RobotState(
            timestamp=reading.tick * TICK_S,
            joint_position=tuple(reading.q[slot] for slot in self._slots),
            joint_velocity=tuple(reading.dq[slot] for slot in self._slots),
            joint_torque=tuple(reading.tau_est[slot] for slot in self._slots),
        )

    def _command(
        self, values: Sequence[float], kp: Sequence[float], kd: Sequence[float]
    ) -> LowCommand:
        """Every slot stated: the contract's joints enabled, the rest off (ADR-0027)."""
        motors = [MotorCommand()] * SLOT_COUNT
        for index, slot in enumerate(self._slots):
            motors[slot] = MotorCommand(
                mode=MOTOR_ENABLED, q=float(values[index]), kp=kp[index], kd=kd[index]
            )
        return LowCommand(
            mode_pr=MODE_PR_SERIES, mode_machine=self._machine, motors=tuple(motors)
        )


def _check_upper_body_profile(profile: RobotProfile) -> None:
    """An upper body over ``rt/lowcmd`` only where no robot can be on the link (ADR-0064).

    The legs of such a contract get ``mode = 0``. A mount holds them in the world this
    profile describes (ADR-0063); on a robot standing on the floor at the low level
    nobody would, and it would fall. A profile that declares ``real`` says it may meet
    such a robot, and is refused before a channel is opened: on hardware the upper
    body goes through ``rt/arm_sdk`` while the vendor's controller keeps the legs.
    """
    if profile.has_real_adapter:
        raise BodyMismatchError(
            f"{profile.name}: контракт — верх тела G1 без ног, а профиль объявляет адаптер "
            f"'real'. По rt/lowcmd ноги получили бы mode = 0, и стоящий робот упал бы; верх "
            f"тела на железе идёт по rt/arm_sdk при работающем балансе (фаза 7). Верх тела "
            f"по rt/lowcmd — только у закреплённого робота в симуляции (ADR-0064)"
        )


def _check_limit_widths(profile: RobotProfile) -> None:
    """Refuse limits that do not line up with the contract, rather than index past them."""
    dof = profile.contract.dof
    bounds = profile.limits.joint_position
    if bounds is not None and (len(bounds.min) != dof or len(bounds.max) != dof):
        raise ValueError(
            f"лимиты позиций на {len(bounds.min)}/{len(bounds.max)} суставов при {dof} в "
            f"контракте; профиль не прошёл бы валидатор (limits.length_mismatch)"
        )


def _outside_bounds(
    profile: RobotProfile,
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> str:
    """Joints outside the envelope, named with value and bounds (ADR-0026)."""
    return "; ".join(
        f"{profile.contract.joint_order[i]} = {value:.4g} при [{lower[i]:.4g}, {upper[i]:.4g}]"
        for i, value in enumerate(values)
        if not (lower[i] <= value <= upper[i])
    )
