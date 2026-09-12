"""``ros2_control`` -- the link to a robot driven by the ROS 2 control stack.

The counterpart of the generator: that one wrote the configuration which brings
the controllers into existence, this one talks to the controllers it created.
Both read their names from ``ros2.names``, which is the whole point of that
module (ADR-0014).

Three things here are not conveniences and should not be softened later.

*Order is restored, not assumed.* A joint state message carries names next to
values, and nothing requires them to arrive in the order the contract declares.
Reading the arrays positionally would produce a robot whose second joint holds
the third joint's angle -- a robot that moves and is wrong, which is the failure
this platform exists to prevent. So every reading is re-indexed by name, and a
joint the contract declares but the message never carries is an error rather
than a zero.

*Limits are enforced by refusal, not by clipping.* The same position the dummy
adapter takes, and for the same reason: silent repair is worse than a stop.

*Nothing is published that was not asked for.* A command whose type disagrees
with the contract's action is refused rather than converted. The controller on
the other end accepts exactly one kind of quantity, and sending it another kind
in the right shape is how a position lands in a velocity interface.
"""

from __future__ import annotations

from typing import Any, Final

from ..core.capabilities import Capability, parse_set
from ..core.enums import ActionType, ControlMode
from ..core.models import Limits, RobotProfile
from ..core.types import Command, RobotState
from ..ros2 import names
from ..ros2.transport import Publication, RclpyTransport, Transport

NAMESPACE_PARAM: Final = "namespace"
"""Configuration key naming the robot's namespace.

Its value is the scenario identifier, because that is what the generated launch
pushes -- never the profile name, which two robots of the same kind share. An
absent namespace means the root, which is a real configuration and not a
missing value (``names.qualified``)."""

DEFAULT_POLL_S: Final = 0.1
"""How long one read waits for a state message before giving up on this cycle.

Not a timeout on the robot: the executor decides when missing state becomes a
stop (ADR-0015). This is only how long a single read blocks, and it is short so
that a command loop keeps its period even while state is late."""


class NotConnectedError(RuntimeError):
    """The adapter was used before ``connect`` or after ``disconnect``."""


class SafetyViolationError(RuntimeError):
    """A command would leave the envelope the profile declares."""


class UnsupportedActionError(RuntimeError):
    """The contract's action has no counterpart among ros2_control interfaces."""


class StateMismatchError(RuntimeError):
    """What arrived on the state topic does not match the contract."""


class Ros2ControlAdapter:
    """Drives a robot through its controllers and reads its broadcaster."""

    def __init__(self, transport: Transport | None = None) -> None:
        self._transport = transport
        self._owns_transport = transport is None
        self._profile: RobotProfile | None = None
        self._namespace = ""
        self._capabilities: frozenset[Capability] = frozenset()
        self._control_mode: ControlMode = ControlMode.STREAMING
        self._limits: Limits = Limits()
        self._enforce_limits = True
        self._poll_s = DEFAULT_POLL_S
        self._state_topic = ""
        self._command_topic = ""
        self._command_message = ""
        self._last: RobotState | None = None
        self._stopped = False

    # -- ControlAdapter -----------------------------------------------------

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        action = profile.contract.action.type
        if names.command_topic("", profile) is None:
            raise UnsupportedActionError(
                f"{profile.name}: действие {action} не отображается на интерфейс "
                f"сустава, и подменять его позициями платформа не станет. "
                f"Отображаются: {', '.join(sorted(names.COMMAND_INTERFACES))}. "
                f"Декартова поза и ширина захвата требуют своего контроллера (ADR-0009)"
            )

        self._profile = profile
        self._namespace = str(config.get(NAMESPACE_PARAM, ""))
        self._capabilities = (
            parse_set(config["capabilities"])
            if "capabilities" in config
            else frozenset(profile.capabilities)
        )
        self._control_mode = ControlMode(config.get("control_mode", profile.control_mode))
        self._limits = profile.limits
        self._enforce_limits = bool(config.get("enforce_limits", True))
        self._poll_s = float(config.get("poll_timeout", DEFAULT_POLL_S))
        self._check_limit_widths(profile.contract.dof)

        plane = names.control_plane(self._namespace, profile)
        self._state_topic = str(plane["state_topic"])
        self._command_topic = str(plane["command_topic"])
        self._command_message = str(plane["command_message"])
        self._last = None
        self._stopped = False

        if self._transport is None:
            self._transport = RclpyTransport(node_name=f"{profile.name}_adapter")
            self._owns_transport = True
        self._transport.subscribe(self._state_topic, names.STATE_MESSAGE)
        self._transport.advertise(self._command_topic, self._command_message)

    def disconnect(self) -> None:
        if self._transport is not None and self._owns_transport:
            self._transport.close()
            self._transport = None
        self._profile = None
        self._last = None

    def read_state(self) -> RobotState:
        """The latest state, re-indexed into the contract's joint order.

        Returns the previous reading unchanged when nothing new arrived, rather
        than blocking or inventing one. Staleness is then visible as a timestamp
        that stopped moving, which is what the executor's watchdog watches.
        """
        profile = self._require_connected()
        reading = self._link().poll(self._state_topic, self._poll_s)
        if reading is None:
            if self._last is None:
                raise StateMismatchError(
                    f"{self._state_topic}: состояние не пришло за {self._poll_s:g} с. "
                    f"Проверьте, что контроллеры запущены: "
                    f"ros2 control list_controllers -c {self._namespace}/controller_manager"
                )
            return self._last

        state = _reindex(reading.fields, reading.stamp, profile, self._state_topic)
        self._last = state
        return state

    def send_command(self, cmd: Command) -> None:
        profile = self._require_connected()
        if self._stopped:
            raise SafetyViolationError(
                "аварийный останов: адаптер не принимает команды до reset()"
            )

        expected = profile.contract.action.type
        if cmd.type is not expected:
            raise UnsupportedActionError(
                f"контракт объявляет действие {expected}, команда несёт {cmd.type}; "
                f"контроллер принимает ровно одну величину, и подменять её платформа "
                f"не станет"
            )
        self._check_width(cmd.values, profile.contract.dof)
        self._check_limits(cmd)
        self._link().publish(self._publication(profile, cmd.values))

    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def control_mode(self) -> ControlMode:
        return self._control_mode

    def safety_limits(self) -> Limits:
        return self._limits

    def emergency_stop(self) -> None:
        """Command the robot to stay where it is, then refuse further commands.

        What "stay" means depends on the interface, and it is derived rather
        than chosen: a position interface is told the position last measured, a
        velocity or effort interface is told zero. Publishing nothing would
        leave the last command standing, which for a velocity interface means
        the robot keeps moving.
        """
        profile = self._require_connected()
        dof = profile.contract.dof
        if profile.contract.action.type is ActionType.JOINT_POSITION:
            held = self._last.joint_position if self._last is not None else (0.0,) * dof
        else:
            held = (0.0,) * dof
        if len(held) == dof:
            self._link().publish(self._publication(profile, held))
        self._stopped = True

    # -- adapter-specific ---------------------------------------------------

    def reset(self) -> None:
        """Clear an emergency stop. Real hardware needs a human here."""
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def transport(self) -> Transport | None:
        """The link this adapter publishes through, or ``None`` before connect."""
        return self._transport

    @property
    def topics(self) -> dict[str, str]:
        """Where this link reads and writes. Printable before a run starts."""
        return {"state": self._state_topic, "command": self._command_topic}

    # -- internals ----------------------------------------------------------

    def _require_connected(self) -> RobotProfile:
        if self._profile is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._profile

    def _link(self) -> Transport:
        if self._transport is None:
            raise NotConnectedError("адаптер не подключён: вызовите connect()")
        return self._transport

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

    @staticmethod
    def _check_width(values: tuple[float, ...], dof: int) -> None:
        if len(values) != dof:
            raise ValueError(f"команда на {len(values)} суставов при {dof} в контракте")

    def _check_limits(self, cmd: Command) -> None:
        if not self._enforce_limits:
            return
        if cmd.type is ActionType.JOINT_POSITION:
            bounds = self._limits.joint_position
            if bounds is None:
                return
            outside = [
                i
                for i, v in enumerate(cmd.values)
                if not (bounds.min[i] <= v <= bounds.max[i])
            ]
            if outside:
                raise SafetyViolationError(
                    f"позиции вне лимитов в суставах {outside}; "
                    f"адаптер не подрезает команду, а отказывает"
                )
        elif cmd.type is ActionType.JOINT_VELOCITY:
            limit = self._limits.joint_velocity
            if limit is None:
                return
            outside = [i for i, v in enumerate(cmd.values) if abs(v) > limit.max[i]]
            if outside:
                raise SafetyViolationError(f"скорости вне лимитов в суставах {outside}")

    def _publication(self, profile: RobotProfile, values: tuple[float, ...]) -> Publication:
        """One command, as the message its controller subscribes to."""
        if self._command_message == names.TRAJECTORY_MESSAGE:
            # The trajectory controller is told to reach the point in one
            # command period: the rate is the contract's, so the platform is
            # not inventing a duration, only restating one it was given.
            fields: dict[str, Any] = {
                "joint_names": list(profile.contract.joint_order),
                "points": [
                    {
                        "positions": [float(v) for v in values],
                        "time_from_start": 1.0 / profile.contract.command_rate_hz,
                    }
                ],
            }
        else:
            fields = {"data": [float(v) for v in values]}
        return Publication(
            topic=self._command_topic, message_type=self._command_message, fields=fields
        )


def _reindex(
    fields: dict[str, Any], stamp: float, profile: RobotProfile, topic: str
) -> RobotState:
    """Put an arriving joint state into the order the contract declares.

    Everything the message carries beyond the contract goes to ``extra``: those
    are the joints the description declares and the contract does not (ADR-0012)
    -- the platform reads them and cannot command them, and dropping them here
    would be the same defect at the other end of the link.
    """
    arrived: list[str] = list(fields.get("name", []))
    if not arrived:
        raise StateMismatchError(
            f"{topic}: в сообщении нет имён суставов, а без них значения нельзя "
            f"поставить в порядок контракта"
        )
    index = {name: position for position, name in enumerate(arrived)}

    missing = [name for name in profile.contract.joint_order if name not in index]
    if missing:
        raise StateMismatchError(
            f"{topic}: контракт объявляет суставы, которых нет в состоянии: "
            f"{', '.join(missing)}. Пришли: {', '.join(arrived)}. "
            f"Обычно это значит, что контроллеры поднялись для другого робота "
            f"или что joint_order разошёлся с описанием"
        )

    def full(key: str) -> list[float] | None:
        """One array of the message, or ``None`` if it is not fully populated.

        A short array is not a partly known reading: the broadcaster either
        reports an interface for every joint it carries or reports none, so a
        length that disagrees means the message is not what it claims to be.
        """
        values = fields.get(key) or []
        return [float(v) for v in values] if len(values) == len(arrived) else None

    def column(key: str) -> tuple[float, ...]:
        values = full(key)
        if values is None:
            return ()
        return tuple(values[index[name]] for name in profile.contract.joint_order)

    positions = full("position")
    if positions is None:
        raise StateMismatchError(
            f"{topic}: позиций {len(fields.get('position') or [])} при "
            f"{len(arrived)} именах суставов — сообщение не описывает то, что называет"
        )

    declared = set(profile.contract.joint_order)
    observed = {name: positions[index[name]] for name in arrived if name not in declared}
    return RobotState(
        timestamp=stamp,
        joint_position=column("position"),
        joint_velocity=column("velocity"),
        joint_torque=column("effort"),
        extra={"observed": observed} if observed else {},
    )
