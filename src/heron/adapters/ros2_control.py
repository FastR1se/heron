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

*Each class is what it says it is* (ADR-0060). In the streaming class a command
is published on the controller's topic every tick. In the command class a goal is
one motion, planned from the measured pose within the profile's envelope and the
run's speed scale -- the planner ``dummy`` moves along -- and handed to the
trajectory controller as a goal of its action. The phase is latched ``moving``
from the accepted command until the controller says the goal is over, a second
command meanwhile is refused, and ``disconnect`` cancels what is left (ADR-0049).
A link that reported the profile's class without doing any of this passed every
check before the first tick and failed on it (ADR-0058).

*Radians only.* ROS 2 measures angles in radians (REP 103), and this link does
not convert: which joints are angles is known to the description, which the link
does not read. A contract in degrees is refused on ``connect`` rather than sent
57 times off.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final

from ..core.capabilities import Capability, parse_set
from ..core.enums import ActionType, AngleUnit, ControlMode, MotionPhase
from ..core.environment import Requirement, RequirementKind, needs
from ..core.models import SPEED_SCALE_PARAM, Limits, RobotProfile
from ..core.protocols import CommandRefusedError
from ..core.types import Command, RobotState
from ..ros2 import names
from ..ros2.transport import (
    Goal,
    GoalStatus,
    Publication,
    RclpyTransport,
    Reading,
    Transport,
)
from .common import outside_bounds
from .motion import Motion, plan

REQUIREMENTS: Final[tuple[Requirement, ...]] = (
    *(
        needs(
            RequirementKind.PYTHON,
            module,
            hint=f"пакет ros-<дистрибутив>-{package} (ставит scripts/phase0.sh) и подключённое "
            f"окружение: source /opt/ros/<дистрибутив>/setup.bash",
        )
        for module, package in names.ADAPTER_PACKAGES.items()
    ),
    *(
        needs(
            RequirementKind.PYTHON,
            module,
            because="модули сообщений ROS 2 импортируют его при загрузке типа",
            hint="группа ros2 платформы: uv sync --extra ros2",
        )
        for module in names.ADAPTER_PYTHON_PACKAGES
    ),
)
"""What the process driving a robot through ROS 2 has to find (ADR-0025).

Derived from ``ADAPTER_PACKAGES``, the list the transport's imports are held to, and
``ADAPTER_PYTHON_PACKAGES``, what those imports import in turn, rather than written a
second time: the requirement and the import cannot drift."""

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

DEFAULT_LINK_S: Final = 10.0
"""How long the *first* read waits for the link to come up (ADR-0016).

Establishing a link is not the same as using one. A subscription created a
moment ago has not been matched with its publisher yet: discovery takes as long
as it takes, routinely longer than one command period and occasionally seconds.
Waiting one period for the first state and declaring the robot silent is how a
working scene reads as a broken one.

This is a deadline on a condition, not a pause: the wait ends the moment a state
arrives. A generated sleep would be a guess that breaks on another machine --
the same reason the generated launch waits on its predecessors rather than on a
timer."""


class NotConnectedError(RuntimeError):
    """The adapter was used before ``connect`` or after ``disconnect``."""


class SafetyViolationError(CommandRefusedError):
    """A command would leave the envelope the profile declares."""


class MotionInProgressError(CommandRefusedError):
    """A command arrived while the accepted motion is unfinished (ADR-0049)."""


class GoalRefusedError(CommandRefusedError):
    """The controller did not take the goal, or is not there to take it: nothing moves."""


class UnsupportedActionError(RuntimeError):
    """The contract's action has no counterpart among ros2_control interfaces."""


class StateMismatchError(RuntimeError):
    """What arrived on the state topic does not match the contract."""


class LinkConfigurationError(RuntimeError):
    """The profile or the run asks for something this link cannot do. Raised by ``connect``."""


class Ros2ControlAdapter:
    """Drives a robot through its controllers and reads its broadcaster."""

    requirements: Sequence[Requirement] = REQUIREMENTS

    supports_speed_scale: bool = True
    """In the command class the link plans the motion, so it can slow it; a stream is
    planned by the skill or the policy, and a scale other than 1 is refused on
    ``connect`` (ADR-0052, ADR-0060)."""

    def __init__(self, transport: Transport | None = None) -> None:
        self._transport = transport
        self._owns_transport = transport is None
        self._profile: RobotProfile | None = None
        self._namespace = ""
        self._capabilities: frozenset[Capability] = frozenset()
        self._control_mode = ControlMode.STREAMING
        self._limits: Limits = Limits()
        self._enforce_limits = True
        self._poll_s = DEFAULT_POLL_S
        self._link_s = DEFAULT_LINK_S
        self._speed_scale = 1.0
        self._state_topic = ""
        self._command_topic = ""
        self._command_message = ""
        self._action = ""
        self._action_type = ""
        self._server_seen = False
        self._motion: Motion | None = None
        self._ended = ""
        self._last: RobotState | None = None
        self._stopped = False

    # -- ControlAdapter -----------------------------------------------------

    def connect(self, profile: RobotProfile, config: dict[str, Any]) -> None:
        action = profile.contract.action.type
        if names.command_interface(profile) is None:
            raise UnsupportedActionError(
                f"{profile.name}: действие {action} не отображается на интерфейс "
                f"сустава, и подменять его позициями платформа не станет. "
                f"Отображаются: {', '.join(sorted(names.COMMAND_INTERFACES))}. "
                f"Декартова поза и ширина захвата требуют своего контроллера (ADR-0009)"
            )
        unit = profile.contract.angle_unit
        if unit is not AngleUnit.RAD:
            raise LinkConfigurationError(
                f"{profile.name}: контракт в {unit}, а ROS 2 измеряет углы в радианах "
                f"(REP 103). Связь ros2_control единиц не переводит: какие суставы угловые, "
                f"знает описание, а связь его не читает (ADR-0060). Контракт робота, которого "
                f"ведёт ros2_control, — angle_unit: rad"
            )
        mode = profile.control_mode
        if mode is ControlMode.COMMAND and action is not ActionType.JOINT_POSITION:
            raise LinkConfigurationError(
                f"{profile.name}: в командном классе цель — поза суставов, а действие "
                f"контракта {action} (ADR-0049)"
            )

        limits = profile.limits
        self._check_limit_widths(limits, profile.contract.dof)
        if mode is ControlMode.COMMAND:
            _check_motion_limits(limits)
        scale = _scale_of(config, mode)

        self._profile = profile
        self._namespace = str(config.get(NAMESPACE_PARAM, ""))
        self._capabilities = (
            parse_set(config["capabilities"])
            if "capabilities" in config
            else frozenset(profile.capabilities)
        )
        self._control_mode = mode
        self._limits = limits
        self._enforce_limits = bool(config.get("enforce_limits", True))
        self._poll_s = float(config.get("poll_timeout", DEFAULT_POLL_S))
        self._link_s = float(config.get("link_timeout", DEFAULT_LINK_S))
        self._speed_scale = scale

        plane = names.control_plane(self._namespace, profile)
        self._state_topic = str(plane["state_topic"])
        self._command_topic = str(plane["command_topic"] or "")
        self._command_message = str(plane["command_message"] or "")
        self._action = str(plane["command_action"] or "")
        self._action_type = str(plane["command_action_type"] or "")
        self._server_seen = False
        self._motion = None
        self._ended = ""
        self._last = None
        self._stopped = False

        if self._transport is None:
            self._transport = RclpyTransport(node_name=f"{profile.name}_adapter")
            self._owns_transport = True
        self._transport.subscribe(self._state_topic, names.STATE_MESSAGE)
        if mode is ControlMode.COMMAND:
            self._transport.connect_action(self._action, self._action_type)
        else:
            self._transport.advertise(self._command_topic, self._command_message)

    def disconnect(self) -> None:
        """Revoke an unfinished motion while the link is still there (ADR-0049)."""
        if self._motion is not None and self._transport is not None:
            self._transport.cancel_goal(self._action)
        self._motion = None
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

        The first read is the exception: it waits for the link to come up
        (ADR-0016), because until then there is no previous reading to be stale.

        In the command class the reading carries the phase of the motion. It is
        ``moving`` from the accepted command until the controller has said the goal
        is over *and* a state has arrived after that: the pose that goes with
        ``idle`` is one read after the end, never one from before it (ADR-0050).
        """
        profile = self._require_connected()
        over = self._goal_over()
        reading = (
            self._await_link() if self._last is None
            else self._link().poll(self._state_topic, self._poll_s)
        )
        if reading is None:
            if self._last is None:
                return self._no_state()
            return replace(
                self._last, **self._phase(fresh=False, over=over, extra=self._last.extra)
            )

        state = _reindex(reading.fields, reading.stamp, profile, self._state_topic)
        state = replace(state, **self._phase(fresh=True, over=over, extra=state.extra))
        self._last = state
        return state

    def _await_link(self) -> Reading | None:
        """Poll until the first state arrives or the link deadline passes."""
        link = self._link()
        deadline = time.monotonic() + self._link_s
        while True:
            reading = link.poll(self._state_topic, self._poll_s)
            if reading is not None or time.monotonic() >= deadline:
                return reading

    def _no_state(self) -> RobotState:
        """Report a link that never came up, saying which half is missing.

        "Nobody publishes" and "somebody publishes and nothing arrives" are two
        different failures with two different fixes, and the link can tell them
        apart. Reporting only silence would leave that to guesswork, which is
        what this platform exists not to do.
        """
        publishers = self._link().publisher_count(self._state_topic)
        if publishers == 0:
            detail = (
                "на этот топик никто не публикует. Проверьте, что сцена поднята и "
                f"контроллеры запущены:\n"
                f"    ros2 control list_controllers -c /{self._namespace}/controller_manager\n"
                f"    ros2 topic list | grep {self._namespace}\n"
                f"Если broadcaster активен, а топик называется иначе — разошлись "
                f"пространства имён: сверьте с блоком control в manifest.json"
            )
        else:
            detail = (
                f"публикующих узлов: {publishers}, но сообщения не доходят. "
                f"Обычно это несовпадение QoS или разные домены ROS_DOMAIN_ID.\n"
                f"    ros2 topic hz {self._state_topic}\n"
                f"    ros2 topic info {self._state_topic} --verbose"
            )
        raise StateMismatchError(
            f"{self._state_topic}: состояние не пришло за {self._link_s:g} с — {detail}"
        )

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
        if self._control_mode is ControlMode.COMMAND:
            self._send_goal(profile, cmd.values)
        else:
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

        In the command class the goal is cancelled, and the trajectory controller
        holds where the joints are (``goal_cancelled_callback``: ``set_hold_position``).
        Without a goal it already holds.
        """
        profile = self._require_connected()
        if self._control_mode is ControlMode.COMMAND:
            if self._motion is not None:
                self._link().cancel_goal(self._action)
            self._stopped = True
            return
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
    def speed_scale(self) -> float:
        return self._speed_scale

    @property
    def transport(self) -> Transport | None:
        """The link this adapter publishes through, or ``None`` before connect."""
        return self._transport

    @property
    def topics(self) -> dict[str, str]:
        """Where this link reads and writes. Printable before a run starts."""
        if self._control_mode is ControlMode.COMMAND:
            return {"state": self._state_topic, "action": self._action}
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

    @staticmethod
    def _check_limit_widths(limits: Limits, dof: int) -> None:
        """Refuse limits that do not line up with the contract.

        The validator catches this too, but an adapter that trusts its input and
        then indexes past the end would turn a caught error into a crash.
        """
        widths = {
            "joint_position.min": getattr(limits.joint_position, "min", None),
            "joint_position.max": getattr(limits.joint_position, "max", None),
            "joint_velocity.max": getattr(limits.joint_velocity, "max", None),
            "joint_acceleration.max": getattr(limits.joint_acceleration, "max", None),
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
            outside = outside_bounds(
                self._require_connected(), cmd.values, bounds.min, bounds.max
            )
            if outside:
                raise SafetyViolationError(
                    f"позиции вне лимитов: {outside}; "
                    f"адаптер не подрезает команду, а отказывает"
                )
        elif cmd.type is ActionType.JOINT_VELOCITY:
            limit = self._limits.joint_velocity
            if limit is None:
                return
            too_fast = [i for i, v in enumerate(cmd.values) if abs(v) > limit.max[i]]
            if too_fast:
                raise SafetyViolationError(f"скорости вне лимитов в суставах {too_fast}")

    def _publication(self, profile: RobotProfile, values: tuple[float, ...]) -> Publication:
        """One command of a stream, as the message its controller subscribes to."""
        if self._command_message == names.TRAJECTORY_MESSAGE:
            # In the streaming class the trajectory controller carries only the drive's
            # law of an action closed through gains (ADR-0019), tau = kp * (q* - q) -
            # kd * dq, from the tick the point arrives: desired state (q*, 0) at once. A
            # later time would interpolate a ramp, or hold the state at receipt for a
            # whole period with no torque toward the target (ADR-0045).
            point: dict[str, Any] = {
                "positions": [float(v) for v in values],
                "velocities": [0.0] * len(values),
                "time_from_start": 0.0,
            }
            fields: dict[str, Any] = {
                "joint_names": list(profile.contract.joint_order),
                "points": [point],
            }
        else:
            fields = {"data": [float(v) for v in values]}
        return Publication(
            topic=self._command_topic, message_type=self._command_message, fields=fields
        )

    # -- the command class (ADR-0049, ADR-0060) ------------------------------

    def _send_goal(self, profile: RobotProfile, values: tuple[float, ...]) -> None:
        """Plan the motion from the measured pose and hand it to the controller.

        The command returns once the controller has taken the goal, or after one
        command period without an answer -- the phase is then latched as ``moving``
        either way, and a refusal that comes later ends the motion like any other
        end. A refusal within the period is a refused command: nothing moves.
        """
        if self._motion is not None:
            raise MotionInProgressError(
                "робот выполняет принятое движение; одно движение за раз (ADR-0049)"
            )
        start = self._last
        if start is None:
            raise GoalRefusedError(
                "состояние робота ещё не прочитано, а движение строится от измеренной позы"
            )
        link = self._link()
        if not self._server_seen:
            # Establishing a link is not using it (ADR-0016): the first goal waits for
            # the action server as the first reading waits for the state.
            if not link.action_ready(self._action, self._link_s):
                raise GoalRefusedError(
                    f"{self._action}: сервера действия нет за {self._link_s:g} с — "
                    f"траекторный контроллер не загружен или в другом пространстве имён:\n"
                    f"    ros2 action list | grep {self._namespace}\n"
                    f"    ros2 control list_controllers -c /{self._namespace}/controller_manager"
                )
            self._server_seen = True

        speeds, accelerations = self._limits.joint_velocity, self._limits.joint_acceleration
        assert speeds is not None and accelerations is not None  # checked on connect
        motion = plan(
            start.timestamp,
            start.joint_position,
            values,
            speeds.max,
            accelerations.max,
            self._speed_scale,
        )
        # Sampled at the rate the controller manager runs at, which is the contract's
        # state rate (generator): every update of the controller lands on a point of
        # the plan, and interpolation between points decides nothing.
        samples = motion.samples(1.0 / profile.contract.state_rate_hz)
        link.send_goal(
            Goal(
                action=self._action,
                action_type=self._action_type,
                fields={
                    "trajectory": {
                        "joint_names": list(profile.contract.joint_order),
                        "points": [
                            {
                                "positions": list(sample.position),
                                "velocities": list(sample.velocity),
                                "time_from_start": sample.time,
                            }
                            for sample in samples
                        ],
                    }
                },
            )
        )
        outcome = link.goal_status(self._action, 1.0 / profile.contract.command_rate_hz)
        if outcome.status is GoalStatus.REJECTED:
            said = f": {outcome.detail}" if outcome.detail else ""
            raise GoalRefusedError(
                f"{self._action}: контроллер не принял цель{said}. Контроллер, который не "
                f"активен, отвергает любую цель:\n"
                f"    ros2 control list_controllers -c /{self._namespace}/controller_manager"
            )
        self._motion = motion
        self._ended = ""

    def _goal_over(self) -> bool:
        """Whether the controller has said the accepted goal is over."""
        if self._motion is None:
            return False
        outcome = self._link().goal_status(self._action)
        if not outcome.over:
            return False
        self._ended = str(outcome.status)
        if outcome.detail:
            self._ended += f": {outcome.detail}"
        return True

    def _phase(
        self, *, fresh: bool, over: bool, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """The motion fields of a reading. A stream reports no phase."""
        if self._control_mode is not ControlMode.COMMAND:
            return {}
        if self._motion is not None and over and fresh:
            self._motion = None
        phase = MotionPhase.IDLE if self._motion is None else MotionPhase.MOVING
        fields: dict[str, Any] = {"motion": phase}
        if phase is MotionPhase.IDLE and self._ended:
            fields["extra"] = {**(extra or {}), MOTION_ENDED: self._ended}
        return fields


MOTION_ENDED: Final = "motion_ended"
"""Key of ``extra`` saying how the controller ended the last goal -- ``succeeded``, or
``aborted`` with its words. The core never reads it; a record keeps it."""


def _scale_of(config: dict[str, Any], mode: ControlMode) -> float:
    scale = float(config.get(SPEED_SCALE_PARAM, 1.0))
    if not 0 < scale <= 1:
        raise LinkConfigurationError(f"{SPEED_SCALE_PARAM} = {scale:g}: нужно 0 < s ≤ 1")
    if scale != 1 and mode is ControlMode.STREAMING:
        raise LinkConfigurationError(
            f"{SPEED_SCALE_PARAM} = {scale:g}: потоковую связь адаптер не замедляет — "
            f"движение планирует навык (ADR-0052)"
        )
    return scale


def _check_motion_limits(limits: Limits) -> None:
    """A link planning the motion needs both bounds of it (ADR-0049)."""
    missing = [
        name for name in ("joint_velocity", "joint_acceleration") if getattr(limits, name) is None
    ]
    if missing:
        raise LinkConfigurationError(
            f"командный класс: движение строится по рамке профиля, а в limits нет "
            f"{', '.join(missing)} (ADR-0049)"
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
