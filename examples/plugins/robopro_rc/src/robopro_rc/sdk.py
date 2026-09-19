"""The real channel: the vendor's ``RobotApi`` behind the ``Channel`` protocol.

The only module of the package that touches the SDK, and it imports it when a link
is opened -- never when the package is imported, a plugin resolved or a dry run
planned (ADR-0053, p. 2). Checked against the source of API 1.4.1; every call below
names the behaviour of that version it relies on.

What the SDK does to a process, and what this channel does about it (ADR-0053):

* ``RobotApi.__init__`` sets ``sys.excepthook`` to its own shutdown. The previous
  hook is put back right after construction; the ``atexit`` handler is left, it only
  closes sockets.
* ``StateHandler`` raises from its thread on ``emergency_stop``, ``fault``,
  ``violation`` and ``failure`` and so closes the link. ``ignore_controller_exceptions``
  turns that off: the safety status reaches the adapter in every reading instead.
* The SDK logs to a console handler of its own unless given a logger; it is given
  this module's, so its messages go where the platform's go.
* The confirmation of a position always refuses (ADR-0054, p. 4). In 1.4.1 no
  handler is set by default and the SDK raises instead of asking; ``input()`` is only
  reached if someone calls ``set_confirm_position_callback()`` without an argument.
  The refusing handler is set anyway, so that no path can wait on a terminal.
* ``disconnect`` of the SDK is called on every path out: its two threads are not
  daemons and would keep the process alive.

The real-time packet is read through the SDK's private receiver (ADR-P5B): the
public getters read one field each from separate snapshots, and none of them gives
the timestamp, the buffer or the waypoint counter.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from collections.abc import Sequence
from typing import Any, Final

from .channel import (
    WAYPOINT_COUNTER_MASK,
    ChannelError,
    Identity,
    LinkConfig,
    RtdReading,
    Waypoint,
    WaypointRefusedError,
)

SDK_MODULE: Final = "API.rc_api"
"""Where ``RobotApi`` lives in the vendor's archive (fact 0.3)."""

PACKET_BEGIN: Final = 0x12
PACKET_END: Final = 0x21
"""Marker bytes that open and close an RTD packet (``RTDataPackageBorders``, fact 0.29).
The SDK checks them once, when the receiver connects; a packet read out of step
afterwards is parsed field by field into garbage. So each reading checks them again."""

CONNECT_TIMEOUT_S: Final = 5
"""The SDK's own default for ``RobotApi(timeout=...)``: seconds to open a socket."""

POLL_STEP_S: Final = 0.001
"""How often ``poll`` looks for a new packet while it waits."""

log = logging.getLogger(__name__)


def copy_rtd(data: Any) -> RtdReading:
    """One reading from the SDK's live ``RTD`` object, in one pass (ADR-0053, p. 3).

    The SDK's receiving thread writes the object field by field without a lock
    (fact 0.38). Every field is loaded in a single tuple display: it contains no call
    and no backward jump, so this thread has no point inside it at which it could
    let the receiving thread run. Conversions come after, on the copy. What remains
    is the writer's side: it is only made to release the GIL mid-packet when it has
    held it for a full switch interval, which a loop of sixty assignments does not
    unless the operating system stalls it there (ADR-P5B).
    """
    (
        begin, end, sec, nsec, q, qd, torque, flange, state, safety, mode, fill, counter,
        scale, reason, flags, brakes, cycle,
    ) = (
        data.packet_begin, data.packet_end, data.sec, data.nsec, data.act_q, data.act_qd,
        data.act_trq, data.act_flange_x, data.state, data.safety, data.motion_mode,
        data.buff_fill, data.wp_cntr, data.act_vel_scale, data.act_vel_limit_reason,
        data.state_flags, data.joint_state, data.cycle_time,
    )  # fmt: skip
    if int(begin) != PACKET_BEGIN or int(end) != PACKET_END:
        raise ChannelError(
            f"пакет RTD разобран со сдвигом: маркеры {int(begin):#04x}/{int(end):#04x} вместо "
            f"{PACKET_BEGIN:#04x}/{PACKET_END:#04x}. SDK проверяет границы только при "
            f"подключении; показание отброшено"
        )
    return RtdReading(
        sec=int(sec),
        nsec=int(nsec),
        q=_floats(q),
        qd=_floats(qd),
        torque=_floats(torque),
        flange=_floats(flange),
        controller_state=int(state),
        safety=int(safety),
        motion_mode=int(mode),
        buffer_fill=int(fill),
        waypoint_counter=int(counter),
        velocity_scale=float(scale),
        velocity_limit_reason=int(reason),
        state_flags=int(flags),
        brakes_released=tuple(int(value) for value in brakes),
        cycle_time=float(cycle),
    )


def _floats(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def refuse_confirmation(discrepancies: Sequence[Any]) -> bool:
    """The position is never confirmed by a program (ADR-0054, p. 4)."""
    named = "; ".join(
        f"сустав {item.joint_number}: фактически {item.actual_position:.2f}°, "
        f"сохранено {item.saved_position:.2f}°, порог {item.allowed_discrepancy:.1f}°"
        for item in discrepancies
    )
    log.error(
        "контроллер просит подтвердить позицию (%s) — адаптер не подтверждает: это делает "
        "человек у робота в «Пульсе»",
        named or "без подробностей",
    )
    return False


class SdkChannel:
    """``RobotApi`` in ``full`` mode, opened by ``open``."""

    def __init__(self, api: Any, link: LinkConfig) -> None:
        self._api = api
        self._link = link
        self._identity = Identity(model="")
        self._receiver: Any = None
        self._last_stamp: tuple[int, int] | None = None
        self._lost: str | None = None
        self._closed = False

    @classmethod
    def open(cls, link: LinkConfig, *, logger: logging.Logger | None = None) -> SdkChannel:
        """Import the SDK, connect in ``full`` mode and learn what answers.

        Every failure after ``RobotApi`` exists goes through the SDK's ``disconnect``.
        """
        try:
            module = importlib.import_module(SDK_MODULE)
        except ImportError as exc:
            raise ChannelError(
                f"SDK производителя не найден ({SDK_MODULE}): {exc}. Поставьте архив API 1.4.1 "
                f"«Робопро»: pip install <каталог архива>; env check называет недостающее"
            ) from exc
        previous = sys.excepthook
        try:
            api = module.RobotApi(
                link.host,
                ignore_controller_exceptions=True,
                read_only=False,
                autoconnect=False,
                timeout=CONNECT_TIMEOUT_S,
                logger=logger if logger is not None else log,
            )
        finally:
            # The SDK installs its shutdown as the hook of the whole process.
            sys.excepthook = previous
        channel = cls(api, link)
        try:
            channel._connect()
        except BaseException:
            channel.close()
            raise
        return channel

    def _connect(self) -> None:
        api = self._api
        api.controller_state.set_confirm_position_callback(refuse_confirmation)
        api.set_disconnection_callback(self._on_disconnected)
        try:
            connected = api.connect(read_only=False)
        except Exception as exc:
            raise ChannelError(f"{self._link.host}: SDK не подключился — {exc}") from exc
        if not connected:
            raise ChannelError(
                f"{self._link.host}: SDK не подключился в режиме full. Обычно это адрес, сеть "
                f"или «Пульс», который держит управление (факт 0.5); подробности — в журнале SDK"
            )
        info = api.get_robot_info()
        # On a failed read 1.4.1 returns the default RobotInfo(): robot_model "rc10" and
        # zero DH parameters -- a plausible name for a body nobody read.
        if info == type(info)():
            raise ChannelError(
                f"{self._link.host}: get_robot_info не прочитан — SDK вернул значение по "
                f"умолчанию (robot_model {info.robot_model!r}, нулевые DH)"
            )
        self._identity = Identity(
            model=str(info.robot_model), client_version=str(info.client_version)
        )
        receiver = getattr(api, "_rtd_receiver", None)
        if receiver is None or not callable(getattr(receiver, "get_data", None)):
            raise ChannelError(
                "у RobotApi нет приёмника RTD _rtd_receiver.get_data: канал написан по "
                "исходнику API 1.4.1, у этой версии устройство другое (ADR-P5B)"
            )
        self._receiver = receiver

    def _on_disconnected(self) -> None:
        """Called by the SDK from its thread, after it closed the sockets itself."""
        self._lost = "SDK закрыл соединение с контроллером (обрыв сети или ошибка приёма RTD)"
        log.error("%s: %s", self._link.host, self._lost)

    # -- Channel ---------------------------------------------------------------

    def identity(self) -> Identity:
        return self._identity

    def lost(self) -> str | None:
        if self._lost is None and not self._closed and not self._api.is_connected():
            self._lost = "SDK сообщает, что соединения нет"
        return self._lost

    def poll(self, timeout: float) -> RtdReading | None:
        deadline = time.monotonic() + timeout
        while True:
            if self.lost() is not None or self._receiver is None:
                return None
            try:
                reading = copy_rtd(self._receiver.get_data())
            except ChannelError as exc:
                log.error("%s: %s", self._link.host, exc)
                reading = None
            if reading is not None and (reading.sec, reading.nsec) != self._last_stamp:
                self._last_stamp = (reading.sec, reading.nsec)
                return reading
            if time.monotonic() >= deadline:
                return None
            time.sleep(POLL_STEP_S)

    def set_scale(self, velocity: float, acceleration: float) -> None:
        self._call("шкала скорости", self._api.motion.scale_setup.set, velocity, acceleration)

    def scale(self) -> tuple[float, float]:
        answer = self._call("чтение шкалы", self._api.motion.scale_setup.get)
        # The control socket answers () on a receive timeout, and the SDK passes it on.
        if not isinstance(answer, tuple | list) or len(answer) != 2:
            raise ChannelError(f"шкала скорости не прочитана: контроллер ответил {answer!r}")
        return float(answer[0]), float(answer[1])

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """``joint.add_new_waypoint`` in radians, with every argument stated.

        Nothing is left to the SDK's module-wide defaults (``MOTION_SETUP``, fact 0.39).
        A refusal is decided by the robot's own counter, not by the text of the SDK's
        exception: if ``wp_cntr`` did not move, the point is not in the buffer.
        """
        before = self._counter()
        try:
            taken = self._api.motion.joint.add_new_waypoint(
                angle_pose=list(waypoint.target),
                speed=waypoint.speed,
                accel=waypoint.acceleration,
                blend=0.0,
                units="rad",
            )
        except Exception as exc:
            if self._counter() == before:
                raise WaypointRefusedError(f"контроллер не принял точку: {exc}") from exc
            raise ChannelError(f"точка ушла, но SDK сообщил об ошибке: {exc}") from exc
        if not taken:
            raise ChannelError("точка не добавлена: SDK потерял соединение, ожидая её")

    def start(self) -> None:
        # ``mode.set("move")`` refuses to send when the buffer is empty and returns False.
        if not self._call("режим move", self._api.motion.mode.set, "move"):
            raise ChannelError("режим move не установлен: буфер пуст или связь потеряна")

    def hold(self) -> None:
        # Waits for the mode up to SET_MOTION_MODE_AWAIT_SEC (5 s) and raises past it.
        if not self._call("режим hold", self._api.motion.mode.set, "hold"):
            raise ChannelError("режим hold не подтверждён: связь потеряна")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._api.disconnect()
        except Exception as exc:  # pragma: no cover - the SDK's disconnect does not raise
            log.error("%s: disconnect SDK: %s", self._link.host, exc)

    # -- internals ------------------------------------------------------------

    def _counter(self) -> int | None:
        if self._receiver is None:
            return None
        return int(self._receiver.get_data().wp_cntr) & WAYPOINT_COUNTER_MASK

    def _call(self, what: str, method: Any, *args: Any) -> Any:
        lost = self.lost()
        if lost is not None:
            raise ChannelError(f"{what}: связь потеряна — {lost}")
        try:
            return method(*args)
        except Exception as exc:
            raise ChannelError(f"{what}: {exc}") from exc
