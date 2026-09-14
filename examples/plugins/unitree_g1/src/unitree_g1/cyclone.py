"""The real channel: CycloneDDS directly, types and CRC from the vendor's SDK (ADR-0029).

The only module of this package that imports either, and the only one no test
in the repository executes. Everything decidable without them was decided
before a value reached it, so what is left is: build the configuration, create
the entities, convert.

The SDK's own channel is not used (ADR-0022, p. 4): its configuration template
brings CycloneDDS 0.10.2 down, and it hides the reader whose match status tells
silence from loss. Every call below is one the probes of stage 1 made against
the vendor's twin, except the blocking ``take_one``, which is read from the
SDK's own reader and has not run here.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Final

from .channel import (
    COMMAND_TOPIC,
    QUEUE_DEPTH,
    STATE_TOPIC,
    ChannelError,
    LinkConfig,
    LowCommand,
    LowStateReading,
)

CANNOT_SHARE_A_PROCESS: Final = ("rclpy",)
"""Modules whose presence in the process stops the 0.10.2 binding from creating a
domain: rclpy on ``rmw_cyclonedds_cpp`` loads another ``libddsc`` (ADR-0022, p. 3)."""


def _import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ChannelError(
            f"{module} не импортируется ({exc}). Связи с G1 нужны cyclonedds 0.10.2 и "
            f"unitree_sdk2py в окружении этого процесса; недостающее назовёт "
            f"env check <сценарий>"
        ) from exc


class CycloneChannel:
    """One participant, one reader of ``rt/lowstate``, one writer of ``rt/lowcmd``."""

    def __init__(self, link: LinkConfig) -> None:
        loaded = [name for name in CANNOT_SHARE_A_PROCESS if name in sys.modules]
        if loaded:
            raise ChannelError(
                f"в процессе уже загружен {', '.join(loaded)}: обвязка CycloneDDS 0.10.2 не "
                f"создаёт домен рядом с ним. Адаптер G1 запускается в процессе без ROS 2"
            )
        core = _import("cyclonedds.core")
        domain = _import("cyclonedds.domain")
        pub = _import("cyclonedds.pub")
        sub = _import("cyclonedds.sub")
        topic = _import("cyclonedds.topic")
        self._duration = _import("cyclonedds.util").duration
        self._invalid = _import("cyclonedds.internal").InvalidSample
        messages = _import("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        self._new_command = _import("unitree_sdk2py.idl.default").unitree_hg_msg_dds__LowCmd_
        self._crc = _import("unitree_sdk2py.utils.crc").CRC()
        self._matched = 0

        xml = link.cyclonedds_xml()
        try:
            self._domain = domain.Domain(link.domain, xml)
            self._participant = domain.DomainParticipant(link.domain)
            self._reader = sub.DataReader(
                self._participant,
                topic.Topic(self._participant, STATE_TOPIC, messages.LowState_),
                listener=core.Listener(on_subscription_matched=self._on_matched),
            )
            self._writer = pub.DataWriter(
                self._participant,
                topic.Topic(self._participant, COMMAND_TOPIC, messages.LowCmd_),
            )
        except Exception as exc:
            raise ChannelError(
                f"домен DDS {link.domain} не открыт на интерфейсе {link.interface!r}: {exc}. "
                f"Конфигурация: {xml}"
            ) from exc

    def publish(self, command: LowCommand) -> None:
        message = self._new_command()
        message.mode_pr = command.mode_pr
        message.mode_machine = command.mode_machine
        for slot, motor in enumerate(command.motors):
            target = message.motor_cmd[slot]
            target.mode = motor.mode
            target.q = motor.q
            target.dq = motor.dq
            target.tau = motor.tau
            target.kp = motor.kp
            target.kd = motor.kd
        message.crc = self._crc.Crc(message)
        self._writer.write(message)

    def poll(self, timeout: float) -> LowStateReading | None:
        newest = self._drain()
        if newest is None:
            try:
                sample = self._reader.take_one(timeout=self._duration(seconds=max(timeout, 0.0)))
            except TimeoutError:
                return None
            if not isinstance(sample, self._invalid):
                newest = sample
            newest = self._drain() or newest
        return None if newest is None else _reading(newest)

    def publisher_count(self) -> int:
        return self._matched

    def close(self) -> None:
        # CycloneDDS entities are deleted when the last reference goes; dropping
        # the writer and reader before the participant and the domain keeps the
        # order the library created them in, reversed.
        self._writer = None
        self._reader = None
        self._participant = None
        self._domain = None

    # -- internals ------------------------------------------------------------

    def _on_matched(self, reader: Any, status: Any) -> None:
        self._matched = int(status.current_count)

    def _drain(self) -> Any:
        """The newest valid sample already waiting, or ``None``."""
        valid = [s for s in self._reader.take(N=QUEUE_DEPTH) if not isinstance(s, self._invalid)]
        return valid[-1] if valid else None


def _reading(sample: Any) -> LowStateReading:
    motors = sample.motor_state
    return LowStateReading(
        tick=int(sample.tick),
        mode_machine=int(sample.mode_machine),
        mode_pr=int(sample.mode_pr),
        q=tuple(float(motor.q) for motor in motors),
        dq=tuple(float(motor.dq) for motor in motors),
        tau_est=tuple(float(motor.tau_est) for motor in motors),
    )
