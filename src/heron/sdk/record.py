"""A run on disk, written as it happens (ADR-0038).

The executor hands every state it read and every command it sent to a recorder;
this one writes them to MCAP as JSON, on two clocks. ``log_time`` is the
executor's own clock, ``publish_time`` is the world's: the stamp of a state, and
for a command the stamp of the state it was computed from. Nothing here knows a
robot's protocol -- a record is values in the contract's order, the same for
every robot.

``mcap`` is imported when a record is opened, never at import: the core
installs and runs without it.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Final

from .. import __version__
from ..core.branding import APP_NAME
from ..core.enums import ActionType, MotionPhase
from ..core.types import Command, Pose, RobotState
from .executor import RunResult

RECORD_SCHEMA_VERSION: Final = 1
STATE_TOPIC: Final = "/state"
COMMAND_TOPIC: Final = "/command"
RUN_METADATA: Final = "run"
RESULT_METADATA: Final = "result"
INTERRUPTED: Final = "interrupted"
"""Status of a record whose run never returned a result."""

WALL_ORIGIN_KEY: Final = "wall_origin_ns"
NS: Final = 1_000_000_000

_NUMBER: Final = {"type": ["number", "null"]}
_VECTOR: Final = {"type": "array", "items": _NUMBER}
_POSE: Final = {
    "type": ["object", "null"],
    "properties": {"position": _VECTOR, "orientation": _VECTOR},
}
STATE_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "number"},
        "joint_position": _VECTOR,
        "joint_velocity": _VECTOR,
        "joint_torque": _VECTOR,
        "base": _POSE,
        "motion": {"type": ["string", "null"]},
        "extra": {"type": "object"},
    },
    "required": ["timestamp", "joint_position", "joint_velocity", "joint_torque"],
}
COMMAND_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "values": _VECTOR,
        "pose": _POSE,
        "extra": {"type": "object"},
    },
    "required": ["type", "values"],
}


class RecordError(RuntimeError):
    """A record cannot be written or read."""


def _mcap(part: str) -> Any:
    try:
        if part == "writer":
            from mcap import writer

            return writer
        from mcap import reader

        return reader
    except ImportError as exc:
        raise RecordError(
            f"запись прогона требует пакет mcap ({exc}): pip install '{APP_NAME}[record]'"
        ) from exc


def require_writer() -> None:
    """Refuse before a robot is connected, not after, when a record cannot be written."""
    _mcap("writer")


# -- values as JSON ------------------------------------------------------------


def _number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _vector(values: Sequence[float]) -> list[float | None]:
    return [_number(float(v)) for v in values]


def _pose(pose: Pose | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    return {"position": _vector(pose.position), "orientation": _vector(pose.orientation)}


def state_fields(state: RobotState) -> dict[str, Any]:
    return {
        "timestamp": state.timestamp,
        "joint_position": _vector(state.joint_position),
        "joint_velocity": _vector(state.joint_velocity),
        "joint_torque": _vector(state.joint_torque),
        "base": _pose(state.base),
        "motion": None if state.motion is None else str(state.motion),
        "extra": state.extra,
    }


def command_fields(command: Command) -> dict[str, Any]:
    return {
        "type": str(command.type),
        "values": _vector(command.values),
        "pose": _pose(command.pose),
        "extra": command.extra,
    }


def _floats(values: Sequence[float | None]) -> tuple[float, ...]:
    return tuple(math.nan if v is None else float(v) for v in values)


def _pose_of(data: Mapping[str, Any] | None) -> Pose | None:
    if data is None:
        return None
    position = _floats(data["position"])
    orientation = _floats(data["orientation"])
    return Pose(
        position=(position[0], position[1], position[2]),
        orientation=(orientation[0], orientation[1], orientation[2], orientation[3]),
    )


def state_of(data: Mapping[str, Any]) -> RobotState:
    return RobotState(
        timestamp=float(data["timestamp"]),
        joint_position=_floats(data["joint_position"]),
        joint_velocity=_floats(data["joint_velocity"]),
        joint_torque=_floats(data["joint_torque"]),
        base=_pose_of(data.get("base")),
        motion=None if data.get("motion") is None else MotionPhase(data["motion"]),
        extra=dict(data.get("extra") or {}),
    )


def command_of(data: Mapping[str, Any]) -> Command:
    return Command(
        type=ActionType(data["type"]),
        values=_floats(data["values"]),
        pose=_pose_of(data.get("pose")),
        extra=dict(data.get("extra") or {}),
    )


def _json(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


# -- writing -------------------------------------------------------------------


class McapRecorder:
    """Writes one run to one MCAP file, message by message.

    ``origin_ns`` places the executor's clock on the wall: the monotonic clock a
    real run uses counts from boot, and a record read in a viewer should show the
    day it was made. It is stored in the record, so the executor's own seconds are
    recoverable exactly.
    """

    def __init__(
        self, path: Path, run: Mapping[str, str], *, origin_ns: int | None = None
    ) -> None:
        writer = _mcap("writer")
        self.path = Path(path)
        self._origin_ns = (
            origin_ns if origin_ns is not None else time.time_ns() - time.monotonic_ns()
        )
        try:
            self._stream: BinaryIO = self.path.open("wb")
        except OSError as exc:
            raise RecordError(f"{self.path}: запись не открывается — {exc}") from exc
        self._writer = writer.Writer(self._stream)
        self._writer.start(profile="", library=f"{APP_NAME} {__version__}")
        state_schema = self._writer.register_schema(
            name="RobotState", encoding="jsonschema", data=_json(STATE_SCHEMA)
        )
        command_schema = self._writer.register_schema(
            name="Command", encoding="jsonschema", data=_json(COMMAND_SCHEMA)
        )
        self._state = self._writer.register_channel(
            topic=STATE_TOPIC, message_encoding="json", schema_id=state_schema
        )
        self._command = self._writer.register_channel(
            topic=COMMAND_TOPIC, message_encoding="json", schema_id=command_schema
        )
        self._writer.add_metadata(
            RUN_METADATA,
            {
                **run,
                "record_schema_version": str(RECORD_SCHEMA_VERSION),
                WALL_ORIGIN_KEY: str(self._origin_ns),
            },
        )
        self._sequence = 0
        self._closed = False

    def state(self, wall: float, state: RobotState) -> None:
        self._add(self._state, wall, state.timestamp, state_fields(state))

    def command(self, wall: float, world: float, command: Command) -> None:
        self._add(self._command, wall, world, command_fields(command))

    def close(self, result: RunResult | None, *, error: BaseException | None = None) -> None:
        """Write how the run ended and finish the file. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if result is not None:
            outcome = {
                "status": str(result.status),
                "ticks": str(result.ticks),
                "commands": str(len(result.commands)),
                "reason": result.reason or "",
            }
        else:
            outcome = {"status": INTERRUPTED, "reason": repr(error) if error is not None else ""}
        try:
            self._writer.add_metadata(RESULT_METADATA, outcome)
            self._writer.finish()
        finally:
            self._stream.close()

    def _add(self, channel: int, wall: float, world: float, data: dict[str, Any]) -> None:
        self._writer.add_message(
            channel_id=channel,
            log_time=self._origin_ns + round(wall * NS),
            publish_time=round(world * NS),
            data=_json(data),
            sequence=self._sequence,
        )
        self._sequence += 1


# -- reading -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Recorded:
    """One message of a record: when, on both clocks, and in what order it was written."""

    sequence: int
    wall: float
    """Seconds of the executor's clock."""
    world: float
    value: Any


@dataclass(slots=True)
class Record:
    """A record read back."""

    run: dict[str, str]
    result: dict[str, str] | None
    states: list[Recorded] = field(default_factory=list)
    commands: list[Recorded] = field(default_factory=list)


def _recorded(message: Any, origin: int, value: RobotState | Command) -> Recorded:
    return Recorded(
        sequence=message.sequence,
        wall=(message.log_time - origin) / NS,
        world=message.publish_time / NS,
        value=value,
    )


def read_record(path: Path) -> Record:
    """Everything a record holds, as the values the executor handled."""
    reader_module = _mcap("reader")
    with Path(path).open("rb") as stream:
        reader = reader_module.make_reader(stream)
        metadata = {item.name: dict(item.metadata) for item in reader.iter_metadata()}
        if RUN_METADATA not in metadata:
            raise RecordError(f"{path}: нет метаданных {RUN_METADATA!r} — это не запись прогона")
        run = metadata[RUN_METADATA]
        origin = int(run.get(WALL_ORIGIN_KEY, "0"))
        record = Record(run=run, result=metadata.get(RESULT_METADATA))
        for _, channel, message in reader.iter_messages(log_time_order=False):
            data = json.loads(message.data)
            if channel.topic == STATE_TOPIC:
                record.states.append(_recorded(message, origin, state_of(data)))
            elif channel.topic == COMMAND_TOPIC:
                record.commands.append(_recorded(message, origin, command_of(data)))
    record.states.sort(key=lambda item: item.sequence)
    record.commands.sort(key=lambda item: item.sequence)
    return record
