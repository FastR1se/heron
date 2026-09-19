"""What the platform's contract suite needs to hold this adapter (ADR-0056).

A robot made of readings: the adapter over a recording channel, and the readings a
running controller would give it. Shipped with the package, so that its own tests and
its harness for the contract suite build the adapter one way.

The stub laid a ``profile.yaml`` beside the sources and read it here. The profile of
the real robot lives with the other examples (``examples/robots/rc5``), and a package
must not reach into a repository it is not shipped with, so the test profile is built
in code: the RC5's joints and the class of its link, with an envelope narrow enough
that the suite's probe value (99 in the first joint) lies outside it.
"""

from __future__ import annotations

from typing import Any, Final

from heron.core.capabilities import Capability
from heron.core.contract import ActionSpec, Contract, ObservationTerm
from heron.core.enums import ActionType, AngleUnit, ControlMode, ObservationType
from heron.core.models import AdapterSpec, JointMax, JointRange, Limits, RobotProfile

from .adapter import RoboproRcAdapter
from .channel import Identity, RecordingChannel

PLUGIN: Final = "robopro_rc"
"""The adapter's entry-point name, as a profile writes it."""

MODEL: Final = "rc5_recording"
"""What the recording controller calls itself. Not a real model name: that comes
from probe 0.22 (ADR-0054)."""

JOINTS: Final = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

LINK_PARAMS: Final[dict[str, Any]] = {
    "host": "192.0.2.10",  # TEST-NET-1: an address that is nobody's
    "model": MODEL,
    # A recording channel answers at once; the real deadlines would only slow a test.
    "link_timeout": 0.05,
    "poll_timeout": 0.0,
}


def profile(**overrides: Any) -> RobotProfile:
    """An RC5-shaped profile in degrees, with an envelope of ±90° per joint."""
    dof = len(JOINTS)
    fields: dict[str, Any] = {
        "schema_version": 1,
        "name": "rc5_test",
        "description": {"urdf": "rc5.urdf"},
        "contract": Contract(
            revision=1,
            joint_order=list(JOINTS),
            angle_unit=AngleUnit.DEG,
            state_rate_hz=10.0,
            command_rate_hz=2.0,
            observation=[ObservationTerm(type=ObservationType.JOINT_POS, scale=1.0)],
            action=ActionSpec(type=ActionType.JOINT_POSITION),
        ),
        "capabilities": frozenset({Capability("arm.joint_position")}),
        "control_mode": ControlMode.COMMAND,
        "limits": Limits(
            joint_position=JointRange(min=[-90.0] * dof, max=[90.0] * dof),
            joint_velocity=JointMax(max=[60.0] * dof),
            joint_acceleration=JointMax(max=[80.0] * dof),
        ),
        "adapters": {"real": AdapterSpec(plugin=PLUGIN, params=dict(LINK_PARAMS))},
    }
    fields.update(overrides)
    return RobotProfile(**fields)


def recording_channel(**fields: Any) -> RecordingChannel:
    return RecordingChannel(who=Identity(model=MODEL, client_version="1.4.0.33555712/4"), **fields)


def recording_adapter() -> RoboproRcAdapter:
    return RoboproRcAdapter(recording_channel())


def channel_of(adapter: RoboproRcAdapter) -> RecordingChannel:
    channel = adapter.channel
    if not isinstance(channel, RecordingChannel):
        raise TypeError("обвязка работает только с записывающим каналом")
    return channel


def prime(adapter: RoboproRcAdapter, profile: RobotProfile, count: int = 32) -> None:
    """Queue ``count`` readings of a controller in ``run`` holding still."""
    channel = channel_of(adapter)
    for _ in range(count):
        channel.deliver()


def finish(adapter: RoboproRcAdapter, profile: RobotProfile) -> None:
    """The controller reports the accepted motion over: buffer empty, back in ``hold``."""
    channel_of(adapter).finish_motion()
