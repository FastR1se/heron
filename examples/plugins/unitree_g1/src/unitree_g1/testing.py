"""What a test of this adapter needs: a profile, a link and a robot made of readings.

Shipped with the package rather than kept in its tests, so that a test of the
package and its harness for the platform's contract suite build the adapter one
way (ADR-0056).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from heron.core.capabilities import Capability
from heron.core.contract import ActionSpec, Contract, ObservationTerm
from heron.core.enums import ActionType, AngleUnit, ControlMode, ObservationType
from heron.core.models import AdapterSpec, JointRange, Limits, RobotProfile

from .adapter import G1DdsAdapter
from .channel import RecordingChannel
from .slots import (
    BODY_23DOF,
    BODY_29DOF_LOCK_WAIST,
    LAYOUT,
    UPPER_BODY,
    is_upper_body,
    machine_for,
)

PLUGIN: Final = "unitree_g1_dds"
"""The adapter's entry-point name, as a profile writes it."""

JOINTS_23DOF: Final[tuple[str, ...]] = tuple(name for name in LAYOUT if name in BODY_23DOF)

LINK_PARAMS: Final[dict[str, Any]] = {
    "domain": 1,
    "interface": "lo",
    "peers": ["127.0.0.1"],
    # A recording channel answers at once; the real deadline would spend ten
    # seconds finding out that nothing was queued.
    "link_timeout": 0.05,
}


def profile_23dof(**overrides: Any) -> RobotProfile:
    """A 23-DoF G1 profile. Gains differ per joint, so a permuted slot shows."""
    joints = list(JOINTS_23DOF)
    dof = len(joints)
    fields: dict[str, Any] = {
        "schema_version": 1,
        "name": "g1_23dof_test",
        "description": {"urdf": "g1_23dof.urdf"},
        "contract": Contract(
            revision=1,
            joint_order=joints,
            angle_unit=AngleUnit.RAD,
            state_rate_hz=500.0,
            command_rate_hz=50.0,
            observation=[ObservationTerm(type=ObservationType.JOINT_POS, scale=1.0)],
            action=ActionSpec(
                type=ActionType.JOINT_POSITION,
                kp=[40.0 + index for index in range(dof)],
                kd=[1.0 + index / 10 for index in range(dof)],
            ),
        ),
        "capabilities": frozenset({Capability("arm.joint_position")}),
        "control_mode": ControlMode.STREAMING,
        "limits": Limits(joint_position=JointRange(min=[-2.0] * dof, max=[2.0] * dof)),
        "adapters": {"sim.mujoco": AdapterSpec(plugin=PLUGIN, params=dict(LINK_PARAMS))},
    }
    fields.update(overrides)
    return RobotProfile(**fields)


JOINTS_UPPER_BODY_6: Final[tuple[str, ...]] = tuple(
    name for name in LAYOUT if name in BODY_29DOF_LOCK_WAIST & UPPER_BODY
)
"""The upper body of ``mode_machine`` 6: waist yaw and both arms, 15 joints (ADR-0064)."""


def profile_upper_body(**overrides: Any) -> RobotProfile:
    """A mounted G1 driven above the waist: body 6 without its legs, no ``real`` (ADR-0063)."""
    joints = list(JOINTS_UPPER_BODY_6)
    dof = len(joints)
    fields: dict[str, Any] = {
        "name": "g1_upper_body_test",
        "contract": Contract(
            revision=1,
            joint_order=joints,
            angle_unit=AngleUnit.RAD,
            state_rate_hz=500.0,
            command_rate_hz=50.0,
            observation=[ObservationTerm(type=ObservationType.JOINT_POS, scale=1.0)],
            action=ActionSpec(
                type=ActionType.JOINT_POSITION,
                kp=[20.0 + index for index in range(dof)],
                kd=[0.1 + index / 100 for index in range(dof)],
            ),
        ),
        "limits": Limits(joint_position=JointRange(min=[-1.5] * dof, max=[1.5] * dof)),
        "adapters": {"sim.gazebo": AdapterSpec(plugin=PLUGIN, params=dict(LINK_PARAMS))},
    }
    fields.update(overrides)
    return profile_23dof(**fields)


def machine_of(joints: Sequence[str]) -> int:
    """The ``mode_machine`` of the body with exactly these joints, or with this upper body."""
    machine = machine_for(joints, upper_body=is_upper_body(joints))
    if machine is None:
        raise ValueError("ни одно известное тело не состоит ровно из этих суставов")
    return machine


def deliver_states(
    channel: RecordingChannel,
    count: int,
    *,
    mode_machine: int = 4,
    start_tick: int = 0,
    step_ticks: int = 2,
    q: Sequence[float] = (),
) -> None:
    """Queue ``count`` states two milliseconds of world apart, as the twin at 500 Hz."""
    for index in range(count):
        channel.deliver_state(start_tick + index * step_ticks, mode_machine=mode_machine, q=q)


def recording_adapter() -> G1DdsAdapter:
    return G1DdsAdapter(RecordingChannel())


def prime(adapter: G1DdsAdapter, profile: RobotProfile, count: int = 32) -> None:
    """Queue states on an adapter's recording channel, from the body its profile describes."""
    channel = adapter.channel
    if not isinstance(channel, RecordingChannel):
        raise TypeError("prime подаёт показания только в записывающий канал")
    deliver_states(channel, count, mode_machine=machine_of(profile.contract.joint_order))
