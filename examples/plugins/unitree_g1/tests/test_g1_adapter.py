"""The DDS adapter on a recording channel.

Everything that decides what goes on the wire is checkable here: slot, gains,
body, stop, refusal. Nothing here says that a motor accepts the result -- that is
the twin's to show (stage 4 of the phase-4 plan).
"""

from __future__ import annotations

from typing import Any

import pytest
from unitree_g1 import adapter as adapter_module
from unitree_g1.adapter import (
    G1DdsAdapter,
    LinkError,
    NotConnectedError,
    SafetyViolationError,
    StateMismatchError,
    UnsupportedActionError,
)
from unitree_g1.channel import LinkConfig, LinkConfigError, MotorCommand, RecordingChannel
from unitree_g1.slots import LAYOUT, BodyMismatchError, SlotError
from unitree_g1.testing import LINK_PARAMS, deliver_states, profile_23dof

from heron.core.contract import ActionSpec
from heron.core.enums import ActionType
from heron.core.environment import RequirementKind
from heron.core.models import RobotProfile
from heron.core.protocols import CommandRefusedError, ControlAdapter
from heron.core.types import Command


def connected(
    profile: RobotProfile | None = None,
    *,
    states: int = 8,
    mode_machine: int = 4,
    **config: Any,
) -> tuple[G1DdsAdapter, RecordingChannel, RobotProfile]:
    profile = profile or profile_23dof()
    channel = RecordingChannel()
    deliver_states(channel, states, mode_machine=mode_machine)
    adapter = G1DdsAdapter(channel)
    adapter.connect(profile, {**LINK_PARAMS, **config})
    return adapter, channel, profile


def distinct(profile: RobotProfile) -> tuple[float, ...]:
    """One value per joint, all different and inside the limits."""
    return tuple(0.05 * (index + 1) for index in range(profile.contract.dof))


def test_satisfies_the_protocol() -> None:
    assert isinstance(G1DdsAdapter(), ControlAdapter)


# -- the body on the link ------------------------------------------------------


def test_connect_takes_the_body_from_the_first_state_and_reflects_it() -> None:
    adapter, channel, profile = connected()
    zeros = (0.0,) * profile.contract.dof
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=zeros))
    command = channel.published[-1]
    assert adapter.mode_machine == 4
    assert (command.mode_machine, command.mode_pr) == (4, 0)


def test_a_29_dof_body_refuses_a_23_dof_contract() -> None:
    with pytest.raises(BodyMismatchError, match="left_wrist_pitch_joint"):
        connected(mode_machine=5)


def test_an_unknown_body_is_refused() -> None:
    with pytest.raises(BodyMismatchError, match="7"):
        connected(mode_machine=7)


def test_a_refused_body_gets_no_command() -> None:
    channel = RecordingChannel()
    deliver_states(channel, 4, mode_machine=5)
    adapter = G1DdsAdapter(channel)
    with pytest.raises(BodyMismatchError):
        adapter.connect(profile_23dof(), dict(LINK_PARAMS))
    assert channel.published == []
    with pytest.raises(NotConnectedError):
        adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.0,) * 23))


def test_the_body_changing_mid_run_is_an_error() -> None:
    adapter, channel, _ = connected(states=1)
    channel.deliver_state(100, mode_machine=5)
    with pytest.raises(StateMismatchError, match="с 4 на 5"):
        adapter.read_state()


def test_a_joint_outside_the_layout_is_refused_on_connect() -> None:
    profile = profile_23dof()
    order = ["joint0", *profile.contract.joint_order[1:]]
    renamed = profile.model_copy(
        update={"contract": profile.contract.model_copy(update={"joint_order": order})}
    )
    with pytest.raises(SlotError, match="joint0"):
        G1DdsAdapter(RecordingChannel()).connect(renamed, dict(LINK_PARAMS))


# -- commands ------------------------------------------------------------------


def test_each_joint_lands_in_its_slot_with_its_own_gains() -> None:
    adapter, channel, profile = connected()
    values = distinct(profile)
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=values))
    command = channel.published[-1]
    action = profile.contract.action
    assert action.kp is not None and action.kd is not None
    for index, name in enumerate(profile.contract.joint_order):
        assert command.motors[LAYOUT.index(name)] == MotorCommand(
            mode=1, q=values[index], dq=0.0, tau=0.0, kp=action.kp[index], kd=action.kd[index]
        ), name


def test_the_left_shoulder_goes_to_slot_15_where_the_vendor_table_says_13() -> None:
    adapter, channel, profile = connected()
    values = distinct(profile)
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=values))
    shoulder = profile.contract.joint_order.index("left_shoulder_pitch_joint")
    assert channel.published[-1].motors[15].q == values[shoulder]
    assert channel.published[-1].motors[13] == MotorCommand()


def test_empty_slots_are_off() -> None:
    adapter, channel, profile = connected()
    adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=distinct(profile)))
    motors = channel.published[-1].motors
    for slot in (13, 14, 20, 21, 27, 28, *range(29, 35)):
        assert motors[slot] == MotorCommand(), slot


def test_a_command_outside_the_limits_is_refused_by_name_and_nothing_is_sent() -> None:
    adapter, channel, profile = connected()
    values = list(distinct(profile))
    elbow = profile.contract.joint_order.index("right_elbow_joint")
    values[elbow] = 2.5
    with pytest.raises(CommandRefusedError, match=r"right_elbow_joint = 2.5 при \[-2, 2\]"):
        adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=tuple(values)))
    assert channel.published == []


def test_a_command_of_the_wrong_width_is_refused() -> None:
    adapter, channel, _ = connected()
    with pytest.raises(ValueError, match="суставов"):
        adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=(0.0,) * 22))
    assert channel.published == []


def test_a_command_of_another_quantity_is_refused() -> None:
    adapter, _, profile = connected()
    with pytest.raises(UnsupportedActionError):
        adapter.send_command(
            Command(type=ActionType.JOINT_VELOCITY, values=(0.0,) * profile.contract.dof)
        )


def test_an_action_without_gains_is_refused_on_connect() -> None:
    """A LowCmd without kp and kd is a motor producing no torque."""
    profile = profile_23dof()
    bare = profile.model_copy(
        update={
            "contract": profile.contract.model_copy(
                update={"action": ActionSpec(type=ActionType.JOINT_POSITION)}
            )
        }
    )
    with pytest.raises(UnsupportedActionError, match="kp и kd"):
        G1DdsAdapter(RecordingChannel()).connect(bare, dict(LINK_PARAMS))


# -- state ---------------------------------------------------------------------


def test_state_is_read_back_by_name_and_stamped_with_world_time() -> None:
    adapter, channel, profile = connected(states=1)
    channel.deliver_state(
        1234,
        mode_machine=4,
        q=[slot / 10 for slot in range(35)],
        dq=[slot / 100 for slot in range(35)],
        tau_est=[float(slot) for slot in range(35)],
    )
    state = adapter.read_state()
    slots = [LAYOUT.index(name) for name in profile.contract.joint_order]
    assert state.timestamp == pytest.approx(1.234)
    assert state.joint_position == pytest.approx([slot / 10 for slot in slots])
    assert state.joint_velocity == pytest.approx([slot / 100 for slot in slots])
    assert state.joint_torque == pytest.approx([float(slot) for slot in slots])


def test_no_new_state_returns_the_last_one_with_its_stamp() -> None:
    adapter, _, _ = connected(states=1)
    first = adapter.read_state()
    assert adapter.read_state().timestamp == first.timestamp


def test_silence_says_nobody_publishes() -> None:
    adapter = G1DdsAdapter(RecordingChannel())
    with pytest.raises(LinkError, match="никто не публикует") as caught:
        adapter.connect(profile_23dof(), dict(LINK_PARAMS))
    assert "домен 1, интерфейс lo, пиры 127.0.0.1" in str(caught.value)


def test_a_publisher_whose_messages_never_arrive_is_told_apart() -> None:
    channel = RecordingChannel(publishers=1)
    with pytest.raises(LinkError, match="публикующих: 1, но сообщения не доходят"):
        G1DdsAdapter(channel).connect(profile_23dof(), dict(LINK_PARAMS))


# -- link configuration -------------------------------------------------------


def test_the_link_is_read_from_params_before_anything_is_polled() -> None:
    channel = RecordingChannel()
    deliver_states(channel, 3)
    with pytest.raises(LinkConfigError, match=r"params\.domain"):
        G1DdsAdapter(channel).connect(profile_23dof(), {"interface": "lo"})
    assert len(channel.readings) == 3


def test_the_channel_it_opens_gets_the_link_and_is_closed_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[LinkConfig] = []
    channel = RecordingChannel()

    def open_channel(link: LinkConfig) -> RecordingChannel:
        opened.append(link)
        return channel

    monkeypatch.setattr(adapter_module, "open_cyclone", open_channel)
    adapter = G1DdsAdapter()
    with pytest.raises(LinkError):
        adapter.connect(profile_23dof(), dict(LINK_PARAMS))
    assert opened == [LinkConfig(domain=1, interface="lo", peers=("127.0.0.1",))]
    assert channel.closed
    assert adapter.channel is None


def test_disconnect_closes_only_a_channel_it_opened() -> None:
    adapter, channel, _ = connected()
    adapter.disconnect()
    assert not channel.closed
    with pytest.raises(NotConnectedError):
        adapter.read_state()


# -- emergency stop -------------------------------------------------------------


def test_emergency_stop_damps_every_joint_of_the_contract() -> None:
    adapter, channel, profile = connected(states=1)
    channel.deliver_state(50, mode_machine=4, q=[slot / 10 for slot in range(35)])
    measured = adapter.read_state().joint_position
    adapter.emergency_stop()

    command = channel.published[-1]
    kd = profile.contract.action.kd
    assert kd is not None
    for index, name in enumerate(profile.contract.joint_order):
        assert command.motors[LAYOUT.index(name)] == MotorCommand(
            mode=1, q=measured[index], dq=0.0, tau=0.0, kp=0.0, kd=kd[index]
        ), name
    for slot in (13, 14, 20, 21, 27, 28):
        assert command.motors[slot] == MotorCommand()


def test_after_a_stop_commands_are_refused_until_reset() -> None:
    adapter, channel, profile = connected()
    adapter.emergency_stop()
    zeros = Command(type=ActionType.JOINT_POSITION, values=(0.0,) * profile.contract.dof)
    with pytest.raises(SafetyViolationError):
        adapter.send_command(zeros)
    adapter.reset()
    adapter.send_command(zeros)
    assert len(channel.published) == 2


def test_a_stop_that_cannot_be_sent_still_refuses_what_follows() -> None:
    adapter, channel, profile = connected()
    channel.close()
    with pytest.raises(RuntimeError):
        adapter.emergency_stop()
    assert adapter.stopped
    with pytest.raises(CommandRefusedError):
        adapter.send_command(
            Command(type=ActionType.JOINT_POSITION, values=(0.0,) * profile.contract.dof)
        )


# -- environment ---------------------------------------------------------------


def test_requirements_name_the_version_the_robot_runs_and_forbid_the_one_that_crashes_it() -> None:
    declared = {(item.name, item.version, item.forbidden) for item in G1DdsAdapter.requirements}
    assert declared == {
        ("cyclonedds", "==0.10.2", False),
        ("cyclonedds", ">=0.11", True),
        ("unitree_sdk2py", None, False),
    }
    assert all(item.kind is RequirementKind.PYTHON for item in G1DdsAdapter.requirements)
