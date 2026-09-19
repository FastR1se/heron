"""The RC adapter on a recording channel.

Everything that decides what goes to the controller is checkable here: the body, the
scale, the envelope, the refusals, the phase, the stop. Nothing here says the
controller accepts the result -- that is the stand's to show (stage 4 of the plan).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from robopro_rc import adapter as adapter_module
from robopro_rc.adapter import (
    ACCELERATION_LIMIT_RAD_S2,
    SPEED_LIMIT_RAD_S,
    BodyMismatchError,
    LinkConfigurationError,
    LinkError,
    MotionInProgressError,
    NotConnectedError,
    NotReadyError,
    RoboproRcAdapter,
    SafetyViolationError,
    ScaleNotAppliedError,
    StopNotDeliveredError,
)
from robopro_rc.channel import (
    ChannelError,
    ControllerState,
    LinkConfig,
    LinkConfigError,
    MotionMode,
    RecordingChannel,
    SafetyStatus,
    WaypointRefusedError,
    standing,
)
from robopro_rc.testing import LINK_PARAMS, MODEL, profile, recording_channel

from heron.core.enums import ActionType, ControlMode, MotionPhase
from heron.core.environment import RequirementKind
from heron.core.models import SPEED_SCALE_PARAM, JointMax, JointRange, Limits, RobotProfile
from heron.core.protocols import CommandRefusedError, ControlAdapter
from heron.core.types import Command


def connected(
    robot: RobotProfile | None = None,
    channel: RecordingChannel | None = None,
    *,
    readings: int = 1,
    **config: Any,
) -> tuple[RoboproRcAdapter, RecordingChannel, RobotProfile]:
    robot = robot or profile()
    channel = channel or recording_channel()
    for _ in range(readings):
        channel.deliver()
    adapter = RoboproRcAdapter(channel)
    adapter.connect(robot, {**LINK_PARAMS, **config})
    return adapter, channel, robot


def goal(*values: float) -> Command:
    return Command(type=ActionType.JOINT_POSITION, values=tuple(values))


TARGET = goal(10.0, -20.0, 30.0, -40.0, 50.0, -60.0)


def test_satisfies_the_protocol() -> None:
    assert isinstance(RoboproRcAdapter(), ControlAdapter)


# -- the body and the link (ADR-0054) ------------------------------------------


def test_the_link_needs_a_host_and_a_model() -> None:
    assert LinkConfig.from_params({"host": " 10.0.0.2 ", "model": "rc5b2"}) == LinkConfig(
        "10.0.0.2", "rc5b2"
    )
    with pytest.raises(LinkConfigError, match=r"params\.host"):
        LinkConfig.from_params({"model": "rc5b2"})


@pytest.mark.parametrize("model", [None, "", "  "])
def test_a_model_the_profile_does_not_state_is_refused_and_the_probe_is_named(
    model: str | None,
) -> None:
    """The shipped profile says ``model: null`` until probe 0.22: connect must say so."""
    with pytest.raises(LinkConfigError, match=r"0\.22"):
        connected(model=model)


def test_nothing_is_opened_for_a_link_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[LinkConfig] = []
    monkeypatch.setattr(adapter_module, "open_sdk", lambda link: opened.append(link))
    with pytest.raises(LinkConfigError):
        RoboproRcAdapter().connect(profile(), {**LINK_PARAMS, "model": None})
    assert opened == []


def test_another_model_on_the_link_is_refused_by_both_names_and_gets_no_setting() -> None:
    channel = recording_channel()
    with pytest.raises(BodyMismatchError, match=rf"'{MODEL}'.*'rc10b2'"):
        connected(channel=channel, model="rc10b2")
    assert channel.scales_set == [] and channel.waypoints == []
    assert channel.closed is False  # injected: not the adapter's to close


def test_the_model_is_compared_whole_not_by_family() -> None:
    with pytest.raises(BodyMismatchError):
        connected(model=MODEL[:-1])


def test_a_controller_that_does_not_name_its_model_is_refused() -> None:
    channel = RecordingChannel()
    channel.who = type(channel.who)(model="")
    with pytest.raises(BodyMismatchError, match="не назвал"):
        connected(channel=channel)


def test_the_joint_count_of_the_sdk_is_held_against_the_contract() -> None:
    channel = recording_channel(robot=standing(joints=7))
    with pytest.raises(BodyMismatchError, match="7 суставов"):
        connected(channel=channel)


def test_a_controller_not_in_run_is_refused_and_the_brakes_are_left_to_a_human() -> None:
    channel = recording_channel()
    channel.robot = standing()
    for _ in range(3):
        channel.deliver(controller_state=ControllerState.ON)
    adapter = RoboproRcAdapter(channel)
    with pytest.raises(NotReadyError, match=r"on.*«Пульсе»"):
        adapter.connect(profile(), dict(LINK_PARAMS))
    assert channel.scales_set == []


def test_silence_is_named_with_the_address() -> None:
    adapter = RoboproRcAdapter(recording_channel())
    with pytest.raises(LinkError, match=r"192\.0\.2\.10"):
        adapter.connect(profile(), dict(LINK_PARAMS))


def test_a_channel_it_opens_is_closed_when_connect_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    channel = recording_channel()
    channel.deliver(controller_state=ControllerState.OFF)
    monkeypatch.setattr(adapter_module, "open_sdk", lambda link: channel)
    with pytest.raises(NotReadyError):
        RoboproRcAdapter().connect(profile(), dict(LINK_PARAMS))
    assert channel.closed


def test_disconnect_closes_only_a_channel_it_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    injected, _, _ = connected()
    injected.disconnect()
    assert not isinstance(injected.channel, RecordingChannel) or not injected.channel.closed

    owned = recording_channel()
    owned.deliver()
    monkeypatch.setattr(adapter_module, "open_sdk", lambda link: owned)
    adapter = RoboproRcAdapter()
    adapter.connect(profile(), dict(LINK_PARAMS))
    adapter.disconnect()
    assert owned.closed and adapter.channel is None


# -- the profile and the run -----------------------------------------------------


def test_a_streaming_profile_is_refused() -> None:
    with pytest.raises(LinkConfigurationError, match="command"):
        connected(profile(control_mode=ControlMode.STREAMING))


@pytest.mark.parametrize("missing", ["joint_position", "joint_velocity", "joint_acceleration"])
def test_the_envelope_is_required_whole(missing: str) -> None:
    limits = profile().limits.model_copy(update={missing: None})
    with pytest.raises(LinkConfigurationError, match=missing):
        connected(profile(limits=limits))


def test_one_speed_and_one_acceleration_the_slowest_joint_keeps_in_radians() -> None:
    limits = Limits(
        joint_position=JointRange(min=[-90.0] * 6, max=[90.0] * 6),
        joint_velocity=JointMax(max=[60.0, 45.0, 60.0, 60.0, 60.0, 60.0]),
        joint_acceleration=JointMax(max=[80.0, 80.0, 80.0, 80.0, 70.0, 80.0]),
    )
    adapter, channel, _ = connected(profile(limits=limits))
    adapter.send_command(TARGET)
    waypoint = channel.waypoints[-1]
    assert waypoint.speed == pytest.approx(math.radians(45.0))
    assert waypoint.acceleration == pytest.approx(math.radians(70.0))
    assert adapter.motion_bounds == (waypoint.speed, waypoint.acceleration)


def test_an_envelope_faster_than_the_sdk_accepts_is_refused_on_connect() -> None:
    """3.14 rad/s, not pi: the passport's 180 °/s is beyond what the SDK takes."""
    assert math.radians(180.0) > SPEED_LIMIT_RAD_S
    limits = profile().limits.model_copy(update={"joint_velocity": JointMax(max=[180.0] * 6)})
    with pytest.raises(LinkConfigurationError, match=r"3\.14"):
        connected(profile(limits=limits))
    fast = JointMax(max=[math.degrees(ACCELERATION_LIMIT_RAD_S2) + 1] * 6)
    with pytest.raises(LinkConfigurationError, match=r"26\.18"):
        connected(profile(limits=profile().limits.model_copy(update={"joint_acceleration": fast})))


# -- the speed scale (ADR-0052, ADR-0057) ----------------------------------------------


def test_the_scale_is_set_as_s_and_s_squared() -> None:
    adapter, channel, _ = connected(**{SPEED_SCALE_PARAM: 0.1})
    velocity, acceleration = channel.scales_set[-1]
    assert (velocity, acceleration) == (0.1, pytest.approx(0.01))
    assert adapter.speed_scale == 0.1


def test_a_run_at_full_speed_states_it_to_the_controller_too() -> None:
    """A scale left by someone else is not the run's: 1 is set, not assumed."""
    _, channel, _ = connected()
    assert channel.scales_set == [(1.0, 1.0)]


@pytest.mark.parametrize("applied", [(0.2, 0.01), (0.1, 0.02)])
def test_a_controller_applying_a_faster_scale_is_refused_on_connect(
    applied: tuple[float, float],
) -> None:
    channel = recording_channel(applied=applied)
    with pytest.raises(ScaleNotAppliedError, match="быстрее"):
        connected(channel=channel, **{SPEED_SCALE_PARAM: 0.1})


def test_a_slower_applied_scale_is_accepted() -> None:
    channel = recording_channel(applied=(0.05, 0.0025))
    connected(channel=channel, **{SPEED_SCALE_PARAM: 0.1})


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5])
def test_a_scale_outside_the_interval_is_refused(scale: float) -> None:
    with pytest.raises(LinkConfigurationError):
        connected(**{SPEED_SCALE_PARAM: scale})


def test_a_reading_with_a_faster_applied_scale_refuses_the_next_command() -> None:
    adapter, channel, _ = connected(**{SPEED_SCALE_PARAM: 0.1})
    channel.deliver(velocity_scale=0.5)
    adapter.read_state()
    with pytest.raises(SafetyViolationError, match=r"0\.5"):
        adapter.send_command(TARGET)
    assert channel.waypoints == []


# -- a command: refused before anything is sent (ADR-0026, ADR-0053) ----------------


def test_a_command_goes_as_one_waypoint_in_radians_and_move() -> None:
    adapter, channel, _ = connected()
    adapter.send_command(TARGET)
    assert [round(math.degrees(v), 9) for v in channel.waypoints[0].target] == list(TARGET.values)
    assert channel.started == 1


def test_a_target_outside_the_envelope_is_refused_by_name() -> None:
    adapter, channel, _ = connected()
    with pytest.raises(SafetyViolationError, match=r"joint3 = 95 при \[-90, 90\]"):
        adapter.send_command(goal(0, 0, 95, 0, 0, 0))
    assert channel.waypoints == [] and channel.started == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"safety": SafetyStatus.EMERGENCY_STOP},
        {"safety": SafetyStatus.SAFEGUARD_STOP},
        {"safety": SafetyStatus.RECOVERY},
        {"safety": 42},
        {"controller_state": ControllerState.ON},
    ],
)
def test_a_robot_that_cannot_move_refuses_the_command(changes: dict[str, int]) -> None:
    adapter, channel, _ = connected()
    channel.deliver(**changes)
    state = adapter.read_state()
    with pytest.raises(CommandRefusedError):
        adapter.send_command(TARGET)
    assert channel.waypoints == []
    assert state.motion is MotionPhase.IDLE  # nothing accepted, nothing to report


def test_reduced_is_a_working_status() -> None:
    adapter, channel, _ = connected()
    channel.deliver(safety=SafetyStatus.REDUCED)
    adapter.read_state()
    adapter.send_command(TARGET)
    assert len(channel.waypoints) == 1


def test_a_waypoint_the_controller_refuses_is_a_refusal_and_latches_nothing() -> None:
    adapter, channel, _ = connected()
    channel.refusal = "Waypoint can not be reached with current limits"
    with pytest.raises(WaypointRefusedError, match="current limits"):
        adapter.send_command(TARGET)
    channel.deliver()
    assert adapter.read_state().motion is MotionPhase.IDLE


def test_a_failed_start_drops_the_buffer() -> None:
    adapter, channel, _ = connected()
    channel.failing_start = True
    with pytest.raises(ChannelError):
        adapter.send_command(TARGET)
    assert channel.holds == 1


def test_a_command_of_another_quantity_or_width_is_an_error_not_a_refusal() -> None:
    adapter, _, _ = connected()
    with pytest.raises(ValueError, match="joint_position"):
        adapter.send_command(Command(type=ActionType.JOINT_VELOCITY, values=(0.0,) * 6))
    with pytest.raises(ValueError, match="5 суставов"):
        adapter.send_command(goal(0, 0, 0, 0, 0))


def test_use_before_connect_is_refused() -> None:
    with pytest.raises(NotConnectedError):
        RoboproRcAdapter().send_command(TARGET)


# -- the state and the phase (ADR-0049, ADR-0050) --------------------------------------


def test_state_is_in_the_contracts_unit_and_stamped_with_the_controllers_clock() -> None:
    adapter, channel, _ = connected()
    reading = channel.deliver(
        q=(math.pi / 2, 0.0, -math.pi / 4, 0.0, 0.0, math.pi),
        qd=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        sec=1_726_000_000,
        nsec=250_000_000,
    )
    state = adapter.read_state()
    assert state.joint_position == pytest.approx((90.0, 0.0, -45.0, 0.0, 0.0, 180.0))
    assert state.joint_velocity[0] == pytest.approx(math.degrees(0.1))
    assert state.timestamp == pytest.approx(1_726_000_000.25)
    assert state.timestamp == reading.stamp


def test_safety_and_the_controller_are_reported_in_extra() -> None:
    adapter, channel, _ = connected()
    channel.deliver(safety=SafetyStatus.EMERGENCY_STOP, state_flags=1, velocity_scale=0.3)
    extra = adapter.read_state().extra
    assert extra["safety"] == "emergency_stop"
    assert extra["controller_state"] == "run"
    assert extra["warning"] == "protective_stop"
    assert extra["velocity_scale"] == 0.3


def test_no_new_reading_returns_the_last_one_with_its_stamp() -> None:
    adapter, _, _ = connected(readings=1)
    first = adapter.read_state()
    assert adapter.read_state().timestamp == first.timestamp


def test_a_motion_is_moving_from_acceptance_to_the_controllers_end() -> None:
    adapter, channel, _ = connected(readings=1)
    assert adapter.read_state().motion is MotionPhase.IDLE
    adapter.send_command(TARGET)
    # Nothing new arrived yet: the old reading must not say "done".
    assert adapter.read_state().motion is MotionPhase.MOVING
    channel.deliver()  # the controller executes: mode move, one point in the buffer
    assert adapter.read_state().motion is MotionPhase.MOVING
    channel.finish_motion()
    assert adapter.read_state().motion is MotionPhase.IDLE


def test_a_stale_reading_from_before_the_point_was_taken_is_not_the_end() -> None:
    """Buffer empty and ``hold`` -- but the counter says the point was not seen yet."""
    adapter, channel, _ = connected(readings=1)
    before = channel.robot
    adapter.send_command(TARGET)
    channel.readings.append(before)
    assert adapter.read_state().motion is MotionPhase.MOVING


def test_a_second_command_during_a_motion_is_refused() -> None:
    adapter, channel, _ = connected()
    adapter.send_command(TARGET)
    with pytest.raises(MotionInProgressError):
        adapter.send_command(TARGET)
    assert len(channel.waypoints) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"motion_mode": MotionMode.PAUSE, "state_flags": 1},
        {"safety": SafetyStatus.SAFEGUARD_STOP},
        {"safety": SafetyStatus.EMERGENCY_STOP, "motion_mode": MotionMode.HOLD, "buffer_fill": 0},
        {"controller_state": ControllerState.ON, "motion_mode": MotionMode.HOLD, "buffer_fill": 0},
        {"motion_mode": MotionMode.ZERO_GRAVITY, "buffer_fill": 0},
    ],
)
def test_an_accepted_motion_that_cannot_execute_is_halted(changes: dict[str, int]) -> None:
    """A buffer dropped by a stop looks like the end; the status says it is not."""
    adapter, channel, _ = connected()
    adapter.send_command(TARGET)
    channel.deliver(**changes)
    assert adapter.read_state().motion is MotionPhase.HALTED


def test_without_our_motion_the_phase_is_what_the_robot_is_doing() -> None:
    adapter, channel, _ = connected()
    channel.deliver(motion_mode=MotionMode.MOVE, buffer_fill=2)
    assert adapter.read_state().motion is MotionPhase.MOVING
    channel.deliver(motion_mode=MotionMode.HOLD, buffer_fill=2)
    assert adapter.read_state().motion is MotionPhase.HALTED
    with pytest.raises(MotionInProgressError):
        adapter.send_command(TARGET)


def test_the_waypoint_counter_wraps_at_sixteen_bits() -> None:
    channel = recording_channel()
    channel.robot = standing()
    channel.deliver(waypoint_counter=0xFFFF)
    adapter, _, _ = connected(channel=channel, readings=0)
    adapter.send_command(TARGET)
    assert channel.robot.waypoint_counter == 0
    channel.finish_motion()
    assert adapter.read_state().motion is MotionPhase.IDLE


# -- the stop (ADR-0061) and the end of the link ------------------------------------------


def test_emergency_stop_is_hold_and_refuses_what_follows() -> None:
    adapter, channel, _ = connected()
    adapter.emergency_stop()
    assert channel.holds == 1 and adapter.stopped
    with pytest.raises(SafetyViolationError, match="reset"):
        adapter.send_command(TARGET)


def test_a_stopped_motion_stays_halted_until_reset() -> None:
    adapter, channel, _ = connected()
    adapter.send_command(TARGET)
    adapter.emergency_stop()
    channel.finish_motion()  # hold dropped the buffer: it looks like an end
    assert adapter.read_state().motion is MotionPhase.HALTED
    adapter.reset()
    channel.deliver()
    assert adapter.read_state().motion is MotionPhase.IDLE


def test_a_stop_with_the_link_gone_says_so_and_still_refuses() -> None:
    adapter, channel, _ = connected()
    adapter.send_command(TARGET)
    channel.gone = "SDK закрыл соединение"
    with pytest.raises(StopNotDeliveredError, match="ведёт контроллер"):
        adapter.emergency_stop()
    assert adapter.stopped
    with pytest.raises(SafetyViolationError):
        adapter.send_command(TARGET)


def test_a_lost_link_reads_as_silence_for_the_watchdog() -> None:
    """The executor's watchdog notices a stamp that stopped (ADR-0050, p. 3)."""
    adapter, channel, _ = connected()
    before = adapter.read_state()
    channel.gone = "SDK закрыл соединение"
    channel.deliver()
    assert adapter.read_state().timestamp == before.timestamp


def test_disconnect_revokes_an_unfinished_motion() -> None:
    adapter, channel, _ = connected()
    adapter.send_command(TARGET)
    adapter.disconnect()
    assert channel.holds == 1


def test_disconnect_after_the_end_sends_nothing() -> None:
    adapter, channel, _ = connected()
    adapter.send_command(TARGET)
    channel.finish_motion()
    adapter.read_state()
    adapter.disconnect()
    assert channel.holds == 0


def test_disconnect_on_a_dead_link_does_not_raise_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = recording_channel()
    owned.deliver()
    monkeypatch.setattr(adapter_module, "open_sdk", lambda link: owned)
    adapter = RoboproRcAdapter()
    adapter.connect(profile(), dict(LINK_PARAMS))
    adapter.send_command(TARGET)
    owned.gone = "обрыв"
    adapter.disconnect()
    assert owned.closed and owned.holds == 0


# -- what the adapter declares ------------------------------------------------------------


def test_requirements_name_the_vendors_api_by_its_version() -> None:
    by_name = {item.name: item for item in RoboproRcAdapter.requirements}
    assert set(by_name) == {"API", "numpy", "scipy"}
    assert all(
        item.kind is RequirementKind.PYTHON and not item.forbidden
        for item in by_name.values()
    )
    assert by_name["API"].version == "==1.4.1"
    assert by_name["API"].admits("1.4.1") and not by_name["API"].admits("1.4.0")


def test_the_class_declares_that_it_executes_the_scale() -> None:
    assert RoboproRcAdapter.supports_speed_scale is True


def test_it_is_a_command_link() -> None:
    adapter, _, _ = connected()
    assert adapter.control_mode() is ControlMode.COMMAND
