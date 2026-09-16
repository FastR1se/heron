"""``goto_joints``: one motion, for a robot that plans it itself (ADR-0049, ADR-0050)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from conftest import make_command_profile, make_profile
from heron.adapters.dummy import DummyAdapter
from heron.core.enums import ActionType, MotionPhase, Status
from heron.core.models import RobotProfile
from heron.core.types import Command, RobotState
from heron.sdk.executor import run_skill
from heron.sdk.runner import IncompatibleSkillError, check_runnable
from heron.skills.goto_joints import GotoJointsSkill
from heron.skills.library import SkillLibrary
from heron.skills.move_joints import MoveJointsSkill


@dataclass
class Context:
    profile: RobotProfile
    sent: list[Command] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)

    def send(self, cmd: Command) -> None:
        self.sent.append(cmd)

    def now(self) -> float:
        return 0.0

    def log(self, message: str, **fields: Any) -> None:
        self.log_lines.append(" ".join([message, *(f"{k}={v}" for k, v in fields.items())]))


def _started(goal: dict[str, Any] | None = None) -> tuple[GotoJointsSkill, Context]:
    skill = GotoJointsSkill()
    context = Context(make_command_profile())
    skill.start(goal or {"target": [1.0, 0.0, 0.0], "tolerance": 0.01}, context)
    return skill, context


def _state(motion: MotionPhase | None, *pose: float) -> RobotState:
    return RobotState(timestamp=0.0, joint_position=pose or (0.0, 0.0, 0.0), motion=motion)


def _connected(**config: Any) -> tuple[DummyAdapter, RobotProfile]:
    profile = make_command_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, config)
    return adapter, profile


def test_the_robot_arrives_with_one_command() -> None:
    adapter, profile = _connected()
    goal = {"target": [1.0, -0.5, 0.25], "tolerance": 1e-6}

    status, seen = run_skill(GotoJointsSkill(), adapter, profile, goal)

    assert status is Status.SUCCESS
    assert adapter.commands == [Command(ActionType.JOINT_POSITION, (1.0, -0.5, 0.25))]
    assert seen[-1].joint_position == pytest.approx((1.0, -0.5, 0.25))
    assert MotionPhase.MOVING in {state.motion for state in seen}


def test_the_skill_is_in_the_library() -> None:
    assert SkillLibrary().spec("goto_joints") is GotoJointsSkill.spec


@pytest.mark.parametrize(
    "goal",
    [{"target": [1.0, 0.0, 0.0]}, {"target": [1.0, 0.0, 0.0], "tolerance": 0}],
    ids=["no tolerance", "zero tolerance"],
)
def test_a_tolerance_is_required_and_positive(goal: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        _started(goal)


def test_a_target_outside_the_envelope_is_refused_before_anything_moves() -> None:
    with pytest.raises(ValueError, match="вне лимитов"):
        _started({"target": [9.0, 0.0, 0.0], "tolerance": 0.01})


def test_a_link_without_phases_fails_before_commanding() -> None:
    skill, context = _started()
    assert skill.tick(_state(None)) is Status.FAILURE
    assert not context.sent
    assert "фазу движения" in context.log_lines[0]


@pytest.mark.parametrize("phase", [MotionPhase.MOVING, MotionPhase.HALTED])
def test_a_robot_already_moving_is_not_commanded(phase: MotionPhase) -> None:
    skill, context = _started()
    assert skill.tick(_state(phase)) is Status.FAILURE
    assert not context.sent


def test_one_command_then_only_reading() -> None:
    skill, context = _started()
    assert skill.tick(_state(MotionPhase.IDLE)) is Status.RUNNING
    for _ in range(3):
        assert skill.tick(_state(MotionPhase.MOVING, 0.5, 0.0, 0.0)) is Status.RUNNING
    assert len(context.sent) == 1


def test_a_halted_motion_fails() -> None:
    skill, context = _started()
    skill.tick(_state(MotionPhase.IDLE))
    assert skill.tick(_state(MotionPhase.HALTED, 0.5, 0.0, 0.0)) is Status.FAILURE
    assert "остановлено" in context.log_lines[-1]


def test_a_motion_over_outside_the_tolerance_fails_naming_the_joints() -> None:
    """A buffer emptied halfway looks like a finished motion: the pose decides arrival."""
    skill, context = _started()
    skill.tick(_state(MotionPhase.IDLE))
    assert skill.tick(_state(MotionPhase.IDLE, 0.5, 0.0, 0.0)) is Status.FAILURE
    assert "joint0 0.5" in context.log_lines[-1]


def test_a_motion_over_within_the_tolerance_succeeds() -> None:
    skill, _ = _started()
    skill.tick(_state(MotionPhase.IDLE))
    assert skill.tick(_state(MotionPhase.IDLE, 0.995, 0.0, 0.0)) is Status.SUCCESS


def test_an_emergency_stop_mid_motion_ends_in_failure() -> None:
    adapter, _ = _connected()
    skill, context = _started({"target": [2.0, 0.0, 0.0], "tolerance": 0.01})
    assert skill.tick(adapter.read_state()) is Status.RUNNING
    adapter.send_command(context.sent[0])
    adapter.read_state()
    adapter.emergency_stop()
    assert skill.tick(adapter.read_state()) is Status.FAILURE


def test_the_speed_scale_slows_the_run() -> None:
    goal = {"target": [2.0, 0.0, 0.0], "tolerance": 1e-6}
    full, profile = _connected()
    slow, _ = _connected(speed_scale=0.5)
    _, fast_states = run_skill(GotoJointsSkill(), full, profile, goal)
    _, slow_states = run_skill(GotoJointsSkill(), slow, profile, goal)
    # Each end is read to one state period (20 Hz).
    assert slow_states[-1].timestamp == pytest.approx(2 * fast_states[-1].timestamp, abs=0.15)


# -- modes compare by equality (ADR-0049) --------------------------------------


def test_a_command_skill_is_refused_on_a_streaming_robot_and_link() -> None:
    profile = make_profile(skills=("goto_joints",))
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    report = check_runnable(GotoJointsSkill(), profile, adapter)

    found = {item.code: item.detail for item in report.errors}
    expected = {"required": "command", "actual": "streaming"}
    assert found["skill.control_mode_incompatible"] == expected
    assert found["adapter.control_mode_incompatible"] == expected
    with pytest.raises(IncompatibleSkillError):
        run_skill(GotoJointsSkill(), adapter, profile, {"target": [0.0] * 3, "tolerance": 1})
    assert not adapter.commands


def test_a_streaming_skill_is_refused_on_a_command_link() -> None:
    adapter, profile = _connected()
    report = check_runnable(MoveJointsSkill(), profile, adapter)
    found = {item.code: item.detail for item in report.errors}
    assert found["adapter.control_mode_incompatible"] == {
        "required": "streaming",
        "actual": "command",
    }
