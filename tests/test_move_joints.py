"""``move_joints``: the first skill that actually moves something."""

from __future__ import annotations

import pytest

from conftest import make_profile
from heron.adapters.dummy import DummyAdapter
from heron.core.enums import ActionType, Status
from heron.sdk.executor import Executor, ManualClock, run_skill
from heron.skills.library import SkillLibrary
from heron.skills.move_joints import MoveJointsSkill


def test_the_robot_ends_up_where_the_goal_says() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    status, seen = run_skill(
        MoveJointsSkill(), adapter, profile, {"target": [0.5, -0.5, 0.25], "duration": 0.5}
    )

    assert status is Status.SUCCESS
    assert adapter.read_state().joint_position == pytest.approx((0.5, -0.5, 0.25))
    assert len(seen) > 1


def test_the_motion_is_a_ramp_and_not_a_step() -> None:
    """A group position controller forwards what it is given, so a single
    command at the target would ask the joint to teleport."""
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    run_skill(MoveJointsSkill(), adapter, profile, {"target": [1.0, 0.0, 0.0], "duration": 0.5})

    first = adapter.commands[0].values[0]
    middle = adapter.commands[len(adapter.commands) // 2].values[0]
    assert first == pytest.approx(0.0, abs=1e-9)
    assert 0.2 < middle < 0.8, f"середина движения на {middle}, а не примерно посередине"
    assert adapter.commands[-1].values[0] == pytest.approx(1.0)
    assert all(cmd.type is ActionType.JOINT_POSITION for cmd in adapter.commands)


def test_the_ramp_starts_where_the_robot_is_not_where_it_was_commanded() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {"initial_position": [0.2, 0.2, 0.2]})

    run_skill(MoveJointsSkill(), adapter, profile, {"target": [0.4, 0.2, 0.2], "duration": 0.2})
    assert adapter.commands[0].values[0] == pytest.approx(0.2)


def test_a_target_of_the_wrong_width_is_refused_before_anything_moves() -> None:
    profile = make_profile(dof=3)
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    with pytest.raises(ValueError, match="суставов"):
        run_skill(MoveJointsSkill(), adapter, profile, {"target": [0.0, 0.0]})
    assert not adapter.commands


def test_a_target_outside_the_limits_is_refused_before_anything_moves() -> None:
    """Invariant 6: not discovered halfway through, with the arm left mid-swing."""
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    with pytest.raises(ValueError, match="лимитов"):
        run_skill(MoveJointsSkill(), adapter, profile, {"target": [99.0, 0.0, 0.0]})
    assert not adapter.commands


def test_a_goal_without_a_target_says_what_is_missing() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    with pytest.raises(ValueError, match="target"):
        run_skill(MoveJointsSkill(), adapter, profile, {})


def test_a_duration_of_zero_is_a_step_and_is_refused() -> None:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    with pytest.raises(ValueError, match="ступень"):
        run_skill(
            MoveJointsSkill(), adapter, profile, {"target": [0.1, 0.0, 0.0], "duration": 0.0}
        )


def test_the_skill_is_in_the_library_under_its_name() -> None:
    spec = SkillLibrary().spec("move_joints")
    assert spec.name == "move_joints"
    assert "arm.joint_position" in spec.required_capabilities


def test_pacing_does_not_change_where_the_robot_ends_up() -> None:
    """The same skill, the same goal, a real period instead of none."""
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    result = Executor(adapter, profile, clock=ManualClock()).run(
        MoveJointsSkill(), {"target": [0.5, -0.5, 0.25], "duration": 0.5}
    )
    assert result.ok
    assert adapter.read_state().joint_position == pytest.approx((0.5, -0.5, 0.25))
