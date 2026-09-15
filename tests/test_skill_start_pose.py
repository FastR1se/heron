"""A skill starting from a pose that rests past a limit (ADR-0032).

The Linux run of stage 4 (journal 2026-09-15): the vendor's twin let the robot
fall before the first command, five joints came to rest a few thousandths of a
radian past their stops, and ``move_joints`` commanded the pose it measured. The
adapter refused, rightly, and the skill never started. The dummy adapter shows the
same thing without a simulator: its initial position is not checked, its commands
are.
"""

from __future__ import annotations

import pytest

from conftest import make_profile
from heron.adapters.dummy import DummyAdapter, SafetyViolationError
from heron.core.enums import ActionType, Status
from heron.core.types import Command
from heron.sdk.executor import run_skill
from heron.skills.base import admissible
from heron.skills.hold import HoldSkill
from heron.skills.move_joints import MoveJointsSkill

RESTING = [-2.003, 0.0, 2.0005]
"""Limits are ±2: two joints past their stops by as much as the fallen robot's."""


def _fallen() -> tuple[object, DummyAdapter]:
    profile = make_profile()
    adapter = DummyAdapter()
    adapter.connect(profile, {"initial_position": RESTING})
    return profile, adapter


def test_the_measured_pose_itself_is_a_command_the_adapter_refuses() -> None:
    _, adapter = _fallen()
    with pytest.raises(SafetyViolationError, match="вне лимитов"):
        adapter.send_command(Command(type=ActionType.JOINT_POSITION, values=tuple(RESTING)))


def test_move_joints_starts_from_the_nearest_admissible_pose() -> None:
    profile, adapter = _fallen()
    status, _ = run_skill(
        MoveJointsSkill(), adapter, profile, {"target": [0.0, 0.5, 1.0], "duration": 0.2}
    )

    assert status is Status.SUCCESS
    assert adapter.commands[0].values == pytest.approx((-2.0, 0.0, 2.0))
    assert adapter.read_state().joint_position == pytest.approx((0.0, 0.5, 1.0))


def test_hold_holds_the_nearest_admissible_pose() -> None:
    profile, adapter = _fallen()
    status, _ = run_skill(HoldSkill(), adapter, profile, {"duration": 0.05})

    assert status is Status.SUCCESS
    assert {command.values for command in adapter.commands} == {(-2.0, 0.0, 2.0)}


def test_a_pose_inside_the_envelope_is_left_as_it_is() -> None:
    profile = make_profile()
    assert admissible(profile, (0.1, -0.2, 1.9)) == ((0.1, -0.2, 1.9), [])


def test_every_joint_moved_is_named_with_both_values() -> None:
    profile = make_profile()
    _, moved = admissible(profile, RESTING)
    assert moved == ["joint0 -2.003 -> -2", "joint2 2.0005 -> 2"]
