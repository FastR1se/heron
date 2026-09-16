"""The training environment is derived from the contract, and refuses what it cannot derive.

Everything here runs on a machine with no GPU and no simulator, which is the
whole claim of ADR-0044: the environment is data, so it is checkable before an
hour of training rather than after it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_policy_contract, make_profile, write_urdf
from heron.core.contract import ActionSpec, ObservationTerm
from heron.core.enums import ActionOffset, ActionType, ObservationType
from heron.core.models import Completion, RobotProfile, SkillSpec
from heron.train.spec import SUPPORTED_TERMS, decimation_of, derive, description_text
from heron.validator.train import check_contract, check_goal_range

JOINTS = ["joint0", "joint1", "joint2"]
COMPLETION = Completion(tolerance=0.05, hold_s=0.2, horizon_s=4.0)
GAINS = {"kp": [10.0, 20.0, 30.0], "kd": [1.0, 2.0, 3.0]}


def trainable_contract(**overrides: object) -> object:
    """A contract an environment can be built from: mapped action, gains, goal term."""
    action = ActionSpec(
        type=ActionType.JOINT_POSITION,
        clip=(-1.0, 1.0),
        scale=[0.5, 0.5, 0.5],
        offset=ActionOffset.DEFAULT_POSE,
        **GAINS,
    )
    fields: dict[str, object] = {"action": action}
    fields.update(overrides)
    return make_policy_contract(3, **fields)


@pytest.fixture
def profile() -> RobotProfile:
    return make_profile(contract=trainable_contract())


@pytest.fixture
def description(tmp_path: Path) -> Path:
    return write_urdf(tmp_path, "test_arm", JOINTS)


def build(profile: RobotProfile, description: Path) -> object:
    return derive(
        scenario="bench",
        robot="arm1",
        backend="isaaclab",
        task="reach",
        profile=profile,
        description=description,
        skill=SkillSpec(
            name="reach", required_capabilities=profile.capabilities
        ),
        completion=COMPLETION,
    )


# -- what is derived --------------------------------------------------------


def test_rates_become_a_step_and_a_decimation(profile: RobotProfile, description: Path) -> None:
    """100 Hz of state is the physics step; 50 Hz of commands is one tick in two steps."""
    spec = build(profile, description)
    assert spec.dt == pytest.approx(0.01)
    assert spec.decimation == 2
    assert spec.policy_rate_hz == 50.0


def test_the_action_mapping_is_resolved_per_joint(
    profile: RobotProfile, description: Path
) -> None:
    """Unstated defaults are written down once here, not re-invented by every reader."""
    spec = build(profile, description)
    assert spec.action.scale == [0.5, 0.5, 0.5]
    assert spec.action.offset == list(profile.contract.default_pose or [])
    assert spec.action.clip == 1.0


def test_an_unstated_scale_is_derived_as_ones(description: Path) -> None:
    contract = trainable_contract(
        action=ActionSpec(type=ActionType.JOINT_POSITION, clip=(-1.0, 1.0), **GAINS)
    )
    spec = build(make_profile(contract=contract), description)
    assert spec.action.scale == [1.0, 1.0, 1.0]
    assert spec.action.offset == [0.0, 0.0, 0.0]


def test_observation_widths_are_derived_not_declared(
    profile: RobotProfile, description: Path
) -> None:
    spec = build(profile, description)
    assert [slot.type for slot in spec.observation] == [
        term.type for term in profile.contract.observation
    ]
    assert [slot.size for slot in spec.observation] == [3, 3, 3, 3]
    assert spec.observation_size == 12
    assert spec.action_size == 3


def test_the_drive_comes_from_the_contract(profile: RobotProfile, description: Path) -> None:
    spec = build(profile, description)
    assert spec.drive.kp == GAINS["kp"]
    assert spec.drive.kd == GAINS["kd"]


def test_the_spec_agrees_with_the_contract_it_records(
    profile: RobotProfile, description: Path
) -> None:
    """A derivation recorded next to its source must not drift from it."""
    spec = build(profile, description)
    contract = spec.contract
    assert spec.contract_id == contract.contract_id
    assert spec.joint_order == contract.joint_order
    assert spec.action.scale == list(contract.action_scale())
    assert spec.action.offset == list(contract.origin())
    assert spec.observation_size == contract.observation_size
    assert spec.action_size == contract.action_size
    assert spec.dt * spec.decimation == pytest.approx(1.0 / contract.command_rate_hz)


def test_the_spec_round_trips_through_json(profile: RobotProfile, description: Path) -> None:
    """It is written to disk before the process starts, and read back by it."""
    from heron.train.spec import TrainSpec

    spec = build(profile, description)
    again = TrainSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert again.contract.contract_id == spec.contract_id


def test_a_free_base_is_derived_from_the_description(
    profile: RobotProfile, description: Path
) -> None:
    """The platform does not fix a base; it reads whether the description did (ADR-0048)."""
    assert build(profile, description).fix_base is False


def test_a_fixed_base_is_derived_from_the_description(
    profile: RobotProfile, tmp_path: Path
) -> None:
    anchored = write_urdf(
        tmp_path,
        "test_arm",
        JOINTS,
        extra=(
            '  <link name="world"/>\n'
            '  <joint name="to_world" type="fixed">\n'
            '    <parent link="world"/>\n'
            '    <child link="base_link"/>\n'
            "  </joint>\n"
        ),
    )
    assert build(profile, anchored).fix_base is True


def test_the_plan_is_printable(profile: RobotProfile, description: Path) -> None:
    """A dry run shows a human what was derived, before anything costs GPU time."""
    lines = "\n".join(build(profile, description).lines())
    assert "decimation 2" in lines
    assert "joint_pos_rel" in lines


# -- what is refused --------------------------------------------------------


def test_rates_that_do_not_divide_are_refused() -> None:
    contract = trainable_contract(state_rate_hz=100.0, command_rate_hz=30.0)
    assert decimation_of(contract) is None
    report = check_contract(contract, where="x")
    assert "train.rates_not_integer" in report.codes()


def test_decimal_rates_that_do_divide_are_accepted() -> None:
    """100/50 must not depend on how a binary float rounds."""
    assert decimation_of(trainable_contract(state_rate_hz=500.0, command_rate_hz=50.0)) == 10


def test_an_action_without_gains_is_refused() -> None:
    """A policy learned against one stiffness is not the same policy against another."""
    contract = trainable_contract(
        action=ActionSpec(type=ActionType.JOINT_POSITION, clip=(-1.0, 1.0))
    )
    assert "train.gains_missing" in check_contract(contract, where="x").codes()


def test_an_action_that_is_not_per_joint_is_refused() -> None:
    contract = trainable_contract(
        action=ActionSpec(type=ActionType.CARTESIAN_POSE, clip=(-1.0, 1.0))
    )
    codes = check_contract(contract, where="x").codes()
    assert "train.action_unsupported" in codes
    assert "train.gains_missing" not in codes  # one refusal per problem


def test_an_asymmetric_clip_is_refused() -> None:
    """Training bounds the raw output with one number, so a lopsided clip is not
    reproducible there -- and a clip that differs between worlds is the failure
    invariant 6 exists against."""
    contract = trainable_contract(
        action=ActionSpec(
            type=ActionType.JOINT_POSITION,
            clip=(-0.5, 1.0),
            scale=[0.5, 0.5, 0.5],
            offset=ActionOffset.DEFAULT_POSE,
            **GAINS,
        )
    )
    assert "train.clip_not_symmetric" in check_contract(contract, where="x").codes()


def test_every_term_of_the_registry_is_supported_today() -> None:
    """A term nobody can build would be a silently shorter observation vector."""
    assert frozenset(ObservationType) == SUPPORTED_TERMS


def test_a_clean_contract_is_refused_by_nothing(profile: RobotProfile) -> None:
    assert check_contract(profile.contract, where="x").ok


def test_derive_refuses_what_check_contract_refuses(description: Path) -> None:
    contract = trainable_contract(
        action=ActionSpec(type=ActionType.JOINT_POSITION, clip=(-1.0, 1.0))
    )
    with pytest.raises(ValueError, match=r"train\.gains_missing"):
        build(make_profile(contract=contract), description)


# -- the goal a task draws from (ADR-0043) ----------------------------------


def test_a_goal_inside_both_bounds_passes(profile: RobotProfile) -> None:
    pose = profile.contract.default_pose or []
    report = check_goal_range(
        [value - 0.2 for value in pose], [value + 0.2 for value in pose], profile, where="task"
    )
    assert report.ok, report.codes()


def test_a_goal_outside_the_envelope_is_refused(profile: RobotProfile) -> None:
    report = check_goal_range([-9.0, -9.0, -9.0], [9.0, 9.0, 9.0], profile, where="task")
    assert "train.goal_outside_limits" in report.codes()


def test_a_goal_the_action_cannot_reach_is_refused(profile: RobotProfile) -> None:
    """Inside the envelope and outside what ``clip * scale`` can command: the policy
    would be punished for missing a goal no output of it maps to."""
    pose = profile.contract.default_pose or []
    report = check_goal_range(
        [value - 1.5 for value in pose], [value + 1.5 for value in pose], profile, where="task"
    )
    assert "train.goal_unreachable" in report.codes()
    assert "train.goal_outside_limits" not in report.codes()


def test_a_goal_of_the_wrong_width_is_refused(profile: RobotProfile) -> None:
    assert "train.goal_range_width" in check_goal_range([0.0], [1.0], profile, where="t").codes()


def test_a_backwards_goal_range_is_refused(profile: RobotProfile) -> None:
    codes = check_goal_range([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], profile, where="t").codes()
    assert "train.goal_range_width" in codes


# -- the description the converter reads (ADR-0048) -------------------------


def test_the_copy_anchors_meshes_and_drops_control(
    profile: RobotProfile, tmp_path: Path
) -> None:
    """A copy read from a run directory resolves its meshes against the input."""
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "link.stl").write_text("solid\n", encoding="utf-8")
    description = write_urdf(
        tmp_path,
        "test_arm",
        JOINTS,
        extra=(
            '  <visual><geometry><mesh filename="../meshes/link.stl"/></geometry></visual>\n'
            '  <ros2_control name="left_over" type="system"/>\n'
        ),
    )
    text = description_text(profile, description)
    assert "file:///" in text
    assert "../meshes/link.stl" not in text
    assert "ros2_control" not in text
    assert description.read_text(encoding="utf-8").count("../meshes/link.stl") == 1


def test_the_input_is_never_touched(profile: RobotProfile, description: Path) -> None:
    before = description.read_bytes()
    description_text(profile, description)
    assert description.read_bytes() == before


# -- the term the observation carries the goal in ---------------------------


def test_a_contract_without_a_goal_term_still_derives(description: Path) -> None:
    """The environment builds; it is ``LearnedSkill`` that refuses a policy whose
    completion is undefined (``policy.completion_undefined``)."""
    contract = trainable_contract(
        observation=[ObservationTerm(type=ObservationType.JOINT_POS_REL)]
    )
    spec = build(make_profile(contract=contract), description)
    assert spec.observation_size == 3
