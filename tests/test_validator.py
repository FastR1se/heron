"""The validator catches mismatch before a run (invariant 6).

Findings are asserted by ``code``, never by wording: the code is the promise,
the message is prose. Experiment 4 of section 13 will count caught errors the
same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_contract, make_policy_contract, make_profile
from heron.core.capabilities import Capability
from heron.core.contract import ActionSpec, Contract, ObservationTerm
from heron.core.enums import ActionOffset, ActionType, ControlMode, ObservationType
from heron.core.models import (
    AdapterSpec,
    Completion,
    JointMax,
    JointRange,
    Limits,
    OnnxSignature,
    PolicyManifest,
    RobotProfile,
    Scenario,
    SkillSpec,
)
from heron.core.requires import Requirements
from heron.registry.lock import ContractLock
from heron.registry.registry import Registry
from heron.validator.rules import (
    check_policy_against_profile,
    check_profile,
    check_scenario,
    check_skill_against_profile,
)


def codes(report: object) -> list[str]:
    return sorted(report.codes())  # type: ignore[attr-defined]


# -- profile ---------------------------------------------------------------


def test_clean_profile_has_no_errors() -> None:
    assert check_profile(make_profile()).ok


def test_limits_must_match_joint_count() -> None:
    profile = make_profile(
        dof=3,
        limits=Limits(joint_position=JointRange(min=[-1.0, -1.0], max=[1.0, 1.0])),
    )
    report = check_profile(profile)
    assert not report.ok
    assert "limits.length_mismatch" in report.codes()


def test_image_namespace_is_not_stored_in_data() -> None:
    """Section 17: a rename must not have to rewrite user profiles."""
    report = check_profile(make_profile(image="heron/robot-arm:latest"))
    assert "profile.image_has_namespace" in report.codes()
    assert not report.ok


def test_bare_image_name_is_fine() -> None:
    assert check_profile(make_profile(image="robot-arm:latest")).ok


def test_commands_faster_than_state_is_a_warning() -> None:
    profile = make_profile(contract=make_contract(state_rate_hz=10.0, command_rate_hz=50.0))
    report = check_profile(profile)
    assert report.ok  # suspicious, not forbidden
    assert "contract.command_faster_than_state" in report.codes()


def test_uninstalled_adapter_plugin_is_a_warning_not_an_error() -> None:
    """CI runs without vendor packages by design (section 10).

    The plugin named here is deliberately one no package registers: the built-in
    ones are installed by this very repository, so using one would test nothing.
    """
    profile = make_profile(adapters={"sim.gazebo": AdapterSpec(plugin="vendor_sdk")})
    report = check_profile(profile)
    assert report.ok
    assert "adapter.plugin_not_installed" in report.codes()


# -- contract lock (ADR-0004) ----------------------------------------------


def test_silent_contract_change_is_an_error(tmp_path: Path) -> None:
    profile = make_profile()
    lock = ContractLock(tmp_path / "contracts.lock.json")
    lock.record(profile.name, profile.contract)

    reordered = make_profile(
        contract=make_contract(joint_order=["joint1", "joint0", "joint2"])
    )
    report = check_profile(reordered, lock=lock)
    assert not report.ok
    assert "contract.undeclared_change" in report.codes()


def test_declared_contract_change_passes(tmp_path: Path) -> None:
    profile = make_profile()
    lock = ContractLock(tmp_path / "contracts.lock.json")
    lock.record(profile.name, profile.contract)

    declared = make_profile(
        contract=make_contract(joint_order=["joint1", "joint0", "joint2"], revision=2)
    )
    assert check_profile(declared, lock=lock).ok


def test_unchanged_contract_passes(tmp_path: Path) -> None:
    profile = make_profile()
    lock = ContractLock(tmp_path / "contracts.lock.json")
    lock.record(profile.name, profile.contract)
    assert check_profile(make_profile(), lock=lock).ok


def test_lock_round_trips(tmp_path: Path) -> None:
    profile = make_profile()
    lock = ContractLock(tmp_path / "contracts.lock.json")
    lock.record(profile.name, profile.contract)
    lock.save()

    reloaded = ContractLock.load(tmp_path)
    entry = reloaded.get(profile.name)
    assert entry is not None
    assert entry.contract_id == profile.contract.contract_id


# -- skill against robot ---------------------------------------------------


def test_missing_capability_is_an_error() -> None:
    skill = SkillSpec(
        name="pick",
        required_capabilities=frozenset(
            {Capability("arm.joint_position"), Capability("gripper.binary")}
        ),
    )
    report = check_skill_against_profile(skill, make_profile())
    assert not report.ok
    assert "skill.capabilities_missing" in report.codes()


def test_streaming_skill_rejected_on_command_robot() -> None:
    """Section 3: a policy with a fixed-rate loop is unrunnable here."""
    skill = SkillSpec(
        name="pick",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        required_control_mode=ControlMode.STREAMING,
    )
    profile = make_profile(control_mode=ControlMode.COMMAND)
    report = check_skill_against_profile(skill, profile)
    assert "skill.control_mode_incompatible" in report.codes()


def test_command_skill_runs_on_streaming_robot() -> None:
    skill = SkillSpec(
        name="pick",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        required_control_mode=ControlMode.COMMAND,
    )
    assert check_skill_against_profile(skill, make_profile()).ok


def test_quantitative_requirement_unmet() -> None:
    skill = SkillSpec(
        name="pick",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        requires=Requirements({"dof": ">=6"}),
    )
    report = check_skill_against_profile(skill, make_profile(dof=3))
    assert not report.ok
    assert "skill.requirement_unmet" in report.codes()


def test_absent_quantity_fails_rather_than_passes() -> None:
    """"Unknown" and "satisfied" must not look alike (ADR-0001)."""
    skill = SkillSpec(
        name="pick",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        requires=Requirements({"gripper_width_m": ">=0.06"}),
    )
    profile = make_profile(limits=Limits(joint_velocity=JointMax(max=[1.0, 1.0, 1.0])))
    report = check_skill_against_profile(skill, profile)
    assert "skill.requirement_unmet" in report.codes()
    assert report.errors[0].detail["actual"] is None


def test_requirement_met() -> None:
    skill = SkillSpec(
        name="pick",
        required_capabilities=frozenset({Capability("arm.joint_position")}),
        requires=Requirements({"dof": ">=3", "command_rate_hz": ">=50"}),
    )
    assert check_skill_against_profile(skill, make_profile(dof=3)).ok


def test_unknown_quantity_rejected() -> None:
    with pytest.raises(ValueError, match="неизвестная величина"):
        Requirements({"payload_kg": ">=5"})


def test_expressions_rejected() -> None:
    """ADR-0001 forbids growing the language beyond three comparisons."""
    with pytest.raises(ValueError, match="сравнение"):
        Requirements({"dof": ">=6 and <=8"})


# -- policy against robot --------------------------------------------------


def make_manifest(**overrides: object) -> PolicyManifest:
    contract = overrides.pop("contract", make_policy_contract())
    assert isinstance(contract, Contract)
    fields: dict[str, object] = {
        "schema_version": 1,
        "name": "reach_v1",
        "contract": contract,
        "contract_id": contract.contract_id,
        "capabilities": frozenset({Capability("arm.joint_position")}),
        "completion": Completion(tolerance=0.01, hold_s=0.05, horizon_s=2.0),
        "onnx": OnnxSignature(
            sha256="0" * 64,
            input="obs",
            output="actions",
            observation_size=contract.observation_size or 1,
            action_size=contract.action_size or 1,
            opset=18,
        ),
    }
    fields.update(overrides)
    return PolicyManifest(**fields)  # type: ignore[arg-type]


def trained_for() -> RobotProfile:
    """A robot whose contract is the one the manifests above were made against."""
    return make_profile(contract=make_policy_contract())


def test_matching_policy_passes() -> None:
    assert check_policy_against_profile(make_manifest(), trained_for()).ok


def test_policy_with_reordered_joints_is_rejected_by_field() -> None:
    """The mismatch is reported by field, not as "hashes differ" (ADR-0004)."""
    manifest = make_manifest(
        contract=make_policy_contract(joint_order=["joint1", "joint0", "joint2"])
    )
    report = check_policy_against_profile(manifest, trained_for())
    assert not report.ok
    assert "contract.field_mismatch" in report.codes()
    assert report.errors[0].detail["field"] == "joint_order"


def test_tampered_manifest_is_rejected() -> None:
    """The identifier is computed, so editing it cannot fake compatibility."""
    manifest = make_manifest(contract_id="0000000000000000")
    report = check_policy_against_profile(manifest, trained_for())
    assert "policy.manifest_inconsistent" in report.codes()


def test_recorded_widths_are_held_against_the_contract() -> None:
    """Widths are derived, not declared: a manifest that declares another one lies."""
    contract = make_policy_contract()
    manifest = make_manifest(
        contract=contract,
        onnx=OnnxSignature(
            sha256="0" * 64,
            input="obs",
            output="actions",
            observation_size=(contract.observation_size or 0) + 1,
            action_size=contract.action_size or 1,
            opset=18,
        ),
    )
    report = check_policy_against_profile(manifest, trained_for())
    assert "policy.manifest_inconsistent" in report.codes()


# -- what the one implementation of a trained skill cannot execute (ADR-0047) --


def test_a_policy_over_the_contract_of_a_reaching_task_passes() -> None:
    contract = make_policy_contract()
    profile = make_profile(contract=contract)
    assert check_policy_against_profile(make_manifest(contract=contract), profile).ok


def test_a_policy_without_a_goal_term_has_no_completion() -> None:
    contract = make_policy_contract(
        observation=[ObservationTerm(type=ObservationType.JOINT_POS_REL)]
    )
    report = check_policy_against_profile(
        make_manifest(contract=contract), make_profile(contract=contract)
    )
    assert "policy.completion_undefined" in report.codes()


def test_a_policy_observing_the_base_velocity_is_rejected() -> None:
    """``RobotState`` does not carry it, and inventing it would be a lie (ADR-0042)."""
    contract = make_policy_contract(
        observation=[
            ObservationTerm(type=ObservationType.BASE_LIN_VEL),
            ObservationTerm(type=ObservationType.TARGET_JOINT_POS),
        ]
    )
    report = check_policy_against_profile(
        make_manifest(contract=contract), make_profile(contract=contract)
    )
    finding = next(f for f in report.errors if f.code == "policy.observation_unavailable")
    assert finding.detail["terms"] == ["base_lin_vel"]


def test_a_policy_that_does_not_command_positions_is_rejected_in_this_phase() -> None:
    contract = make_policy_contract(
        action=ActionSpec(type=ActionType.JOINT_VELOCITY, clip=(-1.0, 1.0))
    )
    report = check_policy_against_profile(
        make_manifest(contract=contract), make_profile(contract=contract)
    )
    assert "policy.action_unsupported" in report.codes()


# -- what a policy may command (ADR-0041) ----------------------------------


def test_targets_a_policy_can_reach_lie_inside_the_envelope() -> None:
    contract = make_policy_contract()
    profile = make_profile(contract=contract)  # limits +-2, targets 0.1 +- 0.5
    assert check_profile(profile).ok


def test_a_scale_that_reaches_past_the_envelope_is_an_error() -> None:
    contract = make_policy_contract(
        action=ActionSpec(
            type=ActionType.JOINT_POSITION,
            clip=(-1.0, 1.0),
            scale=[3.0, 0.5, 0.5],
            offset=ActionOffset.DEFAULT_POSE,
        )
    )
    report = check_profile(make_profile(contract=contract))
    finding = next(f for f in report.errors if f.code == "contract.action_outside_limits")
    assert list(finding.detail["joints"]) == ["joint0"]
    assert finding.detail["joints"]["joint0"]["targets"] == [-2.9, 3.1]


def test_a_mapping_without_a_clip_leaves_the_targets_unbounded() -> None:
    """The output of a trained network is not bounded by anything else (ADR-0041)."""
    contract = make_policy_contract(
        action=ActionSpec(
            type=ActionType.JOINT_POSITION,
            scale=[0.5, 0.5, 0.5],
            offset=ActionOffset.DEFAULT_POSE,
        )
    )
    report = check_profile(make_profile(contract=contract))
    assert "contract.action_outside_limits" in report.codes()


def test_without_an_envelope_there_is_nothing_to_check() -> None:
    contract = make_policy_contract()
    report = check_profile(make_profile(contract=contract, limits=Limits()))
    assert report.ok
    assert "contract.action_envelope_unknown" in report.codes()


def test_a_reference_pose_outside_the_envelope_is_an_error() -> None:
    contract = make_policy_contract(default_pose=[9.0, -0.1, 0.2])
    report = check_profile(make_profile(contract=contract))
    finding = next(f for f in report.errors if f.code == "contract.default_pose_outside_limits")
    assert list(finding.detail["joints"]) == ["joint0"]


# -- scenario --------------------------------------------------------------


def scenario_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": 1,
        "name": "bench",
        "world": {"backend": "dummy"},
        "robots": [{"id": "arm1", "profile": "test_arm", "role": "learner"}],
    }
    data.update(overrides)
    return data


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    (tmp_path / "test_arm.yaml").write_text(
        make_profile().model_dump_json(indent=2), encoding="utf-8"
    )
    return Registry([tmp_path])


def test_scenario_passes(registry: Registry) -> None:
    assert check_scenario(Scenario.model_validate(scenario_data()), registry).ok


def test_missing_profile_is_an_error(registry: Registry) -> None:
    data = scenario_data(robots=[{"id": "arm1", "profile": "no_such_robot"}])
    report = check_scenario(Scenario.model_validate(data), registry)
    assert not report.ok
    assert "scenario.profile_not_found" in report.codes()


def test_robot_without_backend_adapter_is_an_error(registry: Registry) -> None:
    """One scene, one physics authority: no adapter means no participation."""
    data = scenario_data(world={"backend": "gazebo"})
    report = check_scenario(Scenario.model_validate(data), registry)
    assert not report.ok
    assert "scenario.backend_adapter_missing" in report.codes()


def test_two_learners_rejected(registry: Registry) -> None:
    """Joint multi-agent training is deliberately postponed (section 15)."""
    data = scenario_data(
        robots=[
            {"id": "arm1", "profile": "test_arm", "role": "learner"},
            {"id": "arm2", "profile": "test_arm", "role": "learner"},
        ]
    )
    report = check_scenario(Scenario.model_validate(data), registry)
    assert "scenario.multiple_learners" in report.codes()


def test_shared_profile_is_reported_once(registry: Registry) -> None:
    data = scenario_data(
        robots=[
            {"id": "arm1", "profile": "test_arm", "role": "learner"},
            {"id": "arm2", "profile": "test_arm", "role": "static"},
        ]
    )
    report = check_scenario(Scenario.model_validate(data), registry)
    assert report.codes().count("adapter.plugin_not_installed") <= 1
