"""A trained policy as a skill: checked before the link, one mapping per tick (ADR-0047).

The network here is built, not trained (``tests/policies.py``), and it is built to
command the target it is shown. That is deliberate: what has to be proved is the
platform around a policy -- the checks that refuse a wrong one before anything
connects, the mapping of its output, the observation it is given and when the skill
is done -- and none of that becomes truer with a network nobody can predict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import (
    make_contract,
    make_policy_contract,
    make_profile,
    write_profile,
    write_urdf,
)
from heron.adapters.dummy import DummyAdapter
from heron.cli.run import RunError, execute
from heron.core.contract import ActionSpec, ObservationTerm
from heron.core.enums import (
    ActionOffset,
    ActionType,
    ObservationType,
    Status,
)
from heron.core.models import RobotProfile, Scenario
from heron.core.types import Pose, RobotState
from heron.registry.registry import Registry
from heron.sdk.executor import Executor, ManualClock
from heron.skills.learned import (
    LearnedSkill,
    ObservationError,
    PolicyRejectedError,
    map_action,
    observation_vector,
)
from heron.store.policies import ONNX_FILE, PolicyArtifact, PolicyStore
from policies import store_policy, zero_weights

pytest.importorskip("onnx", reason="группа dev не установлена: сеть тестов не построить")
pytest.importorskip("onnxruntime", reason="группа onnx не установлена")

TARGET = [0.3, 0.2, -0.1]
REFERENCE = [0.1, -0.1, 0.2]


@pytest.fixture
def store(tmp_path: Path) -> PolicyStore:
    return PolicyStore(tmp_path / "store")


@pytest.fixture
def artifact(store: PolicyStore, tmp_path: Path) -> PolicyArtifact:
    return store_policy(store, tmp_path / "export", make_policy_contract())


@pytest.fixture
def profile(artifact: PolicyArtifact) -> RobotProfile:
    return make_profile(
        contract=make_policy_contract(), skills=("hold", {"move_joints": str(artifact.ref)})
    )


# -- the mapping, as functions ---------------------------------------------


def test_the_observation_is_the_terms_in_order_each_times_its_scale() -> None:
    contract = make_policy_contract()
    state = RobotState(
        timestamp=0.0,
        joint_position=(0.2, 0.0, 0.2),
        joint_velocity=(1.0, 2.0, 3.0),
        base=Pose(),
    )
    values = observation_vector(contract, state, TARGET, (0.5, 0.0, -0.5))

    assert values == pytest.approx(
        [
            0.1, 0.1, 0.0,  # q - default_pose
            0.05, 0.1, 0.15,  # dq * 0.05
            *TARGET,
            0.5, 0.0, -0.5,  # the previous output, unscaled
        ]
    )
    assert len(values) == contract.observation_size


def test_a_state_that_does_not_fit_the_contract_is_not_padded() -> None:
    contract = make_policy_contract()
    state = RobotState(timestamp=0.0, joint_position=(0.0, 0.0), joint_velocity=(0.0, 0.0))
    with pytest.raises(ObservationError, match="joint_position"):
        observation_vector(contract, state, TARGET, (0.0, 0.0, 0.0))


def test_the_output_is_clipped_before_it_is_scaled() -> None:
    """``clip`` bounds the raw output; the envelope of the joint is not it (ADR-0041)."""
    contract = make_policy_contract()
    clipped, targets = map_action(contract, [5.0, -5.0, 0.4])

    assert clipped == (1.0, -1.0, 0.4)
    assert targets == pytest.approx([0.1 + 0.5, -0.1 - 0.5, 0.2 + 0.2])


def test_without_a_mapping_the_output_is_the_command() -> None:
    contract = make_contract(
        3, observation=[ObservationTerm(type=ObservationType.TARGET_JOINT_POS)]
    )
    assert map_action(contract, [0.3, 0.0, -0.3])[1] == pytest.approx([0.3, 0.0, -0.3])


# -- the skill on a robot made of numbers ----------------------------------


def test_the_policy_drives_the_robot_to_the_goal(
    artifact: PolicyArtifact, profile: RobotProfile
) -> None:
    adapter = DummyAdapter()
    adapter.connect(profile, {})
    skill = LearnedSkill.load(artifact, profile, name="move_joints")

    result = Executor(adapter, profile, clock=ManualClock()).run(skill, {"target": TARGET})

    assert result.status is Status.SUCCESS
    assert result.commands[-1].type is ActionType.JOINT_POSITION
    assert result.commands[-1].values == pytest.approx(TARGET)
    assert result.states[-1].joint_position == pytest.approx(TARGET)


def test_the_previous_action_is_zero_on_the_first_tick_and_the_output_after(
    artifact: PolicyArtifact, profile: RobotProfile
) -> None:
    adapter = DummyAdapter()
    adapter.connect(profile, {})
    skill = LearnedSkill.load(artifact, profile, name="move_joints")
    skill.start({"target": TARGET}, _context(profile, adapter))

    assert skill.previous == (0.0, 0.0, 0.0)
    skill.tick(adapter.read_state())
    # (target - default_pose) / scale, which is what the built network answers.
    assert skill.previous == pytest.approx([0.4, 0.6, -0.6])


def test_a_policy_that_never_arrives_fails_at_the_horizon(
    store: PolicyStore, tmp_path: Path
) -> None:
    """The episode it was trained for is over; there is nothing to wait for (ADR-0043)."""
    contract = make_policy_contract()
    weights, bias = zero_weights(contract)
    idle = store_policy(
        store, tmp_path / "export", contract, name="idle", weights=weights, bias=bias
    )
    profile = make_profile(contract=contract, skills=({"move_joints": str(idle.ref)},))
    adapter = DummyAdapter()
    adapter.connect(profile, {})

    skill = LearnedSkill.load(idle, profile, name="move_joints")
    result = Executor(adapter, profile, clock=ManualClock()).run(skill, {"target": TARGET})

    assert result.status is Status.FAILURE
    assert result.states[-1].timestamp >= idle.manifest.completion.horizon_s
    assert any("горизонт" in line for line in _context_of(skill).log_lines)


def test_the_goal_is_checked_against_the_robot_before_a_command(
    artifact: PolicyArtifact, profile: RobotProfile
) -> None:
    """The same check ``move_joints`` makes, from the same helper (ADR-0043)."""
    adapter = DummyAdapter()
    adapter.connect(profile, {})
    skill = LearnedSkill.load(artifact, profile, name="move_joints")

    with pytest.raises(ValueError, match="вне лимитов"):
        skill.start({"target": [99.0, 0.0, 0.0]}, _context(profile, adapter))
    assert not adapter.commands


# -- every deviation is refused before the first tick ----------------------


DEVIATIONS: dict[str, tuple[dict[str, Any], str]] = {
    "порядок суставов": ({"joint_order": ["joint1", "joint0", "joint2"]}, "joint_order"),
    "поза отсчёта": ({"default_pose": [0.2, -0.1, 0.2]}, "default_pose"),
    "частота состояния": ({"state_rate_hz": 200.0}, "state_rate_hz"),
    "масштаб действия": (
        {
            "action": ActionSpec(
                type=ActionType.JOINT_POSITION,
                clip=(-1.0, 1.0),
                scale=[0.4, 0.5, 0.5],
                offset=ActionOffset.DEFAULT_POSE,
            )
        },
        "action",
    ),
    "смещение действия": (
        {
            "action": ActionSpec(
                type=ActionType.JOINT_POSITION, clip=(-1.0, 1.0), scale=[0.5, 0.5, 0.5]
            )
        },
        "action",
    ),
    "коэффициенты привода": (
        {
            "action": ActionSpec(
                type=ActionType.JOINT_POSITION,
                clip=(-1.0, 1.0),
                scale=[0.5, 0.5, 0.5],
                offset=ActionOffset.DEFAULT_POSE,
                kp=[60.0, 60.0, 60.0],
                kd=[1.0, 1.0, 1.0],
            )
        },
        "action",
    ),
    "состав наблюдения": (
        {
            "observation": [
                ObservationTerm(type=ObservationType.JOINT_POS_REL),
                ObservationTerm(type=ObservationType.TARGET_JOINT_POS),
                ObservationTerm(type=ObservationType.PREVIOUS_ACTION),
            ]
        },
        "observation",
    ),
}


@pytest.mark.parametrize("case", sorted(DEVIATIONS), ids=sorted(DEVIATIONS))
def test_a_policy_trained_under_another_contract_is_refused_by_field(
    case: str, store: PolicyStore, tmp_path: Path
) -> None:
    overrides, field = DEVIATIONS[case]
    trained = store_policy(store, tmp_path / "export", make_policy_contract(**overrides))
    profile = make_profile(
        contract=make_policy_contract(), skills=({"move_joints": str(trained.ref)},)
    )

    with pytest.raises(PolicyRejectedError) as raised:
        LearnedSkill.load(trained, profile, name="move_joints")
    report = raised.value.report
    assert report.codes() == ["contract.field_mismatch"]
    assert report.errors[0].detail["field"] == field


def test_a_replaced_network_is_caught_by_its_hash(
    artifact: PolicyArtifact, profile: RobotProfile
) -> None:
    (artifact.directory / ONNX_FILE).write_bytes(
        (artifact.directory / ONNX_FILE).read_bytes() + b"\0"
    )
    with pytest.raises(PolicyRejectedError) as raised:
        LearnedSkill.load(artifact, profile, name="move_joints")
    assert raised.value.report.codes() == ["policy.artifact_inconsistent"]


def test_a_network_of_the_wrong_width_is_caught_before_the_first_tick(
    store: PolicyStore, tmp_path: Path
) -> None:
    """The manifest agrees with the contract; the graph does not (ADR-0047)."""
    contract = make_policy_contract()
    width = contract.observation_size or 0
    wide = store_policy(
        store,
        tmp_path / "export",
        contract,
        name="wide",
        weights=[[0.0] * width for _ in range(contract.dof + 1)],
        bias=[0.0] * (contract.dof + 1),
    )
    profile = make_profile(contract=contract, skills=({"move_joints": str(wide.ref)},))

    with pytest.raises(PolicyRejectedError) as raised:
        LearnedSkill.load(wide, profile, name="move_joints")
    finding = raised.value.report.errors[0]
    assert finding.code == "policy.onnx_signature_mismatch"
    assert finding.detail["выход"]["found"][0]["shape"] == [1, 4]


# -- through ``run``: the profile's line, and a refusal before the link ----


def bench(tmp_path: Path, profile: RobotProfile) -> tuple[Path, Registry]:
    write_urdf(tmp_path, profile.name, list(profile.contract.joint_order))
    write_profile(tmp_path, profile)
    path = tmp_path / "scene.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "scene",
                "world": {"backend": "dummy"},
                "robots": [{"id": "arm1", "profile": profile.name, "role": "scripted"}],
                "task": {"skill": "move_joints", "target": TARGET},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path, Registry([tmp_path])


def test_run_executes_the_policy_the_profile_binds(
    store: PolicyStore, artifact: PolicyArtifact, profile: RobotProfile, tmp_path: Path
) -> None:
    """Invariant 4: the scenario asks for ``move_joints`` and knows nothing more."""
    path, registry = bench(tmp_path / "bench", profile)
    scenario = Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    lines, result = execute(
        scenario, registry, store=store, scenario_path=path, environ={}
    )

    assert result is not None and result.ok
    assert any(str(artifact.ref) in line for line in lines)


def test_run_refuses_a_policy_the_robot_cannot_run_before_it_connects(
    store: PolicyStore, tmp_path: Path
) -> None:
    contract = make_policy_contract()
    width = contract.observation_size or 0
    wide = store_policy(
        store,
        tmp_path / "export",
        contract,
        name="wide",
        weights=[[0.0] * width for _ in range(contract.dof + 1)],
        bias=[0.0] * (contract.dof + 1),
    )
    profile = make_profile(contract=contract, skills=({"move_joints": str(wide.ref)},))
    path, registry = bench(tmp_path / "bench", profile)
    scenario = Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    with pytest.raises(RunError) as raised:
        execute(scenario, registry, store=store, scenario_path=path, environ={})
    assert raised.value.report is not None
    assert raised.value.report.codes() == ["policy.onnx_signature_mismatch"]


def test_run_says_when_the_artifact_is_not_on_this_machine(
    store: PolicyStore, tmp_path: Path
) -> None:
    profile = make_profile(
        contract=make_policy_contract(), skills=({"move_joints": "reach@" + "1" * 16},)
    )
    path, registry = bench(tmp_path / "bench", profile)
    scenario = Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    lines, result = execute(
        scenario, registry, store=store, scenario_path=path, environ={}, dry_run=True
    )
    assert result is None
    assert any("политика" in line and "не найдена" in line for line in lines)

    with pytest.raises(RunError, match="не найдена"):
        execute(scenario, registry, store=store, scenario_path=path, environ={})


# -- helpers ----------------------------------------------------------------


def _context(profile: RobotProfile, adapter: DummyAdapter) -> Any:
    from heron.sdk.runner import AdapterContext

    return AdapterContext(profile_=profile, adapter=adapter)


def _context_of(skill: LearnedSkill) -> Any:
    return skill.context
