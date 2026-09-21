"""The backend plans a run, writes it down, and turns what it left into an artifact.

The process that trains is never started here. What is asserted is the command it
would be started with, the files it would read, and the artifact made from the
files it would leave -- the same split the simulation backends are tested under
(ADR-0011): a test proves the composition of a command and says nothing about
whether a simulator comes up from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from conftest import make_policy_contract, make_profile, write_profile, write_urdf
from heron.core.contract import ActionSpec
from heron.core.enums import ActionOffset, ActionType, ControlMode
from heron.core.environment import RequirementKind
from heron.core.models import Completion, RobotProfile, Scenario, SkillSpec
from heron.registry.registry import Registry
from heron.sim.process import PlanningLauncher
from heron.store.policies import PolicyStore
from heron.train.isaaclab import (
    BACKEND,
    CONDITIONS_FILE,
    EXPORT_FILE,
    ONNX_FILE,
    PROCESS_MODULE,
    IsaacLabBackend,
    RunConditions,
    TrainError,
    read_run,
)
from heron.train.spec import DESCRIPTION_FILE, SPEC_FILE
from policies import linear_onnx, zero_weights

JOINTS = ["joint0", "joint1", "joint2"]
GAINS = {"kp": [10.0, 20.0, 30.0], "kd": [1.0, 2.0, 3.0]}
TASK = "bench_reach"
COMPLETION = Completion(tolerance=0.05, hold_s=0.2, horizon_s=4.0)


class BenchTask:
    """A task like a user's, minus everything that needs a simulator."""

    backend = BACKEND
    spec = SkillSpec(
        name=TASK,
        required_capabilities=frozenset(make_profile().capabilities),
        required_control_mode=ControlMode.STREAMING,
    )
    completion = COMPLETION


def trainable_profile(**overrides: Any) -> RobotProfile:
    contract = make_policy_contract(
        3,
        action=ActionSpec(
            type=ActionType.JOINT_POSITION,
            clip=(-1.0, 1.0),
            scale=[0.5, 0.5, 0.5],
            offset=ActionOffset.DEFAULT_POSE,
            **GAINS,
        ),
    )
    # A real description, not the stub's template: a training run converts the
    # copy, and a template is refused rather than guessed at (ADR-0009).
    fields: dict[str, Any] = {"description": {"urdf": "urdf/test_arm.urdf"}}
    fields.update(overrides)
    return make_profile(contract=contract, **fields)


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    root = tmp_path / "registry"
    root.mkdir()
    profile = trainable_profile()
    write_profile(root, profile)
    write_urdf(root, profile.name, JOINTS)
    return Registry([root])


@pytest.fixture
def scenario() -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "bench_train",
            "world": {"backend": BACKEND},
            "robots": [
                {"id": "arm1", "profile": "test_arm", "role": "learner", "task": TASK}
            ],
        }
    )


@pytest.fixture
def backend(registry: Registry, tmp_path: Path, tasks: None) -> IsaacLabBackend:
    return IsaacLabBackend(
        registry,
        into=tmp_path / "run",
        launcher=PlanningLauncher(),
        store=PolicyStore(tmp_path / "store"),
    )


@pytest.fixture
def tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The task, installed the way an entry point would install it."""
    monkeypatch.setattr("heron.train.tasks.load", lambda kind, name: BenchTask)
    monkeypatch.setattr("heron.train.tasks.installed_tasks", lambda: {TASK: "tests:BenchTask"})


# -- planning ---------------------------------------------------------------


def test_the_plan_names_the_process_and_the_directory(
    backend: IsaacLabBackend, scenario: Scenario, tmp_path: Path
) -> None:
    plan = backend.build_env(scenario)
    assert plan.command == (sys.executable, "-m", PROCESS_MODULE, str(tmp_path / "run"))
    assert plan.run == tmp_path / "run"


def test_the_learner_appears_where_the_scenario_places_it(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    """One placement for both worlds: the scenario's ``pose`` (ADR-0077)."""
    assert backend.build_env(scenario).spec.base_position == (0.0, 0.0, 0.0)

    robot = scenario.robots[0].model_copy(update={"pose": (0.5, -0.5, 0.8)})
    placed = scenario.model_copy(update={"robots": [robot]})
    plan = backend.build_env(placed)
    assert plan.spec.base_position == (0.5, -0.5, 0.8)
    assert any("появляется в [0.5, -0.5, 0.8]" in line for line in plan.spec.lines())


def test_the_process_module_is_never_spelled_out() -> None:
    """Section 17: the platform's name is built from the package, not typed."""
    from heron.core.branding import APP_NAME

    assert PROCESS_MODULE.startswith(f"{APP_NAME}.train.")


def test_the_plan_carries_the_task_declaration(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    """What the future skill demands travels from the task into the manifest."""
    plan = backend.build_env(scenario)
    assert plan.spec.task == TASK
    assert plan.spec.skill == BenchTask.spec
    assert plan.spec.completion == COMPLETION


def test_a_scenario_that_does_not_validate_builds_nothing(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    """Invariant 6, at the only moment it is still cheap."""
    broken = scenario.model_copy(
        update={
            "robots": (
                scenario.robots[0].model_copy(update={"profile": "no_such_robot"}),
            )
        }
    )
    with pytest.raises(TrainError, match="не проходит проверку"):
        backend.build_env(broken)


def test_a_world_of_another_backend_is_refused(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    other = scenario.model_copy(update={"world": scenario.world.model_copy(
        update={"backend": "gazebo"})})
    with pytest.raises(TrainError, match="владелец один"):
        backend.build_env(other)


def test_a_named_learner_that_is_not_there_is_refused(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    with pytest.raises(TrainError, match="такого ученика нет"):
        backend.build_env(scenario, "arm9")


# -- the run on disk --------------------------------------------------------


def test_training_writes_the_run_before_starting_it(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    """``spec.json`` is on disk before the process exists, so what it reads is what
    a human was shown."""
    plan = backend.build_env(scenario)
    run = backend.train(plan, RunConditions(envs=64, iterations=3, seed=7, device="cpu"))

    assert (run / SPEC_FILE).is_file()
    assert (run / DESCRIPTION_FILE).is_file()
    record = read_run(run)
    assert record.spec == plan.spec
    assert record.conditions.envs == 64
    assert record.conditions.seed == 7
    assert isinstance(backend.launcher, PlanningLauncher)
    assert backend.launcher.commands == [plan.command]


def test_the_written_description_is_the_copy_not_the_input(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    plan = backend.build_env(scenario)
    run = backend.train(plan, RunConditions())
    assert (run / DESCRIPTION_FILE).read_text(encoding="utf-8") == plan.description


def test_a_process_that_fails_is_reported_with_its_directory(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    class Failed:
        command = ("python",)

        def poll(self) -> int:
            return 1

        def interrupt(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 1

    class Failing(PlanningLauncher):
        def start(self, command: Any, **kwargs: Any) -> Any:
            return Failed()

    backend.launcher = Failing()
    plan = backend.build_env(scenario)
    with pytest.raises(TrainError, match="код"):
        backend.train(plan, RunConditions())


# -- the artifact (ADR-0046) ------------------------------------------------


def finish(run: Path, spec: Any, *, sizes: tuple[int, int] | None = None) -> Path:
    """Leave behind what the training process leaves behind."""
    weights, bias = zero_weights(spec.contract)
    linear_onnx(run / ONNX_FILE, weights, bias)
    observation, action = sizes or (spec.observation_size, spec.action_size)
    (run / EXPORT_FILE).write_text(
        json.dumps(
            {
                "file": ONNX_FILE,
                "input": "obs",
                "output": "actions",
                "observation_size": observation,
                "action_size": action,
                "opset": 18,
            }
        ),
        encoding="utf-8",
    )
    (run / CONDITIONS_FILE).write_text(
        json.dumps({"usd_key": "0123456789abcdef", "learn_s": 12.5}), encoding="utf-8"
    )
    return run


def test_export_names_the_artifact_by_its_content(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    plan = backend.build_env(scenario)
    run = finish(backend.train(plan, RunConditions()), plan.spec)

    artifact = backend.export(run)
    manifest = artifact.manifest
    assert artifact.ref == manifest.ref()
    assert manifest.self_consistent
    assert manifest.name == TASK
    assert manifest.contract_id == plan.spec.contract_id
    assert manifest.completion == COMPLETION
    assert manifest.capabilities == BenchTask.spec.required_capabilities


def test_the_conditions_of_training_travel_next_to_the_artifact(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    """Outside its identity: the same policy trained twice is the same policy."""
    plan = backend.build_env(scenario)
    run = finish(backend.train(plan, RunConditions(envs=128, seed=3)), plan.spec)

    artifact = backend.export(run)
    conditions = artifact.train()
    assert conditions["envs"] == 128
    assert conditions["seed"] == 3
    assert conditions["usd_key"] == "0123456789abcdef"
    assert conditions["contract_id"] == plan.spec.contract_id


def test_a_network_of_the_wrong_width_is_refused(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    """The environment was not built from this contract, and an artifact claiming
    otherwise would pass every later check."""
    plan = backend.build_env(scenario)
    run = finish(backend.train(plan, RunConditions()), plan.spec, sizes=(99, 3))
    with pytest.raises(TrainError, match="входов"):
        backend.export(run)


def test_a_run_that_never_exported_says_so(
    backend: IsaacLabBackend, scenario: Scenario
) -> None:
    plan = backend.build_env(scenario)
    run = backend.train(plan, RunConditions())
    with pytest.raises(TrainError, match="описания экспорта"):
        backend.export(run)


# -- the environment the training process needs (ADR-0025) ------------------


def test_the_backend_declares_what_it_needs_and_what_it_forbids() -> None:
    needed = {r.name for r in IsaacLabBackend.requirements if not r.forbidden}
    forbidden = {r.name for r in IsaacLabBackend.requirements if r.forbidden}
    assert needed == {"isaaclab", "isaacsim", "rsl_rl"}
    assert forbidden == {"rclpy"}
    assert all(
        r.kind is RequirementKind.PYTHON for r in IsaacLabBackend.requirements
    )
    assert all(r.because for r in IsaacLabBackend.requirements if r.forbidden)
