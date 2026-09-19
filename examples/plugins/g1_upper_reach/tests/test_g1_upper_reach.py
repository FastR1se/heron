"""What this package promises, held without a simulator.

Everything here runs on a machine with neither Isaac nor a GPU: what the platform reads
before a run has to be readable where there is nothing to run. The derived environment is
read off the platform's own dry run, through its command line -- the path a user takes --
rather than off the platform's internals, which a task package has no business importing
(ADR-0021).
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest
import yaml
from g1_upper_reach import G1UpperReachTask
from g1_upper_reach.task import ARM_SPREAD, LEGS, RESET_SPREAD, WAIST_SPREAD

from heron.core.models import RobotProfile, Scenario
from heron.core.protocols import TrainTask
from heron.core.urdf import parse_urdf
from heron.train.isaaclab_task import IsaacLabTask, TaskContext, check_goal_range

APP = IsaacLabTask.__module__.split(".")[0]
"""The platform's package name, read off a class it exports rather than spelled."""

EXAMPLES = Path(__file__).resolve().parents[3]
ROBOTS = EXAMPLES / "robots"
PROFILE = ROBOTS / "g1_29dof_lock_waist_rev_1_0_mounted" / "profile.yaml"
TRAIN = EXAMPLES / "scenarios" / "g1_mounted_train.yaml"
EXECUTE = EXAMPLES / "scenarios" / "g1_mounted.yaml"
FETCH_HINT = "python examples/robots/g1_29dof_lock_waist_rev_1_0_mounted/fetch_description.py"


def read(path: Path, model: type) -> object:
    """A YAML document as a schema of the platform. The loader is not public API."""
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.fixture
def profile() -> RobotProfile:
    if not PROFILE.is_file():
        pytest.skip("пример закреплённого G1 удалён — проверка инварианта 1")
    profile = read(PROFILE, RobotProfile)
    assert isinstance(profile, RobotProfile)
    return profile


@pytest.fixture
def description(profile: RobotProfile) -> Path:
    path = PROFILE.parent / profile.description.urdf
    if not path.is_file():
        pytest.skip(f"описание не загружено: {FETCH_HINT}")
    return path


@pytest.fixture
def context(profile: RobotProfile) -> TaskContext:
    """What the platform hands the task, minus the framework's entity."""
    return TaskContext(
        profile=profile,
        contract=profile.contract,
        robot=None,
        asset="robot",
        command="target",
        episode_length_s=G1UpperReachTask.completion.horizon_s,
    )


# -- the extension point ------------------------------------------------------------


def test_task_satisfies_both_layers() -> None:
    task = G1UpperReachTask()
    assert isinstance(task, TrainTask)
    assert isinstance(task, IsaacLabTask)


def test_the_task_is_found_by_its_entry_point() -> None:
    """The group the platform searches, the name a scenario writes (ADR-0044)."""
    found = {item.name: item for item in entry_points(group=f"{APP}.train_tasks")}
    assert "g1_upper_reach" in found, "пакет не установлен: pip install --no-deps -e <пакет>"
    assert found["g1_upper_reach"].load() is G1UpperReachTask


def test_reading_the_task_needs_no_simulator() -> None:
    task = G1UpperReachTask()
    assert task.backend == "isaaclab"
    assert task.spec.name == "g1_upper_reach"
    assert task.completion.horizon_s > task.completion.hold_s
    assert not [name for name in sys.modules if name.split(".")[0] in {"isaaclab", "torch"}]


def test_the_scenario_trains_this_task_on_the_mounted_robot() -> None:
    scenario = read(TRAIN, Scenario)
    assert isinstance(scenario, Scenario)
    (learner,) = scenario.learners
    assert learner.task == "g1_upper_reach"
    assert learner.profile == PROFILE.parent.name


# -- the goal (ADR-0043) --------------------------------------------------------------


def test_goal_box_is_reachable_and_admissible(profile: RobotProfile, context: TaskContext) -> None:
    """Every goal this task may draw is one the action can command and the robot may hold."""
    box = G1UpperReachTask().goal_range(context)
    report = check_goal_range(box.min, box.max, profile, where="task:g1_upper_reach")
    assert report.ok, report.codes()


def test_the_waist_is_given_a_narrower_box_than_the_arms(context: TaskContext) -> None:
    box = G1UpperReachTask().goal_range(context)
    pose = context.contract.default_pose or []
    for name, low, high, reference in zip(
        context.contract.joint_order, box.min, box.max, pose, strict=True
    ):
        width = WAIST_SPREAD if name.startswith("waist_") else ARM_SPREAD
        assert (low, high) == pytest.approx((reference - width, reference + width)), name


def test_the_goal_of_the_scenario_lies_in_the_box(context: TaskContext) -> None:
    """ADR-0043, item 5: the pose the coded skill is asked for is one this policy is
    trained to reach, or swapping one for the other would not be one line."""
    scenario = read(EXECUTE, Scenario)
    assert isinstance(scenario, Scenario)
    assert scenario.task is not None
    target = scenario.task.goal["target"]

    box = G1UpperReachTask().goal_range(context)
    outside = [
        name
        for name, value, low, high in zip(
            context.contract.joint_order, target, box.min, box.max, strict=True
        )
        if not low <= value <= high
    ]
    assert not outside, f"цель сцены вне диапазона обучения в суставах {outside}"


def test_reset_covers_the_pose_of_the_world_of_execution(context: TaskContext) -> None:
    """The arms hang at zeros until the first command; training must have seen that pose."""
    pose = context.contract.default_pose or []
    assert max(abs(value) for value in pose) <= RESET_SPREAD
    assert max(ARM_SPREAD, WAIST_SPREAD) < RESET_SPREAD


# -- the robot this is written for ------------------------------------------------------


def test_the_legs_it_resets_are_the_joints_outside_the_contract(
    profile: RobotProfile, description: Path
) -> None:
    """Nothing drives the legs, so the task puts them back -- all of them, and only them."""
    model = parse_urdf(description)
    movable = {joint.name for joint in model.commandable_joints}
    assert movable - set(profile.contract.joint_order) == set(LEGS)


def test_the_robot_is_mounted_by_its_description(description: Path) -> None:
    """ADR-P6A: the base is fixed in the input, and the platform derives it from there."""
    assert parse_urdf(description).fixed_to_world


# -- the environment the platform derives (ADR-0044) -------------------------------------


def test_the_derived_environment_is_the_contracts(
    profile: RobotProfile, description: Path, tmp_path: Path
) -> None:
    """The platform's dry run, as a user runs it, found this task by its entry point and
    derived the rates, the action, the observation, the drive and the fixed base."""
    environment = {
        **os.environ,
        f"{APP.upper()}_HOME": str(tmp_path / "home"),
        f"{APP.upper()}_STORE": str(tmp_path / "store"),
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "-m", f"{APP}.cli.main", "train", "run", str(TRAIN),
         "--registry", str(ROBOTS), "--dry-run"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = result.stdout
    action = profile.contract.action
    assert action.kp is not None

    assert "задача      g1_upper_reach на бэкенде isaaclab" in plan
    assert f"контракт    {profile.contract.contract_id}" in plan
    assert "физика 500 Гц, политика 50 Гц, decimation 10" in plan
    assert "база        закреплена описанием" in plan
    assert "действие    joint_position, clip ±1, ширина 15" in plan
    assert (
        "наблюдение  joint_pos_relx1[15], joint_velx0.05[15], target_joint_posx1[15], "
        "previous_actionx1[15]  ширина 60"
    ) in plan
    assert "привод      kp " + ", ".join(f"{value:g}" for value in action.kp) in plan
    assert "эпизод      5 с; допуск 0.08 rad, выдержка 0.3 с" in plan
    assert not (tmp_path / "home").exists(), "сухой прогон ничего не пишет"


# That this package imports only the platform's public API is held from the other side, by
# the platform's ``tests/test_public_api.py``, over every package in ``examples/plugins``.
