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
from g1_stand import G1StandTask
from g1_stand.task import FALL_HEIGHT, PELVIS, RESET_JOINTS

from heron.core.models import RobotProfile, Scenario
from heron.core.protocols import TrainTask
from heron.core.urdf import parse_urdf
from heron.train.isaaclab_task import IsaacLabTask, TaskContext, check_goal_range

APP = IsaacLabTask.__module__.split(".")[0]
"""The platform's package name, read off a class it exports rather than spelled."""

EXAMPLES = Path(__file__).resolve().parents[3]
ROBOTS = EXAMPLES / "robots"
PROFILE = ROBOTS / "g1_29dof_lock_waist_rev_1_0" / "profile.yaml"
TRAIN = EXAMPLES / "scenarios" / "g1_stand_train.yaml"
FETCH_HINT = "python examples/robots/g1_29dof_lock_waist_rev_1_0/fetch_description.py"


def read(path: Path, model: type) -> object:
    """A YAML document as a schema of the platform. The loader is not public API."""
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.fixture
def profile() -> RobotProfile:
    if not PROFILE.is_file():
        pytest.skip("пример G1 удалён — проверка инварианта 1")
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
        episode_length_s=G1StandTask.completion.horizon_s,
    )


# -- the extension point ------------------------------------------------------------


def test_task_satisfies_both_layers() -> None:
    task = G1StandTask()
    assert isinstance(task, TrainTask)
    assert isinstance(task, IsaacLabTask)


def test_the_task_is_found_by_its_entry_point() -> None:
    """The group the platform searches, the name a scenario writes (ADR-0044)."""
    found = {item.name: item for item in entry_points(group=f"{APP}.train_tasks")}
    assert "g1_stand" in found, "пакет не установлен: uv pip install --no-deps -e <пакет>"
    assert found["g1_stand"].load() is G1StandTask


def test_reading_the_task_needs_no_simulator() -> None:
    task = G1StandTask()
    assert task.backend == "isaaclab"
    assert task.spec.name == "g1_stand"
    assert task.completion.horizon_s > task.completion.hold_s
    assert not [name for name in sys.modules if name.split(".")[0] in {"isaaclab", "torch"}]


def test_the_scenario_trains_this_task_on_the_free_robot_above_the_floor() -> None:
    scenario = read(TRAIN, Scenario)
    assert isinstance(scenario, Scenario)
    (learner,) = scenario.learners
    assert learner.task == "g1_stand"
    assert learner.profile == PROFILE.parent.name
    assert learner.pose[2] > FALL_HEIGHT, "появляется стоя, а не упавшим (ADR-0077)"


# -- the goal (ADR-0043) --------------------------------------------------------------


def test_the_goal_is_the_standing_pose_and_nothing_around_it(
    profile: RobotProfile, context: TaskContext
) -> None:
    box = G1StandTask().goal_range(context)
    assert list(box.min) == list(box.max) == list(profile.contract.default_pose or [])
    report = check_goal_range(box.min, box.max, profile, where="task:g1_stand")
    assert report.ok, report.codes()


def test_the_skill_fits_the_whole_body(profile: RobotProfile) -> None:
    """Twenty-seven joints, as the task demands and the profile states (ADR-0075)."""
    assert profile.contract.dof == 27
    assert G1StandTask.spec.requires is not None


def test_a_reset_stays_inside_the_narrowest_frame(profile: RobotProfile) -> None:
    """The pose a reset scatters to is one the robot may hold -- ankle roll is +-0.2618."""
    pose = profile.contract.default_pose or []
    limits = profile.limits.joint_position
    assert limits is not None
    for name, reference, low, high in zip(
        profile.contract.joint_order, pose, limits.min, limits.max, strict=True
    ):
        assert low <= reference - RESET_JOINTS and reference + RESET_JOINTS <= high, name


# -- the robot this is written for ------------------------------------------------------


def test_the_base_is_free_and_the_pelvis_is_its_root(description: Path) -> None:
    """Nothing holds the robot but the policy (ADR-0075); the pelvis is what falls."""
    model = parse_urdf(description)
    assert not model.fixed_to_world
    assert model.root_link == PELVIS


def test_every_movable_joint_of_the_description_is_commanded(
    profile: RobotProfile, description: Path
) -> None:
    """No passive leg this time: what the policy does not drive, nothing drives."""
    movable = {joint.name for joint in parse_urdf(description).commandable_joints}
    assert movable == set(profile.contract.joint_order)


# -- the environment the platform derives (ADR-0044, ADR-0076, ADR-0077) ------------------


def test_the_derived_environment_is_the_contracts(
    profile: RobotProfile, description: Path, tmp_path: Path
) -> None:
    """The platform's dry run, as a user runs it, found this task by its entry point and
    derived the rates, the action, the body's observation, the drive, the free base and
    where the robot appears."""
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

    assert "задача      g1_stand на бэкенде isaaclab" in plan
    assert f"контракт    {profile.contract.contract_id}" in plan
    assert "физика 500 Гц, политика 50 Гц, decimation 10" in plan
    assert "база        свободна, появляется в [0, 0, 0.8]" in plan
    assert "действие    joint_position, clip ±1, ширина 27" in plan
    assert (
        "наблюдение  base_ang_velx0.25[3], projected_gravityx1[3], joint_pos_relx1[27], "
        "joint_velx0.05[27], target_joint_posx1[27], previous_actionx1[27]  ширина 114"
    ) in plan
    assert "привод      kp " + ", ".join(f"{value:g}" for value in action.kp) in plan
    assert "эпизод      20 с; допуск 0.25 rad, выдержка 10 с" in plan
    assert not (tmp_path / "home").exists(), "сухой прогон ничего не пишет"


# That this package imports only the platform's public API is held from the other side, by
# the platform's ``tests/test_public_api.py``, over every package in ``examples/plugins``.
