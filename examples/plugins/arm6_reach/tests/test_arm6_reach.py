"""What this package promises, held without a simulator.

Everything here runs on a machine that has neither Isaac nor a GPU, and that is
the point of the two-layer protocol: what the platform reads before a run has to
be readable where there is nothing to run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arm6_reach import Arm6ReachTask
from arm6_reach.task import GOAL_SPREAD, RESET_SPREAD

from heron.core.models import RobotProfile, Scenario
from heron.core.protocols import TrainTask
from heron.train.isaaclab_task import IsaacLabTask, TaskContext, check_goal_range


def read(path: Path, model: type) -> object:
    """A YAML document as a schema of the platform. The loader is not public API,
    and a test has no business needing more than the schema itself."""
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

EXAMPLES = Path(__file__).resolve().parents[3]
PROFILE = EXAMPLES / "robots" / "arm6.yaml"
TABLE = EXAMPLES / "scenarios" / "table.yaml"


@pytest.fixture
def profile() -> RobotProfile:
    if not PROFILE.is_file():
        pytest.skip("пример робота удалён — проверка инварианта 1")
    profile = read(PROFILE, RobotProfile)
    assert isinstance(profile, RobotProfile)
    return profile


@pytest.fixture
def context(profile: RobotProfile) -> TaskContext:
    """What the platform hands the task, minus the framework's entity."""
    return TaskContext(
        profile=profile,
        contract=profile.contract,
        robot=None,
        asset="robot",
        command="target",
        episode_length_s=Arm6ReachTask.completion.horizon_s,
    )


def test_task_satisfies_both_layers() -> None:
    task = Arm6ReachTask()
    assert isinstance(task, TrainTask)
    assert isinstance(task, IsaacLabTask)


def test_reading_the_task_needs_no_simulator() -> None:
    """The three members the platform reads before a run are read without Isaac."""
    import sys

    task = Arm6ReachTask()
    assert task.backend == "isaaclab"
    assert task.spec.name == "arm6_reach"
    assert task.completion.horizon_s > task.completion.hold_s
    assert not [name for name in sys.modules if name.split(".")[0] == "isaaclab"]


def test_goal_box_is_reachable_and_admissible(
    profile: RobotProfile, context: TaskContext
) -> None:
    """Every goal this task may draw is one the action can command and the robot may hold."""
    box = Arm6ReachTask().goal_range(context)
    report = check_goal_range(box.min, box.max, profile, where="task:arm6_reach")
    assert report.ok, report.codes()


def test_the_goal_of_the_scenario_lies_in_the_box(context: TaskContext) -> None:
    """ADR-0043, item 5: the pose the coded skill is asked for is one this policy is
    trained to reach, or swapping one for the other would not be one line."""
    if not TABLE.is_file():
        pytest.skip("пример сцены удалён — проверка инварианта 1")
    scenario = read(TABLE, Scenario)
    assert isinstance(scenario, Scenario)
    assert scenario.task is not None
    target = scenario.task.goal["target"]

    box = Arm6ReachTask().goal_range(context)
    outside = [
        name
        for name, value, low, high in zip(
            context.contract.joint_order, target, box.min, box.max, strict=True
        )
        if not low <= value <= high
    ]
    assert not outside, f"цель сцены вне диапазона обучения в суставах {outside}"


def test_reset_covers_the_pose_of_the_world_of_execution(context: TaskContext) -> None:
    """The arm starts a run at zeros; training must have seen that pose."""
    pose = context.contract.default_pose or []
    assert max(abs(value) for value in pose) <= RESET_SPREAD
    assert GOAL_SPREAD < RESET_SPREAD


# That this package imports only the platform's public API is held from the other
# side, by ``tests/test_public_api.py``: one check over every package in
# ``examples/plugins``, so a new package cannot forget to bring its own.
