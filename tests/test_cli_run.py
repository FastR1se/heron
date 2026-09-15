"""``run``: choosing a robot, a link and a skill before anything moves.

The command connects to a world someone else started, so most of what it does
is refuse to start on bad input. That part is entirely checkable here; the part
that drives a real controller is not, and nothing below pretends otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from conftest import make_profile, write_profile, write_urdf
from heron.adapters.dummy import DummyAdapter
from heron.cli.main import EXIT_FOUND_ERRORS, EXIT_OK, app
from heron.core.models import AdapterSpec, JointRange, Limits, ModelDescription, RobotProfile

runner = CliRunner()

JOINTS = ["joint0", "joint1", "joint2"]


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """A registry with one robot that has both a dummy and a gazebo link."""
    profile = make_profile(
        name="arm",
        description=ModelDescription(urdf="urdf/arm.urdf"),
        adapters={
            "sim.dummy": AdapterSpec(plugin="dummy"),
            "sim.gazebo": AdapterSpec(plugin="ros2_control"),
        },
    )
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(tmp_path, profile)
    return tmp_path


def scenario(tmp_path: Path, **overrides: object) -> Path:
    document: dict[str, object] = {
        "schema_version": 1,
        "name": "scene",
        "world": {"backend": "dummy"},
        "robots": [{"id": "arm1", "profile": "arm", "role": "scripted"}],
        "task": {"skill": "move_joints", "target": [0.2, 0.0, 0.0], "duration": 0.1},
    }
    document.update(overrides)
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return path


def run(path: Path, registry: Path, *args: str) -> object:
    return runner.invoke(app, ["run", str(path), "-R", str(registry), *args])


# -- running ---------------------------------------------------------------


def test_the_scenario_task_moves_the_robot(bench: Path) -> None:
    result = run(scenario(bench), bench)
    assert result.exit_code == EXIT_OK, result.output
    assert "выполнено" in result.output
    assert "move_joints" in result.output


def test_a_skill_named_on_the_command_line_wins(bench: Path) -> None:
    result = run(scenario(bench), bench, "--skill", "hold", "-g", "duration=0.05")
    assert result.exit_code == EXIT_OK, result.output
    assert "навык hold" in result.output


def test_goal_values_keep_their_shape(bench: Path) -> None:
    """``target=[0,1,0]`` has to arrive as three numbers, not as a string."""
    result = run(
        scenario(bench), bench, "-g", "target=[0.1, 0.2, 0.3]", "-g", "duration=0.05"
    )
    assert result.exit_code == EXIT_OK, result.output


def test_a_goal_the_robot_cannot_reach_stops_before_it_moves(bench: Path) -> None:
    result = run(scenario(bench), bench, "-g", "target=[99, 0, 0]")
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "лимитов" in result.output + str(result.exception)


# -- choosing --------------------------------------------------------------


def test_dry_run_prints_the_plan_and_connects_to_nothing(bench: Path) -> None:
    path = scenario(bench, world={"backend": "gazebo"})
    result = run(path, bench, "--dry-run")

    assert result.exit_code == EXIT_OK, result.output
    assert "sim.gazebo -> ros2_control" in result.output
    assert "/arm1/joint_states" in result.output
    assert "/arm1/joint_position_controller/commands" in result.output
    assert "ничего не подключено" in result.output


def test_the_link_is_derived_from_the_worlds_backend(bench: Path) -> None:
    """One scenario, one owner of the physics, one adapter key (ADR-0002)."""
    result = run(scenario(bench), bench, "--dry-run")
    assert "sim.dummy -> dummy" in result.output


def test_a_world_without_a_control_plane_is_not_given_one(bench: Path) -> None:
    """A robot made of numbers has no topics, so none are printed for it."""
    result = run(scenario(bench), bench, "--dry-run")
    assert "joint_states" not in result.output


def test_several_driven_robots_require_naming_one(bench: Path) -> None:
    path = scenario(
        bench,
        robots=[
            {"id": "arm1", "profile": "arm", "role": "scripted"},
            {"id": "arm2", "profile": "arm", "role": "scripted"},
        ],
    )
    result = run(path, bench)
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "--robot" in result.output


def test_naming_a_robot_resolves_it(bench: Path) -> None:
    path = scenario(
        bench,
        robots=[
            {"id": "arm1", "profile": "arm", "role": "scripted"},
            {"id": "arm2", "profile": "arm", "role": "scripted"},
        ],
    )
    result = run(path, bench, "--robot", "arm2", "--dry-run")
    assert result.exit_code == EXIT_OK, result.output
    assert "робот     arm2" in result.output


def test_a_scene_of_obstacles_has_nothing_to_drive(bench: Path) -> None:
    path = scenario(bench, robots=[{"id": "box", "profile": "arm", "role": "static"}])
    result = run(path, bench)
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "препятствиями" in result.output


def test_a_scenario_without_a_task_says_so(bench: Path) -> None:
    path = scenario(bench)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    del document["task"]
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")

    result = run(path, bench)
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "не объявляет задачу" in result.output


def test_an_unknown_skill_is_reported_before_connecting(bench: Path) -> None:
    result = run(scenario(bench), bench, "--skill", "no_such_skill")
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "no_such_skill" in result.output


class RefusingAdapter(DummyAdapter):
    """A link that cannot be established.

    Stands in for the commonest real case -- an environment that was never
    sourced -- without depending on this machine lacking ROS 2. A test that
    passed because rclpy happens to be absent would stop testing anything the
    day it is present.
    """

    def connect(self, profile: object, config: dict[str, object]) -> None:
        raise RuntimeError("окружение не подключено")


class UnsourcedAdapter(DummyAdapter):
    """A link whose library fails to import something on first use, inside connect."""

    def connect(self, profile: object, config: dict[str, object]) -> None:
        raise ModuleNotFoundError("No module named 'numpy'")


def test_an_import_failing_inside_connect_is_reported_not_unwound(tmp_path: Path) -> None:
    """The complex scene on Linux: rclpy loaded, a message module did not, and run
    came out as a traceback."""
    profile = make_profile(
        name="arm",
        description=ModelDescription(urdf="urdf/arm.urdf"),
        adapters={"sim.dummy": AdapterSpec(plugin="test_cli_run:UnsourcedAdapter")},
    )
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(tmp_path, profile)

    result = run(scenario(tmp_path), tmp_path)

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "связь 'sim.dummy' не установлена" in result.output
    assert "No module named 'numpy'" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


class NarrowAdapter(DummyAdapter):
    """A link guarding an envelope of ±1, narrower than the profile's ±2.

    A skill starts inside the profile's envelope (ADR-0032), so a refusal is
    provoked by a link whose own limits are tighter than what the skill plans
    against.
    """

    def connect(self, profile: RobotProfile, config: dict[str, object]) -> None:
        dof = profile.contract.dof
        narrow = Limits(joint_position=JointRange(min=[-1.0] * dof, max=[1.0] * dof))
        super().connect(profile.model_copy(update={"limits": narrow}), config)


def test_a_refused_command_is_a_failed_run_not_a_traceback(tmp_path: Path) -> None:
    """The second humanoid run, on a robot made of numbers: the first command is
    refused, and the run ends with the reason instead of a traceback."""
    profile = make_profile(
        name="arm",
        description=ModelDescription(urdf="urdf/arm.urdf"),
        adapters={
            "sim.dummy": AdapterSpec(
                plugin="test_cli_run:NarrowAdapter", params={"initial_position": [0.0, 0.0, 1.5]}
            )
        },
    )
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(tmp_path, profile)

    result = run(scenario(tmp_path), tmp_path)

    assert result.exit_code == EXIT_FOUND_ERRORS, result.output
    assert "отказ" in result.output
    assert "joint2 = 1.5 при [-1, 1]" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_link_that_cannot_be_established_is_reported_not_unwound(tmp_path: Path) -> None:
    """A setup failure is reported; it does not come out as a traceback."""
    profile = make_profile(
        name="arm",
        description=ModelDescription(urdf="urdf/arm.urdf"),
        adapters={"sim.dummy": AdapterSpec(plugin="test_cli_run:RefusingAdapter")},
    )
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(tmp_path, profile)

    result = run(scenario(tmp_path), tmp_path)

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "связь 'sim.dummy' не установлена" in result.output
    assert "окружение не подключено" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
