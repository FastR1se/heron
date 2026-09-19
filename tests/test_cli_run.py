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

from conftest import make_command_profile, make_profile, write_profile, write_urdf
from heron.adapters.dummy import DummyAdapter
from heron.cli.main import EXIT_FOUND_ERRORS, EXIT_OK, app
from heron.cli.run import execute
from heron.core.models import (
    AdapterSpec,
    JointRange,
    Limits,
    ModelDescription,
    RobotProfile,
)
from heron.registry.registry import Registry

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


# -- on hardware, and slowed (ADR-0051, ADR-0052) ------------------------------------


class StreamOnlyAdapter(DummyAdapter):
    """A link that says it cannot slow the robot."""

    supports_speed_scale = False


class ScaledAdapter(DummyAdapter):
    """Remembers the configuration it was connected with."""

    config: dict[str, object] | None = None

    def connect(self, profile: RobotProfile, config: dict[str, object]) -> None:
        type(self).config = dict(config)
        super().connect(profile, config)


def _hardware(tmp_path: Path, plugin: str = "dummy", **overrides: object) -> Path:
    """A command robot with a simulated world of numbers and a link to the real one."""
    fields: dict[str, object] = {
        "description": ModelDescription(urdf="urdf/arm.urdf"),
        "adapters": {"sim.dummy": AdapterSpec(plugin="dummy"), "real": AdapterSpec(plugin=plugin)},
    }
    fields.update(overrides)
    profile = make_command_profile(name="arm", **fields)
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(tmp_path, profile)
    return scenario(
        tmp_path, task={"skill": "goto_joints", "target": [0.2, 0.0, 0.0], "tolerance": 1e-6}
    )


def test_a_run_on_hardware_requires_a_speed_scale(tmp_path: Path) -> None:
    result = run(_hardware(tmp_path), tmp_path, "--target", "real", "--dry-run")

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "шкала скорости обязательна" in result.output


def test_the_plan_of_a_run_on_hardware_names_reality_and_the_scale(tmp_path: Path) -> None:
    path = _hardware(tmp_path)
    result = run(path, tmp_path, "--target", "real", "--speed-scale", "0.1", "--dry-run")

    assert result.exit_code == EXIT_OK, result.output
    assert "мир       действительность, окружение мира не применяется" in result.output
    assert "шкала     скорости 0.1" in result.output


def test_a_simulated_run_is_at_full_speed_unless_told(tmp_path: Path) -> None:
    result = run(_hardware(tmp_path), tmp_path, "--dry-run")

    assert result.exit_code == EXIT_OK, result.output
    assert "мир       dummy  (владелец физики)" in result.output
    assert "шкала     скорости 1" in result.output


def test_a_robot_without_an_envelope_does_not_go_to_hardware(tmp_path: Path) -> None:
    unbounded = Limits(joint_position=JointRange(min=[-2.0] * 3, max=[2.0] * 3))
    path = _hardware(tmp_path, limits=unbounded)
    result = run(path, tmp_path, "--target", "real", "--speed-scale", "1", "--dry-run")

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "[real.limits_missing]" in result.output
    assert "joint_velocity" in result.output


@pytest.mark.parametrize("scale", ["0", "1.5", "-0.5"])
def test_a_speed_scale_outside_its_range_is_refused(tmp_path: Path, scale: str) -> None:
    result = run(_hardware(tmp_path), tmp_path, "--speed-scale", scale, "--dry-run")

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "0 < s ≤ 1" in result.output


def test_an_adapter_that_cannot_slow_the_robot_is_not_handed_a_scale(tmp_path: Path) -> None:
    path = _hardware(tmp_path, plugin="test_cli_run:StreamOnlyAdapter")
    result = run(path, tmp_path, "--target", "real", "--speed-scale", "0.5")

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "не исполняет шкалу скорости" in result.output


def test_a_dry_run_refuses_the_scale_the_run_would_refuse(tmp_path: Path) -> None:
    """The declaration is read off the class: nothing is connected to answer it."""
    path = _hardware(tmp_path, plugin="test_cli_run:StreamOnlyAdapter")
    result = run(path, tmp_path, "--target", "real", "--speed-scale", "0.5", "--dry-run")

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "не исполняет шкалу скорости" in result.output


def test_the_plan_of_a_command_robot_in_gazebo_names_the_action_it_sends_goals_to(
    tmp_path: Path,
) -> None:
    """The command class through ros2_control: goals to the trajectory controller's
    action, not commands on a topic (ADR-0060)."""
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(
        tmp_path,
        make_command_profile(
            name="arm",
            description=ModelDescription(urdf="urdf/arm.urdf"),
            adapters={"sim.gazebo": AdapterSpec(plugin="ros2_control")},
        ),
    )
    path = scenario(
        tmp_path,
        world={"backend": "gazebo"},
        task={"skill": "goto_joints", "target": [0.2, 0.0, 0.0], "tolerance": 0.01},
    )
    result = run(path, tmp_path, "--speed-scale", "0.5", "--dry-run")

    assert result.exit_code == EXIT_OK, result.output
    assert "цели      /arm1/joint_trajectory_controller/follow_joint_trajectory" in result.output
    assert "шкала     скорости 0.5" in result.output


def test_a_stream_is_not_slowed_whatever_its_adapter_says(tmp_path: Path) -> None:
    """Known from the profile: a stream is planned by its skill (ADR-0052)."""
    write_urdf(tmp_path, "arm", JOINTS)
    write_profile(
        tmp_path,
        make_profile(
            name="arm",
            description=ModelDescription(urdf="urdf/arm.urdf"),
            adapters={"sim.dummy": AdapterSpec(plugin="dummy")},
        ),
    )
    result = run(scenario(tmp_path), tmp_path, "--speed-scale", "0.5", "--dry-run")

    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "робот потокового класса" in result.output


def test_a_dry_run_without_the_plugin_says_the_scale_is_unchecked(tmp_path: Path) -> None:
    path = _hardware(tmp_path, plugin="test_cli_run:AbsentAdapter")
    result = run(path, tmp_path, "--target", "real", "--speed-scale", "0.5", "--dry-run")

    assert result.exit_code == EXIT_OK, result.output
    assert "шкала     не сверена с адаптером" in result.output


def test_the_scale_reaches_the_adapter_and_slows_the_run(tmp_path: Path) -> None:
    path = _hardware(tmp_path, plugin="test_cli_run:ScaledAdapter")
    from heron.scenario.loader import load_scenario

    _, full = execute(load_scenario(path), Registry([tmp_path]), target="real", speed_scale=1.0)
    _, slow = execute(load_scenario(path), Registry([tmp_path]), target="real", speed_scale=0.25)

    assert full is not None and full.ok and slow is not None and slow.ok
    assert ScaledAdapter.config is not None and ScaledAdapter.config["speed_scale"] == 0.25
    assert slow.states[-1].timestamp > 3 * full.states[-1].timestamp


def test_a_profile_cannot_state_the_speed_scale() -> None:
    with pytest.raises(ValueError, match="величина прогона"):
        AdapterSpec(plugin="dummy", params={"speed_scale": 0.1})
