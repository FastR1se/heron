"""CLI behaviour, including the exit codes CI depends on.

0 -- nothing found, 1 -- errors found, 2 -- could not read the input. A command
that printed an error and still exited 0 would make the validator useless in a
pipeline, which is where it matters most.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import make_profile
from heron.cli.main import EXIT_FOUND_ERRORS, EXIT_OK, EXIT_UNREADABLE, app

runner = CliRunner()


@pytest.fixture
def profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_arm.yaml"
    path.write_text(make_profile().model_dump_json(indent=2), encoding="utf-8")
    return path


def test_version_reports_paths() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == EXIT_OK
    assert "_HOME" in result.stdout  # the env var, built from the prefix


def test_capabilities_lists_the_closed_registry() -> None:
    result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == EXIT_OK
    assert "arm.joint_position" in result.stdout
    assert "закрыт" in result.stdout


def test_validate_clean_profile_exits_zero(profile_file: Path) -> None:
    result = runner.invoke(app, ["validate", str(profile_file)])
    assert result.exit_code == EXIT_OK


def test_validate_broken_profile_exits_one(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(
        make_profile(image="heron/arm:latest").model_dump_json(indent=2), encoding="utf-8"
    )
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "profile.image_has_namespace" in result.stdout


def test_unreadable_file_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "not_yaml.yaml"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == EXIT_UNREADABLE


def test_contract_id_prints_the_identifier(profile_file: Path) -> None:
    result = runner.invoke(app, ["contract", "id", str(profile_file)])
    assert result.exit_code == EXIT_OK
    assert result.stdout.strip() == make_profile().contract.contract_id


def test_contract_diff_reports_fields(tmp_path: Path, profile_file: Path) -> None:
    other = tmp_path / "other.yaml"
    changed = make_profile(name="test_arm")
    data = changed.model_dump(mode="json")
    data["contract"]["joint_order"] = ["joint1", "joint0", "joint2"]
    import json

    other.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(app, ["contract", "diff", str(profile_file), str(other)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "joint_order" in result.stdout


def test_contract_lock_then_validate(tmp_path: Path, profile_file: Path) -> None:
    assert runner.invoke(app, ["contract", "lock", str(profile_file)]).exit_code == EXIT_OK
    assert (tmp_path / "contracts.lock.json").is_file()
    assert runner.invoke(app, ["validate", str(profile_file)]).exit_code == EXIT_OK


def test_silent_contract_change_fails_validation(tmp_path: Path, profile_file: Path) -> None:
    """The end-to-end version of ADR-0004's rule."""
    runner.invoke(app, ["contract", "lock", str(profile_file)])

    import json

    data = json.loads(profile_file.read_text(encoding="utf-8"))
    data["contract"]["joint_order"] = ["joint1", "joint0", "joint2"]
    profile_file.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(app, ["validate", str(profile_file)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "contract.undeclared_change" in result.stdout


@pytest.mark.parametrize("mode", ["streaming", "command"])
def test_profile_new_creates_a_validating_stub(tmp_path: Path, mode: str) -> None:
    args = ["profile", "new", "my_bot", "--into", str(tmp_path), "--control-mode", mode]
    assert runner.invoke(app, args).exit_code == 0
    stub = tmp_path / "my_bot" / "profile.yaml"
    assert stub.is_file()
    assert (tmp_path / "my_bot" / "src" / "my_bot" / "adapter.py").is_file()
    assert f"control_mode: {mode}" in stub.read_text(encoding="utf-8")
    assert runner.invoke(app, ["validate", str(stub)]).exit_code == EXIT_OK


def test_profile_new_refuses_to_overwrite(tmp_path: Path) -> None:
    runner.invoke(app, ["profile", "new", "my_bot", "--into", str(tmp_path)])
    result = runner.invoke(app, ["profile", "new", "my_bot", "--into", str(tmp_path)])
    assert result.exit_code == EXIT_UNREADABLE


DESCRIBED_ROBOT = """<?xml version="1.0"?>
<robot name="described_arm">
  <link name="world"/><link name="base"/><link name="l1"/><link name="l2"/>
  <link name="l3"/><link name="f1"/><link name="f2"/><link name="tool"/>
  <joint name="mount" type="fixed"><parent link="world"/><child link="base"/></joint>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="l1"/>
    <limit lower="-0.087267" upper="2.8798" effort="88" velocity="32"/></joint>
  <joint name="wrist-roll" type="continuous"><parent link="l1"/><child link="l2"/>
    <limit effort="25" velocity="37"/></joint>
  <joint name="lift" type="prismatic"><parent link="l2"/><child link="l3"/>
    <limit lower="0" upper="0.3" effort="100"/></joint>
  <joint name="finger" type="prismatic"><parent link="l3"/><child link="f1"/>
    <limit lower="0" upper="0.04" effort="5" velocity="0.1"/></joint>
  <joint name="finger_mirror" type="prismatic"><parent link="l3"/><child link="f2"/>
    <limit lower="0" upper="0.04" effort="5" velocity="0.1"/>
    <mimic joint="finger"/></joint>
  <joint name="flange" type="fixed"><parent link="l3"/><child link="tool"/></joint>
</robot>
"""


def _described(tmp_path: Path) -> Path:
    path = tmp_path / "model" / "described_arm.urdf"
    path.parent.mkdir()
    path.write_text(DESCRIBED_ROBOT, encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["streaming", "command"])
def test_profile_new_from_a_description_validates(tmp_path: Path, mode: str) -> None:
    """ADR-0074: the stub read from a description agrees with it on the day it is made."""
    from heron.core.loading import load_model
    from heron.core.models import RobotProfile

    urdf = _described(tmp_path)
    args = ["profile", "new", "described_arm", "--into", str(tmp_path / "stubs"),
            "--control-mode", mode, "--from-urdf", str(urdf)]
    assert runner.invoke(app, args).exit_code == EXIT_OK

    root = tmp_path / "stubs" / "described_arm"
    assert not (root / "urdf").exists()  # referenced, not copied
    profile_path = root / "profile.yaml"
    profile = load_model(profile_path, RobotProfile)
    assert Path(profile.description.urdf).as_posix() == "../../model/described_arm.urdf"
    assert list(profile.contract.joint_order) == ["shoulder", "wrist-roll", "lift", "finger"]
    position = profile.limits.joint_position
    assert position is not None
    assert position.min[0] == -0.087267 and position.max[0] == 2.8798
    assert (position.min[2], position.max[2]) == (0.0, 0.3)
    assert profile.limits.joint_torque is not None
    assert profile.limits.joint_torque.max == [88.0, 25.0, 100.0, 5.0]

    text = profile_path.read_text(encoding="utf-8")
    assert "записано ±π (линейный — ±1.0 м): wrist-roll" in text
    assert "скорость в описании не задана, записано 1.0: lift" in text

    result = runner.invoke(app, ["validate", str(profile_path)])
    assert result.exit_code == EXIT_OK, result.stdout
    assert "ошибка" not in result.stdout


def test_profile_new_from_a_description_maps_the_action_into_its_frame(tmp_path: Path) -> None:
    from heron.core.loading import load_model
    from heron.core.models import RobotProfile

    args = ["profile", "new", "described_arm", "--into", str(tmp_path),
            "--from-urdf", str(_described(tmp_path))]
    assert runner.invoke(app, args).exit_code == EXIT_OK
    profile = load_model(tmp_path / "described_arm" / "profile.yaml", RobotProfile)
    position = profile.limits.joint_position
    assert position is not None
    targets = profile.contract.action_targets()
    assert targets is not None
    for (low, high), lo, hi in zip(targets, position.min, position.max, strict=True):
        assert lo <= low and high <= hi


@pytest.mark.parametrize(
    ("extra", "name"),
    [
        (["--dof", "3"], "described_arm.urdf"),
        ([], "described_arm.urdf.xacro"),
        ([], "missing.urdf"),
    ],
)
def test_profile_new_from_a_description_refuses(
    tmp_path: Path, extra: list[str], name: str
) -> None:
    urdf = _described(tmp_path)
    if name.endswith(".xacro"):
        urdf = urdf.with_name(name)
        urdf.write_text(DESCRIBED_ROBOT, encoding="utf-8")
    elif name != urdf.name:
        urdf = urdf.with_name(name)
    args = ["profile", "new", "some_bot", "--into", str(tmp_path), "--from-urdf", str(urdf)]
    result = runner.invoke(app, args + extra)
    assert result.exit_code == EXIT_UNREADABLE
    assert not (tmp_path / "some_bot").exists()


def test_profile_new_from_a_description_without_driven_joints_is_refused(
    tmp_path: Path,
) -> None:
    urdf = tmp_path / "frame.urdf"
    urdf.write_text(
        '<?xml version="1.0"?><robot name="frame"><link name="a"/><link name="b"/>'
        '<joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint></robot>',
        encoding="utf-8",
    )
    args = ["profile", "new", "frame", "--into", str(tmp_path), "--from-urdf", str(urdf)]
    result = runner.invoke(app, args)
    assert result.exit_code == EXIT_UNREADABLE
    assert "нет управляемых суставов" in result.output


def test_skill_check_reports_incompatibility(tmp_path: Path) -> None:
    from heron.core.enums import ControlMode

    path = tmp_path / "cmd_arm.yaml"
    path.write_text(
        make_profile(control_mode=ControlMode.COMMAND).model_dump_json(), encoding="utf-8"
    )
    result = runner.invoke(app, ["skill", "check", "hold", str(path)])
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "skill.control_mode_incompatible" in result.stdout


def test_skill_list_includes_builtin() -> None:
    result = runner.invoke(app, ["skill", "list"])
    assert result.exit_code == EXIT_OK
    assert "hold" in result.stdout


# -- gen and sim -----------------------------------------------------------


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """A profile, its description and a scenario, all in one directory."""
    from conftest import write_profile, write_urdf
    from heron.core.models import AdapterSpec

    profile = make_profile(
        name="arm",
        description={"urdf": "urdf/arm.urdf"},
        adapters={"sim.gazebo": AdapterSpec(plugin="ros2_control")},
    )
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)

    path = tmp_path / "scene.yaml"
    path.write_text(
        "schema_version: 1\nname: scene\nworld: {backend: gazebo}\n"
        "robots:\n  - {id: arm1, profile: arm, role: scripted}\n",
        encoding="utf-8",
    )
    return path


def test_gen_writes_the_bundle(scene: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(app, ["gen", str(scene), "--into", str(out), "-R", str(tmp_path)])

    assert result.exit_code == EXIT_OK, result.stdout
    assert (out / "bringup.launch.xml").is_file()
    assert (out / "arm1" / "description.urdf").is_file()


def test_gen_check_passes_on_fresh_artifacts(scene: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    runner.invoke(app, ["gen", str(scene), "--into", str(out), "-R", str(tmp_path)])

    result = runner.invoke(
        app, ["gen", str(scene), "--into", str(out), "--check", "-R", str(tmp_path)]
    )
    assert result.exit_code == EXIT_OK
    assert "актуально" in result.stdout


def test_gen_check_fails_on_stale_artifacts(scene: Path, tmp_path: Path) -> None:
    """The cheap CI check of ADR-0010: artifacts that fell behind the profile."""
    out = tmp_path / "out"
    runner.invoke(app, ["gen", str(scene), "--into", str(out), "-R", str(tmp_path)])
    (out / "arm1" / "controllers.yaml").write_text("подменено", encoding="utf-8")

    result = runner.invoke(
        app, ["gen", str(scene), "--into", str(out), "--check", "-R", str(tmp_path)]
    )
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "устарело" in result.stdout


def test_gen_of_a_lone_profile_needs_no_scenario(tmp_path: Path, scene: Path) -> None:
    result = runner.invoke(
        app, ["gen", str(tmp_path / "arm.yaml"), "--into", str(tmp_path / "solo")]
    )
    assert result.exit_code == EXIT_OK, result.stdout
    assert (tmp_path / "solo" / "arm" / "controllers.yaml").is_file()


def test_gen_reports_an_unreadable_input_as_such(tmp_path: Path) -> None:
    result = runner.invoke(app, ["gen", str(tmp_path / "absent.yaml"), "--into", str(tmp_path)])
    assert result.exit_code == EXIT_UNREADABLE


def test_sim_run_dry_shows_the_command_and_starts_nothing(scene: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["sim", "run", str(scene), "--into", str(tmp_path / "out"), "--dry-run",
         "-R", str(tmp_path)],
    )
    assert result.exit_code == EXIT_OK, result.stdout
    assert "ros2 launch" in result.stdout


def test_sim_run_refuses_a_scene_that_does_not_validate(tmp_path: Path) -> None:
    from conftest import write_profile, write_urdf
    from heron.core.models import AdapterSpec

    profile = make_profile(
        name="arm",
        description={"urdf": "urdf/arm.urdf"},
        adapters={"sim.dummy": AdapterSpec(plugin="dummy")},
    )
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)
    path = tmp_path / "scene.yaml"
    path.write_text(
        "schema_version: 1\nname: scene\nworld: {backend: gazebo}\n"
        "robots:\n  - {id: arm1, profile: arm, role: scripted}\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["sim", "run", str(path), "--into", str(tmp_path / "out"), "--dry-run",
         "-R", str(tmp_path)],
    )
    assert result.exit_code == EXIT_FOUND_ERRORS
    assert "scenario.backend_adapter_missing" in result.output
