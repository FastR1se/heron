"""The shipped examples must validate, and must keep demonstrating what they
were written to demonstrate.

These tests skip themselves when ``examples/`` is absent: invariant 1 says the
core survives deleting the robots, and a test suite that fails without them
would contradict it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from heron.core.enums import ControlMode
from heron.core.loading import load_model
from heron.core.models import RobotProfile
from heron.registry.registry import Registry
from heron.scenario.loader import load_scenario
from heron.skills.library import SkillLibrary
from heron.validator.rules import check_profile, check_scenario, check_skill_against_profile


@pytest.fixture
def robots(examples_dir: Path) -> Path:
    return examples_dir / "robots"


def test_every_example_profile_validates(robots: Path) -> None:
    files = sorted(robots.glob("*.yaml"))
    assert files, "примеров профилей нет"
    for path in files:
        report = check_profile(load_model(path, RobotProfile))
        assert report.ok, f"{path.name}: {[f.code for f in report.errors]}"


def test_example_scenario_validates(examples_dir: Path, robots: Path) -> None:
    scenario = load_scenario(examples_dir / "scenarios" / "bench.yaml")
    report = check_scenario(scenario, Registry([robots]))
    assert report.ok, [f.code for f in report.errors]


def test_registry_finds_examples_by_name(robots: Path) -> None:
    registry = Registry([robots])
    assert "arm6" in registry.names()
    assert registry.load("arm6").contract.dof == 6


def test_the_two_examples_differ_in_interface_class(robots: Path) -> None:
    """Section 9 wants reference implementations of different interface classes."""
    registry = Registry([robots])
    assert registry.load("arm6").control_mode is ControlMode.STREAMING
    assert registry.load("rest_arm").control_mode is ControlMode.COMMAND


def test_streaming_skill_is_rejected_on_the_command_robot(robots: Path) -> None:
    """The example that earns its place: same skill, two robots, one refusal."""
    registry = Registry([robots])
    spec = SkillLibrary().spec("hold")

    assert check_skill_against_profile(spec, registry.load("arm6")).ok

    report = check_skill_against_profile(spec, registry.load("rest_arm"))
    assert not report.ok
    assert "skill.control_mode_incompatible" in report.codes()


def test_locked_contracts_match_the_profiles(robots: Path) -> None:
    """The lock file in the repository is current (ADR-0004)."""
    from heron.registry.lock import ContractLock

    lock = ContractLock.load(robots)
    assert lock.entries, "contracts.lock.json пуст — выполните contract lock"
    registry = Registry([robots])
    for name, entry in lock.entries.items():
        assert registry.load(name).contract.contract_id == entry.contract_id


# -- phase 2: the examples carry descriptions and generate worlds ----------


def test_every_example_profile_agrees_with_its_description(robots: Path) -> None:
    """The examples exercise the rules of ADR-0009 on real files, not fixtures."""
    for path in sorted(robots.glob("*.yaml")):
        report = check_profile(load_model(path, RobotProfile), path=path)
        assert report.ok, f"{path.name}: {[f.code for f in report.errors]}"


def test_the_only_description_warning_is_the_gripper(robots: Path) -> None:
    """A warning the example is written to produce: the fingers are driven by
    the adapter, not by the contract's action vector."""
    path = robots / "arm6.yaml"
    report = check_profile(load_model(path, RobotProfile), path=path)
    codes = [code for code in report.codes() if code.startswith("urdf.")]
    assert codes == ["urdf.joint_not_in_contract"]


def test_the_command_robot_is_described_in_degrees(robots: Path) -> None:
    """Its contract says degrees, its description says radians, and the two
    agree only because the comparison converts."""
    from heron.core.enums import AngleUnit

    path = robots / "rest_arm.yaml"
    profile = load_model(path, RobotProfile)
    assert profile.contract.angle_unit is AngleUnit.DEG
    assert check_profile(profile, path=path).ok


def test_the_gazebo_scenario_validates(examples_dir: Path, robots: Path) -> None:
    scenario = load_scenario(examples_dir / "scenarios" / "table.yaml")
    report = check_scenario(scenario, Registry([robots]))
    assert report.ok, [f.code for f in report.errors]


def test_the_gazebo_scenario_generates_a_world(
    examples_dir: Path, robots: Path, tmp_path: Path
) -> None:
    """Phase 2 end to end, minus the simulator: profile -> validator ->
    generators -> artifacts a world can be started from."""
    from heron.generators import build

    path = examples_dir / "scenarios" / "table.yaml"
    bundle = build(
        load_scenario(path), Registry([robots]), into=tmp_path / "out", scenario_path=path
    )
    bundle.write()

    assert (tmp_path / "out" / "bringup.launch.xml").is_file()
    assert (tmp_path / "out" / "arm1" / "controllers.yaml").is_file()
    # The obstacle gets a description and no controllers at all.
    assert (tmp_path / "out" / "arm2" / "description.urdf").is_file()
    assert not (tmp_path / "out" / "arm2" / "controllers.yaml").exists()
    assert bundle.differences() == []
