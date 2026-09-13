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


# -- phase 4: a vendor description, fetched rather than copied -------------

G1 = "g1_23dof"
FETCH_HINT = "python examples/robots/g1_23dof/fetch_description.py"


def _vendor_profile(robots: Path) -> Path:
    """The G1 profile, skipping when its description was never fetched.

    The description is not in the repository (it is the vendor's, and 110 MB),
    so a checkout without the download is a normal state rather than a broken
    one. The tests that do not need the file run regardless.
    """
    from heron.generators.common import resolve_description

    path = robots / G1 / "profile.yaml"
    if not resolve_description(load_model(path, RobotProfile), path).is_file():
        pytest.skip(f"описание {G1} не загружено: {FETCH_HINT}")
    return path


def test_the_vendor_robot_is_found_by_its_directory(robots: Path) -> None:
    profile = Registry([robots]).load(G1)
    assert profile.contract.dof == 23


def test_the_vendor_robot_contract_is_locked(robots: Path) -> None:
    """A profile in its own directory keeps its lock next to it, where
    ``contract lock`` and ``validate`` both look."""
    from heron.registry.lock import ContractLock

    lock = ContractLock.load(robots / G1)
    entry = lock.get(G1)
    assert entry is not None, f"контракт {G1} не зафиксирован — выполните contract lock"
    assert Registry([robots]).load(G1).contract.contract_id == entry.contract_id


def test_the_vendor_robot_agrees_with_its_description(robots: Path) -> None:
    """23 joints, their limits and every mesh, against the vendor's own file.

    The one warning is a condition of the run, not a defect: the base is free
    and the built-in controller is off (ADR-0018)."""
    path = _vendor_profile(robots)
    report = check_profile(load_model(path, RobotProfile), path=path)

    assert report.ok, [f.code for f in report.errors]
    assert [c for c in report.codes() if c.startswith("urdf.")] == [
        "urdf.free_base_uncontrolled"
    ]


def test_the_vendor_robot_generates_with_every_mesh_anchored(
    robots: Path, tmp_path: Path
) -> None:
    """ADR-0017 on a real description: nothing in the copy resolves against a
    directory the simulator will never see."""
    from xml.etree import ElementTree as ET

    from heron.generators import build_profile

    path = _vendor_profile(robots)
    profile = load_model(path, RobotProfile)
    bundle = build_profile(profile, path, into=tmp_path / "out", backend="gazebo")
    description = next(f for f in bundle.files if str(f.path) == f"{G1}/description.urdf")

    tree = ET.fromstring(description.content.split("?>", 1)[1])
    references = [mesh.get("filename", "") for mesh in tree.iter("mesh")]
    assert references
    assert all(ref.startswith("file://") for ref in references), references[:3]


def test_the_vendor_robot_scenario_validates_and_its_goal_fits(
    examples_dir: Path, robots: Path
) -> None:
    """The scene the Linux run starts from: valid, and a goal the skill accepts."""
    from heron.core.enums import Role

    _vendor_profile(robots)
    scenario = load_scenario(examples_dir / "scenarios" / "humanoid.yaml")
    report = check_scenario(scenario, Registry([robots]))
    assert report.ok, [f.code for f in report.errors]

    profile = Registry([robots]).load(G1)
    driven = [robot for robot in scenario.robots if robot.role is not Role.STATIC]
    assert [robot.profile for robot in driven] == [G1]

    assert scenario.task is not None
    target = scenario.task.goal["target"]
    bounds = profile.limits.joint_position
    assert bounds is not None
    assert len(target) == profile.contract.dof
    assert all(lo <= v <= hi for v, lo, hi in zip(target, bounds.min, bounds.max, strict=True))
