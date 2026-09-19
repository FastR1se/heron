"""What a side of a link needs from its machine, and whether the machine has it.

Three things are held apart here, because each fails on its own. The form
refuses requirements no machine could be asked. The check draws the right
conclusion from what a machine reports -- on a machine made of a dictionary,
because this one has neither ROS 2 nor a robot's SDK and a test that passed
because of that would stop testing anything the day they appear. And the local
machine is asked on real files: an ament index laid out in a temporary
directory, the interpreter running the tests, a variable in a mapping.

The case ADR-0022 exists for runs through all three: a DDS binding in a version
that crashes the robot on discovery is both the wrong version and a forbidden
one, and says both.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from conftest import make_profile, write_profile, write_urdf
from heron.cli import main as cli
from heron.cli import run as run_command
from heron.cli.new_robot import write_stub
from heron.core.enums import TwinKind
from heron.core.environment import (
    Requirement,
    RequirementError,
    RequirementKind,
    forbids,
    needs,
)
from heron.core.models import AdapterSpec, ModelDescription, Scenario
from heron.registry.registry import Registry
from heron.validator import environment
from heron.validator.environment import (
    Found,
    LocalMachine,
    UndeclaredRequirementsError,
    check_adapter,
    check_environment,
    check_requirements,
    check_twin,
    declared_requirements,
)

PY = RequirementKind.PYTHON
ROS2 = RequirementKind.ROS2
runner = CliRunner()


class FakeMachine:
    """A machine made of a dictionary: (kind, name) -> what it would report."""

    def __init__(
        self, found: dict[tuple[str, str], Found] | None = None, *, sourced: bool = True
    ) -> None:
        self.found = found or {}
        self.sourced = sourced
        self.asked: list[str] = []

    def find(self, requirement: Requirement) -> Found | None:
        self.asked.append(requirement.name)
        return self.found.get((str(requirement.kind), requirement.name))

    def ros2_sourced(self) -> bool:
        return self.sourced


def _codes(report: Any) -> list[str]:
    return sorted(report.codes())


# -- the form ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: needs(PY, "unitree_sdk2py.core", hint="x"), "имя"),
        (lambda: needs(ROS2, "UnitreeHg", hint="x"), "имя"),
        (lambda: needs(PY, "cyclonedds", hint="  "), "подсказки"),
        (lambda: needs(RequirementKind.EXECUTABLE, "gz", hint="x", version=">=8"), "версию"),
        (lambda: needs(RequirementKind.VARIABLE, "DISPLAY", hint="x", version="==1"), "версию"),
        (lambda: needs(PY, "cyclonedds", hint="x", version="0.10.2"), "PEP 440"),
        (lambda: needs(PY, "cyclonedds", hint="x", version=""), "пустое"),
        (lambda: Requirement(PY, "cyclonedds", hint="x", forbidden=True), "без причины"),
    ],
)
def test_a_requirement_no_machine_can_be_asked_is_refused(build: Any, fragment: str) -> None:
    with pytest.raises(RequirementError, match=fragment):
        build()


def test_a_requirement_reads_the_way_it_is_written() -> None:
    requirement = forbids(PY, "cyclonedds", version=">=0.11", because="роняет 0.10.2", hint="x")
    assert requirement.describe() == "python cyclonedds >=0.11"
    assert requirement.admits("11.0.1")
    assert not requirement.admits("0.10.5")
    assert needs(RequirementKind.VARIABLE, "DISPLAY", hint="x").admits("anything")


# -- the check -----------------------------------------------------------------


def test_what_is_needed_and_absent_is_missing() -> None:
    report = check_requirements(
        [needs(PY, "unitree_sdk2py", hint="pip install -e unitree_sdk2_python")],
        where="adapter:vendor",
        machine=FakeMachine(),
    )
    assert _codes(report) == ["env.missing"]
    assert report.findings[0].hint == "pip install -e unitree_sdk2_python"
    assert report.findings[0].adr == "ADR-0025"


def test_a_ros2_package_missing_from_an_unsourced_shell_says_to_source_it() -> None:
    """Absent from the machine and absent from this shell are fixed differently."""
    requirement = needs(ROS2, "unitree_hg", hint="colcon build unitree_hg")

    unsourced = check_requirements([requirement], where="twin", machine=FakeMachine(sourced=False))
    sourced = check_requirements([requirement], where="twin", machine=FakeMachine(sourced=True))

    assert "source /opt/ros" in (unsourced.findings[0].hint or "")
    assert "не подключено" in unsourced.findings[0].message
    assert "source" not in (sourced.findings[0].hint or "")


@pytest.mark.parametrize(
    ("found", "codes"),
    [
        (Found("site-packages", "0.10.2"), []),
        (Found("site-packages", "0.10.5"), ["env.version_unmet"]),
        (Found("site-packages", None), ["env.version_unmet"]),
    ],
)
def test_a_version_condition_is_met_or_not_and_unknown_is_not(
    found: Found, codes: list[str]
) -> None:
    """A version that cannot be read does not pass a condition (ADR-0008)."""
    machine = FakeMachine({("python", "cyclonedds"): found})
    report = check_requirements(
        [needs(PY, "cyclonedds", version="==0.10.2", hint="x")], where="adapter", machine=machine
    )
    assert _codes(report) == codes


@pytest.mark.parametrize(
    ("found", "codes"),
    [
        (None, []),
        (Found("/opt/ros/jazzy", "0.10.5"), []),
        (Found("/opt/ros/rolling", "0.11.0"), ["env.forbidden"]),
        (Found("/opt/ros/somewhere", None), ["env.forbidden"]),
    ],
)
def test_a_prohibition_names_only_what_it_forbids(found: Found | None, codes: list[str]) -> None:
    """Jazzy's 0.10.5 is fine, 0.11 is not, and an unreadable version is not trusted."""
    machine = FakeMachine({("ros2", "cyclonedds"): found} if found else {})
    report = check_requirements(
        [forbids(ROS2, "cyclonedds", version=">=0.11", because="роняет 0.10.2", hint="убрать")],
        where="twin",
        machine=machine,
    )
    assert _codes(report) == codes
    if codes:
        assert "роняет 0.10.2" in report.findings[0].message
        assert report.findings[0].hint == "убрать"


def test_a_prohibition_without_a_version_forbids_any() -> None:
    requirement = forbids(
        RequirementKind.VARIABLE, "CYCLONEDDS_URI", because="двойник его не читает", hint="unset"
    )
    machine = FakeMachine({("variable", "CYCLONEDDS_URI"): Found("CYCLONEDDS_URI=x")})
    assert _codes(check_requirements([requirement], where="twin", machine=machine)) == [
        "env.forbidden"
    ]


def test_the_binding_of_the_wrong_generation_is_both_wrong_and_forbidden() -> None:
    """The case ADR-0022 was written for, as a robot package would declare it."""
    declared = [
        needs(PY, "cyclonedds", version="==0.10.2", hint="сборка из исходника"),
        forbids(PY, "cyclonedds", version=">=0.11", because="0.10.2 падает при обнаружении",
                hint="pip uninstall cyclonedds"),
    ]
    machine = FakeMachine({("python", "cyclonedds"): Found("venv", "11.0.1")})

    report = check_requirements(declared, where="adapter:vendor", machine=machine)

    assert _codes(report) == ["env.forbidden", "env.version_unmet"]
    assert all(finding.detail["found"] == "11.0.1" for finding in report.findings)


def test_a_side_that_does_not_say_what_it_needs_is_not_a_side_that_needs_nothing() -> None:
    class Silent:
        pass

    class Text:
        requirements = "cyclonedds"

    class Strangers:
        requirements = ("cyclonedds==0.10.2",)

    for owner in (Silent, Text, Strangers):
        with pytest.raises(UndeclaredRequirementsError):
            declared_requirements(owner)
    assert declared_requirements(type("Empty", (), {"requirements": ()})) == ()


# -- the local machine ---------------------------------------------------------


def _ament_prefix(root: Path, package: str, version: str | None) -> Path:
    index = root / "share" / "ament_index" / "resource_index" / "packages"
    index.mkdir(parents=True, exist_ok=True)
    (index / package).write_text("", encoding="utf-8")
    manifest = root / "share" / package / "package.xml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    body = f"<version>{version}</version>" if version else ""
    manifest.write_text(f"<package><name>{package}</name>{body}</package>", encoding="utf-8")
    return root


def test_the_local_machine_reads_the_ament_index_in_prefix_order(tmp_path: Path) -> None:
    import os

    workspace = _ament_prefix(tmp_path / "ws" / "install", "unitree_hg", "0.3.0")
    distro = _ament_prefix(tmp_path / "opt" / "jazzy", "cyclonedds", "0.10.5")
    _ament_prefix(tmp_path / "opt" / "jazzy", "unitree_hg", "0.0.1")
    machine = LocalMachine({"AMENT_PREFIX_PATH": os.pathsep.join([str(workspace), str(distro)])})

    assert machine.ros2_sourced()
    assert machine.find(needs(ROS2, "unitree_hg", hint="x")) == Found(str(workspace), "0.3.0")
    assert machine.find(needs(ROS2, "cyclonedds", hint="x")) == Found(str(distro), "0.10.5")
    assert machine.find(needs(ROS2, "rmw_cyclonedds_cpp", hint="x")) is None


def test_the_local_machine_without_ros2_finds_no_ros2_package() -> None:
    machine = LocalMachine({})
    assert not machine.ros2_sourced()
    assert machine.find(needs(ROS2, "rclpy", hint="x")) is None


def test_the_local_machine_locates_a_module_without_importing_it() -> None:
    import importlib.metadata

    machine = LocalMachine({})
    found = machine.find(needs(PY, "pydantic", hint="x"))

    assert found is not None
    assert found.version == importlib.metadata.version("pydantic")
    assert machine.find(needs(PY, "no_such_module_anywhere", hint="x")) is None


def test_the_local_machine_finds_programs_on_its_path_and_variables_in_its_environment() -> None:
    interpreter = Path(sys.executable)
    machine = LocalMachine({"PATH": str(interpreter.parent), "DISPLAY": ":0", "EMPTY": ""})

    assert machine.find(needs(RequirementKind.EXECUTABLE, interpreter.name, hint="x"))
    assert machine.find(needs(RequirementKind.EXECUTABLE, "no-such-program", hint="x")) is None
    assert machine.find(needs(RequirementKind.VARIABLE, "DISPLAY", hint="x"))
    assert machine.find(needs(RequirementKind.VARIABLE, "EMPTY", hint="x")) is None


# -- sides of a scene ------------------------------------------------------------


def _registry(tmp_path: Path, plugin: str = "ros2_control", *, real: bool = False) -> Registry:
    adapters = {
        "sim.gazebo": AdapterSpec(plugin=plugin),
        "sim.dummy": AdapterSpec(plugin="dummy"),
    }
    if real:
        adapters["real"] = AdapterSpec(plugin="dummy")
    profile = make_profile(
        name="arm",
        description=ModelDescription(urdf="urdf/arm.urdf"),
        adapters=adapters,
    )
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)
    return Registry([tmp_path])


def _scenario(backend: str = "gazebo", roles: tuple[str, ...] = ("scripted",)) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "scene",
            "world": {"backend": backend},
            "robots": [
                {"id": f"arm{index + 1}", "profile": "arm", "role": role}
                for index, role in enumerate(roles)
            ],
            "task": {"skill": "move_joints", "target": [0.2, 0.0, 0.0], "duration": 0.1},
        }
    )


def test_both_sides_of_a_link_are_checked_once_each(tmp_path: Path) -> None:
    machine = FakeMachine(sourced=False)
    result = check_environment(
        _scenario(roles=("scripted", "scripted", "static")), _registry(tmp_path), machine=machine
    )

    assert [side.where for side in result.checked] == [
        "adapter:ros2_control",
        "twin:ros2_control@gazebo",
    ]
    missing = {finding.detail["name"] for finding in result.report.findings}
    assert {"rclpy", "controller_manager", "gz_ros2_control"} <= missing
    assert set(result.report.codes()) == {"env.missing"}


def test_each_command_checks_its_own_side(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    world = check_environment(_scenario(), registry, robots=False, machine=FakeMachine())
    robot = check_environment(_scenario(), registry, world=False, machine=FakeMachine())

    assert [side.where for side in world.checked] == ["twin:ros2_control@gazebo"]
    assert [side.where for side in robot.checked] == ["adapter:ros2_control"]


def test_a_world_the_platform_does_not_start_has_no_twin_side(tmp_path: Path) -> None:
    result = check_environment(
        _scenario(backend="dummy"), _registry(tmp_path), machine=FakeMachine()
    )
    assert [side.where for side in result.checked] == ["adapter:dummy"]
    assert result.report.ok


def test_a_real_target_checks_its_adapter_and_needs_no_world(tmp_path: Path) -> None:
    """On hardware the robot owns the physics (ADR-0051): the scene's twins are not needed."""
    result = check_environment(
        _scenario(), _registry(tmp_path, real=True), target="real", machine=FakeMachine()
    )
    assert [side.where for side in result.checked] == ["adapter:dummy"]
    assert result.report.ok


def test_a_target_the_profile_has_no_adapter_for_is_an_error(tmp_path: Path) -> None:
    result = check_environment(
        _scenario(), _registry(tmp_path), target="real", machine=FakeMachine()
    )
    assert [(finding.code, finding.detail) for finding in result.report.findings] == [
        ("env.target_adapter_missing", {"target": "real", "profile": "arm"})
    ]
    assert result.checked == []


def test_an_obstacle_has_no_link_and_no_requirements(tmp_path: Path) -> None:
    machine = FakeMachine()
    result = check_environment(_scenario(roles=("static",)), _registry(tmp_path), machine=machine)
    assert result.checked == [] and machine.asked == []


def test_an_adapter_this_machine_cannot_load_is_an_error_here(tmp_path: Path) -> None:
    """The validator warns, answering for every machine; this machine cannot drive it."""
    result = check_adapter("vendor_absent", machine=FakeMachine())
    assert _codes(result.report) == ["env.unreadable"]
    assert not result.checked


def test_a_twin_that_does_not_say_what_it_needs_is_unreadable() -> None:
    class Unasked:
        adapter = "vendor_link"
        backend = "gazebo"
        kind = TwinKind.BRIDGE
        requirements = None

    result = check_twin(Unasked(), machine=FakeMachine())  # type: ignore[arg-type]
    assert _codes(result.report) == ["env.unreadable"]


def test_a_generated_adapter_stub_declares_its_requirements(tmp_path: Path) -> None:
    """Every robot added the standard way starts with the member, and empty."""
    adapter = write_stub("some_robot", tmp_path).adapter
    spec = importlib.util.spec_from_file_location("some_robot_adapter", adapter)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert declared_requirements(module.SomeRobotAdapter) == ()


# -- commands --------------------------------------------------------------------


def _scene_file(tmp_path: Path, backend: str) -> Path:
    path = tmp_path / "scene.yaml"
    path.write_text(
        yaml.safe_dump(_scenario(backend=backend).model_dump(mode="json"), allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def bare_machine(monkeypatch: pytest.MonkeyPatch) -> FakeMachine:
    """This process's machine, replaced by one with nothing on it."""
    machine = FakeMachine(sourced=False)
    monkeypatch.setattr(environment, "LocalMachine", lambda: machine)
    return machine


def test_env_check_names_what_is_missing_and_how_to_get_it(
    tmp_path: Path, bare_machine: FakeMachine
) -> None:
    _registry(tmp_path)
    result = runner.invoke(
        cli.app, ["env", "check", str(_scene_file(tmp_path, "gazebo")), "-R", str(tmp_path)]
    )

    assert result.exit_code == cli.EXIT_FOUND_ERRORS, result.output
    assert "проверено  twin:ros2_control@gazebo: требований 7" in result.output
    assert "[env.missing]" in result.output
    assert "controller_manager" in result.output
    assert "ros-<дистрибутив>-ros2-control" in result.output


def test_env_check_of_a_scene_needing_nothing_is_clean(tmp_path: Path) -> None:
    _registry(tmp_path)
    result = runner.invoke(
        cli.app, ["env", "check", str(_scene_file(tmp_path, "dummy")), "-R", str(tmp_path)]
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    assert "проверено  adapter:dummy: требований 0" in result.output


def test_env_check_for_hardware_checks_the_real_adapter_not_the_world(
    tmp_path: Path, bare_machine: FakeMachine
) -> None:
    _registry(tmp_path, real=True)
    scene = str(_scene_file(tmp_path, "gazebo"))
    arguments = ["env", "check", scene, "--target", "real", "-R", str(tmp_path)]
    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "проверено  adapter:dummy: требований 0" in result.output
    assert "twin:" not in result.output


def test_sim_run_starts_no_world_on_a_machine_its_twins_cannot_run_on(
    tmp_path: Path, bare_machine: FakeMachine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("мир не должен строиться")

    monkeypatch.setattr(cli, "SubprocessLauncher", never)
    _registry(tmp_path)
    result = runner.invoke(
        cli.app,
        ["sim", "run", str(_scene_file(tmp_path, "gazebo")), "--into", str(tmp_path / "out"),
         "-R", str(tmp_path)],
    )

    assert result.exit_code == cli.EXIT_FOUND_ERRORS, result.output
    assert "мир не поднимается" in result.output
    assert "[env.missing]" in result.output
    assert "rclpy" not in result.output, "сторона адаптера — не дело sim run"


def test_a_dry_run_says_the_environment_was_not_checked(
    tmp_path: Path, bare_machine: FakeMachine
) -> None:
    _registry(tmp_path)
    result = runner.invoke(
        cli.app,
        ["sim", "run", str(_scene_file(tmp_path, "gazebo")), "--into", str(tmp_path / "out"),
         "-R", str(tmp_path), "--dry-run"],
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    assert "окружение не проверялось" in result.output
    assert bare_machine.asked == []


def test_run_connects_nothing_on_a_machine_its_adapter_cannot_run_on(
    tmp_path: Path, bare_machine: FakeMachine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("адаптер не должен создаваться")

    monkeypatch.setattr(run_command, "build_adapter", never)
    _registry(tmp_path)
    result = runner.invoke(
        cli.app, ["run", str(_scene_file(tmp_path, "gazebo")), "-R", str(tmp_path)]
    )

    assert result.exit_code == cli.EXIT_FOUND_ERRORS, result.output
    assert "запуск отменён" in result.output
    assert "rclpy" in result.output
