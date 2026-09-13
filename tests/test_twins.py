"""The other side of a link in simulation (ADR-0021, ADR-0024).

Three claims are checked here, each where it can fail. The twin is found through
the public plugin group, and the platform's own ``ros2_control`` path is found
exactly that way. The generators assemble a scene with a twin they have never
heard of, by the same code that assembles one with ours. And the validator
refuses, before a world exists, a driven robot nobody answers for and a vendor's
simulator asked to share its scene.

Vendor twins below are written in this file and slipped in where the lookup is
made. That is the honest substitute for a package that does not exist yet: what
is tested is that the platform does not care whose twin it is given.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any, ClassVar
from xml.etree import ElementTree as ET

import pytest
from typer.testing import CliRunner

from conftest import make_profile, write_profile, write_urdf
from heron.adapters.ros2_control import Ros2ControlAdapter
from heron.cli import main as cli
from heron.core import twins as lookup
from heron.core.enums import TwinKind
from heron.core.environment import Requirement, RequirementKind, needs
from heron.core.models import AdapterSpec, RobotProfile, Scenario
from heron.core.plugins import load
from heron.core.protocols import Twin
from heron.core.twins import TwinError, find_twin
from heron.core.urdf import UrdfModel
from heron.generators import GeneratorError, build
from heron.generators import bundle as generation
from heron.generators.common import GeneratedFile, Instance
from heron.generators.ros2_control import Ros2ControlTwin
from heron.generators.twin import GazeboTwin, LaunchNode
from heron.registry.registry import Registry
from heron.sim.process import PlanningLauncher
from heron.validator import rules
from heron.validator.rules import check_scenario

VENDOR = "vendor_link"
runner = CliRunner()


class VendorBridge:
    """A vendor's bridge in the Gazebo world, as a package would ship it."""

    adapter = VENDOR
    backend = "gazebo"
    kind = TwinKind.BRIDGE
    requirements: tuple[Requirement, ...] = (
        needs(RequirementKind.ROS2, "vendor_bridge", hint="пакет производителя"),
    )

    def description_elements(
        self, instance: Instance, description: UrdfModel, root: Path
    ) -> list[ET.Element]:
        joints = str(len(description.commandable_joints))
        return [ET.Element("vendor_bridge", {"robot": instance.id, "joints": joints})]

    def files(self, instance: Instance, root: Path) -> list[GeneratedFile]:
        return [GeneratedFile(path=instance.directory / "bridge.yaml", content="rate: 1000\n")]

    def launch_nodes(self, instance: Instance) -> list[LaunchNode]:
        return [LaunchNode("vendor_bridge", "bridge_node", "bridge", ("--robot", instance.id))]

    def control_plane(self, namespace: str, profile: RobotProfile) -> Mapping[str, str | None]:
        return {"command_topic": "/lowcmd", "state_topic": "/lowstate"}


class VendorSimulator(VendorBridge):
    """A vendor's simulator: it steps the physics itself."""

    kind = TwinKind.OWNS_PHYSICS


class Mute:
    """Declares the pair and contributes nothing a Gazebo scene could use."""

    adapter = VENDOR
    backend = "gazebo"
    kind = TwinKind.BRIDGE
    requirements = ()


class Nameless:
    """Not a twin at all."""


class Undeclared(VendorBridge):
    """Starts a node from a package it does not declare."""

    requirements = ()


class HalfKind:
    adapter = VENDOR
    backend = "gazebo"
    kind = "half"
    requirements = ()


def _only(twin: Any) -> Callable[[str, str], Any]:
    """A lookup that knows exactly one twin."""
    return lambda adapter, backend: (
        twin if (adapter, backend) == (twin.adapter, twin.backend) else None
    )


def _registry(tmp_path: Path, plugin: str = VENDOR) -> Registry:
    profile = make_profile(
        name="arm",
        description={"urdf": "urdf/arm.urdf"},
        adapters={
            "sim.gazebo": AdapterSpec(plugin=plugin),
            "sim.dummy": AdapterSpec(plugin="dummy"),
        },
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
        }
    )


def _vendor_installed(monkeypatch: pytest.MonkeyPatch, twin: Any) -> None:
    """The vendor's adapter counts as installed and its twin is the one found."""
    real = rules.available
    monkeypatch.setattr(
        rules,
        "available",
        lambda kind: {**real(kind), VENDOR: "tests"} if kind == "adapters" else real(kind),
    )
    monkeypatch.setattr(rules, "find_twin", _only(twin))


def _file(bundle: Any, path: str) -> str:
    return next(item.content for item in bundle.files if str(item.path) == path)


# -- lookup ------------------------------------------------------------------


def test_the_builtin_path_is_a_twin_found_through_the_public_group() -> None:
    twin = find_twin("ros2_control", "gazebo")

    assert isinstance(twin, Ros2ControlTwin)
    assert isinstance(twin, Twin)
    assert isinstance(twin, GazeboTwin)
    assert twin.kind == TwinKind.BRIDGE


def test_the_builtin_twin_answers_for_an_installed_adapter() -> None:
    """The name the twin declares is the name the adapter is installed under."""
    assert load("adapters", Ros2ControlTwin.adapter) is Ros2ControlAdapter


def test_a_pair_nobody_answers_for_has_no_twin() -> None:
    assert find_twin("dummy", "gazebo") is None


def test_two_twins_for_one_pair_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lookup, "installed_twins", lambda: [VendorBridge(), VendorBridge()])
    with pytest.raises(TwinError, match="одна"):
        find_twin(VENDOR, "gazebo")


@pytest.mark.parametrize(("attribute", "fragment"), [("Nameless", "не объявляет"),
                                                      ("HalfKind", "неизвестен")])
def test_an_installed_object_that_is_not_a_twin_is_refused(
    monkeypatch: pytest.MonkeyPatch, attribute: str, fragment: str
) -> None:
    entry = EntryPoint(name="broken", value=f"{__name__}:{attribute}", group="twins")
    monkeypatch.setattr(lookup, "entry_points", lambda **_: [entry])
    with pytest.raises(TwinError, match=fragment):
        lookup.installed_twins()


# -- generation ----------------------------------------------------------------


def test_a_vendor_twin_joins_the_scene_the_generators_never_heard_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generation, "find_twin", _only(VendorBridge()))
    bundle = build(
        _scenario(roles=("scripted", "static")), _registry(tmp_path), into=tmp_path / "out"
    )

    driven = _file(bundle, "arm1/description.urdf")
    assert '<vendor_bridge robot="arm1" joints="3"' in driven
    assert "<ros2_control" not in driven

    paths = {str(path) for path in bundle.paths()}
    assert "arm1/bridge.yaml" in paths
    assert "arm1/controllers.yaml" not in paths

    launch = _file(bundle, "bringup.launch.xml")
    assert 'pkg="vendor_bridge" exec="bridge_node" name="bridge" args="--robot arm1"' in launch
    assert "spawner" not in launch

    manifest = json.loads(_file(bundle, "manifest.json"))
    robots = {item["id"]: item for item in manifest["robots"]}
    assert robots["arm1"]["control"] == {"command_topic": "/lowcmd", "state_topic": "/lowstate"}
    assert "control" not in robots["arm2"]
    assert "<static>true</static>" in _file(bundle, "arm2/description.urdf")


def test_a_driven_robot_nobody_answers_for_gets_no_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generation, "find_twin", lambda adapter, backend: None)
    with pytest.raises(GeneratorError, match="ADR-0021"):
        build(_scenario(), _registry(tmp_path), into=tmp_path / "out")


def test_a_simulator_of_its_own_cannot_bridge_into_gazebo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generation, "find_twin", _only(VendorSimulator()))
    with pytest.raises(GeneratorError, match="владеет физикой"):
        build(_scenario(), _registry(tmp_path), into=tmp_path / "out")


def test_a_twin_starting_a_node_it_does_not_declare_gets_no_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0013 holds for a vendor's twin as it holds for ours (ADR-0025)."""
    monkeypatch.setattr(generation, "find_twin", _only(Undeclared()))
    with pytest.raises(GeneratorError, match="vendor_bridge"):
        build(_scenario(), _registry(tmp_path), into=tmp_path / "out")


def test_a_twin_that_says_nothing_to_gazebo_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generation, "find_twin", _only(Mute()))
    with pytest.raises(GeneratorError, match="ADR-0024"):
        build(_scenario(), _registry(tmp_path), into=tmp_path / "out")


def test_an_obstacle_is_not_asked_for_a_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(adapter: str, backend: str) -> None:
        raise AssertionError("препятствию двойник не нужен")

    monkeypatch.setattr(generation, "find_twin", refuse)
    bundle = build(_scenario(roles=("static",)), _registry(tmp_path), into=tmp_path / "out")
    assert "arm1/description.urdf" in {str(path) for path in bundle.paths()}


# -- validation ----------------------------------------------------------------


def test_a_driven_robot_without_a_twin_is_an_error(tmp_path: Path) -> None:
    """``dummy`` is an installed adapter, and nothing answers for it in Gazebo."""
    report = check_scenario(_scenario(), _registry(tmp_path, plugin="dummy"))
    assert "twin.missing" in report.codes()
    assert not report.ok


def test_an_obstacle_needs_no_twin(tmp_path: Path) -> None:
    report = check_scenario(_scenario(roles=("static",)), _registry(tmp_path, plugin="dummy"))
    assert "twin.missing" not in report.codes()


def test_a_world_the_platform_does_not_start_asks_for_no_twin(tmp_path: Path) -> None:
    """A bench scene: the dummy adapter is both sides of its own link."""
    report = check_scenario(_scenario(backend="dummy"), _registry(tmp_path))
    assert not [code for code in report.codes() if code.startswith("twin.")]


def test_an_adapter_not_installed_is_not_also_missing_a_twin(tmp_path: Path) -> None:
    report = check_scenario(_scenario(), _registry(tmp_path, plugin="vendor_absent"))
    assert "adapter.plugin_not_installed" in report.codes()
    assert "twin.missing" not in report.codes()


def test_a_simulator_of_its_own_does_not_share_its_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _vendor_installed(monkeypatch, VendorSimulator())
    report = check_scenario(_scenario(roles=("scripted", "static")), _registry(tmp_path))
    assert "twin.owns_physics_not_alone" in report.codes()


def test_a_simulator_of_its_own_with_one_robot_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _vendor_installed(monkeypatch, VendorSimulator())
    report = check_scenario(_scenario(), _registry(tmp_path))
    assert not [code for code in report.codes() if code.startswith("twin.")]


def test_a_twin_that_cannot_be_loaded_is_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _vendor_installed(monkeypatch, VendorBridge())

    def broken(adapter: str, backend: str) -> None:
        raise TwinError("двойник 'vendor' не загружается")

    monkeypatch.setattr(rules, "find_twin", broken)
    report = check_scenario(_scenario(), _registry(tmp_path))
    assert "twin.unloadable" in report.codes()


# -- sim run -------------------------------------------------------------------


class RecordingBackend:
    """A backend from somebody's package: it starts one process of its own."""

    made_with: ClassVar[dict[str, Any]] = {}

    def __init__(self, registry: Registry, **kwargs: Any) -> None:
        RecordingBackend.made_with = kwargs
        self.launcher = kwargs["launcher"]

    def build_world(self, scenario: Scenario) -> object:
        return object()

    def spawn(self, profile: RobotProfile, pose: Any) -> object:
        return object()

    def start(self) -> None:
        self.launcher.start(["elsewhere-sim", "--scene", "scene"])

    def stop(self) -> None:
        return None


def _scene_file(tmp_path: Path, backend: str) -> Path:
    path = tmp_path / "scene.yaml"
    path.write_text(
        f"schema_version: 1\nname: scene\nworld: {{backend: {backend}}}\n"
        "robots:\n  - {id: arm1, profile: arm, role: scripted}\n",
        encoding="utf-8",
    )
    return path


def test_sim_run_takes_the_backend_from_the_plugin_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[str, str]] = []

    def plugin(kind: str, name: str) -> Any:
        asked.append((kind, name))
        return RecordingBackend

    monkeypatch.setattr(cli, "load", plugin)
    result = runner.invoke(
        cli.app,
        ["sim", "run", str(_scene_file(tmp_path, "elsewhere")), "--into", str(tmp_path / "out"),
         "--dry-run"],
    )

    assert result.exit_code == cli.EXIT_OK, result.output
    assert asked == [("sim", "elsewhere")]
    assert isinstance(RecordingBackend.made_with["launcher"], PlanningLauncher)
    assert "elsewhere-sim --scene scene" in result.output


def test_sim_run_names_a_backend_that_is_not_installed(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["sim", "run", str(_scene_file(tmp_path, "nowhere")), "--into", str(tmp_path / "out"),
         "--dry-run"],
    )
    assert result.exit_code == cli.EXIT_FOUND_ERRORS
    assert "nowhere" in result.output
