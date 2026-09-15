"""A run joins its world in the world's environment (ADR-0039).

The world of a Gazebo scene starts in the environment its twins ask for (ADR-0034). A
process driving one of its robots is a participant of that world and has to be in it too:
started elsewhere, a ROS 2 link waits in another domain for a robot nobody publishes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
from xml.etree import ElementTree as ET

import pytest

from conftest import make_profile, write_profile, write_urdf
from heron.adapters.dummy import DummyAdapter
from heron.cli.run import RunError, execute
from heron.core.enums import TwinKind
from heron.core.models import AdapterSpec, RobotProfile, Scenario
from heron.core.urdf import UrdfModel
from heron.generators import bundle as generation
from heron.generators.common import GeneratedFile, Instance
from heron.generators.twin import LaunchNode
from heron.registry.registry import Registry
from heron.validator import rules

PLUGIN = "test_run_environment:WorldAdapter"
NAMES = ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION")
WORLD = {"RMW_IMPLEMENTATION": "rmw_world", "ROS_DOMAIN_ID": "7"}
GOAL = {"target": [0.2, 0.0, 0.0], "duration": 0.05}


class WorldAdapter(DummyAdapter):
    """Remembers the environment its process had when it connected."""

    seen: ClassVar[dict[str, str | None] | None] = None

    def connect(self, profile: RobotProfile, config: dict[str, object]) -> None:
        type(self).seen = {name: os.environ.get(name) for name in NAMES}
        super().connect(profile, config)


class WorldTwin:
    """A bridge whose world runs in an environment of its own."""

    adapter = PLUGIN
    backend = "gazebo"
    kind = TwinKind.BRIDGE
    requirements = ()

    def description_elements(
        self, instance: Instance, description: UrdfModel, root: Path
    ) -> list[ET.Element]:
        return []

    def files(self, instance: Instance, root: Path) -> list[GeneratedFile]:
        return []

    def launch_nodes(self, instance: Instance) -> list[LaunchNode]:
        return []

    def control_plane(self, namespace: str, profile: RobotProfile) -> Mapping[str, str | None]:
        return {"command_topic": f"/{namespace}/cmd", "state_topic": f"/{namespace}/state"}

    def environment(self, instance: Instance) -> Mapping[str, str]:
        return WORLD


def _lookup(adapter: str, backend: str) -> WorldTwin | None:
    return WorldTwin() if (adapter, backend) == (PLUGIN, "gazebo") else None


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Registry:
    """A registry with one robot whose twin asks its world for an environment."""
    monkeypatch.setattr(generation, "find_twin", _lookup)
    monkeypatch.setattr(rules, "find_twin", _lookup)
    for name in NAMES:
        # Registered first, so whatever the run sets is undone after the test.
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    WorldAdapter.seen = None

    for name, plugin in (("arm", PLUGIN), ("other", "absent_vendor")):
        profile = make_profile(
            name=name,
            description={"urdf": f"urdf/{name}.urdf"},
            adapters={"sim.gazebo": AdapterSpec(plugin=plugin)},
        )
        write_urdf(tmp_path, name, list(profile.contract.joint_order))
        write_profile(tmp_path, profile)
    return Registry([tmp_path])


def _scenario(*robots: tuple[str, str]) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "scene",
            "world": {"backend": "gazebo"},
            "robots": [
                {"id": robot, "profile": profile, "role": "scripted"} for robot, profile in robots
            ],
            "task": {"skill": "move_joints", **GOAL},
        }
    )


def test_the_run_connects_in_the_environment_of_its_world(world: Registry) -> None:
    _, result = execute(_scenario(("arm1", "arm")), world)

    assert result is not None and result.ok, result
    assert WorldAdapter.seen == WORLD


def test_the_plan_names_the_worlds_environment_and_changes_nothing(world: Registry) -> None:
    process: dict[str, str] = {}
    lines, _ = execute(_scenario(("arm1", "arm")), world, dry_run=True, environ=process)

    assert "окружение RMW_IMPLEMENTATION=rmw_world, ROS_DOMAIN_ID=7" in lines
    assert process == {}


def test_a_process_in_another_environment_is_refused_before_it_connects(
    world: Registry,
) -> None:
    with pytest.raises(RunError) as refusal:
        execute(_scenario(("arm1", "arm")), world, environ={"ROS_DOMAIN_ID": "0"})

    assert "ROS_DOMAIN_ID: у процесса '0', у мира '7'" in str(refusal.value)
    assert WorldAdapter.seen is None


def test_the_same_value_is_not_a_difference(world: Registry) -> None:
    process = {"ROS_DOMAIN_ID": "7"}
    _, result = execute(_scenario(("arm1", "arm")), world, environ=process)

    assert result is not None and result.ok
    assert process == WORLD


def test_a_world_whose_twin_is_not_on_this_machine_is_not_joined(world: Registry) -> None:
    """The second robot's adapter is not installed here, so neither is its twin: the
    world's environment is unknown, which is not the same as empty (ADR-0008)."""
    scenario = _scenario(("arm1", "arm"), ("other1", "other"))

    with pytest.raises(RunError, match="не выводится"):
        execute(scenario, world, robot_id="arm1", environ={})
    assert WorldAdapter.seen is None

    lines, _ = execute(scenario, world, robot_id="arm1", dry_run=True, environ={})
    assert any(line.startswith("окружение не выводится") for line in lines)


def test_a_world_the_generators_do_not_produce_asks_for_nothing(tmp_path: Path) -> None:
    profile = make_profile(name="arm", description={"urdf": "urdf/arm.urdf"})
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    write_profile(tmp_path, profile)
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "bench",
            "world": {"backend": "dummy"},
            "robots": [{"id": "arm1", "profile": "arm", "role": "scripted"}],
            "task": {"skill": "move_joints", **GOAL},
        }
    )
    process: dict[str, str] = {}
    lines, result = execute(scenario, Registry([tmp_path]), environ=process)

    assert result is not None and result.ok
    assert not any(line.startswith("окружение") for line in lines)
    assert process == {}
