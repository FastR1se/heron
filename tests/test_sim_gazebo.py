"""The simulation backend, checked without a simulator.

ADR-0011 says plainly what these tests can and cannot show. They show that the
backend builds the commands it meant to build and refuses the scenes it meant to
refuse. They show nothing about whether a simulator starts from those commands:
that is proved on the Linux machine, by a run.

What they do prove is the part that matters before a run: a scenario the
validator rejects does not become a world.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import make_profile, write_profile, write_urdf
from heron.core.models import AdapterSpec, Scenario
from heron.core.protocols import SimBackend
from heron.core.types import Pose
from heron.registry.registry import Registry
from heron.sim.gazebo import GazeboBackend, SimulationError
from heron.sim.process import PlanningLauncher


def _robot(tmp_path: Path, name: str = "arm", **overrides: Any) -> Any:
    overrides.setdefault("description", {"urdf": f"urdf/{name}.urdf"})
    overrides.setdefault("adapters", {"sim.gazebo": AdapterSpec(plugin="ros2_control")})
    profile = make_profile(name=name, **overrides)
    write_urdf(tmp_path, name, list(profile.contract.joint_order))
    write_profile(tmp_path, profile)
    return profile


def _scenario(backend: str = "gazebo", profile: str = "arm") -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "scene",
            "world": {"backend": backend},
            "robots": [{"id": "arm1", "profile": profile, "role": "scripted"}],
        }
    )


def _backend(tmp_path: Path, **kwargs: Any) -> tuple[GazeboBackend, PlanningLauncher]:
    planner = PlanningLauncher()
    backend = GazeboBackend(
        Registry([tmp_path]), into=tmp_path / "out", launcher=planner, **kwargs
    )
    return backend, planner


def test_the_backend_satisfies_the_protocol(tmp_path: Path) -> None:
    backend, _ = _backend(tmp_path)
    assert isinstance(backend, SimBackend)


def test_building_a_world_writes_the_bundle(tmp_path: Path) -> None:
    _robot(tmp_path)
    backend, _ = _backend(tmp_path)

    world = backend.build_world(_scenario())

    assert world.name == "scene"
    assert world.launch_file.is_file()
    assert (world.root / "arm1" / "controllers.yaml").is_file()


def test_a_scene_the_validator_rejects_does_not_become_a_world(tmp_path: Path) -> None:
    """Invariant 6 in one method: a world that cannot start is better than a
    world that starts wrong."""
    _robot(tmp_path, adapters={"sim.dummy": AdapterSpec(plugin="dummy")})
    backend, planner = _backend(tmp_path)

    with pytest.raises(SimulationError) as caught:
        backend.build_world(_scenario())

    assert caught.value.report is not None
    assert "scenario.backend_adapter_missing" in caught.value.report.codes()
    assert planner.commands == [], "ни один процесс не должен быть запущен"
    assert not (tmp_path / "out").exists()


def test_a_scene_owned_by_another_engine_is_refused(tmp_path: Path) -> None:
    _robot(tmp_path)
    backend, _ = _backend(tmp_path)

    with pytest.raises(SimulationError) as caught:
        backend.build_world(_scenario(backend="mujoco"))
    assert "владелец" in str(caught.value)


def test_starting_runs_the_generated_launch_file(tmp_path: Path) -> None:
    _robot(tmp_path)
    backend, planner = _backend(tmp_path)
    world = backend.build_world(_scenario())
    backend.start()

    assert planner.commands == [
        ("ros2", "launch", str(world.launch_file), "headless:=false", "rviz:=true")
    ]


def test_headless_and_visualization_reach_the_command_line(tmp_path: Path) -> None:
    _robot(tmp_path)
    backend, planner = _backend(tmp_path, gui=False, rviz=False)
    backend.build_world(_scenario())
    backend.start()

    assert planner.commands[0][-2:] == ("headless:=true", "rviz:=false")


def test_a_world_is_not_started_twice(tmp_path: Path) -> None:
    _robot(tmp_path)
    backend, _ = _backend(tmp_path)
    backend.build_world(_scenario())
    backend.start()

    with pytest.raises(SimulationError):
        backend.start()


def test_starting_before_building_is_refused(tmp_path: Path) -> None:
    backend, _ = _backend(tmp_path)
    with pytest.raises(SimulationError):
        backend.start()


def test_stopping_releases_every_process(tmp_path: Path) -> None:
    _robot(tmp_path)
    backend, _ = _backend(tmp_path)
    backend.build_world(_scenario())
    backend.start()

    backend.stop()
    assert backend.processes == []


def test_spawning_places_one_more_robot_in_a_running_world(tmp_path: Path) -> None:
    profile = _robot(tmp_path)
    backend, planner = _backend(tmp_path)
    world = backend.build_world(_scenario())
    backend.start()

    robot = backend.spawn(profile, Pose(position=(1.0, 2.0, 0.5)))

    assert robot.id == profile.name
    assert robot.description.is_file()
    command = planner.commands[-1]
    assert command[:4] == ("ros2", "run", "ros_gz_sim", "create")
    assert "-world" in command and world.name in command
    assert command[-6:] == ("-x", "1", "-y", "2", "-z", "0.5")


def test_spawning_a_robot_without_an_adapter_is_refused(tmp_path: Path) -> None:
    _robot(tmp_path)
    stranger = make_profile(name="stranger", adapters={"sim.dummy": AdapterSpec(plugin="dummy")})
    backend, _ = _backend(tmp_path)
    backend.build_world(_scenario())

    with pytest.raises(SimulationError):
        backend.spawn(stranger, Pose())


def test_spawning_before_building_is_refused(tmp_path: Path) -> None:
    profile = _robot(tmp_path)
    backend, _ = _backend(tmp_path)

    with pytest.raises(SimulationError):
        backend.spawn(profile, Pose())
