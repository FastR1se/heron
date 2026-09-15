"""The MuJoCo backend, checked without MuJoCo and without any vendor's twin (ADR-0030).

Like the Gazebo backend's tests (ADR-0011), these show which commands the backend
means to start and which scenes it refuses to start at all. Whether a simulator
comes up from those commands is shown on the Linux machine, by a run.

The twin is written here and slipped in where the lookup is made: what is tested
is that the backend does not care whose simulator it starts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import make_profile, write_profile, write_urdf
from heron.cli import main as cli
from heron.core.enums import TwinKind
from heron.core.environment import Requirement, RequirementKind, needs
from heron.core.models import AdapterSpec, Scenario
from heron.core.plugins import load
from heron.core.protocols import SimBackend
from heron.core.types import Pose
from heron.registry.registry import Registry
from heron.sim import mujoco
from heron.sim.base import SimulationError
from heron.sim.mujoco import MujocoBackend
from heron.sim.mujoco_twin import MujocoScene, MujocoTwin, TwinProcess
from heron.sim.process import PlanningLauncher
from test_mjcf import mjcf_text, write

VENDOR = "vendor_link"
runner = CliRunner()


class VendorSimulator:
    """A vendor's simulator: loads the scene itself and reads the link from params."""

    adapter = VENDOR
    backend = "mujoco"
    kind = TwinKind.OWNS_PHYSICS
    requirements: tuple[Requirement, ...] = (
        needs(RequirementKind.EXECUTABLE, "vendor-sim", hint="программа производителя"),
    )

    def __init__(self) -> None:
        self.told: list[MujocoScene] = []

    def processes(self, scene: MujocoScene) -> list[TwinProcess]:
        self.told.append(scene)
        domain = str(scene.params["domain"])
        command = ("vendor-sim", "--scene", str(scene.world), "--domain", domain)
        return [TwinProcess("vendor-sim", command)]


class VendorBridge(VendorSimulator):
    kind = TwinKind.BRIDGE


class Mute:
    adapter = VENDOR
    backend = "mujoco"
    kind = TwinKind.OWNS_PHYSICS
    requirements = ()


class Refusing(VendorSimulator):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def processes(self, scene: MujocoScene) -> list[TwinProcess]:
        raise self.error


class Idle(VendorSimulator):
    def processes(self, scene: MujocoScene) -> list[TwinProcess]:
        return []


def _only(twin: Any) -> Callable[[str, str], Any]:
    return lambda adapter, backend: (
        twin if (adapter, backend) == (twin.adapter, twin.backend) else None
    )


def _scene(tmp_path: Path, roles: tuple[str, ...] = ("scripted",), file: str | None = "scene.xml",
           plugin: str = VENDOR) -> Path:
    """Profile, descriptions, scene and scenario on disk; returns the scenario file."""
    profile = make_profile(
        name="arm",
        description={"urdf": "urdf/arm.urdf", "mjcf": "mjcf/arm.xml"},
        adapters={"sim.mujoco": AdapterSpec(plugin=plugin, params={"domain": 7})},
    )
    write_urdf(tmp_path, "arm", list(profile.contract.joint_order))
    write(tmp_path / "mjcf" / "arm.xml", mjcf_text())
    write_profile(tmp_path, profile)
    write(tmp_path / "worlds" / "scene.xml", '<mujoco><include file="../mjcf/arm.xml"/></mujoco>')
    robots = "".join(
        f"  - {{id: arm{index + 1}, profile: arm, role: {role}}}\n"
        for index, role in enumerate(roles)
    )
    world = f"{{backend: mujoco, file: worlds/{file}}}" if file else "{backend: mujoco}"
    return write(
        tmp_path / "scenario.yaml",
        f"schema_version: 1\nname: scene\nworld: {world}\nrobots:\n{robots}",
    )


def _backend(
    tmp_path: Path, scenario_path: Path, monkeypatch: pytest.MonkeyPatch, twin: Any, **kwargs: Any
) -> tuple[MujocoBackend, PlanningLauncher]:
    monkeypatch.setattr(mujoco, "find_twin", _only(twin))
    planner = PlanningLauncher()
    backend = MujocoBackend(
        Registry([tmp_path]),
        into=tmp_path / "out",
        launcher=planner,
        scenario_path=scenario_path,
        **kwargs,
    )
    return backend, planner


def _load(path: Path) -> Scenario:
    from heron.scenario.loader import load_scenario

    return load_scenario(path)


# -- the backend ---------------------------------------------------------------


def test_the_backend_is_found_through_the_plugin_group_and_satisfies_the_protocol(
    tmp_path: Path,
) -> None:
    assert load("sim", "mujoco") is MujocoBackend
    assert isinstance(MujocoBackend(into=tmp_path), SimBackend)
    assert isinstance(VendorSimulator(), MujocoTwin)


def test_the_twin_is_told_the_scene_and_the_link_and_started_as_it_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    twin = VendorSimulator()
    backend, planner = _backend(tmp_path, path, monkeypatch, twin)

    world = backend.build_world(_load(path))
    backend.start()

    scene = world.scene
    assert scene.world == (tmp_path / "worlds" / "scene.xml").resolve()
    assert scene.description == (tmp_path / "mjcf" / "arm.xml").resolve()
    assert scene.params == {"domain": 7}
    assert (scene.robot, scene.gui) == ("arm1", True)
    assert planner.commands == [("vendor-sim", "--scene", str(scene.world), "--domain", "7")]
    assert planner.planned[0].cwd == tmp_path / "out"
    assert [robot.id for robot in world.robots] == ["arm1"]


def test_a_headless_run_is_the_twins_to_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    twin = VendorSimulator()
    backend, _ = _backend(tmp_path, path, monkeypatch, twin, gui=False)
    backend.build_world(_load(path))
    assert twin.told[0].gui is False


def test_a_scene_the_validator_rejects_is_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path, file=None)
    backend, planner = _backend(tmp_path, path, monkeypatch, VendorSimulator())

    with pytest.raises(SimulationError) as caught:
        backend.build_world(_load(path))
    assert caught.value.report is not None
    assert "mjcf.scene_missing" in caught.value.report.codes()
    assert planner.commands == []


def test_a_scene_owned_by_another_engine_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    backend, _ = _backend(tmp_path, path, monkeypatch, VendorSimulator())
    scenario = _load(path).model_copy(update={"world": _load(path).world.model_copy(
        update={"backend": "gazebo"})})
    with pytest.raises(SimulationError, match="владелец"):
        backend.build_world(scenario)


@pytest.mark.parametrize(("roles", "fragment"), [(("scripted", "static"), "участников 2"),
                                                 (("static",), "препятствие")])
def test_a_mujoco_world_is_one_driven_robot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, roles: tuple[str, ...], fragment: str
) -> None:
    path = _scene(tmp_path, roles=roles)
    backend, _ = _backend(tmp_path, path, monkeypatch, VendorSimulator())
    with pytest.raises(SimulationError, match=fragment):
        backend.build_world(_load(path))


def test_a_robot_nobody_simulates_gets_no_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    backend, _ = _backend(tmp_path, path, monkeypatch, VendorSimulator())
    monkeypatch.setattr(mujoco, "find_twin", lambda adapter, backend: None)
    with pytest.raises(SimulationError, match="адаптер не установлен"):
        backend.build_world(_load(path))


@pytest.mark.parametrize(("twin", "fragment"), [(VendorBridge(), "мост"),
                                                (Mute(), "processes")])
def test_a_twin_that_is_not_a_world_of_its_own_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, twin: Any, fragment: str
) -> None:
    path = _scene(tmp_path)
    backend, planner = _backend(tmp_path, path, monkeypatch, twin)
    with pytest.raises(SimulationError, match=fragment):
        backend.build_world(_load(path))
    assert planner.commands == []


def test_a_twin_refusing_the_scene_is_heard_as_it_said_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    backend, _ = _backend(tmp_path, path, monkeypatch, Refusing(SimulationError("нет окна")))
    with pytest.raises(SimulationError, match=r"^нет окна$"):
        backend.build_world(_load(path))

    backend, _ = _backend(tmp_path, path, monkeypatch, Refusing(ValueError("params.domain")))
    wrapped = r"Refusing не собирает запуск сцены — params\.domain"
    with pytest.raises(SimulationError, match=wrapped):
        backend.build_world(_load(path))


def test_a_world_of_its_own_that_starts_nothing_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    backend, _ = _backend(tmp_path, path, monkeypatch, Idle())
    with pytest.raises(SimulationError, match="ни одного процесса"):
        backend.build_world(_load(path))


def test_nothing_is_spawned_into_a_twins_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    backend, _ = _backend(tmp_path, path, monkeypatch, VendorSimulator())
    backend.build_world(_load(path))
    with pytest.raises(SimulationError, match="ADR-0002"):
        backend.spawn(make_profile(name="stranger"), Pose())


def test_start_and_stop_follow_the_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    backend, _ = _backend(tmp_path, path, monkeypatch, VendorSimulator())
    with pytest.raises(SimulationError, match="не построен"):
        backend.start()

    backend.build_world(_load(path))
    backend.start()
    with pytest.raises(SimulationError, match="уже запущен"):
        backend.start()
    backend.stop()
    assert backend.processes == []


def test_sim_run_prints_what_the_twin_would_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path)
    monkeypatch.setattr(mujoco, "find_twin", _only(VendorSimulator()))
    result = runner.invoke(
        cli.app,
        ["sim", "run", str(path), "--into", str(tmp_path / "out"), "-R", str(tmp_path),
         "--dry-run"],
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    assert "vendor-sim --scene" in result.output
    assert "--domain 7" in result.output
