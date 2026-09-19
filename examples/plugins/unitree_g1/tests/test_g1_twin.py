"""The vendor's MuJoCo twin, as the platform finds and asks it (ADR-0023, ADR-0030).

Nothing here starts a simulator: the twin returns the command, and the command is
what is checked. That the vendor's program comes up from it is the Linux run's.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
from unitree_g1.channel import LinkConfigError
from unitree_g1.slots import LAYOUT
from unitree_g1.testing import LINK_PARAMS, profile_23dof, profile_upper_body
from unitree_g1.twin import ADAPTER, PROGRAM, G1MujocoTwin, reported_machine

from heron.core.contract import ActionSpec, Contract, ObservationTerm
from heron.core.enums import ActionType, AngleUnit, ObservationType, TwinKind
from heron.core.environment import RequirementKind
from heron.core.mjcf import MUJOCO
from heron.core.models import RobotProfile
from heron.sim.base import SimulationError
from heron.sim.mujoco_twin import MujocoScene, MujocoTwin

SCENE_23 = Path("/vendor/unitree_robots/g1/scene_23dof.xml")


def scene(profile: RobotProfile | None = None, **overrides: Any) -> MujocoScene:
    fields: dict[str, Any] = {
        "robot": "g1",
        "profile": profile or profile_23dof(),
        "params": dict(LINK_PARAMS),
        "world": SCENE_23,
        "description": None,
        "gui": True,
    }
    fields.update(overrides)
    return MujocoScene(**fields)


def test_the_package_registers_the_twin_next_to_its_adapter() -> None:
    """What the package promises: an entry point, and the pair it answers for.

    Finding it by that pair is the platform's lookup, which a package may not
    import (ADR-0021); the platform's tests find this twin through it
    (tests/test_examples.py).
    """
    twins = {item.name: item.value for item in entry_points(group="heron.twins")}
    adapters = {item.name: item.value for item in entry_points(group="heron.adapters")}
    assert twins["unitree_g1_mujoco"] == "unitree_g1.twin:G1MujocoTwin"
    assert adapters[ADAPTER] == "unitree_g1.adapter:G1DdsAdapter"

    twin = G1MujocoTwin()
    assert isinstance(twin, MujocoTwin)
    assert (twin.adapter, twin.backend, twin.kind) == (ADAPTER, MUJOCO, TwinKind.OWNS_PHYSICS)


def test_the_twin_is_started_with_the_scene_and_the_link_of_the_adapter() -> None:
    """Domain and interface from params; the peers stay on the adapter's side (ADR-0023, p. 3)."""
    (process,) = G1MujocoTwin().processes(scene())
    assert process.command == (PROGRAM, "-r", "g1", "-s", str(SCENE_23), "-i", "1", "-n", "lo")
    assert "127.0.0.1" not in process.command


def test_a_link_for_the_robot_reaches_the_command_unchanged() -> None:
    (process,) = G1MujocoTwin().processes(
        scene(params={"domain": 0, "interface": "enp3s0"})
    )
    assert process.command[-4:] == ("-i", "0", "-n", "enp3s0")


def test_a_link_that_is_not_one_is_refused_by_name() -> None:
    with pytest.raises(LinkConfigError, match=r"params\.domain"):
        G1MujocoTwin().processes(scene(params={"interface": "lo"}))


def test_the_twin_has_no_headless_mode() -> None:
    with pytest.raises(SimulationError, match="безголового режима"):
        G1MujocoTwin().processes(scene(gui=False))


def test_the_body_is_named_by_the_scene_file_name() -> None:
    assert reported_machine(SCENE_23) == 4
    assert reported_machine(SCENE_23.with_name("scene_29dof.xml")) == 5
    assert reported_machine(SCENE_23.with_name("scene.xml")) == 5


def test_a_scene_for_another_body_is_refused_before_the_adapter_would() -> None:
    with pytest.raises(SimulationError, match="mode_machine 5") as caught:
        G1MujocoTwin().processes(scene(world=SCENE_23.with_name("scene_29dof.xml")))
    assert "mode_machine 4" in str(caught.value)


def test_a_contract_of_no_known_body_is_refused() -> None:
    joints = list(LAYOUT[:12])
    legs = profile_23dof(
        contract=Contract(
            revision=1,
            joint_order=joints,
            angle_unit=AngleUnit.RAD,
            state_rate_hz=500.0,
            command_rate_hz=50.0,
            observation=[ObservationTerm(type=ObservationType.JOINT_POS, scale=1.0)],
            action=ActionSpec(type=ActionType.JOINT_POSITION, kp=[40.0] * 12, kd=[1.0] * 12),
        ),
        limits={},
    )
    with pytest.raises(SimulationError, match="ни одно тело G1"):
        G1MujocoTwin().processes(scene(profile=legs))


def test_an_upper_body_is_refused_because_the_vendors_scene_is_free() -> None:
    """ADR-P6A, item 4: the legs would go limp on a robot nothing holds."""
    with pytest.raises(SimulationError, match="свободной базе"):
        G1MujocoTwin().processes(scene(profile=profile_upper_body()))


def test_the_twin_declares_the_program_and_the_display() -> None:
    declared = {(item.kind, item.name) for item in G1MujocoTwin.requirements}
    assert declared == {
        (RequirementKind.EXECUTABLE, PROGRAM),
        (RequirementKind.VARIABLE, "DISPLAY"),
    }


def test_importing_the_twin_imports_no_dds() -> None:
    """Every twin of the group is loaded on every lookup (ADR-0024): it must stay light."""
    code = (
        "import sys, unitree_g1.twin; "
        "print(sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'cyclonedds', 'unitree_sdk2py'}))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
