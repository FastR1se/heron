"""Fixtures shared by the test suite.

Profiles are built in code rather than read from ``examples/``, because
invariant 1 has to stay testable: deleting the example robots must not break the
core or its tests. Only ``tests/test_examples.py`` looks at the files, and it
skips itself when they are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from heron.core.capabilities import Capability
from heron.core.contract import ActionSpec, Contract, ObservationTerm
from heron.core.enums import ActionType, AngleUnit, ControlMode, ObservationType
from heron.core.models import AdapterSpec, JointMax, JointRange, Limits, RobotProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


def make_contract(dof: int = 3, **overrides: Any) -> Contract:
    fields: dict[str, Any] = {
        "revision": 1,
        "joint_order": [f"joint{i}" for i in range(dof)],
        "angle_unit": AngleUnit.RAD,
        "state_rate_hz": 100.0,
        "command_rate_hz": 50.0,
        "observation": [
            ObservationTerm(type=ObservationType.JOINT_POS, scale=1.0),
            ObservationTerm(type=ObservationType.JOINT_VEL, scale=0.05),
        ],
        "action": ActionSpec(type=ActionType.JOINT_POSITION, clip=(-1.0, 1.0)),
    }
    fields.update(overrides)
    return Contract(**fields)


def make_profile(name: str = "test_arm", dof: int = 3, **overrides: Any) -> RobotProfile:
    fields: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "description": {"urdf": f"urdf/{name}.urdf.xacro"},
        "contract": make_contract(dof),
        "capabilities": frozenset({Capability("arm.joint_position")}),
        "control_mode": ControlMode.STREAMING,
        "limits": Limits(
            joint_position=JointRange(min=[-2.0] * dof, max=[2.0] * dof),
            joint_velocity=JointMax(max=[1.5] * dof),
        ),
        "adapters": {"sim.dummy": AdapterSpec(plugin="dummy")},
    }
    fields.update(overrides)
    return RobotProfile(**fields)


@pytest.fixture
def contract() -> Contract:
    return make_contract()


@pytest.fixture
def profile() -> RobotProfile:
    return make_profile()


@pytest.fixture
def examples_dir() -> Path:
    if not EXAMPLES.is_dir():
        pytest.skip("каталог examples удалён — проверка инварианта 1")
    return EXAMPLES
