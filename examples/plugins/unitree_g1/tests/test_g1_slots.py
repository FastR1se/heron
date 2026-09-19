"""The slot layout, held against the vendor's MJCF rather than the vendor's table (ADR-0027)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from unitree_g1.slots import (
    BODY_23DOF,
    BODY_29DOF,
    BODY_29DOF_LOCK_WAIST,
    LAYOUT,
    LEGS,
    MACHINES,
    SLOT_COUNT,
    UPPER_BODY,
    BodyMismatchError,
    SlotError,
    check_body,
    empty_slots,
    is_upper_body,
    machine_for,
    slots_of,
)

from heron.core.mjcf import parse_mjcf

EXAMPLES = Path(__file__).resolve().parents[3]
ROBOT = EXAMPLES / "robots" / "g1_23dof"
DESCRIPTIONS = ROBOT / "vendor" / "unitree_ros" / "robots" / "g1_description"
TWIN_MJCF = ROBOT / "vendor" / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_23dof.xml"
PROFILE = ROBOT / "profile.yaml"
LOCK_WAIST_PROFILE = EXAMPLES / "robots" / "g1_29dof_lock_waist_rev_1_0" / "profile.yaml"
MOUNTED_PROFILE = EXAMPLES / "robots" / "g1_29dof_lock_waist_rev_1_0_mounted" / "profile.yaml"
NOT_FETCHED = "описание G1 не загружено: python examples/robots/g1_23dof/fetch_description.py"

TABLE_ROW = re.compile(r"^\|\s*`(?P<file>[^`]+)`\s*\|\s*(?P<machine>\d+)\s*\|", re.MULTILINE)
"""A current row of the vendor's table: file name, then ``mode_machine``. A deprecated
row is struck through (``~~`name`~~``) and does not match."""

MACHINE_DESCRIPTIONS = {
    4: "g1_23dof_rev_1_0",
    5: "g1_29dof_rev_1_0",
    6: "g1_29dof_lock_waist_rev_1_0",
}
"""The description the vendor's table names for each value the package knows."""


def actuated_joints(mjcf: Path) -> list[str]:
    """Joints of the actuators in the order the MJCF declares them."""
    actuator = ET.parse(mjcf).getroot().find("actuator")
    assert actuator is not None, f"{mjcf.name}: нет блока actuator"
    return [motor.attrib["joint"] for motor in actuator]


def test_the_layout_is_the_actuator_order_of_the_29_dof_mjcf() -> None:
    """The twin's bridge puts motor_cmd[i] into ctrl[i]: actuator order is slot order.

    unitree_ros declares the same order name for name as the twin's own file, which
    the test below reads.
    """
    mjcf = DESCRIPTIONS / "g1_29dof.xml"
    if not mjcf.is_file():
        pytest.skip(NOT_FETCHED)
    assert tuple(actuated_joints(mjcf)) == LAYOUT


def test_the_mjcf_the_twin_loads_declares_its_actuators_in_slot_order() -> None:
    """The file the vendor's twin loads for the 23-DoF body, fetched from its repository.

    29 actuators, not 23: the six joints this body lacks hang on bodies of their
    own, twenty metres above the floor, and keep their slots. Checked by hand in
    session 12; read by a test since the file is fetched (journal 2026-09-15).
    """
    if not TWIN_MJCF.is_file():
        pytest.skip(NOT_FETCHED)
    model = parse_mjcf(TWIN_MJCF)
    assert tuple(item.joint for item in model.actuators) == LAYOUT
    assert len(model.joints) == len(LAYOUT) + 1, "и свободная база"


def test_the_23_dof_mjcf_of_unitree_ros_is_not_a_slot_order() -> None:
    """Same robot, same file name pattern, 23 actuators in a row: not slots.

    Kept as a test because it is the trap: reading this file's order as slots
    puts the left shoulder at 13, which is exactly the vendor table's mistake.
    """
    mjcf = DESCRIPTIONS / "g1_23dof.xml"
    if not mjcf.is_file():
        pytest.skip(NOT_FETCHED)
    order = actuated_joints(mjcf)
    assert len(order) == 23
    assert order.index("left_shoulder_pitch_joint") == 13
    assert LAYOUT.index("left_shoulder_pitch_joint") == 15


def test_the_23_dof_body_leaves_the_slots_the_vendor_marks_invalid() -> None:
    assert sorted(slot for slot in empty_slots(BODY_23DOF) if slot < len(LAYOUT)) == [
        13, 14, 20, 21, 27, 28,
    ]
    assert empty_slots(BODY_29DOF) == frozenset(range(len(LAYOUT), SLOT_COUNT))


def test_arms_start_at_15_and_22_not_13() -> None:
    assert slots_of(["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]) == (15, 22)


def test_slots_follow_names_not_positions() -> None:
    assert slots_of(["right_elbow_joint", "left_hip_pitch_joint", "waist_yaw_joint"]) == (25, 0, 12)


def test_an_unknown_joint_has_no_slot() -> None:
    with pytest.raises(SlotError, match="joint0"):
        slots_of(["left_knee_joint", "joint0"])


def test_the_example_profile_is_the_23_dof_body() -> None:
    if not PROFILE.is_file():
        pytest.skip("профиля G1 в examples нет")
    order = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))["contract"]["joint_order"]
    check_body(4, order)
    assert frozenset(order) == BODY_23DOF


def test_the_example_locked_waist_profile_is_its_body() -> None:
    if not LOCK_WAIST_PROFILE.is_file():
        pytest.skip("профиля G1 с закреплённой талией в examples нет")
    order = yaml.safe_load(LOCK_WAIST_PROFILE.read_text(encoding="utf-8"))["contract"][
        "joint_order"
    ]
    check_body(6, order)


def test_each_known_body_accepts_its_own_joints() -> None:
    check_body(4, BODY_23DOF)
    check_body(5, BODY_29DOF)
    check_body(6, BODY_29DOF_LOCK_WAIST)


def movable_joints(urdf: Path) -> frozenset[str]:
    return frozenset(
        joint.attrib["name"]
        for joint in ET.parse(urdf).getroot().iter("joint")
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
    )


def test_each_mode_machine_is_the_body_of_the_description_the_vendor_names_for_it() -> None:
    """The vendor's table of descriptions is where the value is stated (ADR-0059)."""
    readme = DESCRIPTIONS / "README.md"
    if not readme.is_file():
        pytest.skip(NOT_FETCHED)
    table = {
        row["file"]: int(row["machine"])
        for row in TABLE_ROW.finditer(readme.read_text(encoding="utf-8"))
    }

    assert set(MACHINE_DESCRIPTIONS) == set(MACHINES)
    for machine, description in MACHINE_DESCRIPTIONS.items():
        assert table[description] == machine, description
        assert movable_joints(DESCRIPTIONS / f"{description}.urdf") == MACHINES[machine]


def test_a_body_the_package_does_not_know_is_refused_with_its_number() -> None:
    with pytest.raises(BodyMismatchError, match="mode_machine = 7"):
        check_body(7, BODY_23DOF)


def test_a_23_dof_contract_on_a_29_dof_body_names_the_joints_left_limp() -> None:
    with pytest.raises(BodyMismatchError, match="left_wrist_pitch_joint") as caught:
        check_body(5, BODY_23DOF)
    assert "в контракте нет" in str(caught.value)


def test_a_29_dof_contract_on_a_23_dof_body_names_the_joints_it_lacks() -> None:
    with pytest.raises(BodyMismatchError, match="у тела нет: waist_roll_joint"):
        check_body(4, BODY_29DOF)


# -- the upper body (ADR-P6B) --------------------------------------------------------


ARM_SDK_JOINTS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
)
"""``arm_joints`` of the vendor's ``g1_arm7_sdk_dds_example.cpp`` (unitree_sdk2 ``9754cd1``),
spelled out: the joints ``rt/arm_sdk`` commands while the robot balances on its own."""


def test_the_upper_body_is_what_the_vendors_arm_sdk_commands() -> None:
    assert frozenset(ARM_SDK_JOINTS) == UPPER_BODY
    assert UPPER_BODY | LEGS == BODY_29DOF and not UPPER_BODY & LEGS
    assert sorted(slots_of(sorted(UPPER_BODY))) == list(range(12, 29))


def test_each_upper_body_names_its_body() -> None:
    """Three different sets, so an upper body is as good an identity as a whole one."""
    uppers = {value: body & UPPER_BODY for value, body in MACHINES.items()}
    assert {value: len(joints) for value, joints in uppers.items()} == {4: 11, 5: 17, 6: 15}
    for value, joints in uppers.items():
        assert is_upper_body(joints)
        assert machine_for(joints, upper_body=True) == value
        assert machine_for(joints) is None
        check_body(value, joints, upper_body=True)


def test_a_whole_body_is_not_an_upper_body() -> None:
    for body in MACHINES.values():
        assert not is_upper_body(body)
    assert not is_upper_body(())


def test_an_upper_body_is_refused_as_a_whole_body() -> None:
    """Without the caller saying the legs are held, a leg-less contract is ADR-0027's case."""
    with pytest.raises(BodyMismatchError, match="left_knee_joint"):
        check_body(6, BODY_29DOF_LOCK_WAIST & UPPER_BODY)


def test_the_upper_body_of_6_on_body_5_names_the_waist_it_would_leave_limp() -> None:
    with pytest.raises(BodyMismatchError, match="верх тела mode_machine = 5") as caught:
        check_body(5, BODY_29DOF_LOCK_WAIST & UPPER_BODY, upper_body=True)
    assert "waist_roll_joint" in str(caught.value)


def test_the_example_mounted_profile_is_the_upper_body_of_6() -> None:
    if not MOUNTED_PROFILE.is_file():
        pytest.skip("профиля закреплённого G1 в examples нет")
    order = yaml.safe_load(MOUNTED_PROFILE.read_text(encoding="utf-8"))["contract"][
        "joint_order"
    ]
    assert is_upper_body(order)
    check_body(6, order, upper_body=True)
    assert list(order) == [name for name in LAYOUT if name in order], "порядок — порядок слотов"
