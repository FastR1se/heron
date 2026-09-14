"""Where each joint of a G1 sits in the vendor's low-level messages (ADR-0027).

``LowCmd`` and ``LowState`` address motors by index into 35 slots, not by name,
and the index is not the position of the joint in any description. It is the
layout of the 29-DoF body: a modification with fewer joints leaves its missing
slots empty rather than closing the gap. The vendor's own table
``g1_joint_index_dds.md`` closes the gap and is wrong -- a command written from
it sends the left shoulder to a dummy waist joint. The layout below is read from
the order of actuators in the twin's MJCF, which its bridge maps one to one onto
``motor_cmd``, and a test holds it against that file.

A robot that moves and is wrong is the failure this module exists to prevent,
so joints are found by name and nothing here is positional.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

SLOT_COUNT: Final = 35
"""Motor slots in ``unitree_hg`` ``LowCmd_`` and ``LowState_``."""

LAYOUT: Final[tuple[str, ...]] = (
    # Legs, 0-11.
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist, 12-14.
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm, 15-21.
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm, 22-28.
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
"""Joint of each slot, in slot order. Slots 29-34 carry no joint of a G1."""

BODY_29DOF: Final = frozenset(LAYOUT)

BODY_23DOF: Final = BODY_29DOF - {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
"""Slots 13, 14, 20, 21, 27 and 28 empty: "INVALID for g1 23dof" in the vendor's
examples, dummy bodies twenty metres above the world in the twin's MJCF."""

MACHINES: Final[dict[int, frozenset[str]]] = {4: BODY_23DOF, 5: BODY_29DOF}
"""The body behind each ``mode_machine`` a robot reports.

Taken from the vendor's C++ twin, which reports 4 for a scene whose file name
contains "23" and 5 otherwise. What the hardware reports is not visible in any
source; a value missing here refuses the connection with the number, and is
added as a fact when it is one (phase 7), not guessed."""


class SlotError(RuntimeError):
    """A joint the layout does not have."""


class BodyMismatchError(RuntimeError):
    """The robot on the link is not the body the contract describes."""


def _layout_order(name: str) -> tuple[int, str]:
    """Joints in slot order, anything outside the layout after them by name."""
    return (LAYOUT.index(name) if name in BODY_29DOF else len(LAYOUT), name)


def slots_of(joint_order: Sequence[str]) -> tuple[int, ...]:
    """The slot of each joint, in the order given."""
    unknown = [name for name in joint_order if name not in BODY_29DOF]
    if unknown:
        raise SlotError(
            f"суставов {', '.join(unknown)} нет в раскладке слотов G1; "
            f"команда для них не может быть адресована ни одному мотору. "
            f"Есть: {', '.join(LAYOUT)}"
        )
    return tuple(LAYOUT.index(name) for name in joint_order)


def empty_slots(joint_order: Iterable[str]) -> frozenset[int]:
    """Slots no joint of the contract occupies."""
    used = set(slots_of(tuple(joint_order)))
    return frozenset(slot for slot in range(SLOT_COUNT) if slot not in used)


def check_body(mode_machine: int, joint_order: Iterable[str]) -> None:
    """Refuse a link whose robot is not the body the contract commands.

    Equal sets, not a subset: at the low access level every motor of the body is
    commanded, and a motor the contract does not name would get zero gains and
    hang limp while the rest of the robot moves.
    """
    body = MACHINES.get(mode_machine)
    if body is None:
        known = ", ".join(f"{value} ({len(joints)} DoF)" for value, joints in MACHINES.items())
        raise BodyMismatchError(
            f"на связи робот с mode_machine = {mode_machine}, а какое это тело, пакету "
            f"неизвестно; известны: {known}. Команды не отправлены"
        )
    declared = frozenset(joint_order)
    if declared == body:
        return
    absent = sorted(body - declared, key=LAYOUT.index)
    foreign = sorted(declared - body, key=_layout_order)
    details = []
    if absent:
        details.append(f"у тела есть, в контракте нет: {', '.join(absent)}")
    if foreign:
        details.append(f"в контракте есть, у тела нет: {', '.join(foreign)}")
    raise BodyMismatchError(
        f"на связи тело mode_machine = {mode_machine} ({len(body)} DoF), контракт "
        f"описывает {len(declared)} суставов — {'; '.join(details)}. "
        f"Подключён не тот робот или не тот профиль; команды не отправлены"
    )
