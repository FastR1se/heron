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

BODY_29DOF_LOCK_WAIST: Final = BODY_29DOF - {
    "waist_roll_joint",
    "waist_pitch_joint",
}
"""The 29-DoF body with its waist locked: roll and pitch are fixed joints of its
description, slots 13 and 14 empty, both wrists whole."""

LEGS: Final = frozenset(LAYOUT[:12])
"""Slots 0-11: what the vendor's balance controller keeps when the upper body is lent out."""

UPPER_BODY: Final = frozenset(LAYOUT[12:])
"""Slots 12-28: waist and arms, the joints the vendor's ``rt/arm_sdk`` example commands
(``g1_arm7_sdk_dds_example.cpp``, unitree_sdk2 ``9754cd1``) while the robot balances on its own.

Inside a body it is the body without its legs. A contract of exactly that is the one
subset of a body the package accepts (ADR-P6B): 11 joints of body 4, 17 of body 5, 15 of
body 6 -- three different sets, so an upper body names its body as a whole body does."""

MACHINES: Final[dict[int, frozenset[str]]] = {
    4: BODY_23DOF,
    5: BODY_29DOF,
    6: BODY_29DOF_LOCK_WAIST,
}
"""The body behind each ``mode_machine`` a robot reports.

Read off the vendor's table of descriptions -- the README of ``robots/g1_description``
in ``unitree_ros``, at the commit the profiles fetch -- which names the description
of each value: 4 ``g1_23dof_rev_1_0``, 5 ``g1_29dof_rev_1_0``, 6
``g1_29dof_lock_waist_rev_1_0``. Each body is the set of movable joints of that
description, and a test holds the two together. The vendor's C++ twin agrees on the
two it can load: 4 for a scene whose file name contains "23", 5 otherwise.

The table names more values than this: earlier revisions and bodies with other
motors. A value missing here refuses the connection with its number, and joins
from the table when a robot reports it -- 6 did (journal 2026-09-18)."""


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


def is_upper_body(joint_order: Iterable[str]) -> bool:
    """Whether a contract names no leg: it can only be the upper body of some body, or nothing."""
    joints = frozenset(joint_order)
    return bool(joints) and joints <= UPPER_BODY


def machine_for(joint_order: Iterable[str], *, upper_body: bool = False) -> int | None:
    """The ``mode_machine`` of the body with exactly these joints, if the package knows one.

    With ``upper_body``, the body whose upper body these joints are (ADR-P6B).
    """
    wanted = frozenset(joint_order)
    for value, body in MACHINES.items():
        if (body & UPPER_BODY if upper_body else body) == wanted:
            return value
    return None


def empty_slots(joint_order: Iterable[str]) -> frozenset[int]:
    """Slots no joint of the contract occupies."""
    used = set(slots_of(tuple(joint_order)))
    return frozenset(slot for slot in range(SLOT_COUNT) if slot not in used)


def check_body(mode_machine: int, joint_order: Iterable[str], *, upper_body: bool = False) -> None:
    """Refuse a link whose robot is not the body the contract commands.

    Equal sets, not a subset: at the low access level every motor of the body is
    commanded, and a motor the contract does not name would get zero gains and
    hang limp while the rest of the robot moves. The one exception is ``upper_body``
    (ADR-P6B): the body without its legs, for a robot whose legs hang from a mount --
    the caller has established that, this function only checks which body it is.
    """
    body = MACHINES.get(mode_machine)
    if body is None:
        known = ", ".join(f"{value} ({len(joints)} DoF)" for value, joints in MACHINES.items())
        raise BodyMismatchError(
            f"на связи робот с mode_machine = {mode_machine}, а какое это тело, пакету "
            f"неизвестно; известны: {known}. Команды не отправлены"
        )
    declared = frozenset(joint_order)
    if upper_body:
        body = body & UPPER_BODY
    if declared == body:
        return
    absent = sorted(body - declared, key=LAYOUT.index)
    foreign = sorted(declared - body, key=_layout_order)
    details = []
    if absent:
        details.append(f"у тела есть, в контракте нет: {', '.join(absent)}")
    if foreign:
        details.append(f"в контракте есть, у тела нет: {', '.join(foreign)}")
    what = "верх тела" if upper_body else "тело"
    raise BodyMismatchError(
        f"на связи {what} mode_machine = {mode_machine} ({len(body)} DoF), контракт "
        f"описывает {len(declared)} суставов — {'; '.join(details)}. "
        f"Подключён не тот робот или не тот профиль; команды не отправлены"
    )
