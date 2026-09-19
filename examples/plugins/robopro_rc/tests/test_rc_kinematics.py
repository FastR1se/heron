"""The hand-written RC5 description is the DH table it claims to be (ADR-0055).

Whether the table's convention is the robot's is the probe's question. Whether the file
says what the table says, under the convention its header assumes, is this file's: a
slip in one origin would otherwise be found on the robot, as a flange pose that does
not match for a reason nobody would look for in the description.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from robopro_rc.kinematics import (
    RC5_DH,
    UrdfChain,
    compare,
    dh_classic,
    dh_modified,
    difference,
    flange_matrix,
    main,
    rpy,
)

EXAMPLES = Path(__file__).resolve().parents[3]
DESCRIPTION = EXAMPLES / "robots" / "rc5" / "rc5.urdf"

pytestmark = pytest.mark.skipif(
    not DESCRIPTION.is_file(), reason="описания RC5 в examples/robots нет"
)


def poses(count: int, seed: int = 5) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-math.pi, math.pi) for _ in range(6)] for _ in range(count)]


def test_the_description_is_the_classic_reading_of_the_table() -> None:
    chain = UrdfChain.from_file(DESCRIPTION)
    assert chain.joints == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    for q in [[0.0] * 6, *poses(50)]:
        distance, angle = difference(chain.pose(q), dh_classic(RC5_DH, q))
        assert distance < 1e-9 and angle < 1e-7, q


def test_the_other_convention_is_told_apart() -> None:
    """The check could fail: under the modified reading the flange is elsewhere."""
    worst = max(
        difference(dh_classic(RC5_DH, q), dh_modified(RC5_DH, q))[0] for q in poses(20)
    )
    assert worst > 0.05


def test_the_zero_pose_is_the_tables() -> None:
    """At zero, worked out by hand from the rows: x = a2 + a3, y = -(d2 + d3 + d4 + d6),
    z = d1 - d5 -- independent of any code that multiplies matrices."""
    pose = UrdfChain.from_file(DESCRIPTION).pose([0.0] * 6)
    a, d = RC5_DH.a, RC5_DH.d
    # Rx(pi/2) after row 1 turns the z of rows 2-4 into -y; row 4's Rx(pi/2) turns row 5's
    # z into -z, and row 5's Rx(-pi/2) turns row 6's back into -y.
    assert pose[0][3] == pytest.approx(a[1] + a[2], abs=1e-6)
    assert pose[1][3] == pytest.approx(-(d[1] + d[2] + d[3]) - d[5], abs=1e-5)
    assert pose[2][3] == pytest.approx(d[0] - d[4], abs=1e-5)


@pytest.mark.parametrize("angles", ["xyz_extrinsic", "rotation_vector"])
def test_a_pose_reported_by_the_controller_is_read_both_ways(angles: str) -> None:
    turn = rpy(0.3, -0.2, 0.9)
    if angles == "xyz_extrinsic":
        reported = (0.1, 0.2, 0.3, 0.3, -0.2, 0.9)
    else:
        angle = math.acos((turn[0][0] + turn[1][1] + turn[2][2] - 1) / 2)
        axis = (turn[2][1] - turn[1][2], turn[0][2] - turn[2][0], turn[1][0] - turn[0][1])
        norm = math.sqrt(sum(value * value for value in axis))
        reported = (0.1, 0.2, 0.3, *(value / norm * angle for value in axis))
    matrix = flange_matrix(reported, angles)
    assert difference(matrix, flange_matrix((0.1, 0.2, 0.3, 0.3, -0.2, 0.9), "xyz_extrinsic"))[
        1
    ] == pytest.approx(0.0, abs=1e-9)


def test_pairs_the_description_explains_compare_to_zero(tmp_path: Path) -> None:
    """Pairs made from the table itself: the classic chains fit, the modified does not."""
    chain = UrdfChain.from_file(DESCRIPTION)
    pairs = []
    for q in poses(5):
        pose = dh_classic(RC5_DH, q)
        roll = math.atan2(pose[2][1], pose[2][2])
        pitch = math.asin(-pose[2][0])
        yaw = math.atan2(pose[1][0], pose[0][0])
        pairs.append({"act_q": q, "act_flange_x": [pose[0][3], pose[1][3], pose[2][3],
                                                   roll, pitch, yaw]})
    worst = compare(pairs, chain)["worst"]
    assert worst["urdf/xyz_extrinsic"]["position_mm"] < 1e-6
    assert worst["urdf/xyz_extrinsic"]["orientation_deg"] < 1e-5
    assert worst["dh_modified/xyz_extrinsic"]["position_mm"] > 10

    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    assert main([str(path), "--urdf", str(DESCRIPTION)]) == 0
