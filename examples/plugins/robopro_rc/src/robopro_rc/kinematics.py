"""Forward kinematics of the RC5 description, for the check on the robot (ADR-0055).

The description ``examples/robots/rc5/rc5.urdf`` is written by hand from the DH table
of the vendor's API, and the convention of that table is named nowhere: it is
*assumed* classic. The check is on the robot: the probe records pairs of joint angles
``act_q`` and the flange pose ``act_flange_x`` the controller computes from them, and
this module computes the flange from the same angles through the description.

Three chains are compared with every pair, so that a mismatch says more than "wrong":

``urdf``          the description as written -- the one the platform uses;
``dh_classic``    the table under the classic (distal) convention;
``dh_modified``   the same table under the modified (proximal) convention.

``urdf`` and ``dh_classic`` agree by construction (a test holds that to 1e-9). If both
miss and ``dh_modified`` fits, the convention was wrong; if all miss by a constant, the
zeros of the joints are offset. Two readings of the controller's three angles are
compared as well: extrinsic X-Y-Z, as the SDK's docstring says with a caveat, and a
rotation vector.

Plain Python on purpose: the platform's core installs without numpy, and so does this
package. Six 4x4 products per pose are not worth a dependency.

    python -m robopro_rc.kinematics pairs.json --urdf examples/robots/rc5/rc5.urdf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class DhTable:
    """One row per joint: twist ``alpha`` (rad), length ``a`` (m), offset ``d`` (m) and
    the zero of the joint angle ``offset`` (rad)."""

    alpha: tuple[float, ...]
    a: tuple[float, ...]
    d: tuple[float, ...]
    offset: tuple[float, ...]


RC5_DH: Final = DhTable(
    alpha=(1.570796, 0.0, 0.0, 1.570796, -1.570796, 0.0),
    a=(0.0, -0.405, -0.3722, 0.0, 0.0, 0.0),
    d=(0.1725, 0.156, -0.1485, 0.1398, 0.1398, 0.1398),
    offset=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
"""``DhParamsManager._RC5_DH_PARAMS`` of API 1.4.1 (fact 0.34), digits as printed there
(``1.570796`` for pi/2). The vendor's "ideal" parameters of the model, not a
calibration of a body; ``theta`` of the table is zero and not listed."""

IDENTITY: Final[Matrix] = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def multiply(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(math.fsum(left[i][k] * right[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def chain(*matrices: Matrix) -> Matrix:
    result = IDENTITY
    for matrix in matrices:
        result = multiply(result, matrix)
    return result


def translation(x: float, y: float, z: float) -> Matrix:
    return ((1.0, 0.0, 0.0, x), (0.0, 1.0, 0.0, y), (0.0, 0.0, 1.0, z), (0.0, 0.0, 0.0, 1.0))


def rotation(axis: Sequence[float], angle: float) -> Matrix:
    """About a unit axis through the origin (Rodrigues)."""
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("ось вращения нулевой длины")
    x, y, z = x / norm, y / norm, z / norm
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def rpy(roll: float, pitch: float, yaw: float) -> Matrix:
    """URDF's ``rpy`` and extrinsic X-Y-Z alike: ``Rz(yaw) · Ry(pitch) · Rx(roll)``."""
    return chain(
        rotation((0, 0, 1), yaw), rotation((0, 1, 0), pitch), rotation((1, 0, 0), roll)
    )


def dh_classic(table: DhTable, q: Sequence[float]) -> Matrix:
    """``Π Rz(θ_i + offset_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)`` -- base to flange."""
    _check_width(table, q)
    return chain(
        *(
            chain(
                rotation((0, 0, 1), q[i] + table.offset[i]),
                translation(table.a[i], 0.0, table.d[i]),
                rotation((1, 0, 0), table.alpha[i]),
            )
            for i in range(len(q))
        )
    )


def dh_modified(table: DhTable, q: Sequence[float]) -> Matrix:
    """``Π Rx(α_i) · Tx(a_i) · Rz(θ_i + offset_i) · Tz(d_i)`` -- the table read the other way."""
    _check_width(table, q)
    return chain(
        *(
            chain(
                rotation((1, 0, 0), table.alpha[i]),
                translation(table.a[i], 0.0, 0.0),
                rotation((0, 0, 1), q[i] + table.offset[i]),
                translation(0.0, 0.0, table.d[i]),
            )
            for i in range(len(q))
        )
    )


def _check_width(table: DhTable, q: Sequence[float]) -> None:
    if len(q) != len(table.alpha):
        raise ValueError(f"{len(q)} углов при {len(table.alpha)} строках таблицы DH")


@dataclass(frozen=True, slots=True)
class Step:
    """One joint of a URDF chain: its fixed origin, then its motion."""

    name: str
    origin: Matrix
    axis: tuple[float, float, float]
    moves: bool


@dataclass(frozen=True, slots=True)
class UrdfChain:
    """The joints from ``base`` to ``tip`` of a description, in order."""

    steps: tuple[Step, ...]

    @classmethod
    def from_file(cls, path: Path, base: str = "world", tip: str = "flange") -> UrdfChain:
        """Walk from ``tip`` back to ``base`` through the parents of each link."""
        root = ET.parse(path).getroot()
        by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                by_child[child.get("link", "")] = joint
        steps: list[Step] = []
        link = tip
        while link != base:
            found = by_child.get(link)
            if found is None:
                raise ValueError(f"{path}: звено {link!r} ни к чему не присоединено — до {base!r} "
                                 f"цепь не доходит")
            steps.append(_step(found))
            parent = found.find("parent")
            link = parent.get("link", "") if parent is not None else ""
        return cls(steps=tuple(reversed(steps)))

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps if step.moves)

    def pose(self, q: Sequence[float]) -> Matrix:
        if len(q) != len(self.joints):
            raise ValueError(f"{len(q)} углов при {len(self.joints)} подвижных суставах цепи")
        angles = iter(q)
        result = IDENTITY
        for step in self.steps:
            result = multiply(result, step.origin)
            if step.moves:
                result = multiply(result, rotation(step.axis, next(angles)))
        return result


def _step(joint: ET.Element) -> Step:
    kind = joint.get("type", "")
    if kind not in {"revolute", "continuous", "fixed"}:
        raise ValueError(f"сустав {joint.get('name')!r} типа {kind!r}: цепь знает вращательные "
                         f"и неподвижные")
    origin = joint.find("origin")
    xyz = _vector(origin.get("xyz") if origin is not None else None)
    angles = _vector(origin.get("rpy") if origin is not None else None)
    axis_element = joint.find("axis")
    axis = _vector(axis_element.get("xyz") if axis_element is not None else "1 0 0")
    return Step(
        name=joint.get("name", ""),
        origin=multiply(translation(*xyz), rpy(*angles)),
        axis=axis,
        moves=kind != "fixed",
    )


def _vector(text: str | None) -> tuple[float, float, float]:
    if text is None:
        return (0.0, 0.0, 0.0)
    values = [float(part) for part in text.split()]
    if len(values) != 3:
        raise ValueError(f"ожидались три числа, получено {text!r}")
    return (values[0], values[1], values[2])


def flange_matrix(pose: Sequence[float], angles: str) -> Matrix:
    """The controller's flange pose as a matrix, under one reading of its three angles.

    ``xyz_extrinsic``    rotations about the fixed X, Y, Z (the SDK's docstring, hedged);
    ``rotation_vector``  axis times angle, as some controllers report orientation.
    """
    x, y, z, a, b, c = (float(value) for value in pose)
    if angles == "xyz_extrinsic":
        turn = rpy(a, b, c)
    elif angles == "rotation_vector":
        angle = math.sqrt(a * a + b * b + c * c)
        turn = IDENTITY if angle == 0 else rotation((a, b, c), angle)
    else:
        raise ValueError(f"{angles!r}: известны xyz_extrinsic и rotation_vector")
    return multiply(translation(x, y, z), turn)


ANGLE_READINGS: Final = ("xyz_extrinsic", "rotation_vector")


def difference(left: Matrix, right: Matrix) -> tuple[float, float]:
    """Distance between the origins (m) and the angle of the relative rotation (rad)."""
    distance = math.sqrt(sum((left[i][3] - right[i][3]) ** 2 for i in range(3)))
    trace = sum(left[k][i] * right[k][i] for i in range(3) for k in range(3))
    return distance, math.acos(max(-1.0, min(1.0, (trace - 1) / 2)))


def compare(pairs: Sequence[dict[str, Any]], urdf: UrdfChain) -> dict[str, Any]:
    """Worst position and orientation error of each model over every pair.

    A pair is ``{"act_q": [6 rad], "act_flange_x": [x, y, z (m), 3 angles (rad)]}``,
    as the probe writes it. Returned as numbers, for a line of the experiment record.
    """
    models: dict[str, Callable[[Sequence[float]], Matrix]] = {
        "urdf": urdf.pose,
        "dh_classic": lambda q: dh_classic(RC5_DH, q),
        "dh_modified": lambda q: dh_modified(RC5_DH, q),
    }
    table: dict[str, Any] = {}
    for name, model in models.items():
        for reading in ANGLE_READINGS:
            worst_m, worst_rad = 0.0, 0.0
            for pair in pairs:
                computed = model([float(v) for v in pair["act_q"]])
                reported = flange_matrix(pair["act_flange_x"], reading)
                distance, angle = difference(computed, reported)
                worst_m, worst_rad = max(worst_m, distance), max(worst_rad, angle)
            table[f"{name}/{reading}"] = {
                "position_mm": worst_m * 1e3,
                "orientation_deg": math.degrees(worst_rad),
            }
    return {"pairs": len(pairs), "worst": table}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("pairs", type=Path, help="JSON пробы: список пар act_q / act_flange_x")
    parser.add_argument("--urdf", type=Path, required=True, help="описание RC5")
    args = parser.parse_args(argv)
    data = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = data["pairs"] if isinstance(data, dict) else data
    if not pairs:
        print("пар нет: проба не записала ни одной позы", file=sys.stderr)
        return 2
    result = compare(pairs, UrdfChain.from_file(args.urdf))
    print(f"пар: {result['pairs']}; худшее расхождение с act_flange_x контроллера")
    for name, worst in result["worst"].items():
        print(f"  {name:32s} {worst['position_mm']:10.3f} мм {worst['orientation_deg']:10.4f}°")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
