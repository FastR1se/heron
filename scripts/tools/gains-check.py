"""Are a profile's drive gains stable where they are computed? (ADR-0045, item 5; ADR-0063, item 5)

The law ``tau = kp * (q* - q) - kd * dq`` is computed every ``dt = 1 / state_rate_hz`` and held
until the next computation, in Isaac (explicit actuator) and in Gazebo (a controller in the
manager's step). For one joint with inertia ``I`` about its axis that is a double integrator under
a zero-order hold, and its closed loop is stable exactly when

    b = kd * dt / I < 2        and        kp * dt < 2 * kd

The first bound is tightest where ``I`` is least. Neither world adds a rotor's inertia: the URDF
has none, Gazebo reads none from it and the platform's converter writes zero armature. So ``I`` is
the description's own, and the least one the moving part can present is looked for by moving the
contract's joints inside it over the profile's envelope -- on a grid when the subtree is small, by
a seeded sample when it is not (then ``I_min`` is an upper bound of the least, and says so).

``scripts/tools/arm-gains.py`` derives gains for a serial arm; this script checks the gains a
profile already states, on any tree. It prints, per joint, the least and greatest inertia, ``b``
and the damping ratio at both ends, and exits 1 if a joint is unstable at its lightest.

Usage: python scripts/tools/gains-check.py <profile.yaml> [--samples N] [--seed S]
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import yaml

GRID = 5
"""Samples per downstream joint across the envelope, bounds included, when a grid is small."""

GRID_LIMIT = 20_000
"""Largest grid swept exhaustively; a larger subtree is sampled."""


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s, v = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ]
    )


def rpy(values: list[float]) -> np.ndarray:
    roll, pitch, yaw = values
    return (
        rotation(np.array([0.0, 0.0, 1.0]), yaw)
        @ rotation(np.array([0.0, 1.0, 0.0]), pitch)
        @ rotation(np.array([1.0, 0.0, 0.0]), roll)
    )


def numbers(text: str | None, default: str) -> list[float]:
    return [float(item) for item in (text or default).split()]


class Tree:
    """Links with their inertia, joints with their frames: what the sweep walks."""

    def __init__(self, urdf: Path) -> None:
        root = ET.parse(urdf).getroot()
        self.inertial: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
        for link in root.iter("link"):
            body = link.find("inertial")
            inertia = body.find("inertia") if body is not None else None
            mass = body.find("mass") if body is not None else None
            if body is None or inertia is None or mass is None:
                continue
            origin = body.find("origin")
            com = np.array(numbers(origin.get("xyz") if origin is not None else None, "0 0 0"))
            frame = rpy(numbers(origin.get("rpy") if origin is not None else None, "0 0 0"))
            i = inertia.attrib
            tensor = np.array(
                [
                    [float(i["ixx"]), float(i["ixy"]), float(i["ixz"])],
                    [float(i["ixy"]), float(i["iyy"]), float(i["iyz"])],
                    [float(i["ixz"]), float(i["iyz"]), float(i["izz"])],
                ]
            )
            self.inertial[link.get("name", "")] = (
                float(mass.get("value", "0")),
                com,
                frame @ tensor @ frame.T,
            )
        self.joints: dict[str, dict[str, Any]] = {}
        for joint in root.iter("joint"):
            origin = joint.find("origin")
            axis = joint.find("axis")
            parent = joint.find("parent")
            child = joint.find("child")
            assert parent is not None and child is not None
            xyz = origin.get("xyz") if origin is not None else None
            angles = origin.get("rpy") if origin is not None else None
            self.joints[joint.get("name", "")] = {
                "type": joint.get("type"),
                "parent": parent.get("link"),
                "child": child.get("link"),
                "xyz": np.array(numbers(xyz, "0 0 0")),
                "rpy": rpy(numbers(angles, "0 0 0")),
                "axis": np.array(numbers(axis.get("xyz") if axis is not None else None, "1 0 0")),
            }
        self.below: dict[str, list[str]] = {}
        for name, joint in self.joints.items():
            self.below.setdefault(joint["parent"], []).append(name)

    def subtree(self, link: str) -> list[str]:
        """Movable joints hanging below a link, in document order of discovery."""
        found: list[str] = []
        for name in self.below.get(link, []):
            if self.joints[name]["type"] in ("revolute", "continuous"):
                found.append(name)
            found += self.subtree(self.joints[name]["child"])
        return found

    def inertia(self, name: str, angles: dict[str, float]) -> float:
        """Inertia of everything below a joint, about its axis, at these joint angles."""
        joint = self.joints[name]
        axis = joint["axis"] / np.linalg.norm(joint["axis"])
        total = 0.0
        stack = [(joint["child"], np.eye(3), np.zeros(3))]
        while stack:
            link, frame, position = stack.pop()
            if link in self.inertial:
                mass, com, tensor = self.inertial[link]
                p = position + frame @ com
                about = frame @ tensor @ frame.T + mass * (p @ p * np.eye(3) - np.outer(p, p))
                total += float(axis @ about @ axis)
            for child_name in self.below.get(link, []):
                child = self.joints[child_name]
                turn = child["rpy"]
                if child["type"] in ("revolute", "continuous"):
                    turn = turn @ rotation(child["axis"], angles.get(child_name, 0.0))
                stack.append((child["child"], frame @ turn, position + frame @ child["xyz"]))
        return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Устойчивость коэффициентов привода профиля.")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    contract = profile["contract"]
    order: list[str] = contract["joint_order"]
    action = contract["action"]
    if "kp" not in action or "kd" not in action:
        print(f"{args.profile}: у действия нет kp и kd — проверять нечего")
        return 1
    kp: list[float] = [float(v) for v in action["kp"]]
    kd: list[float] = [float(v) for v in action["kd"]]
    dt = 1.0 / float(contract["state_rate_hz"])
    low = profile["limits"]["joint_position"]["min"]
    high = profile["limits"]["joint_position"]["max"]
    envelope = {name: (low[i], high[i]) for i, name in enumerate(order)}

    urdf = args.profile.parent / profile["description"]["urdf"]
    if not urdf.is_file():
        print(f"{urdf}: описания нет (загрузите его скриптом рядом с профилем)")
        return 1
    tree = Tree(urdf)
    rng = random.Random(args.seed)

    print(f"{args.profile}: dt = {dt:g} с; b = kd·dt/I < 2, kp·dt < 2·kd; ζ = kd / (2·√(kp·I))\n")
    print(
        f"{'сустав':28} {'I_min':>9} {'I_max':>9} {'kp':>7} {'kd':>7} {'b':>6} "
        f"{'kp·dt/kd':>8} {'ζ(I_min)':>8} {'ζ(I_max)':>8}  выборка"
    )
    unstable: list[str] = []
    for index, name in enumerate(order):
        # Joints below this one that the envelope moves; anything else stays at zero.
        downstream = [d for d in tree.subtree(tree.joints[name]["child"]) if d in envelope]
        grids = [np.linspace(*envelope[d], GRID) for d in downstream]
        if GRID ** len(downstream) <= GRID_LIMIT:
            poses = [dict(zip(downstream, values, strict=True)) for values in
                     itertools.product(*grids)] if downstream else [{}]
            how = f"сетка {GRID}^{len(downstream)}"
        else:
            poses = [dict.fromkeys(downstream, 0.0)] + [
                {d: rng.uniform(*envelope[d]) for d in downstream} for _ in range(args.samples)
            ]
            how = f"{args.samples} случайных (I_min — оценка сверху)"
        values = [tree.inertia(name, pose) for pose in poses]
        least, most = min(values), max(values)
        b = kd[index] * dt / least
        ratio = kp[index] * dt / kd[index] if kd[index] else math.inf
        zeta_light = kd[index] / (2 * math.sqrt(kp[index] * least)) if kp[index] else math.inf
        zeta_heavy = kd[index] / (2 * math.sqrt(kp[index] * most)) if kp[index] else math.inf
        mark = ""
        if not (b < 2 and ratio < 2):
            unstable.append(name)
            mark = "  НЕУСТОЙЧИВ"
        print(
            f"{name:28} {least:9.5f} {most:9.5f} {kp[index]:7.3g} {kd[index]:7.3g} {b:6.2f} "
            f"{ratio:8.3f} {zeta_light:8.2f} {zeta_heavy:8.2f}  {how}{mark}"
        )

    if unstable:
        print(f"\nнеустойчивы на самой лёгкой нагрузке: {', '.join(unstable)}")
        return 1
    print("\nвсе суставы устойчивы на самой лёгкой нагрузке")
    return 0


if __name__ == "__main__":
    sys.exit(main())
