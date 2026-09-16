"""Drive gains of arm6 from the inertias of its description (ADR-0045, item 5).

The law is ``tau = kp * (q* - q) - kd * dq``, computed every ``dt = 1 / state_rate_hz`` and
held until the next computation. For one joint with inertia ``I`` about its axis that is a
double integrator under a zero-order hold, and its closed loop is stable exactly when

    b = kd * dt / I < 2        and        a = kp * dt**2 / I < 2 * b

The second condition does not depend on ``I``; the first is tightest where ``I`` is least.
So each joint is designed at the least inertia its moving part can present inside the
profile's envelope, found here by sweeping the joints after it over that envelope:

    kd = I_min / dt                  b = 1 at the lightest load, half the bound
    kp = I_min / (4 * dt**2)         critically damped there, natural frequency 1 / (2 dt)

A heavier configuration is only more stable; its damping ratio falls as sqrt(I_min / I).
Coupling between joints, gravity and contact are outside this criterion and are what a run
measures, not what this script claims.

Usage: python scripts/phase6-arm6-gains.py [profile]   (default examples/robots/arm6.yaml)
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import yaml

GRID = 9
"""Samples per downstream joint across the envelope, bounds included."""


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


def main() -> None:
    profile_path = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/robots/arm6.yaml")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    contract = profile["contract"]
    order: list[str] = contract["joint_order"]
    dt = 1.0 / float(contract["state_rate_hz"])
    low = profile["limits"]["joint_position"]["min"]
    high = profile["limits"]["joint_position"]["max"]

    root = ET.parse(profile_path.parent / profile["description"]["urdf"]).getroot()
    inertial: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for link in root.iter("link"):
        body = link.find("inertial")
        if body is None:
            continue
        origin = body.find("origin")
        com = np.array(numbers(origin.get("xyz") if origin is not None else None, "0 0 0"))
        frame = rpy(numbers(origin.get("rpy") if origin is not None else None, "0 0 0"))
        i = body.find("inertia").attrib  # type: ignore[union-attr]
        tensor = np.array(
            [
                [float(i["ixx"]), float(i["ixy"]), float(i["ixz"])],
                [float(i["ixy"]), float(i["iyy"]), float(i["iyz"])],
                [float(i["ixz"]), float(i["iyz"]), float(i["izz"])],
            ]
        )
        mass = float(body.find("mass").get("value"))  # type: ignore[union-attr, arg-type]
        inertial[link.get("name")] = (mass, com, frame @ tensor @ frame.T)  # type: ignore[index]

    joints = {}
    for joint in root.iter("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints[joint.get("name")] = {
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),  # type: ignore[union-attr]
            "child": joint.find("child").get("link"),  # type: ignore[union-attr]
            "xyz": np.array(numbers(origin.get("xyz") if origin is not None else None, "0 0 0")),
            "rpy": rpy(numbers(origin.get("rpy") if origin is not None else None, "0 0 0")),
            "axis": np.array(numbers(axis.get("xyz") if axis is not None else None, "1 0 0")),
        }
    below: dict[str, list[str]] = {}
    for name, joint in joints.items():
        below.setdefault(joint["parent"], []).append(name)

    print(f"{profile_path}: dt = {dt:g} s, сетка {GRID} точек на сустав в рамке профиля\n")
    header = f"{'сустав':8} {'I_min':>10} {'I_max':>10} {'kd':>9} {'kp':>9} {'b':>5} {'a':>6} {'ζ(I_max)':>9}"
    print(header)
    rows = []
    for index, name in enumerate(order):
        joint = joints[name]
        axis = joint["axis"] / np.linalg.norm(joint["axis"])
        downstream = order[index + 1 :]
        grids = [np.linspace(low[order.index(d)], high[order.index(d)], GRID) for d in downstream]
        least, most = math.inf, 0.0
        for values in itertools.product(*grids) if grids else [()]:
            angles = dict(zip(downstream, values, strict=True))
            total = 0.0
            # Walk the subtree from the joint's frame; the joint's own angle does not change
            # the inertia about its own axis.
            stack = [(joint["child"], np.eye(3), np.zeros(3))]
            while stack:
                link, frame, position = stack.pop()
                if link in inertial:
                    mass, com, tensor = inertial[link]
                    p = position + frame @ com
                    about = frame @ tensor @ frame.T + mass * (p @ p * np.eye(3) - np.outer(p, p))
                    total += float(axis @ about @ axis)
                for child_name in below.get(link, []):
                    child = joints[child_name]
                    turn = child["rpy"]
                    if child["type"] in ("revolute", "continuous"):
                        turn = turn @ rotation(child["axis"], angles.get(child_name, 0.0))
                    stack.append(
                        (child["child"], frame @ turn, position + frame @ child["xyz"])
                    )
            least, most = min(least, total), max(most, total)

        kd = float(f"{least / dt:.2g}")
        kp = float(f"{least / (4 * dt * dt):.2g}")
        b = kd * dt / least
        a = kp * dt * dt / least
        assert b < 2 and a < 2 * b, f"{name}: округлённые коэффициенты неустойчивы"
        zeta = kd / (2 * math.sqrt(kp * most))
        rows.append((kp, kd))
        print(f"{name:8} {least:10.5f} {most:10.5f} {kd:9.3g} {kp:9.3g} {b:5.2f} {a:6.3f} {zeta:9.2f}")

    print("\nkp:", [kp for kp, _ in rows])
    print("kd:", [kd for _, kd in rows])


if __name__ == "__main__":
    main()
