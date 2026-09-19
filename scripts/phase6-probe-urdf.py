"""Phase 6, stage 0, fact 0.15: what Isaac Lab makes of a platform description.

Run with Isaac Lab's launcher (``isaaclab.sh -p``) from ``scripts/phase6-probe.sh urdf``. The
description is converted to USD, spawned as an articulation with a fixed base, and the facts the
contract depends on are written to JSON: the DOF order PhysX assigns (fact 0.4 says nothing else
decides it), the bodies left after merging fixed joints, and the limits and drive gains the
converter took from the URDF. Nothing is set from the profile on purpose -- the question is what
arrives by default, before the platform says anything.

Phase 6, stage 7 (ADR-0063) asks one more thing of it: where the bodies are in the world after the
reset and after ``--steps`` physics steps. A description bolted to the world by a ``world`` link
is converted with ``fix_base``, and whether the importer keeps that joint's origin -- the height
of the G1's pelvis on its stand -- or fixes the root link where the articulation stands is not in
Isaac Lab's source (ADR-0048 left it open). The positions answer it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", required=True, help="description to convert")
parser.add_argument("--usd-dir", required=True, help="where the converted USD goes")
parser.add_argument("--merge-fixed-joints", choices=["yes", "no"], required=True)
parser.add_argument("--out", required=True, help="JSON with the facts")
parser.add_argument("--steps", type=int, default=0, help="steps before a second reading")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
print("[probe] запуск Kit", flush=True)
simulation_app = AppLauncher(args).app

# Isaac Lab modules need a running Kit application: imported only after the launcher.
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402


def first(tensor: object) -> object:
    """The first environment's values as plain lists."""
    return tensor[0].tolist() if tensor is not None else None  # type: ignore[index, attr-defined]


def mark(stage: str) -> None:
    """Say where the probe is, at once: a silent log is indistinguishable from a hung process."""
    print(f"[probe] {stage}", flush=True)


def main() -> None:
    mark("приложение запущено, создаётся контекст симуляции")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
    )
    spawn = sim_utils.UrdfFileCfg(
        asset_path=str(Path(args.urdf).resolve()),
        usd_dir=str(Path(args.usd_dir).resolve()),
        force_usd_conversion=True,
        fix_base=True,
        merge_fixed_joints=args.merge_fixed_joints == "yes",
        joint_drive=drive,
    )
    mark(f"конвертация URDF в USD и артикуляция, merge_fixed_joints={args.merge_fixed_joints}")
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=spawn,
            actuators={
                "all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)
            },
        )
    )
    mark("сброс симуляции: PhysX строит артикуляцию")
    sim.reset()
    mark("артикуляция готова, факты записываются")
    data = robot.data
    facts = {
        "merge_fixed_joints": args.merge_fixed_joints == "yes",
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "joint_pos_limits": first(data.joint_pos_limits),
        "joint_vel_limits": first(data.joint_vel_limits),
        "joint_effort_limits": first(data.joint_effort_limits),
        "joint_stiffness": first(data.joint_stiffness),
        "joint_damping": first(data.joint_damping),
        "default_joint_pos": first(data.default_joint_pos),
        "root_pos_w": first(data.root_pos_w),
        "body_pos_w": first(data.body_pos_w),
    }
    if args.steps:
        mark(f"{args.steps} шагов физики")
        for _ in range(args.steps):
            sim.step()
            robot.update(sim.get_physics_dt())
        facts["steps"] = args.steps
        facts["root_pos_w_after"] = first(data.root_pos_w)
        facts["body_pos_w_after"] = first(data.body_pos_w)
    Path(args.out).write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))


try:
    main()
finally:
    print("[probe] закрытие Kit", flush=True)
    simulation_app.close()
