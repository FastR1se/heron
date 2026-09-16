"""The process that actually trains. Started as ``python -m`` with a run directory.

This is the only module of the platform that imports a simulator, and it is a
script rather than a library because the framework it drives demands one: the
application has to be launched before anything of it may be imported, it takes
over the process it is launched in, and it decides when that process ends. A
library that did this would take a command's process hostage; a process does not
have to be anything else.

It reads what was derived for it and builds nothing of its own. Everything the
environment is -- rates, action mapping, observation terms, drive gains, whether
the base is fixed -- arrives in ``spec.json``, computed once by the platform
(``train.spec``) and written to disk before this process started. What this
module adds is the translation into the framework's objects, plus the two things
only the framework can do: converting the description to USD and running the
algorithm.

The goal is the platform's, not the task's (ADR-0044): a command term draws one
joint pose per episode from the box the task declares, the observation carries
it as ``target_joint_pos``, and completion is measured against it. The task
supplies rewards, terminations, events and that box.

Nothing here is covered by the test suite, and that is stated rather than hidden:
a test that could run this would need a GPU and a simulator. What *is* covered is
everything it reads -- the derivation, the refusals, the artifact -- so that a
failure in this file is a failure of translation and not of meaning.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import field
from pathlib import Path
from typing import Any

# -- the application, before anything of the framework is imported ---------

parser = argparse.ArgumentParser(description="Обучение по каталогу прогона.")
parser.add_argument("run", type=Path, help="Каталог прогона со spec.json.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
_args = parser.parse_args()
_args.headless = True
"""Headless always: cameras in observations are forbidden and a window is a cost
nobody asked for (ADR-0005)."""

_launcher = AppLauncher(_args)
simulation_app = _launcher.app

# -- now the framework -----------------------------------------------------

import gymnasium  # noqa: E402, F401
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    mdp,
)
from isaaclab.managers import (  # noqa: E402
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ..core.branding import data_home  # noqa: E402
from ..core.contract import canonical_dumps, digest  # noqa: E402
from ..core.enums import ObservationType  # noqa: E402
from .isaaclab import CONDITIONS_FILE, EXPORT_FILE, ONNX_FILE, TrainRun, read_run  # noqa: E402
from .isaaclab_task import JointBox, TaskContext, check_goal_range  # noqa: E402
from .spec import DESCRIPTION_FILE, TrainSpec  # noqa: E402
from .tasks import find_task  # noqa: E402

ASSET = "robot"
"""Name of the articulation in the scene. Terms of the task resolve it by this."""

GOAL = "target"
"""Name of the goal's command term. What ``target_joint_pos`` reads."""

ACTION = "joints"
"""Name of the one action term. What ``previous_action`` reads."""

ONNX_INPUT = "obs"
ONNX_OUTPUT = "actions"
ONNX_OPSET = 18
"""What the framework's exporter writes, recorded so the manifest states it (факт 0.17)."""

ENV_SPACING = 2.5
NUM_STEPS_PER_ENV = 24

TERM_FUNCS = {
    ObservationType.JOINT_POS: "joint_pos",
    ObservationType.JOINT_POS_REL: "joint_pos_rel",
    ObservationType.JOINT_VEL: "joint_vel",
    ObservationType.JOINT_TORQUE: "joint_effort",
    ObservationType.BASE_LIN_VEL: "base_lin_vel",
    ObservationType.BASE_ANG_VEL: "base_ang_vel",
    ObservationType.BASE_ORIENTATION: "root_quat_w",
}
"""Contract term -> the framework's observation function. The two per-joint terms
the platform owns -- the goal and the previous action -- are not here: they read a
command term and an action term by name."""


# -- the goal (ADR-0043) ----------------------------------------------------


class JointPoseCommand(CommandTerm):
    """One joint pose per episode, drawn uniformly from the task's box.

    Belongs to the platform rather than to the task, because it is the same
    quantity in three places: what the observation carries, what a reward is
    written against, and what ``LearnedSkill`` compares the measured pose with
    when it decides the skill is done.
    """

    cfg: JointPoseCommandCfg

    def __init__(self, cfg: JointPoseCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.joint_ids, _ = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        self.low = torch.tensor(cfg.range_min, device=self.device)
        self.high = torch.tensor(cfg.range_max, device=self.device)
        self.goal = torch.zeros(self.num_envs, len(cfg.joint_names), device=self.device)
        self.metrics["max_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal

    def _resample_command(self, env_ids: Any) -> None:
        draw = torch.rand(len(env_ids), self.goal.shape[1], device=self.device)
        self.goal[env_ids] = self.low + draw * (self.high - self.low)

    def _update_command(self) -> None:
        return None

    def _update_metrics(self) -> None:
        measured = self.robot.data.joint_pos[:, self.joint_ids]
        self.metrics["max_error"] = torch.max(torch.abs(measured - self.goal), dim=1).values


@configclass
class JointPoseCommandCfg(CommandTermCfg):
    """Where the goal comes from, as data."""

    class_type: type = JointPoseCommand
    asset_name: str = ASSET
    # Always supplied by the platform. The empty defaults are per-instance
    # factories, as the framework's configclass makes of every mutable default;
    # the linter cannot see that through the decorator.
    joint_names: list[str] = field(default_factory=list)
    range_min: list[float] = field(default_factory=list)
    range_max: list[float] = field(default_factory=list)


@configclass
class TrainingSceneCfg(InteractiveSceneCfg):
    """Ground, light, robot. The scene is the platform's; the task never sees it."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot: ArticulationCfg | None = None


# -- the description, converted (ADR-0048) ----------------------------------


def converter_settings(spec: TrainSpec) -> dict[str, Any]:
    """Everything about the conversion that could change the result.

    Part of the cache key, so a changed setting is a different USD and not the
    same one silently reused.
    """
    return {
        "fix_base": spec.fix_base,
        "merge_fixed_joints": False,
        "link_density": 0.0,
        "stiffness": 0.0,
        "damping": 0.0,
        "collider_type": "convex_hull",
        "self_collision": False,
    }


def usd_key(description: str, settings: dict[str, Any], version: str) -> str:
    """``sha256(описание + настройки + версия)[:16]`` -- the name of a cache entry."""
    return digest(canonical_dumps({"urdf": description, "convert": settings, "isaacsim": version}))


def to_usd(run: Path, spec: TrainSpec) -> tuple[Path, str]:
    """The description's copy as USD, from the cache or converted into it."""
    description = run / DESCRIPTION_FILE
    settings = converter_settings(spec)
    version = _version("isaacsim")
    key = usd_key(description.read_text(encoding="utf-8"), settings, version)
    cache = data_home() / "cache" / "usd" / key
    target = cache / f"{spec.profile.name}.usd"
    if target.is_file():
        return target, key

    cache.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(description),
        usd_dir=str(cache),
        usd_file_name=target.name,
        fix_base=spec.fix_base,
        merge_fixed_joints=settings["merge_fixed_joints"],
        link_density=settings["link_density"],
        collider_type=settings["collider_type"],
        self_collision=settings["self_collision"],
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=settings["stiffness"], damping=settings["damping"]
            ),
        ),
    )
    UrdfConverter(cfg)
    return Path(cfg.usd_path), key


# -- the environment, translated --------------------------------------------


def articulation(spec: TrainSpec, usd: Path) -> ArticulationCfg:
    """The robot: the reference pose it starts from, and one explicit PD drive.

    Explicit on purpose (ADR-0045): the drive law then runs in the actuator model
    at the physics rate, which is the rate of the contract's state, and it is the
    same law a trajectory controller computes in a world of execution. An implicit
    actuator would hand the law to the solver's continuous integration -- steadier,
    and a different law than the one the robot will meet.
    """
    joints = spec.joint_order
    pose = spec.default_pose
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(zip(joints, pose, strict=True)),
            joint_vel=dict.fromkeys(joints, 0.0),
        ),
        actuators={
            "drive": IdealPDActuatorCfg(
                joint_names_expr=list(joints),
                stiffness=dict(zip(joints, spec.drive.kp, strict=True)),
                damping=dict(zip(joints, spec.drive.kd, strict=True)),
                # Left unset: the effort limit is the description's, in both worlds.
            )
        },
    )


def action_term(spec: TrainSpec) -> ActionTermCfg:
    """``q* = clip(a) * scale + offset``, per joint, in the contract's order (ADR-0041).

    The clip is not here: it is applied to the raw output before the environment
    sees it, so that what the observation remembers as the previous action is the
    clipped output and not something the policy never sent (ADR-0042).
    """
    joints = spec.joint_order
    return mdp.JointPositionActionCfg(
        asset_name=ASSET,
        joint_names=list(joints),
        scale=dict(zip(joints, spec.action.scale, strict=True)),
        offset=dict(zip(joints, spec.action.offset, strict=True)),
        use_default_offset=False,
        preserve_order=True,
    )


def observation_group(spec: TrainSpec) -> ObservationGroupCfg:
    """The contract's terms, in the contract's order, each times its scale.

    No noise and no clip: ADR-0042 keeps both out of the contract, and a term the
    contract does not describe is a term the executing side cannot reproduce.
    """
    group = ObservationGroupCfg()
    group.concatenate_terms = True
    group.enable_corruption = False
    entity = SceneEntityCfg(ASSET, joint_names=spec.joint_order, preserve_order=True)
    for index, slot in enumerate(spec.observation):
        if slot.type is ObservationType.TARGET_JOINT_POS:
            term = ObservationTermCfg(
                func=mdp.generated_commands, params={"command_name": GOAL}, scale=slot.scale
            )
        elif slot.type is ObservationType.PREVIOUS_ACTION:
            term = ObservationTermCfg(
                func=mdp.last_action, params={"action_name": ACTION}, scale=slot.scale
            )
        else:
            func = getattr(mdp, TERM_FUNCS[slot.type])
            params = {"asset_cfg": entity} if slot.type.startswith("joint") else {}
            term = ObservationTermCfg(func=func, params=params, scale=slot.scale)
        setattr(group, f"term{index}_{slot.type}", term)
    return group


def environment(spec: TrainSpec, task: Any, usd: Path, envs: int) -> ManagerBasedRLEnvCfg:
    """The whole environment: the platform's half derived, the task's half asked for."""
    profile = spec.profile
    context = TaskContext(
        profile=profile,
        contract=spec.contract,
        robot=SceneEntityCfg(ASSET, joint_names=spec.joint_order, preserve_order=True),
        asset=ASSET,
        command=GOAL,
        episode_length_s=spec.completion.horizon_s,
    )
    box = task.goal_range(context)
    report = check_goal_range(box.min, box.max, profile, where=f"task:{spec.task}.goal_range")
    if not report.ok:
        raise SystemExit("\n".join(str(item) for item in report.errors))

    scene = TrainingSceneCfg(num_envs=envs, env_spacing=ENV_SPACING)
    scene.robot = articulation(spec, usd)

    cfg = ManagerBasedRLEnvCfg()
    cfg.scene = scene
    cfg.decimation = spec.decimation
    cfg.episode_length_s = spec.completion.horizon_s
    cfg.sim.dt = spec.dt
    cfg.sim.render_interval = spec.decimation
    cfg.actions = {ACTION: action_term(spec)}
    cfg.observations = {"policy": observation_group(spec)}
    cfg.commands = {GOAL: _goal_term(spec, box)}
    cfg.rewards = task.rewards(context)
    cfg.terminations = task.terminations(context)
    cfg.events = task.events(context)
    cfg.curriculum = {}
    return cfg


def _goal_term(spec: TrainSpec, box: JointBox) -> JointPoseCommandCfg:
    horizon = spec.completion.horizon_s
    return JointPoseCommandCfg(
        resampling_time_range=(horizon, horizon),
        joint_names=list(spec.joint_order),
        range_min=list(box.min),
        range_max=list(box.max),
    )


def agent_config(spec: TrainSpec, record: TrainRun, task: Any, log_dir: Path) -> dict[str, Any]:
    """The learning agent. The backend's default, overridden by whatever the task states."""
    conditions = record.conditions
    agent: dict[str, Any] = {
        "seed": conditions.seed,
        "device": conditions.device,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "max_iterations": conditions.iterations,
        "save_interval": max(1, conditions.iterations // 10),
        "experiment_name": spec.task,
        "run_name": "",
        "logger": "tensorboard",
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        # The one clip of ADR-0041, applied to the raw output before the
        # environment sees it -- the same place the executing side applies it.
        "clip_actions": spec.action.clip,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.001,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
    context = TaskContext(
        profile=spec.profile,
        contract=spec.contract,
        robot=None,
        asset=ASSET,
        command=GOAL,
        episode_length_s=spec.completion.horizon_s,
    )
    overrides = task.agent(context)
    if overrides:
        agent.update(dict(overrides))
    agent["log_dir"] = str(log_dir)
    return agent


# -- the run ----------------------------------------------------------------


def main(run: Path) -> int:
    started = time.monotonic()
    record = read_run(run)
    spec = record.spec
    task = find_task(spec.task, spec.backend)

    usd, key = to_usd(run, spec)
    converted = time.monotonic()

    cfg = environment(spec, task, usd, record.conditions.envs)
    cfg.seed = record.conditions.seed
    cfg.sim.device = record.conditions.device

    log_dir = run / "logs"
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg), clip_actions=spec.action.clip)
    agent = agent_config(spec, record, task, log_dir)
    runner = OnPolicyRunner(env, agent, log_dir=str(log_dir), device=record.conditions.device)
    runner.learn(
        num_learning_iterations=record.conditions.iterations, init_at_random_ep_len=True
    )
    learned = time.monotonic()

    policy = runner.alg.policy
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    export_policy_as_onnx(policy, path=str(run), normalizer=normalizer, filename=ONNX_FILE)

    (run / EXPORT_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "file": ONNX_FILE,
                "input": ONNX_INPUT,
                "output": ONNX_OUTPUT,
                "observation_size": spec.observation_size,
                "action_size": spec.action_size,
                "opset": ONNX_OPSET,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run / CONDITIONS_FILE).write_text(
        json.dumps(
            {
                "usd_key": key,
                "python": platform.python_version(),
                "machine": platform.platform(),
                "isaacsim": _version("isaacsim"),
                "isaaclab": _version("isaaclab"),
                "rsl_rl": _version("rsl_rl"),
                "torch": _version("torch"),
                "envs": record.conditions.envs,
                "iterations": record.conditions.iterations,
                "convert_s": round(converted - started, 1),
                "learn_s": round(learned - converted, 1),
                "total_s": round(time.monotonic() - started, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env.close()
    return 0


def _version(module: str) -> str:
    from importlib.metadata import PackageNotFoundError, packages_distributions, version

    for distribution in packages_distributions().get(module, []):
        try:
            return version(distribution)
        except PackageNotFoundError:  # pragma: no cover
            continue
    return "не определяется"


if __name__ == "__main__":
    code = main(_args.run)
    simulation_app.close()
    sys.exit(code)
