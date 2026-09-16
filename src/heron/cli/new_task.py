"""Generating a stub for a new training task (ADR-0044, item 6).

The tenth criterion of phase 6 is that somebody trains *their* robot for *their*
task without editing the core. A stub is how that claim stops being a claim: what
it writes is a package, an entry point and a scenario, and none of it is a file of
the platform.

Like the robot stub, it is deliberately minimal rather than plausible. The reward
is one term and it is obviously a placeholder, because a generated reward that
looks finished is a reward nobody reads before training against it for six hours.

What the stub does not generate is anything the contract already states: no
action, no observation, no rates, no gains. A task has no member through which
they could be set, and the template says so where a reader will look for them.
"""

from __future__ import annotations

from pathlib import Path

from ..core.branding import APP_NAME, ENTRY_POINT_GROUPS
from ..core.models import NAME_RE

TASK_TEMPLATE = '''\
"""Training task {name}: what is learned, and when it is done.

Generated stub. Fill in the reward -- the rest is already decided elsewhere:

* the action, the observation, the rates and the drive come from the robot's
  contract, and this file has no way to change them (ADR-0044);
* the goal is a joint pose drawn from ``goal_range``; the platform draws it,
  carries it in the observation and measures completion against it (ADR-0043).

The framework is imported inside the methods, never at module level: the platform
reads ``backend``, ``spec`` and ``completion`` on a machine that has no simulator,
and an import at the top would make that impossible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from {package}.core.capabilities import Capability
from {package}.core.enums import ControlMode
from {package}.core.models import Completion, SkillSpec
from {package}.train.isaaclab_task import BACKEND, JointBox, TaskContext


class {klass}:
    """Reach a joint pose. TODO: say what this task actually is."""

    backend = BACKEND

    spec = SkillSpec(
        name="{name}",
        # TODO: what the robot must be able to do for this skill to make sense.
        required_capabilities=frozenset({{Capability("arm.joint_position")}}),
        required_control_mode=ControlMode.STREAMING,
    )

    completion = Completion(
        tolerance=0.05,   # TODO: in the contract's angle_unit
        hold_s=0.2,
        horizon_s=4.0,    # length of one episode: past it the policy is not trained
    )

    def goal_range(self, ctx: TaskContext) -> JointBox:
        """Where an episode's goal is drawn from, per joint, in joint_order.

        Checked against the profile's envelope and against the set of targets the
        action can command: a goal outside either is one the policy cannot reach.
        """
        pose = ctx.contract.default_pose or [0.0] * ctx.contract.dof
        spread = 0.3   # TODO: the real working range of this task
        return JointBox(
            min=[value - spread for value in pose],
            max=[value + spread for value in pose],
        )

    def rewards(self, ctx: TaskContext) -> object:
        """TODO: the reward. This one only asks the robot to be near the goal."""
        from isaaclab.envs import mdp
        from isaaclab.managers import RewardTermCfg
        from isaaclab.utils import configclass

        @configclass
        class Rewards:
            joint_velocity = RewardTermCfg(
                func=mdp.joint_vel_l2, weight=-1e-4, params={{"asset_cfg": ctx.robot}}
            )
            action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-4)

        return Rewards()

    def terminations(self, ctx: TaskContext) -> object:
        """When an episode ends. Without a time out it never does."""
        from isaaclab.envs import mdp
        from isaaclab.managers import TerminationTermCfg
        from isaaclab.utils import configclass

        @configclass
        class Terminations:
            time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

        return Terminations()

    def events(self, ctx: TaskContext) -> object:
        """Resets and randomization. The reset pose must cover the pose the robot
        starts from in the world it will be executed in."""
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg
        from isaaclab.utils import configclass

        @configclass
        class Events:
            reset_joints = EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={{"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)}},
            )

        return Events()

    def agent(self, ctx: TaskContext) -> Mapping[str, Any] | None:
        """Overrides for the learning agent, or None for the backend's own."""
        return None
'''

PYPROJECT_TEMPLATE = """\
# Package of the training task {name}. Installed with `pip install -e .`, found
# through the entry point below -- the core never imports it by path.
[project]
name = "{name}"
version = "0.1.0"
description = "Training task {name}"
requires-python = ">=3.11"
dependencies = []

# The sixth extension point. The group name is the platform's; the task's name is
# what a scenario writes in the `task` field of its learner.
[project.entry-points."{group}"]
{name} = "{name}.task:{klass}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
"""

SCENARIO_TEMPLATE = """\
# Сцена обучения задачи {name}. Владелец физики — бэкенд обучения (ADR-0002);
# учится ровно один участник, и он объявляет, чему учится (ADR-0044).
schema_version: 1
name: {name}_train

world:
  backend: {backend}

robots:
  - id: learner
    profile: TODO   # имя профиля робота из реестра
    role: learner
    task: {name}
"""


def write_stub(name: str, into: Path, *, backend: str = "isaaclab") -> tuple[Path, ...]:
    """Write a task package and the scenario that trains it. Nothing is overwritten."""
    if not NAME_RE.match(name):
        raise ValueError(f"{name!r}: имя задачи — нижний регистр, [a-z][a-z0-9_]*")

    klass = "".join(part.capitalize() for part in name.split("_")) + "Task"
    root = into / name
    package = root / "src" / name
    files = {
        root / "pyproject.toml": PYPROJECT_TEMPLATE.format(
            name=name, klass=klass, group=ENTRY_POINT_GROUPS["tasks"]
        ),
        package / "__init__.py": f'"""Training task {name}."""\n',
        package / "task.py": TASK_TEMPLATE.format(name=name, klass=klass, package=APP_NAME),
        root / f"{name}.yaml": SCENARIO_TEMPLATE.format(name=name, backend=backend),
    }
    existing = [path for path in files if path.exists()]
    if existing:
        raise FileExistsError(f"уже есть: {', '.join(str(p) for p in existing)}")

    package.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    return tuple(files)
