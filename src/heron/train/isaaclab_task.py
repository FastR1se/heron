"""What a training task contributes to this backend's environment (ADR-0044).

The second layer of the sixth extension point, in the vocabulary of one training
environment -- as a Gazebo twin's contribution is phrased in elements of a
description and nodes of a launch (ADR-0024). A task package implements
:class:`IsaacLabTask`; the platform hands it a context and takes back
configuration objects.

What the platform gives:

* the robot's articulation, resolved over the joints of the contract and in the
  contract's order, so that a reward over joints reads them in the order the
  policy commands them;
* the name of the goal's command term, because the goal is the platform's: it is
  what the observation carries (``target_joint_pos``) and what completion is
  measured against (ADR-0042, ADR-0043);
* the contract and the profile, so a reward can be written in the robot's own
  quantities.

What it does not give, and has no member to give: the scene's robot, the drive,
the rates, the action manager and the observation manager. Those come from the
contract, and a task that could change them would be able to train a policy the
contract does not describe -- the exact failure invariant 6 exists against.

Nothing here imports the simulator. The configuration objects a task returns are
built inside the training process, by the task, when it is asked; typed here as
``object`` because the platform never looks inside them. A task package must
therefore import its framework inside these methods, the way an adapter imports
its transport inside ``connect`` -- otherwise the platform could not so much as
read what skill the task declares.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..core.contract import Contract
from ..core.models import RobotProfile
from ..core.protocols import TrainTask
from ..validator.train import check_goal_range

__all__ = ["BACKEND", "IsaacLabTask", "JointBox", "TaskContext", "check_goal_range"]
"""``check_goal_range`` is re-exported rather than kept where it is computed: a
box is a value of this vocabulary, and a task package that declares one must be
able to check it without reaching into anything private (ADR-0021)."""

BACKEND = "isaaclab"
"""What a task of this layer declares as its ``backend``."""


@dataclass(frozen=True, slots=True)
class JointBox:
    """A rectangle in joint space, ordered like ``contract.joint_order``.

    The goal of an episode is drawn from it uniformly. Data, not a sampler: what
    a goal is drawn from is the task's to say, how it is drawn and held for an
    episode is the platform's (ADR-0044).
    """

    min: Sequence[float]
    max: Sequence[float]


@dataclass(frozen=True, slots=True)
class TaskContext:
    """What a task is told about the robot it is written for."""

    profile: RobotProfile
    contract: Contract
    robot: Any
    """``SceneEntityCfg`` over the contract's joints, ``preserve_order=True``."""

    asset: str
    """Name of the articulation in the scene, for a term that resolves its own."""

    command: str
    """Name of the goal's command term, for a reward or a termination that reads it."""

    episode_length_s: float
    """Length of one episode, from the task's own ``completion.horizon_s``."""


@runtime_checkable
class IsaacLabTask(TrainTask, Protocol):
    """A task trained in this backend."""

    def goal_range(self, ctx: TaskContext) -> JointBox:
        """Where an episode's goal is drawn from. Checked against the envelope and
        against the set of targets the action can reach (``train.goal_*``)."""

    def rewards(self, ctx: TaskContext) -> object:
        """The reward terms, as a configuration object of the framework."""

    def terminations(self, ctx: TaskContext) -> object:
        """When an episode ends. A task that never times out never ends."""

    def events(self, ctx: TaskContext) -> object:
        """Resets and randomization."""

    def agent(self, ctx: TaskContext) -> Mapping[str, Any] | None:
        """Overrides for the learning agent, or ``None`` for the backend's own."""
