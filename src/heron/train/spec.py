"""The training environment, derived from the profile and the scenario (ADR-0044).

Nothing here imports a simulator, a learning framework or a tensor library. That
is the point: what a training environment *is* -- its rates, what the policy
emits, what it observes, how the joints are driven -- is already stated by the
contract, so building it is a derivation over data and not a configuration
somebody writes twice. It can therefore be printed by a dry run, asserted on by a
test on a machine with no GPU, and written to disk before anything is started.

The derivation is deliberately total: every refusal is a finding with a code
(ADR-0007), so a contract that cannot be trained from says so before a process is
launched rather than in the middle of an hour of GPU time.

What is derived from what:

* ``dt = 1 / state_rate_hz`` -- the physics step, so that the drive law is
  computed at the rate it is computed at in a world of execution (ADR-0045);
* ``decimation = state_rate_hz / command_rate_hz`` -- how many physics steps one
  policy tick lasts. Not an integer means the two rates of the contract cannot
  both be true in one environment (``train.rates_not_integer``);
* the action's mapping -- ``clip``, ``scale`` and the origin -- from ADR-0041,
  resolved per joint so that the unstated defaults are written down once here
  instead of being re-invented by every reader;
* the observation -- the terms of the contract, in order, each with its scale and
  its derived width (ADR-0042);
* the drive -- ``kp`` and ``kd`` of the action, which a trained policy may not be
  without: a policy learned against one stiffness is not the same policy against
  another (ADR-0019, ADR-0045);
* whether the base is fixed -- from the description, by the one rule that holds in
  every world (ADR-0018, ADR-0048). The platform does not fix a base.

The spec carries the contract as well as what was derived from it. That is not a
second statement of the same thing but a derivation recorded next to its source,
as ``contract_id`` is recorded next to the contract in a manifest (ADR-0046); a
test holds the two together.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Final, Literal
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

from ..core.contract import JOINT_ACTIONS, Contract
from ..core.enums import ActionType, ObservationType
from ..core.models import Completion, RobotProfile, SkillSpec
from ..core.urdf import parse_urdf
from ..generators.description import anchor_meshes

ADR: Final = "ADR-0044"

DESCRIPTION_FILE: Final = "description.urdf"
"""Name of the description's copy inside a run directory. The input is never
touched (ADR-0009); what is converted to USD is this copy (ADR-0048)."""

SPEC_FILE: Final = "spec.json"
"""What the training process is started with, on disk before it starts."""

RATE_TOLERANCE: Final = 1e-9
"""How far from whole a rate ratio may fall and still be called whole. Rates are
written as decimals in YAML, and 100/50 must not depend on binary floats."""

SUPPORTED_TERMS: Final = frozenset(
    {
        ObservationType.JOINT_POS,
        ObservationType.JOINT_POS_REL,
        ObservationType.JOINT_VEL,
        ObservationType.JOINT_TORQUE,
        ObservationType.TARGET_JOINT_POS,
        ObservationType.PREVIOUS_ACTION,
        ObservationType.BASE_LIN_VEL,
        ObservationType.BASE_ANG_VEL,
        ObservationType.BASE_ORIENTATION,
    }
)
"""Observation terms this backend can build. Every member of the registry is
here today; the set exists so that adding a term to the registry without teaching
the backend to produce it is a refusal and not a silently shorter vector."""


class ActionMapping(BaseModel):
    """What the policy's raw output becomes, per joint, in both worlds (ADR-0041).

    ``q* = clip(a) * scale + offset``. Resolved: an unstated scale is written out
    as ones, and the origin as the reference pose or as zeros, so that a reader
    of this document never has to know what "unstated" meant.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ActionType
    clip: float | None = None
    """The raw output's symmetric bound. ``None`` when the contract states none."""

    scale: list[float] = Field(min_length=1)
    offset: list[float] = Field(min_length=1)


class ObservationSlot(BaseModel):
    """One term of the observation vector: what it is, its scale, its width."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ObservationType
    scale: float
    size: int = Field(gt=0)


class Drive(BaseModel):
    """The law a joint is driven by: ``tau = kp * (q* - q) - kd * dq`` (ADR-0045)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kp: list[float] = Field(min_length=1)
    kd: list[float] = Field(min_length=1)


class TrainSpec(BaseModel):
    """A training environment as data. Written to disk before a run starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    backend: str
    scenario: str
    robot: str
    """The learner's identifier in the scenario."""

    profile: RobotProfile
    """The robot, whole. A run directory is read by a process that has no registry
    and no profile path: what the environment was built from travels with it."""

    task: str
    """Plugin name of the training task, and the name the artifact is stored under."""

    contract_id: str
    """Identity of the representation the environment was derived from, recorded
    next to it as a manifest records it next to a contract (ADR-0046)."""

    fix_base: bool
    """Whether the description bolts the robot to the world (ADR-0048)."""

    dt: float = Field(gt=0)
    decimation: int = Field(gt=0)
    action: ActionMapping
    observation: list[ObservationSlot] = Field(min_length=1)
    drive: Drive
    skill: SkillSpec
    completion: Completion

    @property
    def contract(self) -> Contract:
        """The representation everything here was derived from."""
        return self.profile.contract

    @property
    def joint_order(self) -> list[str]:
        return list(self.contract.joint_order)

    @property
    def dof(self) -> int:
        return self.contract.dof

    @property
    def default_pose(self) -> list[float]:
        """The reference pose the articulation starts from and observes against."""
        return list(self.contract.default_pose or [0.0] * self.dof)

    @property
    def observation_size(self) -> int:
        return sum(slot.size for slot in self.observation)

    @property
    def action_size(self) -> int:
        return len(self.action.scale)

    @property
    def policy_rate_hz(self) -> float:
        """How often the policy is asked. One tick is ``decimation`` physics steps."""
        return self.contract.command_rate_hz

    def lines(self) -> list[str]:
        """The derivation as a human reads it in a dry run."""
        contract = self.contract
        terms = ", ".join(f"{slot.type}x{slot.scale:g}[{slot.size}]" for slot in self.observation)
        return [
            f"сценарий    {self.scenario}, ученик {self.robot} ({self.profile.name})",
            f"задача      {self.task} на бэкенде {self.backend}",
            f"контракт    {self.contract_id}  revision {contract.revision}",
            f"суставы     {self.dof}: {', '.join(self.joint_order)}",
            f"частоты     физика {1 / self.dt:g} Гц, политика {self.policy_rate_hz:g} Гц, "
            f"decimation {self.decimation}",
            f"эпизод      {self.completion.horizon_s:g} с; допуск "
            f"{self.completion.tolerance:g} {contract.angle_unit}, выдержка "
            f"{self.completion.hold_s:g} с",
            f"база        {'закреплена описанием' if self.fix_base else 'свободна'}",
            f"действие    {self.action.type}, clip "
            f"{f'±{self.action.clip:g}' if self.action.clip is not None else '—'}, "
            f"ширина {self.action_size}",
            f"  scale     {', '.join(f'{v:g}' for v in self.action.scale)}",
            f"  offset    {', '.join(f'{v:g}' for v in self.action.offset)}",
            f"наблюдение  {terms}  ширина {self.observation_size}",
            f"привод      kp {', '.join(f'{v:g}' for v in self.drive.kp)}",
            f"            kd {', '.join(f'{v:g}' for v in self.drive.kd)}",
        ]


# -- the derivation ---------------------------------------------------------


def decimation_of(contract: Contract) -> int | None:
    """Physics steps per policy tick, or ``None`` when the ratio is not whole."""
    ratio = contract.state_rate_hz / contract.command_rate_hz
    rounded = round(ratio)
    if rounded < 1 or abs(ratio - rounded) > RATE_TOLERANCE:
        return None
    return int(rounded)


def derive(
    *,
    scenario: str,
    robot: str,
    backend: str,
    task: str,
    profile: RobotProfile,
    description: Path,
    skill: SkillSpec,
    completion: Completion,
) -> TrainSpec:
    """The environment this contract describes. Call only on a contract that checks out.

    ``ValueError`` if it does not. The answer a human needs is the validator's --
    ``validator.train.check_contract`` gives it with codes and hints -- and this is
    the terse guard that stops a spec from being built from a contract that never
    passed it.
    """
    contract = profile.contract
    decimation = decimation_of(contract)
    action = contract.action
    if decimation is None:
        raise ValueError("train.rates_not_integer")
    if action.type not in JOINT_ACTIONS:
        raise ValueError("train.action_unsupported")
    if action.kp is None or action.kd is None:
        raise ValueError("train.gains_missing")
    unsupported = [t.type for t in contract.observation if t.type not in SUPPORTED_TERMS]
    if unsupported:
        raise ValueError("train.observation_term_unsupported")

    return TrainSpec(
        backend=backend,
        scenario=scenario,
        robot=robot,
        profile=profile,
        task=task,
        contract_id=contract.contract_id,
        fix_base=parse_urdf(description).fixed_to_world,
        dt=1.0 / contract.state_rate_hz,
        decimation=decimation,
        action=ActionMapping(
            type=action.type,
            clip=None if action.clip is None else abs(action.clip[1]),
            scale=list(contract.action_scale()),
            offset=list(contract.origin()),
        ),
        observation=[
            ObservationSlot(
                type=term.type,
                scale=term.scale,
                # A term of the registry always has a width here: a term whose
                # width the contract cannot derive is not in SUPPORTED_TERMS.
                size=contract.term_size(term.type) or 0,
            )
            for term in contract.observation
        ],
        drive=Drive(kp=list(action.kp), kd=list(action.kd)),
        skill=skill,
        completion=completion,
    )


def description_text(profile: RobotProfile, description: Path) -> str:
    """The input description, copied with its meshes anchored (ADR-0017, ADR-0048).

    The copy is what a converter reads, and it reads it from a run directory that
    is nowhere near the input, so a relative mesh reference would resolve against
    nothing. Nothing else is changed: a control block belongs to a world that
    executes, and a training environment is not one.
    """
    parsed = parse_urdf(description)
    tree = copy.deepcopy(parsed.tree)
    anchor_meshes(tree, description, profile.description.mesh_dir)
    for existing in tree.findall("ros2_control"):
        tree.remove(existing)
    return ET.tostring(tree, encoding="unicode") + "\n"
