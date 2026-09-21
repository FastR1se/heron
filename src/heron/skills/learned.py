"""``LearnedSkill`` -- the one implementation of a skill that is a trained policy (ADR-0047).

Policies of different robots are data of one implementation: a network, the
contract it was trained under, and when the skill it implements is done. The
declaration comes from the manifest, under the name the profile binds it to, so a
mission cannot tell this skill from ``move_joints`` (invariant 4).

Everything decidable is decided in ``load``, before the executor checks the link
and before anything connects: the artifact against its name, the contract against
the robot's, what this implementation can execute, and the network's signature. A
mismatch is a report with codes, never a robot that goes mad on the first tick.

A tick is one mapping and nothing more (ADR-0041, ADR-0042, ADR-0043)::

    observation from state, target and the previous output
    a  = network(observation)
    u  = clip(a)                      remembered as the next previous output
    q* = u * scale + origin           sent as joint_position
    done when |q - target| <= tolerance has held for hold_s, failed after horizon_s

No projection onto the envelope (ADR-0032 does not apply): a policy does not
command a pose it measured, and the set of targets it can command was checked
against the envelope before the run (``contract.action_outside_limits``).

``onnxruntime`` and ``numpy`` are imported when a network is opened, never with this
module: the core installs without them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Final

from ..core.contract import Contract
from ..core.enums import ActionType, ObservationType, Status
from ..core.models import PolicyManifest, RobotProfile
from ..core.types import Command, RobotState
from ..store.policies import PolicyArtifact
from ..validator.policy import check_policy_against_profile
from ..validator.report import Report, error
from .base import BaseSkill, goal_target

ONNX_GROUP: Final = "onnx"
"""Optional dependency group that installs the runtime (``pyproject.toml``)."""

ONNX_FLOAT: Final = "tensor(float)"
BATCH: Final = 1
PROVIDERS: Final = ("CPUExecutionProvider",)
THREADS: Final = 1
"""One thread: the network is 64x64, and the spread of a tick matters more than throughput."""


class PolicyRejectedError(RuntimeError):
    """A policy may not drive this robot, and the report says why."""

    def __init__(self, report: Report) -> None:
        self.report = report
        super().__init__("\n".join(str(item) for item in report.errors))


class PolicyRuntimeError(RuntimeError):
    """The network cannot be run on this machine."""


class ObservationError(ValueError):
    """The state read this tick does not carry what the contract observes."""


# -- the mapping, as functions of data ------------------------------------------


def observation_vector(
    contract: Contract,
    state: RobotState,
    target: Sequence[float],
    previous: Sequence[float],
) -> list[float]:
    """The observation the contract describes, term by term, each times its scale (ADR-0042)."""
    dof = contract.dof
    position = _joints(state.joint_position, "joint_position", dof)
    values: list[float] = []
    for term in contract.observation:
        kind = term.type
        part: Sequence[float]
        if kind is ObservationType.JOINT_POS:
            part = position
        elif kind is ObservationType.JOINT_POS_REL:
            reference = contract.default_pose or (0.0,) * dof
            part = [q - d for q, d in zip(position, reference, strict=True)]
        elif kind is ObservationType.JOINT_VEL:
            part = _joints(state.joint_velocity, "joint_velocity", dof)
        elif kind is ObservationType.JOINT_TORQUE:
            part = _joints(state.joint_torque, "joint_torque", dof)
        elif kind is ObservationType.TARGET_JOINT_POS:
            part = target
        elif kind is ObservationType.PREVIOUS_ACTION:
            part = previous
        elif kind is ObservationType.BASE_ORIENTATION:
            part = _measured(state.base_orientation, kind)
        elif kind is ObservationType.PROJECTED_GRAVITY:
            part = projected_gravity(_measured(state.base_orientation, kind))
        elif kind is ObservationType.BASE_ANG_VEL:
            part = _measured(state.base_angular_velocity, kind)
        else:
            raise ObservationError(f"{kind}: в состоянии робота этой величины нет")
        values.extend(float(value) * term.scale for value in part)
    return values


def map_action(
    contract: Contract, raw: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The clipped output and the command it maps to: ``clip(a) * scale + origin`` (ADR-0041)."""
    clip = contract.action.clip
    if clip is None:
        clipped = tuple(float(value) for value in raw)
    else:
        low, high = clip
        clipped = tuple(min(max(float(value), low), high) for value in raw)
    targets = tuple(
        u * s + o
        for u, s, o in zip(clipped, contract.action_scale(), contract.origin(), strict=True)
    )
    return clipped, targets


def projected_gravity(orientation: Sequence[float]) -> tuple[float, float, float]:
    """``R(q)^T (0, 0, -1)``: which way is down, seen from the base (ADR-0076).

    ``q`` is ``w, x, y, z`` and need not be exactly unit: a measured quaternion is
    normalized first, so a sensor's rounding does not tilt the answer. The third
    row of ``R(q)``, negated -- the same quantity Isaac Lab's ``projected_gravity``
    computes with ``quat_apply_inverse``.
    """
    w, x, y, z = (float(value) for value in orientation)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ObservationError("base_orientation: нулевой кватернион — ориентации нет")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    )


def _measured(values: Sequence[float] | None, kind: ObservationType) -> Sequence[float]:
    if values is None:
        raise ObservationError(f"{kind}: адаптер не сообщает эту величину корпуса")
    return values


def _joints(values: Sequence[float], field: str, dof: int) -> Sequence[float]:
    if len(values) != dof:
        raise ObservationError(f"{field}: {len(values)} значений при {dof} суставах контракта")
    return values


# -- the network -----------------------------------------------------------------


def open_network(artifact: PolicyArtifact) -> Any:
    """An inference session over the artifact's graph, on CPU, one thread."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise PolicyRuntimeError(
            f"onnxruntime не установлен: исполнение политики требует необязательной группы "
            f"{ONNX_GROUP!r} (pip install -e '.[{ONNX_GROUP}]')"
        ) from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = THREADS
    try:
        return onnxruntime.InferenceSession(
            str(artifact.onnx_path), sess_options=options, providers=list(PROVIDERS)
        )
    except Exception as exc:  # the runtime's own exception hierarchy, not Python's
        raise PolicyRejectedError(
            Report(
                [
                    error(
                        "policy.onnx_signature_mismatch",
                        f"policy:{artifact.ref}",
                        f"граф не загружается: {exc}",
                        adr="ADR-0047",
                    )
                ]
            )
        ) from exc


def check_signature(session: Any, manifest: PolicyManifest, *, where: str) -> Report:
    """One input ``[1, observation_size]`` float32, one output ``[1, action_size]`` (ADR-0047)."""
    contract = manifest.contract
    problems: list[str] = []
    detail: dict[str, Any] = {}
    for side, nodes, name, width in (
        ("вход", session.get_inputs(), manifest.onnx.input, contract.observation_size),
        ("выход", session.get_outputs(), manifest.onnx.output, contract.action_size),
    ):
        wanted = [BATCH, width]
        found = [
            {"name": node.name, "shape": list(node.shape), "type": node.type} for node in nodes
        ]
        detail[side] = {
            "expected": {"name": name, "shape": wanted, "type": ONNX_FLOAT},
            "found": found,
        }
        if len(nodes) != 1:
            problems.append(f"{side}ов {len(nodes)}, ожидался один")
            continue
        node = found[0]
        if node["name"] != name:
            problems.append(f"{side} назван {node['name']!r}, манифест называет {name!r}")
        if node["shape"] != wanted:
            problems.append(f"{side} формы {node['shape']}, контракт выводит {wanted}")
        if node["type"] != ONNX_FLOAT:
            problems.append(f"{side} типа {node['type']}, ожидался {ONNX_FLOAT}")
    report = Report()
    if problems:
        report.add(
            error(
                "policy.onnx_signature_mismatch",
                where,
                "; ".join(problems),
                hint="ширины выводятся из контракта; оси сети не динамические",
                adr="ADR-0047",
                detail=detail,
            )
        )
    return report


def check_network(artifact: PolicyArtifact) -> Report:
    """Open the artifact's graph and check its signature. Needs the ``onnx`` group."""
    try:
        session = open_network(artifact)
    except PolicyRejectedError as exc:
        return exc.report
    return check_signature(session, artifact.manifest, where=f"policy:{artifact.ref}")


# -- the skill -------------------------------------------------------------------


class LearnedSkill(BaseSkill):
    """A trained policy behind the ``Skill`` protocol."""

    def __init__(self, manifest: PolicyManifest, session: Any, *, name: str) -> None:
        super().__init__()
        self.spec = manifest.skill_spec(name)
        self.manifest = manifest
        self._session = session
        self.target: tuple[float, ...] = ()
        self.previous: tuple[float, ...] = ()
        self.started_at: float | None = None
        self.within_since: float | None = None
        self.slowest_s = 0.0
        """Longest inference so far, wall time (ADR-0047, item 7)."""

    @classmethod
    def load(cls, artifact: PolicyArtifact, profile: RobotProfile, *, name: str) -> LearnedSkill:
        """Check the artifact against the robot, open the network, check its signature.

        Raises ``PolicyRejectedError`` with every finding, or ``PolicyRuntimeError``
        when the runtime is not installed.
        """
        report = check_policy_against_profile(artifact.manifest, profile, artifact=artifact)
        if not report.ok:
            raise PolicyRejectedError(report)
        session = open_network(artifact)
        report.extend(
            check_signature(session, artifact.manifest, where=f"policy:{artifact.ref}").findings
        )
        if not report.ok:
            raise PolicyRejectedError(report)
        return cls(artifact.manifest, session, name=name)

    @property
    def contract(self) -> Contract:
        return self.manifest.contract

    def on_start(self) -> None:
        self.target = goal_target(self.spec.name, self.goal, self.context.profile)
        # Zeros on the first tick, as after a reset in training (ADR-0042).
        self.previous = (0.0,) * (self.contract.action_size or 0)
        self.started_at = None
        self.within_since = None
        self.slowest_s = 0.0

    def on_tick(self, state: RobotState) -> Status:
        if self.started_at is None:
            self.started_at = state.timestamp
        try:
            observation = observation_vector(self.contract, state, self.target, self.previous)
        except ObservationError as exc:
            self._finish("состояние не годится для наблюдения политики", reason=str(exc))
            return Status.FAILURE

        began = time.perf_counter()
        raw = self._infer(observation)
        self.slowest_s = max(self.slowest_s, time.perf_counter() - began)

        clipped, values = map_action(self.contract, raw)
        self.context.send(Command(type=ActionType.JOINT_POSITION, values=values))
        self.previous = clipped

        completion = self.manifest.completion
        now = state.timestamp
        deviation = max(
            abs(q - t) for q, t in zip(state.joint_position, self.target, strict=True)
        )
        if deviation <= completion.tolerance:
            if self.within_since is None:
                self.within_since = now
            if now - self.within_since >= completion.hold_s:
                self._finish("цель достигнута", deviation=f"{deviation:.4g}")
                return Status.SUCCESS
        else:
            self.within_since = None
        if now - self.started_at >= completion.horizon_s:
            self._finish(
                "цель не достигнута за горизонт обучения",
                deviation=f"{deviation:.4g}",
                horizon_s=f"{completion.horizon_s:g}",
            )
            return Status.FAILURE
        return Status.RUNNING

    def _infer(self, observation: Sequence[float]) -> list[float]:
        import numpy

        onnx = self.manifest.onnx
        batch = numpy.asarray([observation], dtype=numpy.float32)
        (output,) = self._session.run([onnx.output], {onnx.input: batch})
        return [float(value) for value in output[0]]

    def _finish(self, message: str, **fields: Any) -> None:
        self.context.log(
            message,
            skill=self.spec.name,
            policy=str(self.manifest.ref()),
            inference_max_ms=f"{self.slowest_s * 1e3:.3f}",
            **fields,
        )
