"""Tiny policies, built here rather than trained (ADR-0047).

A policy is a file, so a test that needs one has to write one. The network below is
a single ``Gemm``: it reads the observation the contract describes and answers with
an action, which is enough to exercise everything the platform does around a policy
-- the artifact, the manifest, the checks, the mapping and the tick.

``reach_weights`` builds the one policy whose behaviour a test can predict: it
commands the target it is shown. That is not a trained policy and does not pretend
to be one; it is the network for which "did the skill reach the goal" has an answer
that does not depend on training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from heron.core.capabilities import Capability
from heron.core.contract import Contract
from heron.core.enums import ObservationType
from heron.core.models import Completion, OnnxSignature, PolicyManifest
from heron.store.policies import PolicyArtifact, PolicyStore, file_sha256

INPUT = "obs"
OUTPUT = "actions"
OPSET = 18
IR_VERSION = 10
"""Pinned: a newer one than the installed runtime knows is refused when it loads."""

COMPLETION = Completion(tolerance=0.01, hold_s=0.05, horizon_s=2.0)


def linear_onnx(
    path: Path,
    weights: list[list[float]],
    bias: list[float],
    *,
    input_name: str = INPUT,
    output_name: str = OUTPUT,
    opset: int = OPSET,
) -> Path:
    """``actions = weights @ obs + bias`` as an ONNX graph with static shapes."""
    import numpy
    from onnx import TensorProto, checker, helper, numpy_helper, save

    matrix = numpy.asarray(weights, dtype=numpy.float32)
    offsets = numpy.asarray(bias, dtype=numpy.float32)
    outputs, inputs = matrix.shape

    graph = helper.make_graph(
        [helper.make_node("Gemm", [input_name, "W", "b"], [output_name], transB=1)],
        "policy",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, inputs])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, outputs])],
        [numpy_helper.from_array(matrix, "W"), numpy_helper.from_array(offsets, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = IR_VERSION
    checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(path))
    return path


def term_offsets(contract: Contract) -> list[tuple[ObservationType, int, int]]:
    """Where each observation term starts and how wide it is."""
    found: list[tuple[ObservationType, int, int]] = []
    start = 0
    for term in contract.observation:
        width = contract.term_size(term.type) or 0
        found.append((term.type, start, width))
        start += width
    return found


def reach_weights(contract: Contract) -> tuple[list[list[float]], list[float]]:
    """A policy whose command is the target it observes.

    Inverts the mapping of ADR-0041: ``q* = u * scale + origin`` reaches ``target``
    when ``u = (target - origin) / scale``, and the target arrives in the observation
    multiplied by the term's scale.
    """
    dof = contract.dof
    width = contract.observation_size or 0
    scale = contract.action_scale()
    origin = contract.origin()
    weights = [[0.0] * width for _ in range(dof)]
    bias = [-origin[i] / scale[i] for i in range(dof)]
    for term, start, _ in term_offsets(contract):
        if term is not ObservationType.TARGET_JOINT_POS:
            continue
        declared = next(item for item in contract.observation if item.type is term)
        for i in range(dof):
            weights[i][start + i] = 1.0 / (scale[i] * declared.scale)
    return weights, bias


def zero_weights(contract: Contract) -> tuple[list[list[float]], list[float]]:
    """A policy that always commands the reference pose: it never reaches anything else."""
    width = contract.observation_size or 0
    return [[0.0] * width for _ in range(contract.dof)], [0.0] * contract.dof


def make_manifest(
    contract: Contract,
    onnx_path: Path,
    *,
    name: str = "reach",
    capabilities: frozenset[Capability] | None = None,
    completion: Completion = COMPLETION,
    **overrides: Any,
) -> PolicyManifest:
    """A manifest describing the network at ``onnx_path`` under ``contract``."""
    signature = OnnxSignature(
        sha256=file_sha256(onnx_path),
        input=INPUT,
        output=OUTPUT,
        observation_size=contract.observation_size or 0,
        action_size=contract.action_size or 0,
        opset=OPSET,
    )
    fields: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "contract": contract,
        "contract_id": contract.contract_id,
        "capabilities": capabilities or frozenset({Capability("arm.joint_position")}),
        "completion": completion,
        "onnx": signature,
    }
    fields.update(overrides)
    return PolicyManifest(**fields)


def store_policy(
    store: PolicyStore,
    workdir: Path,
    contract: Contract,
    *,
    name: str = "reach",
    weights: list[list[float]] | None = None,
    bias: list[float] | None = None,
    train: dict[str, Any] | None = None,
    **manifest_fields: Any,
) -> PolicyArtifact:
    """Build a reaching policy for this contract and put it in the store."""
    if weights is None or bias is None:
        weights, bias = reach_weights(contract)
    onnx_path = linear_onnx(workdir / f"{name}.onnx", weights, bias)
    manifest = make_manifest(contract, onnx_path, name=name, **manifest_fields)
    return store.put(manifest, onnx_path, train or {"seed": 1, "envs": 512})
