"""Phase 6, stage 0, facts 0.17 and 0.18: what an exported policy looks like from outside.

Three modes, run by ``scripts/phase6-probe.sh onnx`` in two different interpreters -- which is
the point: the policy crosses the boundary between worlds as a file (invariant 2), so the file
has to give the same answer to a reader that shares nothing with the writer.

``describe``  (probe venv, onnxruntime) names, shapes and types of inputs and outputs, opset.
``torch``     (training venv) runs the TorchScript export on seeded random inputs and saves
              inputs and outputs.
``compare``   (probe venv) runs the ONNX file on the same inputs, row by row as its static
              shape demands, and prints the largest difference.
"""

from __future__ import annotations

import argparse
import json
import sys

TOLERANCE = 1e-4
"""Float32 graphs from two runtimes agree to about 1e-6; a hundred times that is still noise,
and a wrong input order or a missing normalizer is orders of magnitude above it."""

ROWS = 64


def describe(args: argparse.Namespace) -> int:
    import onnx
    import onnxruntime as ort

    model = onnx.load(args.onnx)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    def shape(items: list) -> list[dict]:
        return [{"name": i.name, "shape": i.shape, "type": i.type} for i in items]

    opsets = [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import]
    facts = {
        "opset": opsets,
        "inputs": shape(session.get_inputs()),
        "outputs": shape(session.get_outputs()),
        "producer": f"{model.producer_name} {model.producer_version}",
    }
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(facts, file, indent=2)
    print(json.dumps(facts, indent=2))
    return 0


def run_torch(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    with open(args.describe, encoding="utf-8") as file:
        facts = json.load(file)
    dim = int(facts["inputs"][0]["shape"][-1])
    rng = np.random.default_rng(args.seed)
    inputs = rng.standard_normal((ROWS, dim)).astype(np.float32)
    policy = torch.jit.load(args.jit, map_location="cpu")
    policy.eval()
    with torch.no_grad():
        outputs = policy(torch.from_numpy(inputs)).numpy()
    np.savez(args.out, inputs=inputs, outputs=outputs)
    print(f"TorchScript: вход {inputs.shape}, выход {outputs.shape}, torch {torch.__version__}")
    return 0


def compare(args: argparse.Namespace) -> int:
    import numpy as np
    import onnxruntime as ort

    saved = np.load(args.inputs)
    inputs, expected = saved["inputs"], saved["outputs"]
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    got = np.concatenate([session.run(None, {name: row[None, :]})[0] for row in inputs])
    worst = float(np.max(np.abs(got - expected)))
    print(f"onnxruntime: выход {got.shape}; наибольшее расхождение с TorchScript {worst:.3g}")
    print(f"диапазон выхода: [{float(got.min()):.3g}, {float(got.max()):.3g}]")
    return 0 if worst <= TOLERANCE else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    one = modes.add_parser("describe")
    one.add_argument("--onnx", required=True)
    one.add_argument("--out", required=True)
    two = modes.add_parser("torch")
    two.add_argument("--jit", required=True)
    two.add_argument("--describe", required=True)
    two.add_argument("--seed", type=int, default=42)
    two.add_argument("--out", required=True)
    three = modes.add_parser("compare")
    three.add_argument("--onnx", required=True)
    three.add_argument("--inputs", required=True)
    args = parser.parse_args()
    return {"describe": describe, "torch": run_torch, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
