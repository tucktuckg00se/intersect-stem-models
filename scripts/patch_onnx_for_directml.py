"""Rewrite third-party ONNX stem models so ONNX Runtime's DirectML provider loads them and computes them correctly.

Every rewrite is semantics-preserving; the script verifies that the patched model's CPU output is
bit-identical to the original's before writing it.

Rewrites (each applied only where it is provably a no-op on CPU):
  * Reshape(allowzero=1) -> allowzero removed, when the target shape is a constant with no zeros.
    DirectML rejects allowzero=1 at session creation ("80070057 The parameter is incorrect").
    Seen in: Mini-BS-RoFormer V2.5 (HiDolen) - 88 nodes.
  * Dead nodes (no output used by any node or graph output) -> removed.
    DirectML's graph fusion fails on them (DmlGraphFusionHelper.cpp, 80070057).
    Seen in: SCNet int8 (audiojs) - 6 Transpose nodes.
  * Split -> one Slice per output (static shapes only).
    DirectML's Split gives wrong values when its outputs feed a GLU (Split -> Sigmoid -> Mul):
    SCNet's output was off by 250x the signal peak. With Slice it matches CPU to ~1e-3 (conv rounding).
    Seen in: SCNet int8 (audiojs) - 21 nodes.

Usage:
    python scripts/patch_onnx_for_directml.py INPUT.onnx OUTPUT.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def constant_values(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    values = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute and node.attribute[0].type == onnx.AttributeProto.TENSOR:
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    return values


def clear_reshape_allowzero(graph: onnx.GraphProto) -> int:
    consts = constant_values(graph)
    changed = 0
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        for attr in list(node.attribute):
            if attr.name == "allowzero" and attr.i == 1:
                shape = consts.get(node.input[1])
                if shape is not None and not (shape == 0).any():
                    node.attribute.remove(attr)
                    changed += 1
    return changed


def remove_dead_nodes(graph: onnx.GraphProto) -> int:
    removed = 0
    while True:
        used = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
        dead = [n for n in graph.node if all((not o) or o not in used for o in n.output)]
        if not dead:
            return removed
        for node in dead:
            graph.node.remove(node)
        removed += len(dead)


def split_to_slices(model: onnx.ModelProto) -> int:
    graph = model.graph
    if not any(n.op_type == "Split" for n in graph.node):
        return 0
    # Infer on a copy without value_info: some exports carry stale declarations (e.g. fp16 weights
    # declared as fp32) that make onnx's inference refuse the model.
    scratch = onnx.ModelProto()
    scratch.CopyFrom(model)
    del scratch.graph.value_info[:]
    inferred = shape_inference.infer_shapes(scratch)
    value_info = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}

    def static_dims(name: str) -> list[int] | None:
        v = value_info.get(name)
        if v is None or not v.type.tensor_type.HasField("shape"):
            return None
        dims = v.type.tensor_type.shape.dim
        return [d.dim_value for d in dims] if all(d.HasField("dim_value") for d in dims) else None

    new_nodes, new_inits, replaced = [], [], 0
    for node in graph.node:
        in_dims = static_dims(node.input[0]) if node.op_type == "Split" else None
        out_dims = [static_dims(o) for o in node.output] if in_dims is not None else []
        if in_dims is None or any(d is None for d in out_dims):
            new_nodes.append(node)
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0) % len(in_dims)
        start = 0
        for j, (out, dims) in enumerate(zip(node.output, out_dims)):
            names = [f"__dml_split{replaced}_{j}_{k}" for k in ("starts", "ends", "axes")]
            for name, value in zip(names, (start, start + dims[axis], axis)):
                new_inits.append(numpy_helper.from_array(np.array([value], dtype=np.int64), name))
            new_nodes.append(helper.make_node("Slice", [node.input[0], *names], [out], name=f"{node.name}_slice{j}"))
            start += dims[axis]
        replaced += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return replaced


def cpu_outputs(model_bytes: bytes, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    return session.run(None, feed)


def random_feed(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    initializer_names = {t.name for t in model.graph.initializer}
    feed = {}
    for graph_input in model.graph.input:
        if graph_input.name in initializer_names:
            continue
        dims = [d.dim_value if d.HasField("dim_value") else 1 for d in graph_input.type.tensor_type.shape.dim]
        feed[graph_input.name] = (rng.standard_normal(dims) * 0.1).astype(np.float32)
    return feed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-verify", action="store_true", help="skip the CPU bit-identity check")
    args = parser.parse_args()

    original_bytes = args.input.read_bytes()
    model = onnx.load_from_string(original_bytes)

    allowzero = clear_reshape_allowzero(model.graph)
    dead = remove_dead_nodes(model.graph)
    splits = split_to_slices(model)
    print(f"Reshape allowzero cleared: {allowzero}; dead nodes removed: {dead}; Split -> Slice: {splits}")

    onnx.checker.check_model(model)
    patched_bytes = model.SerializeToString()

    if not args.skip_verify:
        feed = random_feed(model)
        before = cpu_outputs(original_bytes, feed)
        after = cpu_outputs(patched_bytes, feed)
        for a, b in zip(before, after):
            if a.shape != b.shape or not np.array_equal(a, b):
                print("ERROR: patched model's CPU output differs from the original", file=sys.stderr)
                return 1
        print("CPU output bit-identical to the original")

    args.output.write_bytes(patched_bytes)
    print(f"Wrote {args.output} ({len(patched_bytes)} bytes, sha256 {hashlib.sha256(patched_bytes).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
