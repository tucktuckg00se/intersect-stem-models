"""Rewrite the Split nodes that feed a GLU (Split -> Sigmoid -> Mul) as Slice nodes.

DirectML's graph fusion miscomputes ``Mul(a, Sigmoid(b))`` when ``a`` and ``b`` are different outputs
of the same Split: it treats them as one tensor (same producer node, output slot ignored) and returns
``b * sigmoid(b)`` - a SiLU - instead of ``a * sigmoid(b)``. Explicit split sizes or a positive axis
don't avoid it; Slice does. Upstream: https://github.com/microsoft/onnxruntime/issues/32146
(open; reproduced on AMD, NVIDIA and Intel, ORT 1.22-1.28, DirectML 1.15.4/1.15.5).

Only Splits in that pattern are rewritten; other Splits (e.g. band splits) are left alone because the
equivalent Slices are slower on some providers.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .graphinfo import GraphInfo, attr, constant_values, consumers


def _through_identity(name: str, producer: dict[str, onnx.NodeProto]) -> str:
    seen = 0
    while name in producer and producer[name].op_type == "Identity" and seen < 16:
        name = producer[name].input[0]
        seen += 1
    return name


def _forward_identity(name: str, cons: dict[str, list[onnx.NodeProto]]) -> list[str]:
    names, frontier = [name], [name]
    while frontier:
        nxt = []
        for t in frontier:
            for c in cons.get(t, []):
                if c.op_type == "Identity":
                    nxt.append(c.output[0])
        names += nxt
        frontier = nxt
    return names


def glu_splits(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """Split nodes where Mul(out_i, Sigmoid(out_j)) with i != j (Identity hops allowed, any operand order)."""
    producer = {o: n for n in graph.node for o in n.output if o}
    cons = consumers(graph)
    found = []
    for split in (n for n in graph.node if n.op_type == "Split"):
        outs = {o: k for k, o in enumerate(split.output) if o}
        hit = False
        for o_j, j in outs.items():
            for t in _forward_identity(o_j, cons):
                for sig in (c for c in cons.get(t, []) if c.op_type == "Sigmoid"):
                    for s in _forward_identity(sig.output[0], cons):
                        for mul in (c for c in cons.get(s, []) if c.op_type == "Mul"):
                            other = mul.input[1] if mul.input[0] == s else mul.input[0]
                            src = _through_identity(other, producer)
                            if src in outs and outs[src] != j:
                                hit = True
        if hit:
            found.append(split)
    return found


def _split_sizes(node: onnx.NodeProto, dim: int, consts: dict[str, np.ndarray]) -> list[int] | None:
    if len(node.input) > 1 and node.input[1]:
        v = consts.get(node.input[1])
        return v.astype(np.int64).tolist() if v is not None else None
    if attr(node, "split") is not None:  # opset < 13
        return list(attr(node, "split"))
    k = len(node.output)
    chunk = -(-dim // k)  # opset 18 num_outputs: ceil division, last chunk smaller
    return [chunk] * (k - 1) + [dim - chunk * (k - 1)]


def rewrite_glu_splits(model: onnx.ModelProto, info: GraphInfo, all_splits: bool = False) -> int:
    """Replace GLU Splits (or every Split with all_splits) by Slices with the same output names."""
    graph = model.graph
    # Key by output names: protobuf hands out a new Python wrapper per access, so id() isn't stable.
    targets = {tuple(n.output) for n in (graph.node if all_splits else glu_splits(graph)) if n.op_type == "Split"}
    if not targets:
        return 0
    consts = constant_values(graph)
    new_nodes, new_inits, replaced = [], [], 0
    for node in graph.node:
        dims = info.static(node.input[0]) if tuple(node.output) in targets else None
        if dims is None:
            new_nodes.append(node)
            continue
        axis = (attr(node, "axis", 0) or 0) % len(dims)
        sizes = _split_sizes(node, dims[axis], consts)
        if sizes is None or sum(sizes) != dims[axis] or len(sizes) != len(node.output):
            new_nodes.append(node)
            continue
        p = f"__dml_glu{replaced}"
        new_inits.append(numpy_helper.from_array(np.array([axis], np.int64), p + "_axis"))
        start = 0
        for j, (out, size) in enumerate(zip(node.output, sizes)):
            if out:
                new_inits += [numpy_helper.from_array(np.array([start], np.int64), f"{p}_s{j}"),
                              numpy_helper.from_array(np.array([start + size], np.int64), f"{p}_e{j}")]
                new_nodes.append(helper.make_node("Slice", [node.input[0], f"{p}_s{j}", f"{p}_e{j}", p + "_axis"], [out],
                                                  name=f"{node.name or p}_slice{j}"))
            start += size
        replaced += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(info.add_constant(t) for t in new_inits)
    return replaced
