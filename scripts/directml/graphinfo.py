"""Tensor shapes, element types and constant-ness for a graph, shared by the rewrite passes."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}


class GraphInfo:
    """Shapes and element types by tensor name, plus the set of tensors that depend only on constants.

    ``shapes`` comes from ``shapes.annotate`` (static inference or one profiled CPU run). Passes that
    create tensors register them with ``add`` so later passes can size them.
    """

    def __init__(self, model: onnx.ModelProto, shapes: dict[str, tuple[int, list[int]]]):
        g = model.graph
        self.elem: dict[str, int] = {}
        self.dims: dict[str, list[int]] = {}
        for name, (elem, dims) in shapes.items():
            self.elem[name] = elem
            self.dims[name] = list(dims)
        for t in g.initializer:
            self.elem[t.name] = t.data_type
            self.dims[t.name] = list(t.dims)
        for vi in g.input:
            tt = vi.type.tensor_type
            self.elem.setdefault(vi.name, tt.elem_type)
        self.const: set[str] = {t.name for t in g.initializer}
        for n in g.node:
            if all(i == "" or i in self.const for i in n.input):
                self.const.update(o for o in n.output if o)

    def add(self, name: str, elem: int, dims: list[int]) -> None:
        self.elem[name] = elem
        self.dims[name] = list(dims)

    def add_constant(self, tensor: onnx.TensorProto) -> onnx.TensorProto:
        """Register an initializer a pass creates, so later passes treat it as a constant."""
        self.add(tensor.name, tensor.data_type, list(tensor.dims))
        self.const.add(tensor.name)
        return tensor

    def static(self, name: str) -> list[int] | None:
        d = self.dims.get(name)
        return d if d and all(isinstance(x, int) and x > 0 for x in d) else None

    def numel(self, name: str) -> int:
        d = self.static(name)
        return int(np.prod(d)) if d else 0

    def nbytes(self, name: str) -> int:
        return self.numel(name) * (4 if self.elem.get(name) == onnx.TensorProto.FLOAT else 2)

    def is_data(self, name: str) -> bool:
        """Activation tensors that matter for GPU job size; small or integer shape plumbing doesn't."""
        return name not in self.const and self.elem.get(name) in FLOAT_TYPES and self.numel(name) >= 4096


def attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def constant_values(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    values = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute and node.attribute[0].type == onnx.AttributeProto.TENSOR:
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    return values


def consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for i in n.input:
            if i:
                out.setdefault(i, []).append(n)
    return out
