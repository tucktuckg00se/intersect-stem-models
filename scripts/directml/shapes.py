"""Concrete tensor shapes for a model: ONNX static inference, or one ONNX Runtime CPU run with profiling.

Static inference can't resolve graphs exported with dynamic reshapes (BS-RoFormer SW and MVSep Mega 53
infer no MatMul shapes at all). For those, one CPU run with ORT profiling records every node's actual
output shapes (``output_type_shape``). Inputs get random data of the given shape, so the shapes are
those of one chunk/segment; INTERSECT always feeds that fixed size.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, shape_inference

_ORT_TYPES = {
    "float": TensorProto.FLOAT, "float16": TensorProto.FLOAT16, "double": TensorProto.DOUBLE,
    "int64": TensorProto.INT64, "int32": TensorProto.INT32, "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8, "bool": TensorProto.BOOL,
}


def input_shapes(model: onnx.ModelProto, overrides: dict[str, list[int]]) -> dict[str, list[int]]:
    init = {t.name for t in model.graph.initializer}
    out = {}
    for vi in model.graph.input:
        if vi.name in init:
            continue
        if vi.name in overrides:
            out[vi.name] = overrides[vi.name]
            continue
        dims = [d.dim_value if d.HasField("dim_value") else 0 for d in vi.type.tensor_type.shape.dim]
        if not dims or any(d <= 0 for d in dims):
            raise SystemExit(f"input '{vi.name}' has symbolic dims; pass --input-shape {vi.name}=d1,d2,...")
        out[vi.name] = dims
    return out


def _static(model: onnx.ModelProto, feeds: dict[str, list[int]]) -> dict[str, tuple[int, list[int]]]:
    scratch = onnx.ModelProto()
    scratch.CopyFrom(model)
    # Stale declarations in some exports (fp16 weights declared as fp32) make inference refuse the model.
    del scratch.graph.value_info[:]
    for vi in scratch.graph.input:
        if vi.name in feeds:
            del vi.type.tensor_type.shape.dim[:]
            vi.type.tensor_type.shape.dim.extend(onnx.TensorShapeProto.Dimension(dim_value=d) for d in feeds[vi.name])
    try:
        inferred = shape_inference.infer_shapes(scratch, data_prop=True)
    except Exception:  # noqa: BLE001 - fall back to profiling
        return {}
    out = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            out[vi.name] = (vi.type.tensor_type.elem_type, [d.dim_value for d in dims])
    return out


def _profiled(model: onnx.ModelProto, feeds: dict[str, list[int]]) -> dict[str, tuple[int, list[int]]]:
    import onnxruntime as ort

    scratch = onnx.ModelProto()
    scratch.CopyFrom(model)
    for i, n in enumerate(scratch.graph.node):
        if not n.name:
            n.name = f"__unnamed_{i}"
    workdir = tempfile.mkdtemp(prefix="dml_shapes_")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep nodes 1:1 with the file
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(workdir, "prof")
    session = ort.InferenceSession(scratch.SerializeToString(), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    session.run(None, {k: (rng.standard_normal(v) * 0.1).astype(np.float32) for k, v in feeds.items()})
    prof = session.end_profiling()
    with open(prof) as f:
        events = json.load(f)
    os.remove(prof)
    os.rmdir(workdir)
    by_node = {e["name"][: -len("_kernel_time")]: e["args"].get("output_type_shape", [])
               for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")}
    out = {}
    for n in scratch.graph.node:
        for name, ts in zip(n.output, by_node.get(n.name, [])):
            (t, dims), = ts.items()
            if name and t in _ORT_TYPES:
                out[name] = (_ORT_TYPES[t], list(dims))
    for k, v in feeds.items():
        out[k] = (TensorProto.FLOAT, list(v))
    return out


def annotate(model: onnx.ModelProto, overrides: dict[str, list[int]]) -> tuple[dict[str, tuple[int, list[int]]], str]:
    """Return ({tensor: (elem_type, dims)}, method). Uses static inference when it sizes every MatMul."""
    feeds = input_shapes(model, overrides)
    shapes = _static(model, feeds)
    matmul_inputs = [n.input[0] for n in model.graph.node if n.op_type in ("MatMul", "Conv", "ConvTranspose", "Split")]
    if matmul_inputs and all(i in shapes or i in feeds for i in matmul_inputs):
        return shapes, "static"
    return _profiled(model, feeds), "profiled"
