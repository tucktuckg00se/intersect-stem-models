"""Converts a stem model from ONNX to a native Core ML ML Program (.mlpackage), fp32, same inputs/outputs.

Route: ONNX -> constant folding (ONNX Runtime, basic level: standard ONNX only) -> PyTorch (onnx2torch)
-> torch.jit.trace -> coremltools. Checks the PyTorch copy against ONNX Runtime before converting.

usage: python convert_coreml.py <model.onnx> <out.mlpackage> <input shape, e.g. 1,690,8196>
"""
import os, sys, tempfile
import numpy as np
import onnx
import onnxruntime as ort
import torch
import coremltools as ct
from onnx import numpy_helper
from onnx2torch import convert
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx2torch_lstm  # noqa: F401  (registers the LSTM converter)


def snr_db(ref, test):
    err = (ref.astype(np.float64) - test.astype(np.float64))
    return 10 * np.log10((ref.astype(np.float64) ** 2).sum() / max((err ** 2).sum(), 1e-300))


def fix_scalar_unsqueeze_axes(model):
    g = model.graph
    tensors = {i.name: i for i in g.initializer}
    made = {}
    for n in g.node:
        if n.op_type == "Unsqueeze" and len(n.input) > 1 and n.input[1] in tensors and len(tensors[n.input[1]].dims) == 0:
            name = n.input[1] + "__axes1d"
            if name not in made:
                made[name] = numpy_helper.from_array(numpy_helper.to_array(tensors[n.input[1]]).reshape(1), name)
                g.initializer.append(made[name])
            n.input[1] = name
    return model


REDUCE_OPS = {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
              "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"}   # ReduceSum already takes axes as an input in 13


def downgrade_to_opset17(model):
    """Rewrites the opset-18 forms onnx2torch can't read into their opset-13..17 equivalents (same math)."""
    from onnx import helper, shape_inference
    g = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    shapes = {}
    for v in shape_inference.infer_shapes(model).graph.value_info:
        shapes[v.name] = [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
    for v in list(g.input) + list(g.output):
        shapes[v.name] = [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
    for n in g.node:
        attrs = {a.name: a for a in n.attribute}
        if n.op_type in REDUCE_OPS:
            if len(n.input) > 1 and n.input[1]:
                if n.input[1] not in consts:
                    raise RuntimeError(f"{n.name}: reduce axes aren't constant")
                n.attribute.append(helper.make_attribute("axes", [int(a) for a in consts[n.input[1]].reshape(-1)]))
                del n.input[1:]
            if "noop_with_empty_axes" in attrs:
                if attrs["noop_with_empty_axes"].i:
                    raise RuntimeError(f"{n.name}: noop_with_empty_axes isn't in opset 17")
                n.attribute.remove(attrs["noop_with_empty_axes"])
        elif n.op_type == "Split" and "num_outputs" in attrs:
            axis = attrs["axis"].i if "axis" in attrs else 0
            dims = shapes.get(n.input[0])
            if not dims or dims[axis] is None:
                raise RuntimeError(f"{n.name}: Split input size unknown")
            k = attrs["num_outputs"].i
            size = dims[axis]; part = -(-size // k)
            sizes = [part] * (k - 1) + [size - part * (k - 1)]
            name = n.name + "__split_sizes"
            g.initializer.append(numpy_helper.from_array(np.array(sizes, dtype=np.int64), name))
            n.input.append(name)
            n.attribute.remove(attrs["num_outputs"])
        elif n.op_type == "Reshape" and "allowzero" in attrs and attrs["allowzero"].i:
            # allowzero only changes how a 0 in the target shape is read; without a 0 it does nothing.
            if n.input[1] not in consts or (consts[n.input[1]] == 0).any():
                raise RuntimeError(f"{n.name}: Reshape allowzero=1 with a zero or unknown shape")
            n.attribute.remove(attrs["allowzero"])
        elif n.op_type == "Pad" and len(n.input) > 3 and n.input[3]:
            raise RuntimeError(f"{n.name}: Pad with axes isn't in opset 17")
        elif n.op_type in ("ScatterND", "ScatterElements") and "reduction" in attrs and attrs["reduction"].s in (b"max", b"min"):
            raise RuntimeError(f"{n.name}: {n.op_type} max/min reduction isn't in opset 17")
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            o.version = 17
    return model


def main():
    src, out, shape = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(",")]
    work = tempfile.mkdtemp(prefix="coreml-convert-")
    folded = os.path.join(work, "folded.onnx")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = folded
    so.log_severity_level = 3
    ref_session = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    in_name = ref_session.get_inputs()[0].name
    out_name = ref_session.get_outputs()[0].name

    x = (np.random.default_rng(63).standard_normal(shape) * 0.1).astype(np.float32)
    plain = ort.SessionOptions(); plain.log_severity_level = 3
    ref = ort.InferenceSession(src, plain, providers=["CPUExecutionProvider"]).run(None, {in_name: x})[0]
    print(f"input {in_name} {shape}, output {out_name} {list(ref.shape)}")

    model = fix_scalar_unsqueeze_axes(onnx.load(folded))
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    if opset > 17 or any(n.op_type == "Reshape" for n in model.graph.node):   # opset-18 forms, allowzero
        model = downgrade_to_opset17(model)
        down = os.path.join(work, "opset17.onnx")
        onnx.save(model, down)
        y17 = ort.InferenceSession(down, plain, providers=["CPUExecutionProvider"]).run(None, {in_name: x})[0]
        print(f"opset {opset} -> 17 rewrite vs original: {'bit-identical' if np.array_equal(ref, y17) else f'SNR {snr_db(ref, y17):.1f} dB'}")
    for i, n in enumerate(model.graph.node):   # node names become module names: avoid clashes like "to"
        n.name = f"n{i}_{n.op_type}"
    torch_model = convert(model).eval()
    with torch.no_grad():
        y = torch_model(torch.from_numpy(x))
    y = (y[0] if isinstance(y, (tuple, list)) else y).numpy()
    print(f"PyTorch copy vs ONNX Runtime: SNR {snr_db(ref, y):.1f} dB, max abs error {np.abs(ref - y).max():.2e}")

    with torch.no_grad():
        traced = torch.jit.trace(torch_model, torch.from_numpy(x), check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=in_name, shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=out_name, dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = os.path.basename(src) + " (fp32 ML Program, converted from ONNX)"
    mlmodel.save(out)
    ops = {}
    for f in mlmodel._spec.mlProgram.functions.values():
        for block in f.block_specializations.values():
            for op in block.operations:
                ops[op.type] = ops.get(op.type, 0) + 1
    print(f"saved {out}: {sum(ops.values())} MIL ops; most common: {sorted(ops.items(), key=lambda kv: -kv[1])[:12]}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(out)), os.path.basename(out) + ".ref_input.npy"), x)
    np.save(os.path.join(os.path.dirname(os.path.abspath(out)), os.path.basename(out) + ".ref_output.npy"), ref)


if __name__ == "__main__":
    main()
