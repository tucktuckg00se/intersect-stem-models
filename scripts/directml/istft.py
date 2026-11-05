"""Rewrite 1-D ConvTranspose (the usual iSTFT overlap-add) as MatMul + overlap-add.

DirectML's ConvTranspose is extremely slow for iSTFT shapes (kernel 2048, stride 512, one output
channel): one batch item of BS-RoFormer SW's iSTFT took 3.1 s on a GTX 745, enough on its own to trip
Windows' 2-second GPU timeout. The equivalent

    frames = MatMul(Transpose(X, [0, 2, 1]), W.reshape(Cin, Cout * K))     # [N, F, Cout*K]
    y      = overlap-add of the R = K / stride sub-blocks (Slice, Pad, Sum)  # [N, Cout, (F+R-1)*stride]

uses a fast GEMM. Applies when group=1, dilation=1, pads=0, no bias/output_padding/output_shape,
K % stride == 0 and the weight is a constant. Bit-identical to ConvTranspose on the CPU provider.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .graphinfo import GraphInfo, attr, constant_values


def rewrite_convtranspose_1d(model: onnx.ModelProto, info: GraphInfo) -> int:
    graph = model.graph
    consts = constant_values(graph)
    new_nodes, new_inits, rewritten = [], [], 0
    for n in graph.node:
        w = consts.get(n.input[1]) if n.op_type == "ConvTranspose" and len(n.input) >= 2 else None
        x_dims = info.static(n.input[0]) if w is not None else None
        ok = (w is not None and w.ndim == 3 and x_dims is not None and len(x_dims) == 3
              and (len(n.input) < 3 or n.input[2] == "")
              and attr(n, "group", 1) == 1 and list(attr(n, "dilations", [1])) == [1]
              and list(attr(n, "pads", [0, 0])) == [0, 0]
              and not attr(n, "output_padding") and not attr(n, "output_shape"))
        stride = list(attr(n, "strides", [1]))[0] if ok else 0
        if ok:
            cin, cout, k = w.shape
            ok = k % stride == 0 and x_dims[1] == cin
        if not ok:
            new_nodes.append(n)
            continue
        r = k // stride
        batch, _, frames = x_dims
        elem = info.elem.get(n.input[0], onnx.TensorProto.FLOAT)
        p = f"__dml_istft{rewritten}"
        x, y = n.input[0], n.output[0]
        new_inits += [
            numpy_helper.from_array(w.reshape(cin, cout * k), p + "_w"),
            numpy_helper.from_array(np.array([0, 0, cout, r, stride], np.int64), p + "_shape5"),
            numpy_helper.from_array(np.array([3], np.int64), p + "_axis3"),
            numpy_helper.from_array(np.array([0, 0, -1], np.int64), p + "_shape3"),
        ]
        nodes = [
            helper.make_node("Transpose", [x], [p + "_xt"], perm=[0, 2, 1], name=p + "_xt"),
            helper.make_node("MatMul", [p + "_xt", p + "_w"], [p + "_frames"], name=p + "_matmul"),
            helper.make_node("Reshape", [p + "_frames", p + "_shape5"], [p + "_f5"], name=p + "_f5"),
        ]
        info.add(p + "_xt", elem, [batch, frames, cin])
        info.add(p + "_frames", elem, [batch, frames, cout * k])
        info.add(p + "_f5", elem, [batch, frames, cout, r, stride])
        padded = []
        for j in range(r):
            new_inits += [
                numpy_helper.from_array(np.array([j], np.int64), f"{p}_s{j}"),
                numpy_helper.from_array(np.array([j + 1], np.int64), f"{p}_e{j}"),
                numpy_helper.from_array(np.array([0, j, 0, 0, 0, r - 1 - j, 0, 0], np.int64), f"{p}_pads{j}"),
            ]
            nodes += [
                helper.make_node("Slice", [p + "_f5", f"{p}_s{j}", f"{p}_e{j}", p + "_axis3"], [f"{p}_sl{j}"], name=f"{p}_sl{j}"),
                helper.make_node("Squeeze", [f"{p}_sl{j}", p + "_axis3"], [f"{p}_sq{j}"], name=f"{p}_sq{j}"),
                helper.make_node("Pad", [f"{p}_sq{j}", f"{p}_pads{j}"], [f"{p}_pad{j}"], mode="constant", name=f"{p}_pad{j}"),
            ]
            info.add(f"{p}_sl{j}", elem, [batch, frames, cout, 1, stride])
            info.add(f"{p}_sq{j}", elem, [batch, frames, cout, stride])
            info.add(f"{p}_pad{j}", elem, [batch, frames + r - 1, cout, stride])
            padded.append(f"{p}_pad{j}")
        nodes += [
            helper.make_node("Sum", padded, [p + "_sum"], name=p + "_sum"),
            helper.make_node("Transpose", [p + "_sum"], [p + "_t"], perm=[0, 2, 1, 3], name=p + "_t"),
            helper.make_node("Reshape", [p + "_t", p + "_shape3"], [y], name=p + "_out"),
        ]
        info.add(p + "_sum", elem, [batch, frames + r - 1, cout, stride])
        info.add(p + "_t", elem, [batch, cout, frames + r - 1, stride])
        info.add(y, elem, [batch, cout, (frames + r - 1) * stride])
        new_nodes += nodes
        rewritten += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(info.add_constant(t) for t in new_inits)
    return rewritten
