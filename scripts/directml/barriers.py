"""Split a model into many short GPU jobs so DirectML can't trip Windows' 2-second GPU timeout (TDR).

ONNX Runtime's DirectML provider compiles each fused part of the graph into one command list and
submits it as a single GPU job. For BS-RoFormer models that is the whole model: ~5-7 s per chunk on a
GTX 745. GPUs whose compute work can only be preempted between DMA buffers (NVIDIA Kepler/Maxwell,
e.g. GTX 600/700/900) can't be interrupted inside such a job, so Windows resets the driver after ~2 s
and the run fails with 887A0006 (DXGI_ERROR_DEVICE_HUNG) - or returns zeros, with the error only on
the next run.

A *barrier* at a cut point reads one element of a crossing tensor back to the CPU,

    z = Reshape(Mul(Compress(Slice(t0, first element), [True]), 0), [])     # a rank-0 zero

and gates tensors as ``t' = t + z``. DirectML has no Compress kernel, so ORT runs it on the CPU; the
read-back makes the DirectML provider submit and wait for all GPU work recorded so far. Every node after
the cut must depend on ``z``: most already do through an input computed after the cut, and a node whose
data inputs all come from before the cut gets its smallest float input gated. Everything after the cut
therefore becomes a separate, short GPU job. ``x + 0 == x`` for finite ``x``, so values are unchanged.

Cut points: at most ``max_cross_mib`` of data live across the cut, and either ``budget_gflop / 2`` of
MatMul/Conv work or ``max_nodes`` nodes since the previous cut. No cut separates a MatMul/Gemm/Conv from
the elementwise op ONNX Runtime fuses it with on the CPU (e.g. a bias Add), so CPU output stays
bit-identical and CPU speed is kept.

Ops too big for one job are split: batched MatMuls over ``max_op_gflop`` into the fewest equal pieces
along axis 0, and Conv/ConvTranspose over ``max_conv_gflop`` per batch item, with a barrier between
pieces (rows are independent, so this is exact).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .graphinfo import FLOAT_TYPES as _FLOATS, GraphInfo, attr

# Producer -> consumer pairs that ONNX Runtime's basic (level 1) optimizations fuse: MatMulAddFusion,
# MatmulBNFusion, ConvAddFusion, ConvMulFusion, ConvBNFusion, GemmSumFusion, GemmTransposeFusion.
# A cut between them would block the fusion (slower, and CPU output no longer bit-identical).
_FUSED_PAIRS = {
    "MatMul": {"Add", "BatchNormalization"},
    "Conv": {"Add", "Mul", "BatchNormalization"},
    "Gemm": {"Sum", "Transpose"},
}


@dataclass
class BarrierOptions:
    budget_gflop: float = 40.0
    max_nodes: int = 300
    max_cross_mib: float = 300.0
    max_op_gflop: float = 60.0
    max_conv_gflop: float = 4.0


@dataclass
class BarrierReport:
    cuts: int = 0
    split_ops: int = 0
    barriers: int = 0
    gates: int = 0
    total_gflop: float = 0.0
    split_details: list[str] = field(default_factory=list)


def _flops(n: onnx.NodeProto, info: GraphInfo) -> int:
    if not n.input:
        return 0
    a, y = info.static(n.input[0]), info.static(n.output[0])
    if not (a and y):
        return 0
    if n.op_type == "MatMul":
        return 2 * int(np.prod(y)) * a[-1]
    if n.op_type in ("Conv", "ConvTranspose"):
        k = int(np.prod(attr(n, "kernel_shape", [1])))
        group = attr(n, "group", 1) or 1
        if n.op_type == "Conv":
            return 2 * int(np.prod(y)) * (a[1] // group) * k
        return 2 * int(np.prod(a)) * (y[1] // group) * k
    return 0


def _batch_splittable(n: onnx.NodeProto, info: GraphInfo, opts: BarrierOptions) -> bool:
    if n.op_type not in ("MatMul", "Conv", "ConvTranspose"):
        return False
    a = info.static(n.input[0])
    if not (a and a[0] > 1 and info.static(n.output[0])):
        return False
    f = _flops(n, info)
    if n.op_type == "MatMul":
        return f > opts.max_op_gflop * 1e9 and n.input[1] in info.const and len(info.dims.get(n.input[1], [])) == 2
    return f > opts.max_conv_gflop * 1e9 and all(i == "" or i in info.const for i in n.input[1:])


def insert_barriers(model: onnx.ModelProto, info: GraphInfo, opts: BarrierOptions) -> BarrierReport:
    g = model.graph
    report = BarrierReport()
    new_inits: list[onnx.TensorProto] = []

    def make_barrier(src: str) -> tuple[list[onnx.NodeProto], dict]:
        """CPU read-back of one element of src; returns the nodes and a gating context."""
        report.barriers += 1
        p = f"__dml_tdr{report.barriers}"
        rank = len(info.dims[src])
        src_elem = info.elem.get(src, onnx.TensorProto.FLOAT)
        new_inits.extend([
            numpy_helper.from_array(np.zeros(rank, np.int64), p + "_starts"),
            numpy_helper.from_array(np.ones(rank, np.int64), p + "_ends"),
            numpy_helper.from_array(np.array([True]), p + "_keep"),
            numpy_helper.from_array(np.zeros((), helper.tensor_dtype_to_np_dtype(src_elem)), p + "_zero"),
            numpy_helper.from_array(np.zeros(0, np.int64), p + "_scalar"),
        ])
        nodes = [
            helper.make_node("Slice", [src, p + "_starts", p + "_ends"], [p + "_one"], name=p + "_slice"),
            helper.make_node("Compress", [p + "_one", p + "_keep"], [p + "_cpu"], axis=0, name=p + "_compress"),
            helper.make_node("Mul", [p + "_cpu", p + "_zero"], [p + "_z1"], name=p + "_mul"),
            helper.make_node("Reshape", [p + "_z1", p + "_scalar"], [p + "_z"], name=p + "_reshape"),
        ]
        return nodes, {"prefix": p, "zeros": {src_elem: p + "_z"}, "count": 0}

    def gate(t: str, ctx: dict) -> tuple[list[onnx.NodeProto], str]:
        """t' = t + 0 (the barrier's rank-0 zero): t' depends on the read-back, values unchanged."""
        e = info.elem.get(t, onnx.TensorProto.FLOAT)
        p, zeros, nodes = ctx["prefix"], ctx["zeros"], []
        if e not in zeros:
            zeros[e] = f"{p}_z{e}"
            nodes.append(helper.make_node("Cast", [next(iter(zeros.values()))], [zeros[e]], to=e, name=f"{p}_cast{e}"))
        out = f"{p}_gate{ctx['count']}"
        ctx["count"] += 1
        report.gates += 1
        nodes.append(helper.make_node("Add", [t, zeros[e]], [out], name=out))
        info.add(out, e, info.dims[t])
        return nodes, out

    # Pass 1: cut points.
    nodes = list(g.node)
    producer = {o: n for n in nodes for o in n.output if o}
    last_use: dict[str, int] = {}
    consumers_of: dict[str, list[onnx.NodeProto]] = {}
    for idx, n in enumerate(nodes):
        for i in n.input:
            if i:
                last_use[i] = idx
                consumers_of.setdefault(i, []).append(n)
    for o in g.output:
        last_use[o.name] = len(nodes)

    def fusion_sensitive(t: str) -> bool:
        p = producer.get(t)
        fused = _FUSED_PAIRS.get(p.op_type, set()) if p is not None else set()
        return any(c.op_type in fused for c in consumers_of.get(t, []))

    live = {i.name for i in g.input if info.is_data(i.name)}
    cuts: dict[int, list[str]] = {}
    acc, since = 0.0, 0
    for idx, n in enumerate(nodes):
        f = _flops(n, info)
        acc += f
        report.total_gflop += f / 1e9
        since += 1
        live.update(o for o in n.output if o and info.is_data(o))
        live = {t for t in live if last_use.get(t, -1) > idx}
        if not live or idx == len(nodes) - 1:
            continue
        if acc < opts.budget_gflop * 1e9 * 0.5 and since < opts.max_nodes:
            continue
        if sum(info.nbytes(t) for t in live) > opts.max_cross_mib * 2**20 or any(fusion_sensitive(t) for t in live):
            continue
        cuts[idx] = sorted(live)
        acc, since = 0.0, 0
    report.cuts = len(cuts)

    # Pass 2: rewrite. After each cut, every node must depend on that cut's read-back. Most already do
    # through an input computed after the cut; a node whose data inputs all come from before the cut
    # gets its smallest float input gated. Gating every crossing tensor instead copied ~4 GiB per run
    # (long-lived tensors like the model input were re-gated at every cut).
    split_here = [_batch_splittable(n, info, opts) for n in nodes]
    rename: dict[str, str] = {}
    out_nodes: list[onnx.NodeProto] = []
    active: dict | None = None   # gating context of the most recent cut
    dep: set[str] = set()        # tensors that depend on the most recent cut's read-back

    def ensure_dependency(n: onnx.NodeProto) -> None:
        data = [i for i in n.input if i and i not in info.const]
        if active is None or not data or any(i in dep for i in data):
            return
        floats = [i for i in data if info.elem.get(i) in _FLOATS and info.static(i)]
        if not floats:
            return  # integer shape plumbing; runs on the CPU
        t = min(floats, key=info.numel)
        g_nodes, t2 = gate(t, active)
        out_nodes.extend(g_nodes)
        rename[t] = t2
        dep.add(t2)
        for k, i in enumerate(n.input):
            if i == t:
                n.input[k] = t2

    for idx, n in enumerate(nodes):
        in0 = n.input[0] if n.input else None
        flops = _flops(n, info)
        for k, i in enumerate(n.input):
            if i in rename:
                n.input[k] = rename[i]
        ensure_dependency(n)
        depends = any(i in dep for i in n.input if i)
        if split_here[idx]:
            a = info.static(in0)
            y_name = n.output[0]
            parts = a[0] if n.op_type in ("Conv", "ConvTranspose") else min(a[0], int(np.ceil(flops / (opts.max_op_gflop * 1e9))))
            sizes = [a[0] // parts + (1 if j < a[0] % parts else 0) for j in range(parts)]
            p = f"__dml_split{report.split_ops}"
            new_inits.append(numpy_helper.from_array(np.array(sizes, np.int64), p + "_sizes"))
            slices = [f"{p}_in{j}" for j in range(parts)]
            out_nodes.append(helper.make_node("Split", [n.input[0], p + "_sizes"], slices, axis=0, name=p))
            y_dims = info.static(y_name)
            ys: list[str] = []
            for j in range(parts):
                info.add(slices[j], info.elem.get(in0, onnx.TensorProto.FLOAT), [sizes[j]] + a[1:])
                x_in = slices[j]
                if j > 0:  # each piece is its own GPU job
                    b_nodes, ctx = make_barrier(ys[-1])
                    g_nodes, x_in = gate(slices[j], ctx)
                    out_nodes.extend(b_nodes + g_nodes)
                ys.append(f"{p}_out{j}")
                info.add(ys[-1], info.elem.get(y_name, onnx.TensorProto.FLOAT), [sizes[j]] + y_dims[1:])
                piece = helper.make_node(n.op_type, [x_in] + list(n.input[1:]), [ys[-1]], name=f"{p}_part{j}")
                piece.attribute.extend(n.attribute)
                out_nodes.append(piece)
            out_nodes.append(helper.make_node("Concat", ys, [y_name], axis=0, name=p + "_concat"))
            report.split_ops += 1
            report.split_details.append(f"{n.op_type} {n.name or y_name}: {flops / 1e9:.1f} GFLOP -> {parts} pieces")
        else:
            out_nodes.append(n)
        if depends:
            dep.update(o for o in n.output if o)
        if idx in cuts:
            src = rename.get(max(cuts[idx], key=info.numel), max(cuts[idx], key=info.numel))
            b_nodes, active = make_barrier(src)
            out_nodes.extend(b_nodes)
            dep = set()

    if any(o.name in rename for o in g.output):
        raise RuntimeError("a graph output crosses a cut")
    del g.node[:]
    g.node.extend(out_nodes)
    g.initializer.extend(info.add_constant(t) for t in new_inits)
    return report
