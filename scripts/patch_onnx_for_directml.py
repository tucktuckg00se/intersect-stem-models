"""Rewrite third-party ONNX stem models so ONNX Runtime's DirectML provider loads them, computes them
correctly and doesn't trip Windows' 2-second GPU timeout.

Every rewrite is semantics-preserving; the script verifies that the patched model's CPU output is
bit-identical to the original's before writing it.

Rewrites (each applied only where it is provably a no-op on CPU):
  * Reshape(allowzero=1) -> allowzero removed, when the target shape is a constant with no zeros.
    DirectML rejects allowzero=1 at session creation ("80070057 The parameter is incorrect").
    Seen in: Mini-BS-RoFormer V2.5 (HiDolen) - 88 nodes.
  * Dead nodes (no output used by any node or graph output) -> removed.
    DirectML's graph fusion fails on them (DmlGraphFusionHelper.cpp, 80070057).
    Seen in: SCNet int8 (audiojs) - 6 Transpose nodes.
  * GLU Split -> one Slice per output (directml/glu.py).
    DirectML's graph fusion computes Mul(a, Sigmoid(b)) as b * sigmoid(b) when a and b are outputs of
    the same Split (onnxruntime issue #32146). SCNet's output was off by 250x the signal peak; on a
    real song, BS-RoFormer SW's per-stem error was louder than the stems.
    Seen in: SCNet int8 - 21 nodes (all Splits, --all-splits); BS-RoFormer SW - 372; MVSep Mega 53.
  * 1-D ConvTranspose (iSTFT) -> MatMul + overlap-add (directml/istft.py).
    DirectML ran one batch item of SW's iSTFT in 3.1 s on a GTX 745.
  * TDR barriers (directml/barriers.py): split the model into many short GPU jobs. DirectML submits
    each fused part of the graph as one GPU job; Windows resets the driver if one job runs past ~2 s
    on GPUs that can't be preempted mid-job (GTX 600/700/900), which fails with 887A0006.
    Seen in: Mini-BS-RoFormer V2.5 and BS-RoFormer SW (one ~5-7 s job per chunk on a GTX 745).

Usage:
    python scripts/patch_onnx_for_directml.py INPUT.onnx OUTPUT.onnx [--report report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from directml.barriers import BarrierOptions, insert_barriers  # noqa: E402
from directml.glu import rewrite_glu_splits  # noqa: E402
from directml.graphinfo import GraphInfo, constant_values  # noqa: E402
from directml.istft import rewrite_convtranspose_1d  # noqa: E402
from directml.shapes import annotate, input_shapes  # noqa: E402


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


def cpu_outputs(model_bytes: bytes, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    return session.run(None, feed)


def parse_shape(text: str) -> tuple[str, list[int]]:
    name, _, dims = text.partition("=")
    return name, [int(d) for d in dims.split(",")]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-shape", action="append", default=[], metavar="NAME=D1,D2,...",
                        help="concrete shape for an input with symbolic dims")
    parser.add_argument("--all-splits", action="store_true", help="rewrite every Split as Slices, not only GLU Splits")
    parser.add_argument("--no-barriers", action="store_true", help="skip the TDR barrier pass")
    parser.add_argument("--budget-gflop", type=float, default=BarrierOptions.budget_gflop)
    parser.add_argument("--max-nodes", type=int, default=BarrierOptions.max_nodes)
    parser.add_argument("--max-cross-mib", type=float, default=BarrierOptions.max_cross_mib)
    parser.add_argument("--max-op-gflop", type=float, default=BarrierOptions.max_op_gflop)
    parser.add_argument("--max-conv-gflop", type=float, default=BarrierOptions.max_conv_gflop)
    parser.add_argument("--allow-rounding", type=float, default=0.0, metavar="REL",
                        help="accept a CPU difference up to REL x output peak instead of bit-identity")
    parser.add_argument("--report", type=Path, help="write a JSON summary for the manifest")
    parser.add_argument("--skip-verify", action="store_true", help="skip the CPU output check")
    args = parser.parse_args()

    original_bytes = args.input.read_bytes()
    model = onnx.load_from_string(original_bytes)
    graph = model.graph
    overrides = dict(parse_shape(s) for s in args.input_shape)

    allowzero = clear_reshape_allowzero(graph)
    dead = remove_dead_nodes(graph)
    shapes, shape_method = annotate(model, overrides)
    info = GraphInfo(model, shapes)
    glu = rewrite_glu_splits(model, info, all_splits=args.all_splits)
    istft = rewrite_convtranspose_1d(model, info)
    dead += remove_dead_nodes(graph)
    opts = BarrierOptions(args.budget_gflop, args.max_nodes, args.max_cross_mib, args.max_op_gflop, args.max_conv_gflop)
    barriers = None if args.no_barriers else insert_barriers(model, info, opts)
    if glu or istft or (barriers and barriers.barriers):
        del graph.value_info[:]  # declarations would be stale for rewritten tensors; ORT re-infers them
    print(f"shapes: {shape_method}; Reshape allowzero cleared: {allowzero}; dead nodes removed: {dead}; "
          f"{'Split' if args.all_splits else 'GLU Split'} -> Slice: {glu}; ConvTranspose -> MatMul+OLA: {istft}")
    if barriers:
        print(f"TDR barriers: {barriers.barriers} read-backs, {barriers.gates} gated tensors ({barriers.cuts} cuts, {barriers.split_ops} big ops split; "
              f"{barriers.total_gflop:.0f} GFLOP per run)")
        for line in barriers.split_details:
            print(f"  split {line}")

    onnx.checker.check_model(model)
    patched_bytes = model.SerializeToString()

    verify: dict = {"skipped": True}
    if not args.skip_verify:
        rng = np.random.default_rng(0)
        feed = {k: (rng.standard_normal(v) * 0.1).astype(np.float32) for k, v in input_shapes(model, overrides).items()}
        before = cpu_outputs(original_bytes, feed)
        after = cpu_outputs(patched_bytes, feed)
        worst = 0.0
        for a, b in zip(before, after):
            if a.shape != b.shape:
                print("ERROR: patched model's output shape differs from the original", file=sys.stderr)
                return 1
            peak = float(np.abs(a).max()) or 1.0
            worst = max(worst, float(np.abs(a.astype(np.float64) - b).max()) / peak)
        if worst == 0.0:
            print("CPU output bit-identical to the original")
        elif worst <= args.allow_rounding:
            print(f"CPU output within {worst:.2e} x peak of the original (allowed {args.allow_rounding:g})")
        else:
            print(f"ERROR: patched model's CPU output differs from the original ({worst:.2e} x peak)", file=sys.stderr)
            return 1
        verify = {"cpu_max_rel_diff": worst, "bit_identical": worst == 0.0}

    args.output.write_bytes(patched_bytes)
    digest = sha256(patched_bytes)
    print(f"Wrote {args.output} ({len(patched_bytes)} bytes, sha256 {digest})")
    if args.report:
        report = {
            "source": {"file": args.input.name, "bytes": len(original_bytes), "sha256": sha256(original_bytes)},
            "output": {"file": args.output.name, "bytes": len(patched_bytes), "sha256": digest},
            "shapes": shape_method,
            "rewrites": {
                "reshape_allowzero_cleared": allowzero,
                "dead_nodes_removed": dead,
                "split_to_slice": glu,
                "split_to_slice_scope": "all" if args.all_splits else "glu",
                "convtranspose_1d_to_matmul": istft,
                "tdr_barriers": barriers.barriers if barriers else 0,
                "tdr_cuts": barriers.cuts if barriers else 0,
                "tdr_gated_tensors": barriers.gates if barriers else 0,
                "big_ops_split": barriers.split_ops if barriers else 0,
            },
            "barrier_options": vars(opts) if barriers else None,
            "gflop_per_run": round(barriers.total_gflop, 1) if barriers else None,
            "verify": verify,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
