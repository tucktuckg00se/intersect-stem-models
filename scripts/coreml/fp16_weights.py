"""Keeps fp16 weights fp16 through the conversion (MVSep Mega 53 stores 680M weights as fp16 and casts them
to fp32 when it runs). Two steps, neither changes the maths:

- take_fp16_weights / put_back_weights: while ONNX Runtime folds constants, fp16 weights that feed a Cast
  to float become graph inputs, so it can't fold them into an fp32 copy twice the size (past protobuf's
  2 GB for Mega).
- intersect::fp16_exact_weights (a coremltools pass, run last): an fp32 weight whose values are all exactly
  fp16 is stored as fp16 with a constexpr_cast (macOS 13+), which Core ML turns back into fp32 when it
  loads the model. Compute stays fp32; the package is half the size. Importing this module registers it.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

PASS_NAME = "intersect::fp16_exact_weights"


def take_fp16_weights(model):
    """Moves fp16 initializers that feed Cast(to=FLOAT) out of the graph (they become inputs). Returns them."""
    g = model.graph
    sources = {n.input[0] for n in g.node
               if n.op_type == "Cast" and any(a.name == "to" and a.i == TensorProto.FLOAT for a in n.attribute)}
    if not any(i.data_type == TensorProto.FLOAT16 and i.name in sources for i in g.initializer):
        return []
    hidden, keep = [], []
    for init in g.initializer:
        copy = TensorProto.FromString(init.SerializeToString())
        (hidden if init.data_type == TensorProto.FLOAT16 and init.name in sources else keep).append(copy)
    del g.initializer[:]
    g.initializer.extend(keep)
    for init in hidden:
        g.input.append(helper.make_tensor_value_info(init.name, TensorProto.FLOAT16, list(init.dims)))
    return hidden


def put_back_weights(model, hidden):
    names = {i.name for i in hidden}
    inputs = [onnx.ValueInfoProto.FromString(i.SerializeToString()) for i in model.graph.input if i.name not in names]
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    model.graph.initializer.extend(hidden)


@register_pass(namespace="intersect")
class fp16_exact_weights(AbstractGraphPass):
    min_elements = 1024   # small constants aren't worth an op

    def apply(self, prog):
        self.count = 0
        for f in prog.functions.values():
            self._apply_block(f)

    @block_context_manager
    def _apply_block(self, block):
        for op in list(block.operations):
            for inner in op.blocks:
                self._apply_block(inner)
            if op.op_type != "const":
                continue
            val = op.outputs[0].val
            if not isinstance(val, np.ndarray) or val.dtype != np.float32 or val.size < self.min_elements:
                continue
            half = val.astype(np.float16)
            if not np.array_equal(half.astype(np.float32), val):
                continue
            new = mb.constexpr_cast(source_val=half, output_dtype="fp32", before_op=op)
            block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new)
            block.remove_ops([op])
            self.count += 1
