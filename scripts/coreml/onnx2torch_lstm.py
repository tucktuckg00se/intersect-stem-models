"""onnx2torch converter for ONNX LSTM (opset 7/14): constant W/R/B, layout 0, no sequence_lens, no peepholes.
Importing this module registers it."""
import torch
from torch import nn
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping, OnnxToTorchModule, OperationConverterResult


def _onnx_to_torch_gates(t: torch.Tensor) -> torch.Tensor:
    # ONNX stacks the gates as input, output, forget, cell; PyTorch as input, forget, cell, output.
    i, o, f, c = t.chunk(4, 0)
    return torch.cat([i, f, c, o], 0)


class OnnxLSTM(nn.Module, OnnxToTorchModule):
    def __init__(self, w: torch.Tensor, r: torch.Tensor, b, hidden: int, directions: int):
        super().__init__()
        self.hidden, self.directions = hidden, directions
        self.lstm = nn.LSTM(w.shape[2], hidden, num_layers=1, bidirectional=directions == 2)
        with torch.no_grad():
            for d in range(directions):
                suffix = "_reverse" if d == 1 else ""
                getattr(self.lstm, "weight_ih_l0" + suffix).copy_(_onnx_to_torch_gates(w[d]))
                getattr(self.lstm, "weight_hh_l0" + suffix).copy_(_onnx_to_torch_gates(r[d]))
                wb, rb = (b[d].chunk(2, 0) if b is not None else (torch.zeros(4 * hidden), torch.zeros(4 * hidden)))
                getattr(self.lstm, "bias_ih_l0" + suffix).copy_(_onnx_to_torch_gates(wb))
                getattr(self.lstm, "bias_hh_l0" + suffix).copy_(_onnx_to_torch_gates(rb))

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            out, (hn, cn) = self.lstm(x)
        else:
            out, (hn, cn) = self.lstm(x, (h0, c0))
        seq, batch = x.shape[0], x.shape[1]
        # ONNX Y is [seq, directions, batch, hidden]; PyTorch gives [seq, batch, directions * hidden].
        y = out.reshape(seq, batch, self.directions, self.hidden).permute(0, 2, 1, 3)
        return y, hn, cn


@add_converter(operation_type="LSTM", version=7)
@add_converter(operation_type="LSTM", version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    a = node.attributes
    if a.get("layout", 0) != 0 or a.get("input_forget", 0) or "activations" in a or "clip" in a:
        raise NotImplementedError("LSTM: only layout 0 with default activations is supported")
    names = list(node.input_values) + [""] * (8 - len(node.input_values))
    x, w, r, b, seq_lens, h0, c0, peep = names
    if seq_lens or peep:
        raise NotImplementedError("LSTM: sequence_lens and peepholes aren't supported")
    for name in (w, r) + ((b,) if b else ()):
        if name not in graph.initializers:
            raise NotImplementedError(f"LSTM: {name} must be constant")
    direction = a.get("direction", "forward")
    direction = direction.decode() if isinstance(direction, bytes) else direction
    if direction == "reverse":
        raise NotImplementedError("LSTM: reverse-only isn't supported")
    module = OnnxLSTM(graph.initializers[w].to_torch().float(), graph.initializers[r].to_torch().float(),
                      graph.initializers[b].to_torch().float() if b else None,
                      int(a["hidden_size"]), 2 if direction == "bidirectional" else 1)
    inputs = (x,) + ((h0, c0) if h0 and c0 else ())
    return OperationConverterResult(torch_module=module, onnx_mapping=OnnxMapping(inputs=inputs, outputs=node.output_values))
