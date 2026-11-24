"""onnx2torch converter for ONNX STFT (opset 17): real signal, constant frame_step / window / frame_length.
Importing this module registers it.

No torch.stft (coremltools would need its own complex ops): frames are cut by a reshape (when the frame
length is a whole number of hops, as in every model here) or a gather, then one MatMul with a windowed
DFT basis gives the real and imaginary parts. Inner tensors stay small (frames x frame length), which
keeps them inside what the Core ML GPU path accepts."""
import numpy as np
import torch
from torch import nn
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping, OnnxToTorchModule, OperationConverterResult


class OnnxSTFT(nn.Module, OnnxToTorchModule):
    def __init__(self, frame_step: int, frame_length: int, window: np.ndarray, onesided: bool):
        super().__init__()
        self.step, self.length = frame_step, frame_length
        bins = frame_length // 2 + 1 if onesided else frame_length
        t = np.arange(frame_length, dtype=np.float64)[:, None]
        k = np.arange(bins, dtype=np.float64)[None, :]
        angle = 2.0 * np.pi * ((t * k) % frame_length) / frame_length   # exact phase for large t*k
        w = window.astype(np.float64)[:, None]
        basis = np.concatenate([w * np.cos(angle), -w * np.sin(angle)], axis=1)   # [length, 2 * bins]
        self.bins = bins
        self.register_buffer("basis", torch.from_numpy(basis.astype(np.float32)))

    def forward(self, signal, *unused):
        x = signal.squeeze(-1) if signal.dim() == 3 else signal   # [batch, samples(, 1)] -> [batch, samples]
        hops_per_frame = self.length // self.step
        if self.length % self.step == 0 and x.shape[-1] % self.step == 0:
            # Only constant reshapes and slices: no shape arithmetic for coremltools to trace.
            blocks = x.unflatten(-1, (-1, self.step))   # [batch, frames + hops_per_frame - 1, step]
            framed = torch.cat([blocks[:, j:(j - hops_per_frame + 1) or None] for j in range(hops_per_frame)], dim=2)
        else:
            frames = (x.shape[-1] - self.length) // self.step + 1
            index = torch.arange(frames)[:, None] * self.step + torch.arange(self.length)[None, :]
            framed = x[:, index]
        spectrum = torch.matmul(framed, self.basis)   # [batch, frames, 2 * bins]
        return spectrum.unflatten(-1, (2, self.bins)).transpose(-1, -2)   # [batch, frames, bins, 2]


def _constant(graph: OnnxGraph, name: str) -> np.ndarray:
    if name in graph.initializers:
        return graph.initializers[name].to_torch().numpy()
    raise NotImplementedError(f"STFT input {name} must be a constant (run constant folding first)")


@add_converter(operation_type="STFT", version=17)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    inputs = list(node.input_values)
    frame_step = int(_constant(graph, inputs[1]))
    window = _constant(graph, inputs[2]) if len(inputs) > 2 and inputs[2] else None
    if len(inputs) > 3 and inputs[3]:
        frame_length = int(_constant(graph, inputs[3]))
    elif window is not None:
        frame_length = window.shape[0]
    else:
        raise NotImplementedError("STFT needs a window or a frame_length")
    if window is None:
        window = np.ones(frame_length, dtype=np.float32)
    onesided = bool(node.attributes.get("onesided", 1))
    return OperationConverterResult(
        torch_module=OnnxSTFT(frame_step, frame_length, window, onesided),
        onnx_mapping=OnnxMapping(inputs=(inputs[0],), outputs=node.output_values),
    )
