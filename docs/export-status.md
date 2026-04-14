# Export Status

## Current result

The repo can now:

- download the pinned upstream BS RoFormer SW config and checkpoint,
- construct the PyTorch model locally,
- refactor the model into waveform frontend, separator core, and waveform backend stages,
- export a working **waveform-full ONNX** artifact,
- export a working **spectral-core ONNX** artifact,
- validate both artifacts against PyTorch output,
- persist export metadata under `artifacts/exports/bs-roformer-sw-6stem/`.

## Working artifact

The primary shippable artifact is:

- `artifacts/exports/bs-roformer-sw-6stem/bs_roformer_sw_6stem.onnx`

Waveform contract:

- input name: `audio`
- input shape: `[1, 2, 131072]`
- output name: `stems`
- output shape: `[1, 6, 2, 131072]`

Waveform parity result:

- max absolute error: `2.0265579223632812e-06`
- mean absolute error: `5.373570743927303e-08`

The secondary diagnostic artifact is:

- `artifacts/exports/bs-roformer-sw-6stem/bs_roformer_sw_6stem_spectral_core.onnx`

Current spectral-core contract:

- input name: `spectrum`
- input shape: `[1, 2050, 257, 2]`
- output name: `masked_stems`
- output shape: `[1, 6, 2050, 257, 2]`

Current parity result:

- max absolute error: `5.7220458984375e-05`
- mean absolute error: `6.69371445383149e-07`

## What changed

The original waveform export blockers were removed by replacing the waveform reconstruction path with ONNX-exportable real-valued operations:

- STFT export now uses real-valued output tensors instead of `view_as_real(view_as_complex(...))`
- Mask application stays in explicit real/imag tensor form
- Inverse STFT reconstruction uses precomputed inverse-DFT bases plus overlap-add via `conv_transpose1d`

This keeps the separator behavior intact while avoiding unsupported complex-valued lowering in plain ONNX export.

## Practical implication

The repo now has a plugin-ready waveform model for ONNX Runtime on CPU or GPU across macOS, Windows, and Linux.

The spectral-core artifact is still useful as a lower-level debugging or integration fallback, but it is no longer the primary deployment target.

One remaining implementation risk is future PyTorch removal of `torch.stft(..., return_complex=False)`. If that lands before export support improves, the frontend may need the same kind of explicit real-valued decomposition now used in the inverse path.
