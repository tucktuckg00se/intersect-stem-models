# Export Status

## Current result

The repo can now:

- download the pinned upstream BS RoFormer SW config and checkpoint,
- construct the PyTorch model locally,
- refactor the model into waveform frontend, separator core, and waveform backend stages,
- export a working **spectral-core ONNX** artifact,
- validate the spectral-core artifact against PyTorch output,
- persist export metadata and waveform failure diagnostics under `artifacts/exports/bs-roformer-sw-6stem/`.

## Working artifact

The current shippable artifact is:

- `artifacts/exports/bs-roformer-sw-6stem/bs_roformer_sw_6stem_spectral_core.onnx`

Current spectral-core contract:

- input name: `spectrum`
- input shape: `[1, 2050, 257, 2]`
- output name: `masked_stems`
- output shape: `[1, 6, 2050, 257, 2]`

Current parity result:

- max absolute error: `5.7220458984375e-05`
- mean absolute error: `6.69371445383149e-07`

## Confirmed blockers

The current environment reproduced two distinct export failures:

1. PyTorch dynamo exporter failure during graph decomposition
   - error: `expected compiled_fn to be GraphModule, got <class 'function'>`
2. Legacy exporter failure on STFT lowering
   - error: `STFT does not currently support complex types`

These failures are limited to the **waveform-full** export path. The separator core itself is now exportable.

## Practical implication

The repo now has a usable fallback artifact for plugin integration, but full waveform export still needs an ONNX-native spectral path:

- ONNX-native STFT
- complex-mask handling in ONNX-compatible real-valued tensors
- ONNX-native inverse-STFT reconstruction

Once that path exists, the existing export and validation scaffolding can be reused to promote the waveform-full artifact to the primary distribution target.
