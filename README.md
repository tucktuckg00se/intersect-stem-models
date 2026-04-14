# Intersect Stem Models

This repo packages an open-source source-separation model for downstream use in the sampler VST plugin.

Current default target:

- Model family: BS RoFormer SW by jarredou
- Output stems: vocals, drums, bass, guitar, piano, other
- Runtime contract: stereo 44.1 kHz waveform chunk in, 6 stems waveform chunk out
- Runtime engine: ONNX Runtime on CPU or GPU

Current implemented artifact status:

- Working fallback export: spectral-core ONNX
- Fallback tensor contract: `spectrum [1, 2050, 257, 2] -> masked_stems [1, 6, 2050, 257, 2]`
- Waveform end-to-end ONNX export: not yet working in plain ONNX Runtime because the STFT/ISTFT path still needs an ONNX-native implementation

See `docs/sampler-plugin-integration.md` for the plugin-facing integration contract.
