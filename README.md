# Intersect Stem Models

This repo packages an open-source source-separation model for downstream use in the sampler VST plugin.

Current default target:

- Model family: BS RoFormer SW by jarredou
- Output stems: bass, drums, other, vocals, guitar, piano
- Runtime contract: stereo 44.1 kHz waveform chunk in, 6 stems waveform chunk out
- Runtime engine: ONNX Runtime on CPU or GPU

Primary shipped artifact:

- Waveform ONNX: `bs_roformer_sw_6stem.onnx`
- Release: `v0.1.0`
- Release asset URL: `https://github.com/tucktuckg00se/intersect-stem-models/releases/download/v0.1.0/bs_roformer_sw_6stem.onnx`
- SHA256: `718c3304a4a051bc8b1adfa776ce0aeab54165e207d0a5f5890d683c16188e5e`

Waveform runtime contract:

- Input name: `audio`
- Input shape: `[1, 2, 131072]`
- Output name: `stems`
- Output shape: `[1, 6, 2, 131072]`
- Sample rate: `44.1 kHz`
- Stem order: `bass, drums, other, vocals, guitar, piano`

Distribution contract:

- `models/manifest.json` is the machine-readable source of truth for the download URL and SHA256
- plugin-side download and verification should follow the manifest, not hard-coded URLs
- the sampler plugin should consume the waveform artifact by default

Secondary artifact:

- A spectral-core ONNX export also exists for lower-level debugging or custom STFT/ISTFT ownership
- Intersect should not use the spectral-core path unless it intentionally wants to own the spectral frontend and backend

See `docs/sampler-plugin-integration.md` for the plugin-facing integration contract and `models/manifest.json` for the published artifact metadata.
