# Sampler Plugin Integration

## Runtime contract

The default packaged model is **BS RoFormer SW by jarredou**. The repo now ships a validated **waveform-facing ONNX model** for plain ONNX Runtime.

Primary target contract:

- Input tensor name: `audio`
- Input tensor shape: `[1, 2, 131072]`
- Output tensor name: `stems`
- Output tensor shape: `[1, 6, 2, 131072]`

Current fallback contract:

- Input tensor name: `spectrum`
- Input tensor shape: `[1, 2050, 257, 2]`
- Output tensor name: `masked_stems`
- Output tensor shape: `[1, 6, 2050, 257, 2]`
- Tensor layout: real-valued complex pairs in the last dimension
- Stem order: `vocals, drums, bass, guitar, piano, other`
- Native sample rate: `44.1 kHz`
- Default deployment chunk size: `131072` samples

## Plugin responsibilities

The plugin owns all host-facing audio adaptation:

- Resample incoming audio to `44.1 kHz`
- Coerce to stereo before inference
- Split long audio into fixed model chunks
- Use `131072` input samples per inference call unless a future manifest version changes it
- Use overlap-add to merge output chunks back into continuous stem signals when using the waveform artifact
- Select ONNX Runtime execution provider for CPU or GPU

Recommended chunk aggregation:

- Use `50%` overlap between adjacent chunks
- Use a Hann window for overlap-add
- Keep chunk boundaries aligned in samples, not in musical time
- If the host needs a 4-stem view, fold down with `other_4stem = guitar + piano + other`

## Session setup

Use plain ONNX Runtime. CPU must work on macOS, Windows, and Linux. GPU providers are optional acceleration paths and must preserve the same tensor contract.

Recommended session settings:

- Prefer `CPUExecutionProvider` as the baseline path
- Enable one GPU provider only when the host can confirm it is available
- Reuse the same `InferenceSession` across calls
- Reuse preallocated host buffers where practical

## Optional spectral-core fallback

The repo also exports a lower-level spectral-core artifact:

- Input tensor name: `spectrum`
- Input tensor shape: `[1, 2050, 257, 2]`
- Output tensor name: `masked_stems`
- Output tensor shape: `[1, 6, 2050, 257, 2]`
- Tensor layout: real-valued complex pairs in the last dimension

Only use this path if the plugin wants to own STFT and inverse-STFT itself. For normal integration, use the waveform artifact.

If the plugin consumes the spectral-core artifact, it must implement the same STFT and inverse-STFT settings as the model config:

- `n_fft = 2048`
- `hop_length = 441`
- `win_length = 2048`
- Hann window
- stereo frequency packing into `2050 = 1025 * 2` bins

## Artifact flow

Repo-local source assets:

- `artifacts/sources/bs-roformer-sw-6stem/source/BS-Rofo-SW-Fixed.yaml`
- `artifacts/sources/bs-roformer-sw-6stem/source/BS-Rofo-SW-Fixed.ckpt`

Repo-local export output:

- `artifacts/exports/bs-roformer-sw-6stem/bs_roformer_sw_6stem.onnx`
- `artifacts/exports/bs-roformer-sw-6stem/bs_roformer_sw_6stem_spectral_core.onnx`

Release distribution:

- Publish the ONNX file as a release asset
- Copy the release URL and SHA256 into `models/manifest.json`
- Plugin-side download logic should trust the manifest, not hard-coded URLs

## Current status

The upstream training config uses a much larger chunk size than is practical for direct ONNX export on modest machines. The exporter therefore defaults to a smaller deployment chunk size of `131072` samples.

Current repo state:

- the **waveform-full artifact** exports and validates successfully,
- the **spectral-core artifact** also exports and validates successfully,
- `models/manifest.json` records both artifacts and exposes the waveform model as the primary ONNX payload.

Recommended integration path:

- read `models/manifest.json`
- download `artifacts.release_asset_url` once release assets are published
- verify the ONNX SHA256 from the manifest
- run inference against the waveform contract above
