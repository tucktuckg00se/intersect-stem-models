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
- Stem order: `bass, drums, other, vocals, guitar, piano`
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

## Release series

INTERSECT follows a release series, not a single tag. INTERSECT 0.18.1 and later read the manifest from the newest non-draft, non-prerelease release tagged `v0.2.N`. They find it through the GitHub releases API, and fall back to the tag built into the plugin. A patch release therefore reaches every plugin in the series without a plugin update. The plugin only fetches the manifest by itself; model files download when the user clicks.

Rules for a patch release (`v0.2.N`):

- Attach the complete `intersect_stem_models_manifest.json`, listing every model, including ones whose files live in older releases.
- Never rename a `model_id`. INTERSECT stores it in user settings.
- When a model file's content changes, give the file a new `release_asset_name`. INTERSECT treats a model as installed when a file with that name exists, so a new name is what makes users download the fix.
- List the model's previous file names in `artifacts.replaces`, and describe the change in one line in `artifacts.update_note`. INTERSECT versions that read them show an UPDATE button instead of DOWNLOAD, keep using the old file until the user updates, and delete it afterwards. Only list a file in `replaces` if it has the same inputs and outputs as the new one. Older plugins ignore both fields and show DOWNLOAD.
- Only use runtime kinds and fields that plugins in the series already understand. Plugins skip entries they can't parse, and ignore unknown optional fields.
- Never delete a release, or an asset from one, while any published manifest still points at it. Unchanged models stay in the release where they were first uploaded (SCNet in `v0.2.1`, MVSep Mega 53 in `v0.2.0`).
- Attach `THIRD_PARTY_LICENSES.txt` to every release that hosts model files, and add an entry for each new model.

`scripts/set_manifest_artifact.py` updates one model's entry (file name, URL, size, SHA-256, patch report, `replaces`, `update_note`).

A manifest change that older plugins in the series would misread needs a new minor series (`v0.3.0`) and a plugin release that reads it.

## DirectML builds

Every model INTERSECT runs on Windows GPUs goes through `scripts/patch_onnx_for_directml.py` and `scripts/validate_directml.py` before it is published. The patch script refuses to write a file whose CPU output isn't bit-identical to the original's. The manifest records each patch under `artifacts.directml_patch`: the source URL and SHA-256, and the rewrites applied with their counts.

What the patch script fixes:

- **DirectML rejects the file** ("80070057 The parameter is incorrect"): `Reshape` nodes with `allowzero=1`, and dead nodes.
- **DirectML computes wrong values:** its graph fusion turns a GLU (`Split` → `Sigmoid(b)` → `Mul(a, ·)`) into `b·sigmoid(b)`. This is [onnxruntime#32146](https://github.com/microsoft/onnxruntime/issues/32146), open and unfixed in ORT 1.22–1.28. On a real song it made BS-RoFormer SW's stems wrong by more than the stems' own level. The GLU `Split`s are rewritten as `Slice`s.
- **Windows resets the GPU driver (TDR, `887A0006`):**
  - **Cause:** DirectML sends each fused part of the graph to the GPU as one job. For BS-RoFormer models that is the whole model, about 5–7 s per chunk on a GTX 745. Windows resets the driver when one job runs longer than about 2 s on GPUs that can't be paused mid-job (NVIDIA GTX 600/700/900).
  - **Fix:** the script inserts CPU read-back "barriers" so the model runs as many short GPU jobs. Each job stays under 0.5 s on a GTX 745.
  - **Slow `ConvTranspose`:** DirectML's 1-D `ConvTranspose` (iSTFT) alone took 3 s for one signal, so it is rewritten as MatMul + overlap-add.
  - The barriers cost about 2–3% on CPU and about 6–12% on CUDA.

`scripts/validate_directml.py` runs on Windows with `onnxruntime-directml`. It checks the following, then writes a JSON report per model:

- The session really runs on DirectML.
- The output matches the CPU, on random input and, for waveform models, on real-song chunks (`--wav`).
- No single GPU job exceeds 0.5 s (measured from ORT profiling).
- Windows logged no GPU timeout: Display 4101, nvlddmkm 153, or LiveKernelEvent 141/117.

Run it on the slowest GPU you support. The reference is a GTX 745, whose compute preemption is `DMA_BUFFER`.

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
