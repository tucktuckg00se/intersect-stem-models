"""Checks a converted .mlpackage against the ONNX Runtime reference that convert_coreml.py saved next to it,
and times it. Runs on macOS only (Core ML predictions).

usage: python verify_coreml.py <model.mlpackage> [cpu_and_gpu|cpu_only|all]
"""
import sys, time
import numpy as np
import coremltools as ct

UNITS = {"cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu_only": ct.ComputeUnit.CPU_ONLY, "all": ct.ComputeUnit.ALL}


def main():
    pkg = sys.argv[1].rstrip("/")
    units = sys.argv[2] if len(sys.argv) > 2 else "cpu_and_gpu"
    x = np.load(pkg + ".ref_input.npy")
    ref = np.load(pkg + ".ref_output.npy")
    t = time.time()
    model = ct.models.MLModel(pkg, compute_units=UNITS[units])
    load = time.time() - t
    spec = model.get_spec()
    in_name, out_name = spec.description.input[0].name, spec.description.output[0].name
    times = []
    for _ in range(3):
        t = time.time()
        y = model.predict({in_name: x})[out_name]
        times.append(time.time() - t)
    y = np.asarray(y, dtype=np.float32).reshape(ref.shape)
    err = ref.astype(np.float64) - y
    snr = 10 * np.log10((ref.astype(np.float64) ** 2).sum() / max((err ** 2).sum(), 1e-300))
    ok = bool(np.isfinite(y).all()) and snr >= 60
    print(f"RESULT {pkg.split('/')[-1]} units={units} load_s={load:.1f} predict_s={min(times):.2f} "
          f"snr_db={snr:.1f} max_abs_err={np.abs(err).max():.2e} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
