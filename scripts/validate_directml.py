"""Validate ONNX stem models on ONNX Runtime's DirectML provider (Windows) before publishing them.

For each model it checks, with the session settings INTERSECT uses on Windows:
  * the session really runs on DirectML (ORT's Python API silently falls back to CPU otherwise);
  * DirectML output matches the CPU provider (random input, and real audio chunks for waveform models
    when --wav is given): max abs diff <= --max-rel-diff x output peak. GPUs differ from the CPU by
    normal rounding: ~1e-4 x peak on real music on an NVIDIA GTX 745, ~1e-2 on an Intel UHD iGPU (same
    with or without TDR barriers). The bugs this guards against (e.g. onnxruntime#32146) give errors
    larger than the signal, so the default 3e-2 still separates them by more than 30x;
  * on GPUs whose compute work can only be preempted between DMA buffers or dispatches (NVIDIA Kepler/
    Maxwell; the TDR risk), the longest single GPU job is <= --max-job seconds. GPUs with finer
    preemption can't trip the timeout this way, so there it is reported only. With TDR barriers each job
    ends in a GPU->CPU read-back (MemcpyToHost) whose wait time is the job's GPU time; the last job ends
    in the output read-back after the last node. Without barriers there are no read-backs, so job
    lengths can't be measured and only the TDR check applies;
  * Windows logged no GPU timeout (TDR) during the run: System "Display" 4101, "nvlddmkm" 153, or a
    LiveKernelEvent 141/117 report;
  * peak GPU memory of this process (DXGI QueryVideoMemoryInfo), for information.

Run it on the slowest GPU you support (a GTX 745, compute preemption DMA_BUFFER, is the reference).

usage: python scripts/validate_directml.py MODEL.onnx [MODEL.onnx ...] [--wav SONG.wav] [--report-dir DIR]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ── DXGI (adapter, preemption, process GPU memory) ──────────────────────────────────────────────────

_COMPUTE_PREEMPTION = ["DMA_BUFFER", "DISPATCH", "THREAD_GROUP", "THREAD", "INSTRUCTION"]


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    u = uuid.UUID(text)
    g = _GUID()
    g.Data1, g.Data2, g.Data3 = u.fields[0], u.fields[1], u.fields[2]
    g.Data4[:] = list(u.bytes[8:])
    return g


class _DESC2(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", ctypes.c_uint32), ("LuidHigh", ctypes.c_int32), ("Flags", ctypes.c_uint),
                ("GraphicsPreemptionGranularity", ctypes.c_uint), ("ComputePreemptionGranularity", ctypes.c_uint)]


class _MEMINFO(ctypes.Structure):
    _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64), ("CurrentReservation", ctypes.c_uint64)]


def _vcall(obj, index, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])
    return lambda *a: fn(obj, *a)


class GpuAdapter:
    """The hardware adapter with the most dedicated memory (what ORT's DirectML provider picks by default)."""

    def __init__(self) -> None:
        self.name, self.preemption, self.dedicated_mib, self._adapter3 = "unknown", "unknown", 0, None
        if sys.platform != "win32":
            return
        pp = ctypes.POINTER(ctypes.c_void_p)
        factory = ctypes.c_void_p()
        if ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(_guid("770aae78-f26f-4dba-a829-253c83d1b387")), ctypes.byref(factory)):
            return
        best, index = None, 0
        while True:
            adapter = ctypes.c_void_p()
            if _vcall(factory, 12, ctypes.c_uint, pp)(index, ctypes.byref(adapter)):  # EnumAdapters1
                break
            a2 = ctypes.c_void_p()
            _vcall(adapter, 0, ctypes.POINTER(_GUID), pp)(ctypes.byref(_guid("0AA1AE0A-FA0E-4B84-8644-E05FF8E5ACB5")), ctypes.byref(a2))
            d = _DESC2()
            _vcall(a2, 11, ctypes.POINTER(_DESC2))(ctypes.byref(d))  # GetDesc2
            if not (d.Flags & 2) and d.VendorId != 0x1414 and (best is None or d.DedicatedVideoMemory > best[0].DedicatedVideoMemory):
                best = (d, adapter)
            index += 1
        if best is None:
            return
        d, adapter = best
        self.name = d.Description
        self.dedicated_mib = d.DedicatedVideoMemory // 2**20
        g = d.ComputePreemptionGranularity
        self.preemption = _COMPUTE_PREEMPTION[g] if g < len(_COMPUTE_PREEMPTION) else str(g)
        a3 = ctypes.c_void_p()
        if not _vcall(adapter, 0, ctypes.POINTER(_GUID), pp)(ctypes.byref(_guid("645967A4-1392-4310-A798-8053CE3E93FD")), ctypes.byref(a3)):
            self._adapter3 = a3

    def process_usage_mib(self) -> float:
        if self._adapter3 is None:
            return 0.0
        info = _MEMINFO()
        _vcall(self._adapter3, 14, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(_MEMINFO))(0, 0, ctypes.byref(info))  # QueryVideoMemoryInfo, LOCAL
        return info.CurrentUsage / 2**20


class PeakSampler:
    def __init__(self, adapter: GpuAdapter) -> None:
        self.adapter, self.peak, self._stop = adapter, 0.0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.adapter.process_usage_mib())
            time.sleep(0.05)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# ── Windows event log (GPU timeouts) ────────────────────────────────────────────────────────────────

def tdr_events_since(start: datetime) -> list[str]:
    if sys.platform != "win32":
        return []
    since = start.strftime("%Y-%m-%dT%H:%M:%S")
    ps = (
        f"$s=[datetime]'{since}';"
        "Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$s} -ErrorAction SilentlyContinue | "
        "Where-Object { ($_.ProviderName -eq 'Display' -and $_.Id -eq 4101) -or ($_.ProviderName -eq 'nvlddmkm' -and $_.Id -eq 153) } | "
        "ForEach-Object { \"$($_.TimeCreated.ToString('HH:mm:ss')) $($_.ProviderName) $($_.Id)\" };"
        "Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Windows Error Reporting';StartTime=$s} -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Message -match 'LiveKernelEvent' -and $_.Message -match 'P1: (141|117)' } | "
        "ForEach-Object { \"$($_.TimeCreated.ToString('HH:mm:ss')) LiveKernelEvent\" }"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


# ── Sessions ────────────────────────────────────────────────────────────────────────────────────────

def make_session(path: str, gpu: bool, profile_dir: str | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()  # mirrors INTERSECT's Windows settings (StemSeparationJob.cpp)
    so.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(profile_dir, "prof")
    providers = [("DmlExecutionProvider", {}), "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(path, so, providers=providers)
    want = "DmlExecutionProvider" if gpu else "CPUExecutionProvider"
    if session.get_providers()[0] != want:
        raise RuntimeError(f"session fell back to {session.get_providers()} (wanted {want})")
    return session


def longest_job(profile_path: str) -> tuple[float, float, int]:
    """(longest MemcpyToHost wait, tail after the last node, read-back count) for the last model_run."""
    with open(profile_path) as f:
        events = json.load(f)
    run = [e for e in events if e.get("cat") == "Session" and e["name"] == "model_run"][-1]
    start, end = run["ts"], run["ts"] + run["dur"]
    nodes = sorted((e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time") and start <= e["ts"] <= end),
                   key=lambda e: e["ts"])
    waits = [e["dur"] / 1e6 for e in nodes if e["args"].get("op_name") == "MemcpyToHost"]
    tail = (end - (nodes[-1]["ts"] + nodes[-1]["dur"])) / 1e6 if nodes else run["dur"] / 1e6
    return (max(waits) if waits else 0.0), tail, len(waits)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch, width, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if width == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        a = (((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8) >> 8).astype(np.float32) / 8388608
    else:
        raise SystemExit(f"{path}: only 16/24-bit PCM WAV is supported")
    a = a.reshape(-1, ch)
    return (a if ch == 2 else np.repeat(a[:, :1], 2, axis=1)), sr


def validate(path: Path, args, adapter: GpuAdapter) -> dict:
    print(f"\n== {path.name}")
    result: dict = {"model": path.name, "adapter": adapter.name, "compute_preemption": adapter.preemption}
    started = datetime.now()
    cpu = make_session(str(path), gpu=False)
    inp = cpu.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    feeds = [("random", (np.random.default_rng(0).standard_normal(shape) * 0.1).astype(np.float32))]
    if args.wav and len(shape) == 3 and shape[1] == 2:
        audio, sr = read_wav(args.wav)
        if sr != 44100:
            print(f"  note: {args.wav.name} is {sr} Hz; the models expect 44100 Hz")
        n = shape[2]
        for frac in (0.25, 0.5, 0.75):
            start = int((len(audio) - n) * frac)
            feeds.append((f"wav@{start / sr:.0f}s", audio[start:start + n].T[None].copy()))
    elif args.wav:
        print("  note: real-audio check only applies to waveform models; using random input")

    t = time.perf_counter()
    profile_dir = tempfile.mkdtemp(prefix="dml_validate_")
    gpu = make_session(str(path), gpu=True, profile_dir=profile_dir)
    result["gpu_session_s"] = round(time.perf_counter() - t, 1)
    diffs, times = {}, []
    with PeakSampler(adapter) as sampler:
        for label, x in feeds:
            ref = cpu.run(None, {inp.name: x})
            for _ in range(args.runs):
                t = time.perf_counter()
                out = gpu.run(None, {inp.name: x})
                times.append(time.perf_counter() - t)
            worst = 0.0
            for r, o in zip(ref, out):
                if not np.isfinite(o).all():
                    worst = float("inf")
                    break
                worst = max(worst, float(np.abs(r - o).max()) / (float(np.abs(r).max()) or 1.0))
            diffs[label] = worst
            print(f"  {label:12s} DML vs CPU max diff {worst:.2e} x peak")
    job, tail, readbacks = longest_job(gpu.end_profiling())
    events = tdr_events_since(started)
    result.update({
        "dml_vs_cpu_rel_diff": diffs, "run_s_median": round(float(np.median(times)), 3),
        "longest_job_s": round(max(job, tail), 3), "longest_readback_wait_s": round(job, 3), "tail_s": round(tail, 3),
        "gpu_jobs": readbacks + 1, "peak_gpu_mib": round(sampler.peak), "tdr_events": events,
    })
    checks = {
        "matches_cpu": max(diffs.values()) <= args.max_rel_diff,
        # Enforced only where one long job can trip the TDR, and only measurable with read-backs.
        "short_jobs": (max(job, tail) <= args.max_job) if readbacks and adapter.preemption in ("DMA_BUFFER", "DISPATCH") else None,
        "no_tdr": not events,
    }
    result["checks"], result["pass"] = checks, all(v is not False for v in checks.values())
    print(f"  run {result['run_s_median']:.2f}s, {result['gpu_jobs']} GPU jobs, longest {result['longest_job_s']:.3f}s "
          f"(read-back {job:.3f}s, tail {tail:.3f}s), peak GPU memory {result['peak_gpu_mib']} MiB")
    if not readbacks:
        print("  note: no GPU->CPU read-backs, so single-job lengths can't be measured; relying on the TDR check")
    elif checks["short_jobs"] is None:
        print(f"  note: compute preemption {adapter.preemption}: long jobs can't trip the TDR here, job length is informational")
    print(f"  TDR events: {events or 'none'}")
    print(f"  {'PASS' if result['pass'] else 'FAIL'} {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--wav", type=Path, help="44.1 kHz 16/24-bit WAV for the real-audio check (waveform models)")
    parser.add_argument("--runs", type=int, default=2, help="GPU runs per input")
    parser.add_argument("--max-job", type=float, default=0.5, help="seconds; longest allowed single GPU job")
    parser.add_argument("--max-rel-diff", type=float, default=3e-2, help="allowed DML vs CPU max abs diff, x output peak")
    parser.add_argument("--report-dir", type=Path, help="write <model>.validate.json here")
    args = parser.parse_args()

    print(f"onnxruntime {ort.__version__}, providers {ort.get_available_providers()}")
    adapter = GpuAdapter()
    print(f"GPU: {adapter.name}, {adapter.dedicated_mib} MiB dedicated, compute preemption {adapter.preemption}")
    results = []
    for path in args.models:
        try:
            results.append(validate(path, args, adapter))
        except Exception as e:  # noqa: BLE001 - report and continue with the next model
            print(f"  FAIL: {type(e).__name__}: {str(e)[:2000]}")
            results.append({"model": path.name, "pass": False, "error": str(e)[:2000]})
        if args.report_dir:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / f"{path.stem}.validate.json").write_text(json.dumps(results[-1], indent=2) + "\n")
    print("\nSummary: " + ", ".join(f"{r['model']} {'PASS' if r['pass'] else 'FAIL'}" for r in results))
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
