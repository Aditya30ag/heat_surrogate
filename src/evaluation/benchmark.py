import numpy as np
import torch
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "physics"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))

from heat_solver import make_grid, run_simulation, compute_stable_dt
from boundary_conditions import apply_dirichlet_zero
from cnn import HeatSurrogateCNN


# ─────────────────────────────────────────────────────────────
# Solver benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_solver(
    N          = 64,
    alpha      = 0.01,
    stride     = 10,
    n_repeats  = 200,
    seed       = 0,
):
    """
    Benchmark the finite-difference solver for `stride` steps
    on a single 64x64 initial condition.

    Parameters
    ----------
    N         : int   Grid resolution.
    alpha     : float Thermal diffusivity.
    stride    : int   Number of solver steps to time
                      (must match what the CNN predicts).
    n_repeats : int   Number of timing repetitions.
    seed      : int   RNG seed for reproducible initial condition.

    Returns
    -------
    dict with timing statistics in seconds.
    """
    rng      = np.random.default_rng(seed)
    x        = np.linspace(0, 1, N)
    y        = np.linspace(0, 1, N)
    dx       = 1.0 / (N - 1)
    dt, r    = compute_stable_dt(dx, alpha)

    # fixed initial condition for all repeats
    cx, cy   = rng.uniform(0.2, 0.8, size=2)
    sigma    = rng.uniform(0.05, 0.15)
    X, Y     = np.meshgrid(x, y)
    T0       = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    T0       = apply_dirichlet_zero(T0)

    times = []

    for _ in range(n_repeats):
        T = T0.copy()
        t0 = time.perf_counter()

        # run exactly `stride` steps — same physical prediction as CNN
        for _ in range(stride):
            T_new = T.copy()
            T_new[1:-1, 1:-1] = (
                T[1:-1, 1:-1]
                + r * (
                    T[2:,   1:-1]
                  + T[:-2,  1:-1]
                  + T[1:-1, 2:]
                  + T[1:-1, :-2]
                  - 4 * T[1:-1, 1:-1]
                )
            )
            T = apply_dirichlet_zero(T_new)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean"   : float(np.mean(times)),
        "std"    : float(np.std(times)),
        "median" : float(np.median(times)),
        "min"    : float(np.min(times)),
        "max"    : float(np.max(times)),
        "n_repeats" : n_repeats,
        "stride" : stride,
        "N"      : N,
    }


# ─────────────────────────────────────────────────────────────
# CNN benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_cnn(
    model,
    device,
    N            = 64,
    batch_sizes  = [1, 8, 32, 64, 128],
    n_repeats    = 200,
    n_warmup     = 20,
    seed         = 0,
):
    """
    Benchmark CNN inference at multiple batch sizes.

    Warm-up runs are performed first and excluded from timing.
    Data transfer (CPU→device→CPU) is included in the timing
    for honest wall-clock measurement.

    Parameters
    ----------
    model       : HeatSurrogateCNN  Loaded, eval-mode model.
    device      : torch.device
    N           : int   Grid resolution.
    batch_sizes : list  Batch sizes to benchmark.
    n_repeats   : int   Timing repetitions per batch size.
    n_warmup    : int   Warm-up iterations (excluded from timing).
    seed        : int   RNG seed.

    Returns
    -------
    dict keyed by batch size, each with timing stats (per-sample seconds).
    """
    model.eval()
    rng    = np.random.default_rng(seed)
    results = {}

    for bs in batch_sizes:
        # create a fixed random batch of inputs
        inputs_np = rng.random((bs, 1, N, N)).astype(np.float32)

        # ── warm-up ───────────────────────────────────
        # run n_warmup forward passes before timing
        # this loads weights into cache, triggers JIT compilation, etc.
        with torch.no_grad():
            for _ in range(n_warmup):
                x   = torch.from_numpy(inputs_np).to(device)
                out = model(x)
                # if GPU: force synchronization before next iteration
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # ── timed runs ────────────────────────────────
        times = []
        with torch.no_grad():
            for _ in range(n_repeats):
                t0 = time.perf_counter()

                # include CPU→device transfer in timing
                x   = torch.from_numpy(inputs_np).to(device)
                out = model(x)

                # GPU: must synchronize before stopping timer
                # (GPU operations are async — without sync you measure
                #  dispatch time, not completion time)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                # include device→CPU transfer
                _ = out.cpu().numpy()

                t1 = time.perf_counter()
                times.append(t1 - t0)

        times = np.array(times)

        # per-sample time = total batch time / batch size
        results[bs] = {
            "batch_total_mean"    : float(np.mean(times)),
            "batch_total_std"     : float(np.std(times)),
            "per_sample_mean"     : float(np.mean(times)) / bs,
            "per_sample_std"      : float(np.std(times))  / bs,
            "per_sample_median"   : float(np.median(times)) / bs,
            "n_repeats"           : n_repeats,
            "n_warmup"            : n_warmup,
        }

        print(f"  Batch size {bs:>4}:  "
              f"batch={np.mean(times)*1000:.3f}ms  "
              f"per-sample={np.mean(times)/bs*1000:.4f}ms")

    return results