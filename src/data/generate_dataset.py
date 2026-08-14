import numpy as np
import sys
from pathlib import Path

# make sure Python can find the physics module
sys.path.append(str(Path(__file__).resolve().parents[1] / "physics"))

from heat_solver import make_grid, run_simulation, compute_stable_dt
from boundary_conditions import apply_dirichlet_zero


# ─────────────────────────────────────────────
# Initial condition generators
# ─────────────────────────────────────────────

def random_single_blob(x, y, rng):
    """
    Generate a single Gaussian hot spot at a random position,
    with random amplitude and width.

    Parameters
    ----------
    x, y : np.ndarray  Grid coordinate vectors.
    rng  : np.random.Generator  Seeded random number generator.

    Returns
    -------
    T0 : np.ndarray, shape (N, N)
    """
    # keep blob center away from boundary so it doesn't start at T=0
    cx        = rng.uniform(0.2, 0.8)
    cy        = rng.uniform(0.2, 0.8)
    sigma     = rng.uniform(0.05, 0.15)
    amplitude = rng.uniform(0.5, 1.0)

    X, Y = np.meshgrid(x, y)
    T0   = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    # enforce Dirichlet BC on initial condition too
    T0 = apply_dirichlet_zero(T0)
    return T0


def random_multi_blob(x, y, rng, max_blobs=3):
    """
    Generate a superposition of 1 to max_blobs Gaussian hot spots,
    each with independent random parameters.

    Parameters
    ----------
    x, y      : np.ndarray  Grid coordinate vectors.
    rng       : np.random.Generator
    max_blobs : int  Maximum number of blobs (inclusive).

    Returns
    -------
    T0 : np.ndarray, shape (N, N)
    """
    n_blobs = rng.integers(1, max_blobs + 1)   # 1, 2, or 3
    X, Y   = np.meshgrid(x, y)
    T0     = np.zeros_like(X)

    for _ in range(n_blobs):
        cx        = rng.uniform(0.15, 0.85)
        cy        = rng.uniform(0.15, 0.85)
        sigma     = rng.uniform(0.04, 0.15)
        amplitude = rng.uniform(0.3, 1.0)
        T0       += amplitude * np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)
        )

    # clip to [0, 1] in case overlapping blobs push sum above 1
    T0 = np.clip(T0, 0.0, 1.0)
    T0 = apply_dirichlet_zero(T0)
    return T0


# ─────────────────────────────────────────────
# Core dataset generation
# ─────────────────────────────────────────────

def generate_dataset(
    n_samples   = 10_000,
    N           = 64,
    L           = 1.0,
    alpha       = 0.01,
    stride      = 10,
    max_blobs   = 3,
    seed        = 42,
    save_path   = "../data/heat_dataset.npz",
):
    """
    Generate a supervised dataset for single-step heat diffusion prediction.

    Each sample is a pair:
        input  = T(x, y, t)
        target = T(x, y, t + stride * dt)

    Parameters
    ----------
    n_samples : int    Number of (input, target) pairs to generate.
    N         : int    Grid resolution (N x N).
    L         : float  Domain size.
    alpha     : float  Thermal diffusivity.
    stride    : int    Number of solver steps between input and target.
    max_blobs : int    Maximum number of Gaussian blobs per sample.
    seed      : int    Random seed for reproducibility.
    save_path : str    Where to save the .npz file.

    Returns
    -------
    inputs  : np.ndarray, shape (n_samples, N, N)
    targets : np.ndarray, shape (n_samples, N, N)
    metadata: dict  Simulation parameters for reproducibility.
    """
    rng        = np.random.default_rng(seed)
    x, y, dx   = make_grid(N, L)
    dt, r      = compute_stable_dt(dx, alpha)

    print("=" * 50)
    print("Dataset Generation Parameters")
    print("=" * 50)
    print(f"  Samples       : {n_samples}")
    print(f"  Grid          : {N} x {N}")
    print(f"  alpha         : {alpha}")
    print(f"  dx            : {dx:.6f}")
    print(f"  dt            : {dt:.6f}")
    print(f"  Fourier r     : {r:.4f}  (<= 0.25 required)")
    print(f"  Stride        : {stride} steps")
    print(f"  Physical Δt   : {stride * dt:.6f} per CNN step")
    print(f"  Max blobs     : {max_blobs}")
    print(f"  Random seed   : {seed}")
    print("=" * 50)

    inputs  = np.zeros((n_samples, N, N), dtype=np.float32)
    targets = np.zeros((n_samples, N, N), dtype=np.float32)

    for i in range(n_samples):

        # progress indicator
        if i % 1000 == 0:
            print(f"  Generating sample {i:>6} / {n_samples} ...")

        # generate a random initial condition
        T0 = random_multi_blob(x, y, rng, max_blobs=max_blobs)

        # run solver for exactly `stride` steps
        # save_every=stride means we only store step 0 and step stride
        snaps, _, _, _ = run_simulation(
            T0, alpha, dx, n_steps=stride, save_every=stride
        )

        # snaps[0] = T at step 0  (input)
        # snaps[1] = T at step stride (target)
        inputs[i]  = snaps[0].astype(np.float32)
        targets[i] = snaps[1].astype(np.float32)

    # ── sanity checks before saving ──────────────────
    assert not np.any(np.isnan(inputs)),   "NaN found in inputs!"
    assert not np.any(np.isnan(targets)),  "NaN found in targets!"
    assert not np.any(np.isinf(inputs)),   "Inf found in inputs!"
    assert not np.any(np.isinf(targets)),  "Inf found in targets!"
    assert inputs.max()  <= 1.0 + 1e-5,   "Input values exceed 1.0!"
    assert targets.max() <= 1.0 + 1e-5,   "Target values exceed 1.0!"
    assert targets.max() <= inputs.max() + 1e-5, \
        "Targets hotter than inputs — unphysical for Dirichlet BC!"

    print("\n✓ All sanity checks passed.")

    # ── train / val / test split ──────────────────────
    # 80% train, 10% validation, 10% test
    # split BEFORE saving — never shuffle after splitting
    n_train = int(0.80 * n_samples)   # 8000
    n_val   = int(0.10 * n_samples)   # 1000
    # n_test  = remaining              # 1000

    idx = np.arange(n_samples)

    # important: use the seeded rng to shuffle, so split is reproducible
    rng_split = np.random.default_rng(seed + 1)
    rng_split.shuffle(idx)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    splits = {
        "train_inputs"  : inputs[train_idx],
        "train_targets" : targets[train_idx],
        "val_inputs"    : inputs[val_idx],
        "val_targets"   : targets[val_idx],
        "test_inputs"   : inputs[test_idx],
        "test_targets"  : targets[test_idx],
    }

    # ── metadata for reproducibility ─────────────────
    metadata = {
        "n_samples" : n_samples,
        "N"         : N,
        "L"         : L,
        "alpha"     : alpha,
        "dx"        : dx,
        "dt"        : dt,
        "r"         : r,
        "stride"    : stride,
        "seed"      : seed,
        "max_blobs" : max_blobs,
        "n_train"   : n_train,
        "n_val"     : n_val,
        "n_test"    : len(test_idx),
        "physical_dt_per_cnn_step" : stride * dt,
    }

    # ── save ──────────────────────────────────────────
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(save_path, **splits, **{
        f"meta_{k}": np.array(v) for k, v in metadata.items()
    })

    print(f"\n✓ Dataset saved to: {save_path}")
    print(f"  File size: {save_path.stat().st_size / 1e6:.1f} MB")

    # ── print split summary ───────────────────────────
    print("\nSplit summary:")
    print(f"  Train : {splits['train_inputs'].shape}")
    print(f"  Val   : {splits['val_inputs'].shape}")
    print(f"  Test  : {splits['test_inputs'].shape}")

    return splits, metadata


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    splits, metadata = generate_dataset(
        n_samples = 10_000,
        N         = 64,
        alpha     = 0.01,
        stride    = 10,
        max_blobs = 3,
        seed      = 42,
        save_path = "../data/heat_dataset.npz",
    )