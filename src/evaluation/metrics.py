import numpy as np
import torch


def mse(pred, target):
    """
    Mean Squared Error over the full spatial field.

    Parameters
    ----------
    pred   : np.ndarray, shape (N, N) or (B, N, N)
    target : np.ndarray, shape (N, N) or (B, N, N)

    Returns
    -------
    float or np.ndarray  MSE per sample if batched.
    """
    err = pred - target
    if pred.ndim == 2:
        return float(np.mean(err**2))
    return np.mean(err**2, axis=(-2, -1))   # per sample


def mae(pred, target):
    """Mean Absolute Error."""
    err = np.abs(pred - target)
    if pred.ndim == 2:
        return float(np.mean(err))
    return np.mean(err, axis=(-2, -1))


def rmse(pred, target):
    """Root Mean Squared Error — same units as temperature."""
    return np.sqrt(mse(pred, target))


def relative_l2(pred, target, eps=1e-8):
    """
    Relative L2 error, normalized by the L2 norm of the target.
    eps prevents division by zero for near-zero fields.

    Returns value in [0, inf) — multiply by 100 for percentage.
    """
    if pred.ndim == 2:
        num = np.sqrt(np.sum((pred - target)**2))
        den = np.sqrt(np.sum(target**2)) + eps
        return float(num / den)

    # batched
    num = np.sqrt(np.sum((pred - target)**2, axis=(-2, -1)))
    den = np.sqrt(np.sum(target**2,          axis=(-2, -1))) + eps
    return num / den


def max_absolute_error(pred, target):
    """Maximum absolute error — worst-case pixel."""
    err = np.abs(pred - target)
    if pred.ndim == 2:
        return float(np.max(err))
    return np.max(err, axis=(-2, -1))


def compute_all_metrics(pred, target):
    """
    Compute all metrics for a batch of predictions.

    Parameters
    ----------
    pred   : np.ndarray, shape (B, N, N)
    target : np.ndarray, shape (B, N, N)

    Returns
    -------
    dict with per-sample arrays and aggregate statistics.
    """
    assert pred.shape == target.shape, \
        f"Shape mismatch: pred {pred.shape} vs target {target.shape}"

    per_sample = {
        "mse"        : mse(pred, target),
        "mae"        : mae(pred, target),
        "rmse"       : rmse(pred, target),
        "rel_l2"     : relative_l2(pred, target),
        "max_ae"     : max_absolute_error(pred, target),
    }

    # mean error field — spatial map of where errors concentrate
    per_sample["mean_error_field"] = np.mean(pred - target, axis=0)
    per_sample["mean_abs_error_field"] = np.mean(np.abs(pred - target), axis=0)

    # aggregate statistics
    aggregate = {}
    for k, v in per_sample.items():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            aggregate[k] = {
                "mean"   : float(np.mean(v)),
                "std"    : float(np.std(v)),
                "median" : float(np.median(v)),
                "min"    : float(np.min(v)),
                "max"    : float(np.max(v)),
                "p95"    : float(np.percentile(v, 95)),
            }

    return per_sample, aggregate