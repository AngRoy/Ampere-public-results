from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PchipStatus:
    available: bool
    reason: str


def pchip_status() -> PchipStatus:
    try:
        from scipy.interpolate import PchipInterpolator  # noqa: F401

        return PchipStatus(available=True, reason="scipy.interpolate.PchipInterpolator available")
    except Exception as exc:  # pragma: no cover - environment dependent
        return PchipStatus(available=False, reason=f"PCHIP skipped because scipy is unavailable: {exc!r}")


def observed_xy(
    x: np.ndarray,
    observed_mask: np.ndarray,
    observed_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = observed_mask.astype(bool) & np.isfinite(observed_values)
    return x[valid].astype(float), observed_values[valid].astype(float), valid


def constant_fallback(n_rows: int, value: float = 0.0) -> np.ndarray:
    return np.full(n_rows, float(value), dtype=float)


def mean_per_branch_reconstruct(
    x: np.ndarray,
    observed_mask: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    _, y_obs, valid = observed_xy(x, observed_mask, observed_values)
    if len(y_obs) == 0:
        return constant_fallback(len(x))
    y_hat = constant_fallback(len(x), float(np.mean(y_obs)))
    y_hat[valid] = observed_values[valid]
    return y_hat


def zoh_reconstruct(
    x: np.ndarray,
    observed_mask: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    _, y_obs, valid = observed_xy(x, observed_mask, observed_values)
    if len(y_obs) == 0:
        return constant_fallback(len(x))
    series = pd.Series(np.where(valid, observed_values, np.nan), copy=False)
    y_hat = series.ffill().bfill().to_numpy(dtype=float, copy=True)
    y_hat[valid] = observed_values[valid]
    return y_hat


def linear_reconstruct(
    x: np.ndarray,
    observed_mask: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    x_obs, y_obs, valid = observed_xy(x, observed_mask, observed_values)
    if len(y_obs) == 0:
        return constant_fallback(len(x))
    if len(y_obs) == 1:
        y_hat = constant_fallback(len(x), y_obs[0])
    else:
        y_hat = np.interp(x.astype(float), x_obs, y_obs)
    y_hat[valid] = observed_values[valid]
    return y_hat


def pchip_reconstruct(
    x: np.ndarray,
    observed_mask: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    status = pchip_status()
    if not status.available:
        raise ImportError(status.reason)

    from scipy.interpolate import PchipInterpolator

    x_obs, y_obs, valid = observed_xy(x, observed_mask, observed_values)
    if len(y_obs) == 0:
        return constant_fallback(len(x))
    if len(y_obs) == 1:
        y_hat = constant_fallback(len(x), y_obs[0])
    else:
        interp = PchipInterpolator(x_obs, y_obs, extrapolate=False)
        y_hat = interp(x.astype(float))
        y_hat = np.asarray(y_hat, dtype=float)
        left = x.astype(float) < x_obs[0]
        right = x.astype(float) > x_obs[-1]
        y_hat[left] = y_obs[0]
        y_hat[right] = y_obs[-1]
    y_hat[valid] = observed_values[valid]
    return y_hat


def slope_aware_an_reconstruct(
    x: np.ndarray,
    observed_mask: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    """Bounded slope-aware interpolation used as the practical AN baseline.

    The implementation is a documented approximation, not a claim of matching
    any legacy AN formula. It estimates local slopes from neighboring observed
    points, zeros slopes at sign-changing extrema, clamps them to avoid large
    overshoot, and evaluates a cubic Hermite segment bounded by its endpoints.
    """

    x_obs, y_obs, valid = observed_xy(x, observed_mask, observed_values)
    n_rows = len(x)
    if len(y_obs) == 0:
        return constant_fallback(n_rows)
    if len(y_obs) == 1:
        y_hat = constant_fallback(n_rows, y_obs[0])
        y_hat[valid] = observed_values[valid]
        return y_hat

    dx = np.diff(x_obs)
    dy = np.diff(y_obs)
    secants = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)
    slopes = np.zeros_like(y_obs)
    slopes[0] = secants[0]
    slopes[-1] = secants[-1]

    for idx in range(1, len(y_obs) - 1):
        prev_slope = secants[idx - 1]
        next_slope = secants[idx]
        if prev_slope == 0 or next_slope == 0 or np.sign(prev_slope) != np.sign(next_slope):
            slopes[idx] = 0.0
            continue
        candidate = 0.5 * (prev_slope + next_slope)
        limit = 3.0 * min(abs(prev_slope), abs(next_slope))
        slopes[idx] = np.sign(candidate) * min(abs(candidate), limit)

    if len(secants) > 1:
        if np.sign(slopes[0]) != np.sign(secants[1]):
            slopes[0] = 0.0
        if np.sign(slopes[-1]) != np.sign(secants[-2]):
            slopes[-1] = 0.0

    x_all = x.astype(float)
    y_hat = np.empty(n_rows, dtype=float)
    y_hat[x_all <= x_obs[0]] = y_obs[0]
    y_hat[x_all >= x_obs[-1]] = y_obs[-1]

    interval = np.searchsorted(x_obs, x_all, side="right") - 1
    interval = np.clip(interval, 0, len(x_obs) - 2)
    interior = (x_all > x_obs[0]) & (x_all < x_obs[-1])

    for seg_idx in np.unique(interval[interior]):
        mask = interior & (interval == seg_idx)
        x0 = x_obs[seg_idx]
        x1 = x_obs[seg_idx + 1]
        h = x1 - x0
        if h == 0:
            y_hat[mask] = y_obs[seg_idx]
            continue
        t = (x_all[mask] - x0) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        y0 = y_obs[seg_idx]
        y1 = y_obs[seg_idx + 1]
        segment = h00 * y0 + h10 * h * slopes[seg_idx] + h01 * y1 + h11 * h * slopes[seg_idx + 1]
        low = min(y0, y1)
        high = max(y0, y1)
        y_hat[mask] = np.clip(segment, low, high)

    y_hat[valid] = observed_values[valid]
    return y_hat
