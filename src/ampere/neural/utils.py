from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

try:  # pragma: no cover - exercised indirectly when torch is installed
    import torch
except Exception:  # pragma: no cover
    torch = None


TORCH_AVAILABLE = torch is not None


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for Stage 3B neural reconstruction but is not installed.")
    return torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> str:
    if torch is None:
        return "unavailable"
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class PowerScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "PowerScaler":
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return cls(mean=0.0, std=1.0)
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        if not np.isfinite(std) or std < 1e-9:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_numpy(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean

    def inverse_torch(self, values):
        torch_mod = require_torch()
        mean = torch_mod.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch_mod.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * std + mean
