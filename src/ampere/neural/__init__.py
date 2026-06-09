"""Lightweight neural reconstruction experiments for AMPERE."""

from ampere.neural.datasets import NeuralSeries, WindowedReconstructionDataset, build_neural_series
from ampere.neural.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from ampere.neural.losses import LOSS_VARIANTS
    from ampere.neural.models import CausalTCNReconstructor
    from ampere.neural.train import NeuralTrainingConfig, train_and_predict
else:  # pragma: no cover - local Stage 3B environment has torch
    LOSS_VARIANTS = ()
    CausalTCNReconstructor = None
    NeuralTrainingConfig = None
    train_and_predict = None

__all__ = [
    "CausalTCNReconstructor",
    "LOSS_VARIANTS",
    "NeuralSeries",
    "NeuralTrainingConfig",
    "WindowedReconstructionDataset",
    "build_neural_series",
    "train_and_predict",
]
