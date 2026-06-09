from __future__ import annotations

import copy

from ampere.neural.utils import require_torch

torch = require_torch()
from torch import nn


def make_ema_target_observer(student: nn.Module) -> nn.Module:
    """Clone a student observer as a frozen EMA target observer."""
    target = copy.deepcopy(student)
    target.load_state_dict(student.state_dict())
    set_target_requires_grad(target, False)
    target.eval()
    return target


def set_target_requires_grad(target: nn.Module, requires_grad: bool = False) -> None:
    for parameter in target.parameters():
        parameter.requires_grad_(requires_grad)


@torch.no_grad()
def update_ema_target(student: nn.Module, target: nn.Module, tau: float) -> None:
    if not 0.0 < float(tau) <= 1.0:
        raise ValueError(f"EMA tau must be in (0, 1], got {tau}")
    tau = float(tau)
    target_state = target.state_dict()
    student_state = student.state_dict()
    for name, target_value in target_state.items():
        student_value = student_state[name].detach()
        if target_value.dtype.is_floating_point:
            target_value.mul_(1.0 - tau).add_(student_value, alpha=tau)
        else:
            target_value.copy_(student_value)


def max_parameter_delta(model_a: nn.Module, model_b: nn.Module) -> float:
    max_delta = 0.0
    for first, second in zip(model_a.parameters(), model_b.parameters()):
        delta = torch.max(torch.abs(first.detach() - second.detach())).item()
        max_delta = max(max_delta, float(delta))
    return max_delta


def average_observer_outputs(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    """Average matching tensor outputs for student-teacher ensemble evaluation."""
    averaged: dict[str, object] = {}
    for key, value in first.items():
        other = second.get(key)
        if torch.is_tensor(value) and torch.is_tensor(other) and value.shape == other.shape:
            averaged[key] = 0.5 * (value + other)
        else:
            averaged[key] = value
    return averaged
