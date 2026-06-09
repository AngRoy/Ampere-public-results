from __future__ import annotations

from dataclasses import dataclass

from ampere.neural.observer_dataset import OBSERVER_MODEL_TYPES
from ampere.neural.utils import require_torch

torch = require_torch()
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ObserverModelConfig:
    model_type: str
    branch_count: int
    feature_dim: int
    window_tokens: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    branch_embedding_dim: int = 8
    use_branch_embeddings: bool = True
    gain_bound: float = 2.0


class ObserverBaseModel(nn.Module):
    def __init__(self, config: ObserverModelConfig):
        super().__init__()
        if config.model_type not in OBSERVER_MODEL_TYPES:
            raise ValueError(f"Unknown DwellObserver model type: {config.model_type}")
        self.config = config
        if config.use_branch_embeddings and config.branch_embedding_dim > 0:
            self.branch_embedding = nn.Embedding(config.branch_count, config.branch_embedding_dim)
        else:
            self.branch_embedding = None
        self.prior_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.branch_count),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.branch_count + 1, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.branch_count),
        )
        self.selected_gain_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.branch_count + 1, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.state_update = nn.Sequential(
            nn.Linear(config.hidden_dim + 3 * config.branch_count + 1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    @property
    def per_branch_dim(self) -> int:
        extra = self.config.branch_embedding_dim if self.branch_embedding is not None else 0
        return self.config.feature_dim + extra

    def _append_branch_embeddings(self, x):
        if self.branch_embedding is None:
            return x
        batch, tokens, branches, _ = x.shape
        branch_ids = torch.arange(branches, dtype=torch.long, device=x.device)
        emb = self.branch_embedding(branch_ids).view(1, 1, branches, -1).expand(batch, tokens, branches, -1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, x):
        raise NotImplementedError

    def _gain(self, z, selected_branch_index, innovation):
        one_hot = F.one_hot(selected_branch_index, num_classes=self.config.branch_count).to(dtype=z.dtype)
        raw_gain = self.gain_head(torch.cat([z, one_hot, innovation.view(-1, 1)], dim=1))
        return torch.tanh(raw_gain) * float(self.config.gain_bound)

    def _selected_gain(self, z, selected_branch_index, innovation):
        one_hot = F.one_hot(selected_branch_index, num_classes=self.config.branch_count).to(dtype=z.dtype)
        raw_gain = self.selected_gain_head(torch.cat([z, one_hot, innovation.view(-1, 1)], dim=1))
        selected_gain = torch.sigmoid(raw_gain).view(-1, 1) * float(self.config.gain_bound)
        return one_hot * selected_gain

    def _sparse_offbranch_gain(self, z, selected_branch_index, innovation):
        one_hot = F.one_hot(selected_branch_index, num_classes=self.config.branch_count).to(dtype=z.dtype)
        selected_gain = self._selected_gain(z, selected_branch_index, innovation)
        off_gain = self._gain(z, selected_branch_index, innovation) * (1.0 - one_hot) * 0.5
        return selected_gain + off_gain

    def gain_mode(self) -> str:
        if self.config.model_type in {"observer_prior_only", "observer_mlp_prior_only"}:
            return "prior_only"
        if self.config.model_type in {"observer_fixed_selected_gain", "observer_mlp_fixed_selected_gain"}:
            return "fixed_selected_gain"
        if self.config.model_type == "observer_mlp_learned_selected_gain":
            return "learned_selected_gain"
        if self.config.model_type == "observer_mlp_sparse_offbranch_gain":
            return "sparse_offbranch_gain"
        return "full_learned_gain"

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        x = batch["x"]
        selected = batch["selected_branch_index"]
        y_obs_scaled = batch["y_obs_scaled"]
        pre_base_scaled = batch["pre_base_scaled"]

        z_prior = self.encode(x)
        prior_delta = self.prior_head(z_prior)
        p_prior = pre_base_scaled + prior_delta
        selected_prior = p_prior.gather(1, selected.view(-1, 1)).squeeze(1)
        innovation = y_obs_scaled - selected_prior

        gain_mode = self.gain_mode()
        if gain_mode == "prior_only":
            gain = torch.zeros_like(p_prior)
        elif gain_mode == "fixed_selected_gain":
            gain = F.one_hot(selected, num_classes=self.config.branch_count).to(dtype=p_prior.dtype)
        elif gain_mode == "learned_selected_gain":
            gain = self._selected_gain(z_prior, selected, innovation)
        elif gain_mode == "sparse_offbranch_gain":
            gain = self._sparse_offbranch_gain(z_prior, selected, innovation)
        else:
            gain = self._gain(z_prior, selected, innovation)

        p_post = p_prior + gain * innovation.view(-1, 1)
        update_input = torch.cat(
            [z_prior, p_prior, p_post, gain, innovation.view(-1, 1)],
            dim=1,
        )
        z_post = self.state_update(update_input)
        return {
            "prior_scaled": p_prior,
            "posterior_scaled": p_post,
            "gain": gain,
            "innovation_scaled": innovation,
            "latent_prior": z_prior,
            "latent_posterior": z_post,
        }


class ObserverGRU(ObserverBaseModel):
    def __init__(self, config: ObserverModelConfig):
        super().__init__(config)
        token_dim = config.branch_count * self.per_branch_dim
        self.input_projection = nn.Linear(token_dim, config.hidden_dim)
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=max(1, config.num_layers),
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def encode(self, x):
        x = self._append_branch_embeddings(x)
        batch, tokens, branches, dims = x.shape
        y = self.input_projection(x.reshape(batch, tokens, branches * dims))
        y, _ = self.gru(y)
        return self.output_norm(y[:, -1])


class ObserverMLP(ObserverBaseModel):
    def __init__(self, config: ObserverModelConfig):
        super().__init__(config)
        input_dim = config.window_tokens * config.branch_count * self.per_branch_dim
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(max(1, config.num_layers)):
            layers.extend(
                [
                    nn.Linear(current, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            current = config.hidden_dim
        self.encoder = nn.Sequential(*layers)

    def encode(self, x):
        x = self._append_branch_embeddings(x)
        return self.encoder(x.reshape(x.shape[0], -1))


def make_observer_model(config: ObserverModelConfig) -> nn.Module:
    if config.model_type in {"observer_gru", "observer_prior_only", "observer_fixed_selected_gain"}:
        return ObserverGRU(config)
    if config.model_type in {
        "observer_mlp",
        "observer_mlp_prior_only",
        "observer_mlp_fixed_selected_gain",
        "observer_mlp_learned_selected_gain",
        "observer_mlp_sparse_offbranch_gain",
    }:
        return ObserverMLP(config)
    raise ValueError(f"Unknown DwellObserver model type: {config.model_type}")
