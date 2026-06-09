from __future__ import annotations

from dataclasses import dataclass

from ampere.neural.utils import require_torch

torch = require_torch()
from torch import nn
import torch.nn.functional as F


DWELL_MODEL_TYPES = ("dwell_mlp", "dwell_tcn", "dwell_gru", "dwell_transformer")


@dataclass(frozen=True)
class DwellModelConfig:
    model_type: str
    branch_count: int
    feature_dim: int
    window_tokens: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    branch_embedding_dim: int = 8
    use_branch_embeddings: bool = True
    transformer_heads: int = 4


class DwellBaseModel(nn.Module):
    def __init__(self, config: DwellModelConfig):
        super().__init__()
        self.config = config
        if config.use_branch_embeddings and config.branch_embedding_dim > 0:
            self.branch_embedding = nn.Embedding(config.branch_count, config.branch_embedding_dim)
        else:
            self.branch_embedding = None

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


class DwellMLP(DwellBaseModel):
    """Serious non-sequential scan-cycle baseline over flattened dwell context."""

    def __init__(self, config: DwellModelConfig):
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
        layers.append(nn.Linear(current, config.branch_count))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self._append_branch_embeddings(x)
        return self.net(x.reshape(x.shape[0], -1))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class DwellTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = x + self.net(x)
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class DwellTCN(DwellBaseModel):
    def __init__(self, config: DwellModelConfig):
        super().__init__(config)
        token_dim = config.branch_count * self.per_branch_dim
        self.input_projection = nn.Linear(token_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [DwellTCNBlock(config.hidden_dim, dilation=2**idx, dropout=config.dropout) for idx in range(config.num_layers)]
        )
        self.output = nn.Linear(config.hidden_dim, config.branch_count)

    def forward(self, x):
        x = self._append_branch_embeddings(x)
        batch, tokens, branches, dims = x.shape
        y = self.input_projection(x.reshape(batch, tokens, branches * dims))
        y = y.transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        last = y[:, :, -1]
        return self.output(last)


class DwellGRU(DwellBaseModel):
    def __init__(self, config: DwellModelConfig):
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
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.branch_count))

    def forward(self, x):
        x = self._append_branch_embeddings(x)
        batch, tokens, branches, dims = x.shape
        y = self.input_projection(x.reshape(batch, tokens, branches * dims))
        y, _ = self.gru(y)
        return self.output(y[:, -1])


class DwellFormer(DwellBaseModel):
    def __init__(self, config: DwellModelConfig):
        super().__init__(config)
        token_dim = config.branch_count * self.per_branch_dim
        self.input_projection = nn.Linear(token_dim, config.hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.window_tokens, config.hidden_dim))
        heads = max(1, min(config.transformer_heads, config.hidden_dim))
        while config.hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, config.num_layers))
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.branch_count))

    def forward(self, x):
        x = self._append_branch_embeddings(x)
        batch, tokens, branches, dims = x.shape
        y = self.input_projection(x.reshape(batch, tokens, branches * dims))
        y = y + self.position_embedding[:, :tokens]
        y = self.encoder(y)
        return self.output(y[:, -1])


def make_dwell_model(config: DwellModelConfig) -> nn.Module:
    if config.model_type == "dwell_mlp":
        return DwellMLP(config)
    if config.model_type == "dwell_tcn":
        return DwellTCN(config)
    if config.model_type == "dwell_gru":
        return DwellGRU(config)
    if config.model_type == "dwell_transformer":
        return DwellFormer(config)
    raise ValueError(f"Unknown DwellNet model type: {config.model_type}")
