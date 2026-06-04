"""
LeJEPAModel: MLP encoder and predictor built from Config dims.

Public interface:
    model.encode(x: Tensor[B, input_dim]) -> z: Tensor[B, latent_dim]
    model.predict(z: Tensor[B, latent_dim]) -> z_pred: Tensor[B, latent_dim]
"""
import torch
import torch.nn as nn

from src.types import Config


def _build_mlp(dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class LeJEPAModel(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        cfg = config if config is not None else Config()
        self.encoder = _build_mlp(cfg.encoder_dims)
        self.predictor = _build_mlp(cfg.predictor_dims)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.predict(z)
