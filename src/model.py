"""
LeJEPAModel: MLP encoder and predictor.

Encoder:   Linear(180->64) -> ReLU -> Linear(64->16)
Predictor: Linear(16->32)  -> ReLU -> Linear(32->16)

Public interface:
    model.encode(x: Tensor[B, 180]) -> z: Tensor[B, 16]
    model.predict(z: Tensor[B, 16]) -> z_pred: Tensor[B, 16]
"""
import torch
import torch.nn as nn

from src.types import Config


class LeJEPAModel(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(180, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.predict(z)
