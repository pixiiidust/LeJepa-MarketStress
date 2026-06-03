"""
LeJEPAModel: MLP encoder and predictor.

Encoder:   Linear(180->64) -> ReLU -> Linear(64->16)
Predictor: Linear(16->32)  -> ReLU -> Linear(32->16)

Public interface:
    model.encode(x: Tensor[B, 180]) -> z: Tensor[B, 16]
    model.predict(z: Tensor[B, 16]) -> z_pred: Tensor[B, 16]
"""
