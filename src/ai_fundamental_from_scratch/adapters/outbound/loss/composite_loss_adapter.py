from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ai_fundamental_from_scratch.domain.config import LossComponentConfig


class IndividualLossFactory:
    @staticmethod
    def create_loss(loss_type: str) -> nn.Module:
        loss_type = loss_type.lower()
        if loss_type in ("cross_entropy", "ce"):
            return nn.CrossEntropyLoss()
        elif loss_type in ("mse", "mean_squared_error"):
            return nn.MSELoss()
        elif loss_type in ("l1", "l1_loss"):
            return nn.L1Loss()
        else:
            raise ValueError(f"[LossError] Unsupported loss type: {loss_type}")


class CompositeLoss(nn.Module):
    def __init__(self, components: List[LossComponentConfig]):
        super().__init__()
        self.loss_items = nn.ModuleList()
        self.weights: List[float] = []
        self.names: List[str] = []

        for comp in components:
            loss_fn = IndividualLossFactory.create_loss(comp.type)
            self.loss_items.append(loss_fn)
            self.weights.append(comp.weight)
            self.names.append(comp.name)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=pred.device)
        loss_details = {}

        for name, weight, loss_fn in zip(self.names, self.weights, self.loss_items):
            # L1 규제처럼 모델 가중치 배치가 필요한 특수 Loss는 추가 핸들링 가능
            l_val = loss_fn(pred, target)
            weighted_val = weight * l_val
            total_loss += weighted_val
            loss_details[name] = l_val.item()

        return total_loss, loss_details
