# src/ai_fundamental_from_scratch/adapters/outbound/loss/composite_loss_adapter.py
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_fundamental_from_scratch.domain.config import LossComponentConfig

loss_type_mapping = {
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
}


class IndividualLossFactory:
    @staticmethod
    def create_loss(loss_type: str) -> nn.Module:
        loss_type = loss_type.lower()
        if loss_type in loss_type_mapping:
            return loss_type_mapping[loss_type]
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
            # 1. CrossEntropyLoss 처리 (1D Target [B] 수용)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                l_val = loss_fn(pred, target)

            # 2. L1Loss / MSELoss 등 Target과 Pred의 Shape이 일치해야 하는 손실 함수 처리
            else:
                # pred가 [B, C]이고 target이 [B] 정수 라벨인 경우 One-hot 변환
                if pred.dim() == 2 and target.dim() == 1:
                    num_classes = pred.size(1)
                    target_encoded = F.one_hot(target, num_classes=num_classes).float()
                else:
                    target_encoded = target

                l_val = loss_fn(pred, target_encoded)

            weighted_val = weight * l_val
            total_loss += weighted_val
            loss_details[name] = l_val.item()

        return total_loss, loss_details
