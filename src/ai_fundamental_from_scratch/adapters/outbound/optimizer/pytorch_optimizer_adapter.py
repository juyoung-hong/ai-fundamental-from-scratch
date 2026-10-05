from typing import Tuple

import torch.nn as nn
import torch.optim as optim

from ai_fundamental_from_scratch.domain.config import OptimizerConfig
from ai_fundamental_from_scratch.ports.outbound.optimizer_port import OptimizerPort


class PyTorchOptimizerAdapter(OptimizerPort):
    def create_loss_and_optimizer(
        self, model: nn.Module, optimizer_config: OptimizerConfig
    ) -> Tuple[nn.Module, optim.Optimizer]:
        criterion = nn.CrossEntropyLoss()

        name = optimizer_config.name.lower()
        if name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=optimizer_config.lr)
        elif name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=optimizer_config.lr)
        elif name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=optimizer_config.lr)
        else:
            raise ValueError(
                f"[OptimizerError] Unsupported optimizer '{optimizer_config.name}'"
            )

        return criterion, optimizer
