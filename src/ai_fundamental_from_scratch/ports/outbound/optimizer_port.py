from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch.nn as nn

from ai_fundamental_from_scratch.domain.config import OptimizerConfig


class OptimizerPort(ABC):
    @abstractmethod
    def create_loss_and_optimizer(
        self, model: nn.Module, optimizer_config: OptimizerConfig
    ) -> Tuple[Any, Any]:
        """(Criterion Loss, Optimizer) 인스턴스 쌍 반환"""
        pass
