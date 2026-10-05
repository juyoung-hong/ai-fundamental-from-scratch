from abc import ABC, abstractmethod
from typing import Any, Dict

import torch.nn as nn

from ai_fundamental_from_scratch.domain.metrics import EpochMetrics


class PersistencePort(ABC):
    @abstractmethod
    def save_checkpoint(
        self, model: nn.Module, epoch: int, metrics: EpochMetrics, is_best: bool = False
    ) -> str:
        """모델 가중치 및 학습 상태 저장 후 저장된 경로 반환"""
        pass
