from abc import ABC, abstractmethod

import torch.nn as nn

from ai_fundamental_from_scratch.domain.config import ModelConfig


class ModelPort(ABC):
    @abstractmethod
    def create_model(self, model_config: ModelConfig) -> nn.Module:
        """ModelConfig 수치에 따른 PyTorch nn.Module 인스턴스 생성"""
        pass
