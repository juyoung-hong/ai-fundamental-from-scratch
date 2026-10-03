from abc import ABC, abstractmethod
from typing import Any, Tuple

from ai_fundamental_from_scratch.domain.config import DataConfig, TrainerConfig


class DataLoaderPort(ABC):
    @abstractmethod
    def get_data_loaders(
        self, data_config: DataConfig, trainer_config: TrainerConfig
    ) -> Tuple[Any, Any]:
        """Dataset의 종류와 관계없이 표준화된 (train_loader, test_loader)를 반환"""
        pass
