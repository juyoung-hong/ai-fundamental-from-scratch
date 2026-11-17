from abc import ABC, abstractmethod

from src.ai_fundamental_from_scratch.domain.data import DataConfig


class ConfigLoader(ABC):
    @abstractmethod
    def load_data_config(self) -> DataConfig:
        pass
