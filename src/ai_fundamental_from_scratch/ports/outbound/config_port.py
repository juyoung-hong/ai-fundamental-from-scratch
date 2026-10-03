from abc import ABC, abstractmethod

from ai_fundamental_from_scratch.domain.config import AppConfig


class ConfigRepositoryPort(ABC):
    @abstractmethod
    def load_config(self, overrides: list[str] | None = None) -> AppConfig:
        pass


class ConfigPrinterPort(ABC):
    @abstractmethod
    def print_config(
        self, config: AppConfig, title: str = "Config Verification"
    ) -> None:
        pass
