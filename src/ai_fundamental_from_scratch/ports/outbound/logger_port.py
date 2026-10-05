from abc import ABC, abstractmethod
from typing import Any, Dict


class LoggerPort(ABC):
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """단일 스칼라 지표 기록 (Loss, Accuracy 등)"""
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """딕셔너리 형태의 묶음 지표 기록"""
        pass

    @abstractmethod
    def close(self) -> None:
        """로그 스트림 종료"""
        pass
