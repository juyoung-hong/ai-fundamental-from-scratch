from ai_fundamental_from_scratch.adapters.outbound.logger.tensorboard_adapter import (
    TensorBoardLoggerAdapter,
)
from ai_fundamental_from_scratch.domain.config import LoggerConfig
from ai_fundamental_from_scratch.ports.outbound.logger_port import LoggerPort


class LoggerFactory:
    @staticmethod
    def create_logger(logger_config: LoggerConfig) -> LoggerPort:
        logger_type = logger_config.type.lower()
        if logger_type in ("tensorboard", "tb"):
            return TensorBoardLoggerAdapter(logger_config)
        raise ValueError(f"[LoggerError] Unsupported logger type: {logger_config.type}")
