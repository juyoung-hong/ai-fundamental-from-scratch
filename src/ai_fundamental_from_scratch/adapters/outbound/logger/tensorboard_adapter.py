from typing import Dict

from torch.utils.tensorboard import SummaryWriter

from ai_fundamental_from_scratch.domain.config import LoggerConfig
from ai_fundamental_from_scratch.ports.outbound.logger_port import LoggerPort


class TensorBoardLoggerAdapter(LoggerPort):
    def __init__(self, logger_config: LoggerConfig):
        # omegaconf_adapter에서 생성한 타임스탬프 run_dir 사용 (없으면 log_dir)
        log_path = logger_config.run_dir or logger_config.log_dir
        self.writer = SummaryWriter(log_dir=log_path)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        for key, val in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, val, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
