from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SubConfigBase:
    name: str
    source_file: str = ""

    def get_details(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class DataConfig(SubConfigBase):
    data_dir: str = "./data"


@dataclass
class LossComponentConfig:
    name: str
    type: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig(SubConfigBase):
    components: List[LossComponentConfig] = field(default_factory=list)


@dataclass
class TrainerConfig(SubConfigBase):
    batch_size: int = 64
    epochs: int = 5
    seed: int = 42
    device: str = "auto"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])


@dataclass
class ModelConfig(SubConfigBase):
    input_dim: int = 784
    hidden_dim: int = 512
    output_dim: int = 10


@dataclass
class OptimizerConfig(SubConfigBase):
    lr: float = 0.001


@dataclass
class LoggerConfig(SubConfigBase):
    type: str = "tensorboard"
    log_dir: str = "./runs"
    run_dir: str = (
        ""  # 동적으로 생성될 타임스탬프 포함 전체 경로 (e.g., ./runs/2026-10-03_19-38-06)
    )


@dataclass
class PersistenceConfig(SubConfigBase):
    output_dir: str = "./outputs"
    run_dir: str = (
        ""  # 동적으로 생성될 타임스탬프 포함 전체 경로 (e.g., ./outputs/2026-10-03_19-38-06)
    )
    save_top_k: int = 3
    checkpoint_interval: int = 1


@dataclass
class AppConfig:
    data: DataConfig
    trainer: TrainerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    logger: LoggerConfig
    persistence: PersistenceConfig

    def get_sub_configs(self) -> Dict[str, SubConfigBase]:
        """모든 서브 컨피그들을 딕셔너리로 순회 가능하도록 반환"""
        return {
            "Data": self.data,
            "Model": self.model,
            "Trainer": self.trainer,
            "Optimizer": self.optimizer,
            "Logger": self.logger,
            "Persistence": self.persistence,
        }
