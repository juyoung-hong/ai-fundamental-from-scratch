from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
