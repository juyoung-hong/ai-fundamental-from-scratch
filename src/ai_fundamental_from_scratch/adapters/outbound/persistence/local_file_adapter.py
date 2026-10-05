import os

import torch
import torch.nn as nn

from ai_fundamental_from_scratch.domain.config import PersistenceConfig
from ai_fundamental_from_scratch.domain.metrics import EpochMetrics
from ai_fundamental_from_scratch.ports.outbound.persistence_port import PersistencePort


class LocalFilePersistenceAdapter(PersistencePort):
    def __init__(self, persistence_config: PersistenceConfig):
        self.config = persistence_config
        self.save_dir = persistence_config.run_dir or persistence_config.output_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(
        self, model: nn.Module, epoch: int, metrics: EpochMetrics, is_best: bool = False
    ) -> str:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "train_loss": metrics.train_loss,
            "test_loss": metrics.test_loss,
            "test_metrics": metrics.test_metrics,
        }

        # 1. 일반 주기별 체크포인트
        ckpt_filename = f"checkpoint_epoch_{epoch:03d}.pt"
        ckpt_path = os.path.join(self.save_dir, ckpt_filename)
        torch.save(checkpoint, ckpt_path)

        # 2. 최고 성능 체크포인트 갱신 저장
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

        return ckpt_path
