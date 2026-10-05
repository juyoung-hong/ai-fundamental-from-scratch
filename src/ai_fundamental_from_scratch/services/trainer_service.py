from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_fundamental_from_scratch.adapters.outbound.metrics.metric_calculator import (
    MetricEvaluator,
)
from ai_fundamental_from_scratch.domain.config import TrainerConfig
from ai_fundamental_from_scratch.domain.metrics import EpochMetrics


class TrainerService:
    def __init__(
        self,
        model: nn.Module,
        composite_loss: nn.Module,  # CompositeLoss 인스턴스
        optimizer: torch.optim.Optimizer,
        trainer_config: TrainerConfig,
    ):
        self.model = model
        self.loss_fn = composite_loss
        self.optimizer = optimizer
        self.config = trainer_config
        self.metric_evaluator = MetricEvaluator(trainer_config.metrics)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and trainer_config.device == "auto"
            else "cpu"
        )
        self.model.to(self.device)

    def _run_epoch(
        self, dataloader: DataLoader, is_train: bool
    ) -> tuple[float, Dict[str, float]]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        metric_sums: Dict[str, float] = {m: 0.0 for m in self.config.metrics}
        total_samples = 0

        with torch.set_grad_enabled(is_train):
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                # Forward Pass
                pred = self.model(X)
                loss, _ = self.loss_fn(pred, y)

                # Backward Pass (학습 시)
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Metric 계산 및 누적
                batch_size = X.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                batch_metrics = self.metric_evaluator.compute_all(pred, y)
                for k, v in batch_metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v

        avg_loss = total_loss / total_samples
        avg_metrics = {k: v / total_samples for k, v in metric_sums.items()}
        return avg_loss, avg_metrics

    def fit(
        self, train_loader: DataLoader, test_loader: DataLoader
    ) -> List[EpochMetrics]:
        history = []
        print(f"\n🚀 Training started on device: {self.device}")
        print("=" * 70)

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_metrics = self._run_epoch(train_loader, is_train=True)
            test_loss, test_metrics = self._run_epoch(test_loader, is_train=False)

            metrics_record = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            history.append(metrics_record)

            # 콘솔에 동적으로 모든 메트릭 출력
            train_str = " | ".join(
                [f"{k}: {v*100:.2f}%" for k, v in train_metrics.items()]
            )
            test_str = " | ".join(
                [f"{k}: {v*100:.2f}%" for k, v in test_metrics.items()]
            )

            print(f"Epoch [{epoch:>2d}/{self.config.epochs}]")
            print(f"  ├─ Train Loss: {train_loss:.4f} | {train_str}")
            print(f"  └─ Test  Loss: {test_loss:.4f} | {test_str}")

        print("=" * 70)
        return history
