from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_fundamental_from_scratch.adapters.outbound.metrics.metric_calculator import (
    MetricEvaluator,
)
from ai_fundamental_from_scratch.domain.config import TrainerConfig
from ai_fundamental_from_scratch.domain.metrics import EpochMetrics
from ai_fundamental_from_scratch.ports.outbound.logger_port import LoggerPort
from ai_fundamental_from_scratch.ports.outbound.persistence_port import PersistencePort


class TrainerService:
    def __init__(
        self,
        model: nn.Module,
        composite_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        trainer_config: TrainerConfig,
        logger: Optional[LoggerPort] = None,  # [신규] Logger 주입
        persistence: Optional[PersistencePort] = None,  # [신규] Persistence 주입
    ):
        self.model = model
        self.loss_fn = composite_loss
        self.optimizer = optimizer
        self.config = trainer_config
        self.logger = logger
        self.persistence = persistence

        self.metrics_list = getattr(trainer_config, "metrics", ["accuracy"])
        self.metric_evaluator = MetricEvaluator(self.metrics_list)

        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model.to(self.device)

    def _run_epoch(
        self, dataloader: DataLoader, is_train: bool
    ) -> tuple[float, Dict[str, float]]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        running_component_losses = {}
        metric_sums: Dict[str, float] = {m: 0.0 for m in self.metrics_list}
        total_samples = 0

        with torch.set_grad_enabled(is_train):
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss, loss_details = self.loss_fn(pred, y)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_size = X.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                for comp_name, comp_val in loss_details.items():
                    running_component_losses[comp_name] = (
                        running_component_losses.get(comp_name, 0.0)
                        + comp_val * batch_size
                    )

                batch_metrics = self.metric_evaluator.compute_all(pred, y)
                for k, v in batch_metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v

        avg_loss = total_loss / total_samples
        avg_component_losses = {
            f"loss/{k}": v / total_samples for k, v in running_component_losses.items()
        }
        avg_metrics = {k: v / total_samples for k, v in metric_sums.items()}
        return avg_loss, avg_component_losses, avg_metrics

    def fit(
        self, train_loader: DataLoader, test_loader: DataLoader
    ) -> List[EpochMetrics]:
        history = []
        best_loss = float("inf")

        print(f"\n🚀 Training started on device: {self.device}")
        print("=" * 70)

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_component_losses, train_metrics = self._run_epoch(
                train_loader, is_train=True
            )
            test_loss, test_component_losses, test_metrics = self._run_epoch(
                test_loader, is_train=False
            )

            metrics_record = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            history.append(metrics_record)

            # 1. Logger 연동 (TensorBoard)
            if self.logger:
                for tag, val in train_component_losses.items():
                    self.logger.log_scalar(f"Train/{tag}", val, step=epoch)
                self.logger.log_scalar("Train/loss/total_loss", train_loss, epoch)
                for tag, val in test_component_losses.items():
                    self.logger.log_scalar(f"Test/{tag}", val, step=epoch)
                self.logger.log_scalar("Test/loss/total_loss", test_loss, epoch)
                self.logger.log_metrics(train_metrics, step=epoch, prefix="Train")
                self.logger.log_metrics(test_metrics, step=epoch, prefix="Test")

            # 2. Persistence 연동 (Best Model 및 Checkpoint 저장)
            is_best = test_loss < best_loss
            if is_best:
                best_loss = test_loss

            if self.persistence:
                saved_path = self.persistence.save_checkpoint(
                    model=self.model,
                    epoch=epoch,
                    metrics=metrics_record,
                    is_best=is_best,
                )

            # 콘솔 출력
            train_str = " | ".join(
                [f"{k}: {v*100:.2f}%" for k, v in train_metrics.items()]
            )
            test_str = " | ".join(
                [f"{k}: {v*100:.2f}%" for k, v in test_metrics.items()]
            )
            best_mark = " ⭐ (Best)" if is_best else ""

            print(f"Epoch [{epoch:>2d}/{self.config.epochs}]{best_mark}")
            print(f"  ├─ Train Loss: {train_loss:.4f} | {train_str}")
            print(f"  └─ Test  Loss: {test_loss:.4f} | {test_str}")

        if self.logger:
            self.logger.close()

        print("=" * 70)
        print("✓ Training completed successfully!\n")
        return history
