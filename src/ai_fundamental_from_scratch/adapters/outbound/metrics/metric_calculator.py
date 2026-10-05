from typing import Callable, Dict, List

import torch


class MetricRegistry:
    @staticmethod
    def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        preds = pred.argmax(dim=1)
        correct = (preds == target).float().sum().item()
        return correct / target.size(0)

    @staticmethod
    def compute_top_k_accuracy(
        pred: torch.Tensor, target: torch.Tensor, k: int = 3
    ) -> float:
        _, top_k_preds = pred.topk(k, dim=1, largest=True, sorted=True)
        correct = top_k_preds.eq(target.view(-1, 1).expand_as(top_k_preds))
        return correct.float().sum().item() / target.size(0)


class MetricEvaluator:
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names

    def compute_all(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        results = {}
        batch_size = target.size(0)

        for name in self.metric_names:
            name_lower = name.lower()
            if name_lower == "accuracy":
                results["accuracy"] = (
                    MetricRegistry.compute_accuracy(pred, target) * batch_size
                )
            elif name_lower.startswith("top_") or name_lower == "top_k_accuracy":
                results["top_3_accuracy"] = (
                    MetricRegistry.compute_top_k_accuracy(pred, target, k=3)
                    * batch_size
                )
            # 필요 시 precision, recall, f1_score 추가 가능
        return results
