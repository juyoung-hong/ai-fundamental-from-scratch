import torch.nn as nn

from ai_fundamental_from_scratch.adapters.outbound.model.quickstart_net import (
    QuickstartModelAdapter,
)
from ai_fundamental_from_scratch.domain.config import ModelConfig
from ai_fundamental_from_scratch.ports.outbound.model_port import ModelPort


class ModelAdapterFactory:
    @staticmethod
    def create_model(model_config: ModelConfig) -> nn.Module:
        name = model_config.name.lower().replace("-", "_")
        if name in ("quickstart_net", "quickstart"):
            adapter: ModelPort = QuickstartModelAdapter()
            return adapter.create_model(model_config)

        raise ValueError(f"[ModelError] Unsupported model name '{model_config.name}'")
