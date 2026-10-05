import torch.nn as nn

from ai_fundamental_from_scratch.domain.config import ModelConfig
from ai_fundamental_from_scratch.ports.outbound.model_port import ModelPort


class QuickstartNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


class QuickstartModelAdapter(ModelPort):
    def create_model(self, model_config: ModelConfig) -> nn.Module:
        return QuickstartNet(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
        )
