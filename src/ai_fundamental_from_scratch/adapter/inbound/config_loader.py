# src/ai_fundamental_from_scratch/adapters/infrastructure/yaml_config_adapter.py
from pathlib import Path
from typing import Any, Dict

import yaml

from ai_fundamental_from_scratch.application.port.inbound.config_loader import (
    ConfigLoaderPort,
)
from ai_fundamental_from_scratch.domain.config.dataset_config import DataConfig


class YamlConfigAdapter(ConfigLoaderPort):
    """YAML 파일을 읽어 설정을 구성하는 인프라스트럭처 어댑터"""

    def __init__(self, configs_dir: str = "configs"):
        self.configs_dir = Path(configs_dir)

    def _deep_merge(self, base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def load(self) -> DataConfig:
        if not self.configs_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.configs_dir}")

        merged_data: Dict[str, Any] = {}
        yaml_files = sorted(
            list(self.configs_dir.glob("*.yaml")) + list(self.configs_dir.glob("*.yml"))
        )

        for file_path in yaml_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    merged_data = self._deep_merge(merged_data, data)

        # 도메인 모델로 변환 및 Pydantic 유효성 검증
        return DataConfig(**merged_data)
