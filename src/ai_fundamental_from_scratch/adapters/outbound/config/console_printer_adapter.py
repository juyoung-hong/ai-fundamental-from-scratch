from ai_fundamental_from_scratch.domain.config import AppConfig
from ai_fundamental_from_scratch.ports.outbound.config_port import (
    ConfigPrinterPort,
)


class ConsoleConfigPrinterAdapter(ConfigPrinterPort):
    def print_config(
        self, config: AppConfig, title: str = "Config Verification"
    ) -> None:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

        for category, sub_cfg in config.get_sub_configs().items():
            source = sub_cfg.source_file or "Default / Custom"

            # 1. Category 및 Name 헤더 출력
            print(f"[{category}] {sub_cfg.name}")

            # 2. Source File 정보 한 줄 아래 출력
            print(f"  ├─ Source File : {source}")

            # 3. 세부 하이퍼파라미터 목록을 한 줄씩 들여쓰기하여 출력
            details = sub_cfg.get_details()
            if details:
                print("  └─ Parameters  :")
                for param_key, param_val in details.items():
                    print(f"        • {param_key}: {param_val}")

            print("-" * 70)

        print("=" * 70 + "\n")
