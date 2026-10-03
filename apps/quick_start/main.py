import sys

from ai_fundamental_from_scratch.adapters.outbound.config.console_printer_adapter import (
    ConsoleConfigPrinterAdapter,
)
from ai_fundamental_from_scratch.adapters.outbound.config.omegaconf_adapter import (
    OmegaConfAdapter,
)
from ai_fundamental_from_scratch.adapters.outbound.data.data_factory import (
    DataAdapterFactory,
)


def main():
    cli_args = sys.argv[1:]

    # 1. Config Loader Adapter
    config_adapter = OmegaConfAdapter(config_dir="./configs")
    cfg = config_adapter.load_config(overrides=cli_args)

    # 2. Config Printer Adapter
    printer = ConsoleConfigPrinterAdapter()
    printer.print_config(cfg, title="Experiment Traceability Report")

    # 3. Dynamic Data Adapter Factory
    # main.py는 데이터셋이 Torchvision인지, CSV인지, 로컬 이미지 폴더인지 알 필요가 없습니다.
    print(f"[{cfg.data.name}] 데이터 로더 초기화 중...")
    data_adapter = DataAdapterFactory.create_adapter(cfg.data.name)
    train_loader, test_loader = data_adapter.get_data_loaders(cfg.data, cfg.trainer)

    print(
        f"✓ [{cfg.data.name}] Train DataLoader 준비 완료: {len(train_loader)} batches (Batch Size: {cfg.trainer.batch_size})"
    )
    print(f"✓ [{cfg.data.name}] Test DataLoader 준비 완료 : {len(test_loader)} batches")

    # 4. 데이터 샘플 검증 (학습 루프 진입 전 차원 및 배치 검증)
    for X, y in train_loader:
        print("\n--- 데이터 배치 Sample Verification ---")
        print(f"Dataset Name          : {cfg.data.name}")
        print(f"Shape of X [N, C, H, W]: {X.shape} (Dtype: {X.dtype})")
        print(f"Shape of y            : {y.shape} (Dtype: {y.dtype})")
        break


if __name__ == "__main__":
    main()
