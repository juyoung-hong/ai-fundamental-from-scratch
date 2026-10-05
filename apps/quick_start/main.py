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
from ai_fundamental_from_scratch.adapters.outbound.loss.composite_loss_adapter import (
    CompositeLoss,
)
from ai_fundamental_from_scratch.adapters.outbound.model.model_factory import (
    ModelAdapterFactory,
)
from ai_fundamental_from_scratch.adapters.outbound.optimizer.pytorch_optimizer_adapter import (
    PyTorchOptimizerAdapter,
)
from ai_fundamental_from_scratch.services.trainer_service import TrainerService


def main():
    cli_args = sys.argv[1:]

    # 1. Config Loader & Traceability Printer (설정 및 시각 스냅샷 생성)
    config_adapter = OmegaConfAdapter(config_dir="./configs")
    cfg = config_adapter.load_config(overrides=cli_args)

    printer = ConsoleConfigPrinterAdapter()
    printer.print_config(cfg, title="Hexagonal Experiment Traceability Report")

    # 2. Dynamic Data Loader Adapter 생성
    print(f"[{cfg.data.name}] Data Loader 초기화 중...")
    data_adapter = DataAdapterFactory.create_adapter(cfg.data.name)
    train_loader, test_loader = data_adapter.get_data_loaders(cfg.data, cfg.trainer)

    # 3. Dynamic Model Adapter 생성
    print(f"[{cfg.model.name}] PyTorch 신경망 모델 생성 중...")
    model = ModelAdapterFactory.create_model(cfg.model)

    # 4. Composite Loss & Optimizer Adapter 생성
    # 기본 단일 CE Loss 또는 복합 Weighted Loss 구성
    loss_components = getattr(cfg, "loss", None)
    if loss_components and hasattr(loss_components, "components"):
        composite_loss = CompositeLoss(loss_components.components)
    else:
        # 별도 loss config가 없으면 CrossEntropy 기본 구성 적용
        from ai_fundamental_from_scratch.domain.config import LossComponentConfig

        composite_loss = CompositeLoss(
            [LossComponentConfig(name="ce_loss", type="cross_entropy", weight=1.0)]
        )

    opt_adapter = PyTorchOptimizerAdapter()
    _, optimizer = opt_adapter.create_loss_and_optimizer(model, cfg.optimizer)

    # 5. Core Trainer Service Engine 주입
    trainer = TrainerService(
        model=model,
        composite_loss=composite_loss,
        optimizer=optimizer,
        trainer_config=cfg.trainer,
    )

    # 6. Training & Evaluation Fit Loop 실행
    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()
