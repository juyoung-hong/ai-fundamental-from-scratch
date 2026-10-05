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
from ai_fundamental_from_scratch.adapters.outbound.logger.logger_factory import (
    LoggerFactory,
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
from ai_fundamental_from_scratch.adapters.outbound.persistence.local_file_adapter import (
    LocalFilePersistenceAdapter,
)
from ai_fundamental_from_scratch.services.trainer_service import TrainerService


def main():
    cli_args = sys.argv[1:]

    # 1. Config Loader & Printer
    config_adapter = OmegaConfAdapter(config_dir="./configs")
    cfg = config_adapter.load_config(overrides=cli_args)

    printer = ConsoleConfigPrinterAdapter()
    printer.print_config(cfg, title="Hexagonal Experiment Traceability Report")

    # 2. Data Loader
    data_adapter = DataAdapterFactory.create_adapter(cfg.data.name)
    train_loader, test_loader = data_adapter.get_data_loaders(cfg.data, cfg.trainer)

    # 3. Model
    if hasattr(cfg.data, "input_dim") and cfg.data.input_dim:
        cfg.model.input_dim = cfg.data.input_dim
    model = ModelAdapterFactory.create_model(cfg.model)

    # 4. Composite Loss & Optimizer
    loss_components = getattr(cfg, "loss", None)
    if loss_components and hasattr(loss_components, "components"):
        composite_loss = CompositeLoss(loss_components.components)
    else:
        from ai_fundamental_from_scratch.domain.config import LossComponentConfig

        composite_loss = CompositeLoss(
            [LossComponentConfig(name="ce_loss", type="cross_entropy", weight=1.0)]
        )

    opt_adapter = PyTorchOptimizerAdapter()
    _, optimizer = opt_adapter.create_loss_and_optimizer(model, cfg.optimizer)

    # 5. Logger & Persistence Adapters [신규]
    logger = LoggerFactory.create_logger(cfg.logger)
    persistence = LocalFilePersistenceAdapter(cfg.persistence)

    # 6. Core Trainer Engine Execution
    trainer = TrainerService(
        model=model,
        composite_loss=composite_loss,
        optimizer=optimizer,
        trainer_config=cfg.trainer,
        logger=logger,
        persistence=persistence,
    )

    trainer.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()
