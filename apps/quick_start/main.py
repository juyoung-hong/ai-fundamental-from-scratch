import sys

from ai_fundamental_from_scratch.adapters.outbound.config.console_printer_adapter import (
    ConsoleConfigPrinterAdapter,
)
from ai_fundamental_from_scratch.adapters.outbound.config.omegaconf_adapter import (
    OmegaConfAdapter,
)


def main():
    cli_args = sys.argv[1:]

    # 1. Config Loader Adapter
    config_adapter = OmegaConfAdapter(config_dir="./configs")
    cfg = config_adapter.load_config(overrides=cli_args)

    # 2. Config Printer Adapter
    printer = ConsoleConfigPrinterAdapter()

    # 3. Print
    printer.print_config(cfg, title="Experiment Traceability Report")


if __name__ == "__main__":
    main()
