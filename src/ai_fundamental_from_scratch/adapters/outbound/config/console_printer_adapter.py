from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from ai_fundamental_from_scratch.domain.config import AppConfig
from ai_fundamental_from_scratch.ports.outbound.config_port import ConfigPrinterPort


class ConsoleConfigPrinterAdapter(ConfigPrinterPort):
    def __init__(self):
        self.console = Console()

    def print_config(
        self, config: AppConfig, title: str = "Config Verification"
    ) -> None:
        self._print_rich(config, title)

    def _print_rich(self, config: AppConfig, title: str) -> None:
        root_tree = Tree(f"[bold cyan]{title}[/bold cyan]")

        for category, sub_cfg in config.get_sub_configs().items():
            source = getattr(sub_cfg, "source_file", None) or "Default / Custom"

            cat_node = root_tree.add(
                f"[bold yellow][{category.upper()}][/bold yellow] [bold white]{sub_cfg.name}[/bold white]"
            )
            cat_node.add(f"[dim]Source File:[/dim] [green]{source}[/green]")

            filtered_details = {
                k: v
                for k, v in sub_cfg.get_details().items()
                if k not in ("name", "source_file")
            }

            if filtered_details:
                param_node = cat_node.add("[dim]Parameters:[/dim]")
                for param_key, param_val in filtered_details.items():
                    param_node.add(
                        f"[bold magenta]{param_key}[/bold magenta]: [bright_blue]{param_val}[/bright_blue]"
                    )

        # 아름다운 Panel 감싸기
        panel = Panel(
            root_tree,
            title="[bold green]Hexagonal Traceability Report[/bold green]",
            subtitle="[dim]Resolved Configuration Snapshot[/dim]",
            border_style="bright_blue",
            padding=(1, 2),
        )
        self.console.print("\n")
        self.console.print(panel)
        self.console.print("\n")
