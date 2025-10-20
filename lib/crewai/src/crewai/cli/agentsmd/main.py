from pathlib import Path
from typing import Literal

from rich.console import Console


console = Console()

# IDE configuration mapping
IDE_CONFIGS = {
    "cursor": {
        "rules_dir": ".cursor/rules",
        "file_name": "crewai.md",
        "display_name": "Cursor",
    },
    "windsurf": {
        "rules_dir": ".windsurf/rules",
        "file_name": "crewai.md",
        "display_name": "Windsurf",
    },
    "claude-code": {
        "rules_dir": "",
        "file_name": "CLAUDE.md",
        "display_name": "Claude Code",
    },
    "gemini-cli": {
        "rules_dir": "",
        "file_name": "GEMINI.md",
        "display_name": "Gemini CLI",
    },
}

IDEName = Literal["cursor", "windsurf", "claude-code", "gemini-cli"]

class AgentsMDCommand:
    """
    A class to handle agents.md installation for various IDEs.
    """

    def install(self, ide_name: IDEName) -> None:
        """
        Install agents.md file to the specified IDE's rules directory.

        Args:
            ide_name: The name of the IDE (cursor, windsurf, claude-code, gemini-cli)
            agents_md_content: Optional content for agents.md. If None, looks for agents.md in current directory.
        """
        if ide_name not in IDE_CONFIGS:
            console.print(
                f"[bold red]Error: Unknown IDE '{ide_name}'.[/bold red]\n"
                f"Supported IDEs: {', '.join(IDE_CONFIGS.keys())}",
            )
            raise SystemExit(1)

        ide_config = IDE_CONFIGS[ide_name]
        current_dir = Path(__file__).resolve().parent

        # Get agents.md content
        agents_md_path = current_dir / "AGENTS.md"
        if not agents_md_path.exists():
            console.print(
                "[bold red]Error: AGENTS.md file not found in current directory.[/bold red]\n"
                )
            raise SystemExit(1)

        try:
            agents_md_content = agents_md_path.read_text(encoding="utf-8")
        except Exception as e:
            console.print(
                f"[bold red]Error reading AGENTS.md: {e}[/bold red]",
            )
            raise SystemExit(1) from e

        # Create IDE rules directory
        installation_dir = Path.cwd()
        rules_dir = installation_dir / ide_config["rules_dir"]
        try:
            rules_dir.mkdir(parents=True, exist_ok=True)
            console.print(
                f"[green]Created directory: {ide_config['rules_dir']}[/green]"
            )
        except Exception as e:
            console.print(
                f"[bold red]Error creating directory {rules_dir}: {e}[/bold red]",
            )
            raise SystemExit(1) from e

        # Write the rules file
        rules_file = rules_dir / ide_config["file_name"]
        try:
            rules_file.write_text(agents_md_content, encoding="utf-8")
            console.print(
                f"[bold green]✓ Successfully installed CrewAI rules for {ide_config['display_name']}[/bold green]\n"
                f"[blue]File created: {rules_file.relative_to(installation_dir)}[/blue]",
            )
        except Exception as e:
            console.print(
                f"[bold red]Error writing rules file {rules_file}: {e}[/bold red]",
            )
            raise SystemExit(1) from e
