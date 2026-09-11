"""`yafg` command line.

Command groups mirror the lifecycle: author a persona, provision an account by hand,
declare an experiment, run it, enrich it, analyse it.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
account_app = typer.Typer(no_args_is_help=True, help="Provision and inspect synthetic accounts.")
persona_app = typer.Typer(no_args_is_help=True, help="Author and validate personas.")
app.add_typer(account_app, name="account")
app.add_typer(persona_app, name="persona")

console = Console()


@account_app.command("login")
def account_login(label: str) -> None:
    """Open a HEADED browser so YOU can sign in. yafg types nothing and reads no credential."""
    raise NotImplementedError("P1")


@account_app.command("list")
def account_list() -> None:
    """Show every account, its bound persona, and its last known session health."""
    raise NotImplementedError("P1")


@account_app.command("check")
def account_check(label: str | None = None) -> None:
    """Verify sessions are still signed in and unchallenged, without running an experiment."""
    raise NotImplementedError("P1")


@persona_app.command("lint")
def persona_lint(path: Path = Path("configs/personas")) -> None:
    """Validate persona YAML against the schema and report weight/interest sanity."""
    raise NotImplementedError("P1")


@persona_app.command("show")
def persona_show(persona_id: str) -> None:
    """Render the resolved persona and the exact system prompt it produces."""
    raise NotImplementedError("P2")


@app.command()
def run(experiment: Path, dry_run: bool = False) -> None:
    """Execute an experiment. `--dry-run` resolves and validates everything, launches
    no browser, and prints the arm matrix and estimated LLM cost."""
    raise NotImplementedError("P3")


@app.command()
def enrich(limit: int = 500) -> None:
    """Backfill video metadata via the YouTube Data API for observations seen so far."""
    raise NotImplementedError("P3")


@app.command()
def export(run_id: str, out: Path = Path("exports"), fmt: str = "parquet") -> None:
    """Export observations and steps for offline analysis."""
    raise NotImplementedError("P4")


@app.command()
def analyze(experiment_id: str) -> None:
    """Compute rank-weighted lean, cross-persona overlap, and drift against baseline."""
    raise NotImplementedError("P4")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
