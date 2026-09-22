"""`yafg` command line."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy.ext.asyncio import AsyncEngine

from yafg.agent.prompt import PROMPT_VERSION, render_persona_prompt
from yafg.browser.driver import ChallengeDetected, LoginRequired, YouTubeDriver
from yafg.enrich import EnrichmentWorker, PublicTranscriptClient, YouTubeMetadataClient
from yafg.experiment.runner import (
    estimate_llm_cost,
    execute_experiment,
    plan_runs,
    resolve_experiment,
    validate_accounts,
)
from yafg.identity.registry import AccountInUse, AccountRegistry
from yafg.identity.schema import Account, ProxyConfig
from yafg.personas.schema import Persona
from yafg.settings import settings
from yafg.store.database import ensure_schema, make_engine

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
account_app = typer.Typer(no_args_is_help=True, help="Provision and inspect synthetic accounts.")
persona_app = typer.Typer(no_args_is_help=True, help="Author and validate personas.")
app.add_typer(account_app, name="account")
app.add_typer(persona_app, name="persona")

console = Console()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


async def _registry() -> tuple[AccountRegistry, AsyncEngine]:
    settings.ensure_dirs()
    engine = make_engine()
    await ensure_schema(engine)
    return AccountRegistry(engine), engine


@account_app.command("login")
def account_login(
    label: str,
    persona: str | None = typer.Option(None, "--persona", help="Optional single-persona provenance binding."),
    locale: str = typer.Option("en-US", help="Browser locale."),
    timezone: str = typer.Option("UTC", help="IANA timezone, e.g. Africa/Algiers."),
    country: str | None = typer.Option(None, help="ISO-3166 alpha-2 country code."),
    latitude: float | None = typer.Option(None, help="Pinned geolocation latitude; provide with --longitude."),
    longitude: float | None = typer.Option(None, help="Pinned geolocation longitude; provide with --latitude."),
    proxy_server: str | None = typer.Option(None, help="Pinned proxy URL; no password is accepted here."),
    proxy_username: str | None = typer.Option(None, help="Proxy username."),
    proxy_password_env: str | None = typer.Option(None, help="Env var name containing the proxy password."),
) -> None:
    """Open a HEADED browser so YOU can sign in; yafg never types credentials."""
    if (latitude is None) != (longitude is None):
        raise typer.BadParameter("--latitude and --longitude must be provided together")

    async def _run() -> None:
        registry, engine = await _registry()
        try:
            existing = await registry.get(label)
            if existing is not None:
                account = existing.model_copy(update={"status": "unprovisioned"})
            else:
                proxy = None
                if proxy_server:
                    proxy = ProxyConfig(
                        server=proxy_server,
                        username=proxy_username,
                        password_env=proxy_password_env,
                    )
                account = Account(
                    label=label,
                    persona_id=persona,
                    profile_dir=settings.profile_dir / label,
                    locale=locale,
                    timezone=timezone,
                    country=country.upper() if country else None,
                    geolocation=(latitude, longitude) if latitude is not None and longitude is not None else None,
                    proxy=proxy,
                )
                await registry.save(account)
            lease_id = str(uuid.uuid4())
            await registry.acquire_lease(label, lease_id)
            try:
                if existing is not None:
                    await registry.save(account)
                console.print(f"Opening headed browser for [bold]{label}[/bold]. Sign in manually in that window.")
                try:
                    await YouTubeDriver.interactive_login(account)
                except ChallengeDetected:
                    await registry.save(account.model_copy(update={"status": "challenged"}), checked=True)
                    raise
                except LoginRequired:
                    await registry.save(account.model_copy(update={"status": "logged_out"}), checked=True)
                    raise
                await registry.save(account.model_copy(update={"status": "active"}), checked=True)
                console.print(f"[green]Verified[/green] signed-in session for {label}.")
            finally:
                await registry.release_lease(label, lease_id)
        finally:
            await engine.dispose()

    try:
        asyncio.run(_run())
    except (ValidationError, ValueError, RuntimeError) as exc:
        console.print(f"[red]Account login failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@account_app.command("list")
def account_list() -> None:
    """Show every account, its optional persona provenance, and session health."""

    async def _run() -> None:
        registry, engine = await _registry()
        try:
            accounts = await registry.list()
            leases = {account.label: await registry.lease_owner(account.label) for account in accounts}
        finally:
            await engine.dispose()
        table = Table("Label", "Status", "Persona", "Locale", "Timezone", "Profile", "Lease")
        for account in accounts:
            table.add_row(
                account.label,
                account.status,
                account.persona_id or "—",
                account.locale,
                account.timezone,
                str(account.profile_dir),
                leases[account.label] or "—",
            )
        console.print(table)
        if not accounts:
            console.print("No provisioned accounts.")

    asyncio.run(_run())


@account_app.command("unlock")
def account_unlock(
    label: str,
    lease_id: str = typer.Option(..., "--lease-id", help="Exact lease token shown by `account list`."),
) -> None:
    """Release an orphaned lease after verifying no process still owns the profile."""

    async def _run() -> None:
        registry, engine = await _registry()
        try:
            owner = await registry.lease_owner(label)
            if owner is None:
                raise ValueError(f"account {label!r} has no active lease")
            if owner != lease_id:
                raise ValueError(f"account {label!r} lease changed; run `yafg account list` again")
            await registry.release_lease(label, lease_id)
        finally:
            await engine.dispose()

    try:
        asyncio.run(_run())
    except (KeyError, ValueError) as exc:
        console.print(f"[red]Account unlock failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Released[/green] orphaned lease for {label}.")


@account_app.command("check")
def account_check(label: str | None = None) -> None:
    """Verify saved sessions are signed in and unchallenged, without running a study."""

    async def _check_one(registry: AccountRegistry, account: Account) -> tuple[str, str]:
        lease_id = str(uuid.uuid4())
        try:
            await registry.acquire_lease(account.label, lease_id)
        except AccountInUse as exc:
            return "busy", str(exc)
        try:
            try:
                async with YouTubeDriver(account, headless=settings.headless) as driver:
                    identity = await driver.assert_signed_in()
                await registry.save(account.model_copy(update={"status": "active"}), checked=True)
                return "active", identity
            except ChallengeDetected as exc:
                await registry.save(account.model_copy(update={"status": "challenged"}), checked=True)
                return "challenged", str(exc)
            except LoginRequired as exc:
                await registry.save(account.model_copy(update={"status": "logged_out"}), checked=True)
                return "logged_out", str(exc)
        finally:
            await registry.release_lease(account.label, lease_id)

    async def _run() -> int:
        registry, engine = await _registry()
        try:
            if label:
                account = await registry.get(label)
                if account is None:
                    console.print(f"[red]Unknown account:[/red] {label}")
                    return 1
                accounts = [account]
            else:
                accounts = await registry.list()
            failed = False
            for account in accounts:
                status, detail = await _check_one(registry, account)
                style = "green" if status == "active" else "red"
                console.print(f"[{style}]{account.label}: {status}[/{style}] — {detail}")
                failed |= status != "active"
            return int(failed)
        finally:
            await engine.dispose()

    code = asyncio.run(_run())
    if code:
        raise typer.Exit(code=code)


@persona_app.command("lint")
def persona_lint(path: Path = Path("configs/personas")) -> None:
    """Validate persona YAML and filename identity, reporting all failures."""
    files = [path] if path.is_file() else sorted(path.glob("*.yaml"))
    if not files:
        console.print(f"[red]No persona YAML files found at {path}[/red]")
        raise typer.Exit(code=1)
    failures = 0
    for file in files:
        try:
            persona = Persona.model_validate(_load_yaml(file))
            if persona.id != file.stem:
                raise ValueError(f"id {persona.id!r} must match filename {file.stem!r}")
            console.print(f"[green]✓[/green] {file}: {persona.display_name} ({len(persona.interests)} interests)")
        except (OSError, ValueError, ValidationError) as exc:
            failures += 1
            console.print(f"[red]✗[/red] {file}: {exc}")
    if failures:
        raise typer.Exit(code=1)


@persona_app.command("show")
def persona_show(persona_id: str) -> None:
    """Render a validated persona and the exact versioned system prompt."""
    path = settings.config_dir / "personas" / f"{persona_id}.yaml"
    try:
        persona = Persona.model_validate(_load_yaml(path))
        if persona.id != persona_id:
            raise ValueError(f"persona id {persona.id!r} does not match requested id {persona_id!r}")
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f"[red]Persona load failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(Panel.fit(f"[bold]{persona.display_name}[/bold]\nID: {persona.id}\nPrompt: {PROMPT_VERSION}"))
    console.print(render_persona_prompt(persona))


@app.command()
def run(
    experiment: Path,
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print the full arm schedule."),
    headed: bool = typer.Option(False, "--headed", help="Show each browser while the run executes."),
    resume: bool = typer.Option(False, "--resume", help="Resume the latest incomplete run set with this manifest."),
) -> None:
    """Run the P3 arm matrix with bounded concurrency and safe checkpoints."""

    async def _run() -> list[str]:
        resolved = resolve_experiment(experiment)
        plan = plan_runs(resolved)
        estimate = estimate_llm_cost(resolved, plan)
        settings.ensure_dirs()
        engine = make_engine()
        try:
            await ensure_schema(engine)
            if resume and not dry_run:
                registry = AccountRegistry(engine)
                accounts = [await registry.get(item.account_label) for item in plan]
            else:
                accounts = await validate_accounts(engine, resolved, plan)
            console.print(
                f"Manifest [bold]{resolved.manifest_hash}[/bold] — prompt {PROMPT_VERSION}, "
                f"{len(plan)} run(s), concurrency {resolved.experiment.concurrency}"
            )
            table = Table("#", "Repetition", "Arm", "Account", "Behavior seed", "Account status")
            for item, account in zip(plan, accounts, strict=True):
                table.add_row(
                    str(item.index),
                    str(item.repetition),
                    item.arm,
                    item.account_label,
                    str(item.behavior_seed),
                    account.status if account else "missing",
                )
            console.print(table)
            cost = (
                f"~${estimate.estimated_usd:.4f}"
                if estimate.estimated_usd is not None
                else "USD unavailable; set YAFG_LLM_INPUT_USD_PER_MILLION and YAFG_LLM_OUTPUT_USD_PER_MILLION"
            )
            console.print(
                f"LLM envelope: {estimate.llm_calls} calls, ~{estimate.estimated_input_tokens:,} input tokens, "
                f"≤{estimate.max_output_tokens:,} output tokens; {cost}."
            )
            console.print(f"Estimate assumption: {estimate.assumption}.")
            if dry_run:
                console.print("[green]Dry run valid.[/green] No browser or LLM call was made.")
                return []
            return await execute_experiment(
                engine,
                resolved,
                headless=False if headed else None,
                resume=resume,
            )
        finally:
            await engine.dispose()

    try:
        run_ids = asyncio.run(_run())
    except Exception as exc:
        console.print(f"[red]Run failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc
    if run_ids:
        console.print(f"[green]Completed[/green] {len(run_ids)} run(s): {', '.join(run_ids)}")


@app.command()
def enrich(
    limit: int = typer.Option(500, min=1, help="Maximum unresolved video IDs to process."),
    transcripts: bool = typer.Option(
        False,
        "--transcripts",
        help="Also attempt public caption-track retrieval. Metadata-only is the default.",
    ),
) -> None:
    """Backfill video metadata, with optional best-effort public transcripts."""

    async def _run() -> None:
        if not settings.youtube_api_key:
            raise ValueError("YOUTUBE_API_KEY is required for enrichment")
        settings.ensure_dirs()
        engine = make_engine()
        worker: EnrichmentWorker | None = None
        try:
            await ensure_schema(engine)
            metadata = YouTubeMetadataClient(settings.youtube_api_key)
            transcript_client = (
                PublicTranscriptClient(max_requests_per_hour=settings.transcript_max_requests_per_hour)
                if transcripts
                else None
            )
            worker = EnrichmentWorker(engine, metadata, transcripts=transcript_client)
            stats = await worker.run(limit=limit, include_transcripts=transcripts)
            console.print(
                f"Discovered {stats.discovered}; metadata complete={stats.metadata_complete}, "
                f"not-found={stats.metadata_not_found}, errors={stats.metadata_errors}."
            )
            if transcripts:
                console.print(
                    f"Transcripts complete={stats.transcripts_complete}, "
                    f"unavailable={stats.transcripts_unavailable}, errors={stats.transcript_errors}."
                )
        finally:
            if worker is not None:
                await worker.aclose()
            await engine.dispose()

    try:
        asyncio.run(_run())
    except Exception as exc:
        console.print(f"[red]Enrichment failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def export(run_id: str, out: Path = Path("exports"), fmt: str = "parquet") -> None:
    raise NotImplementedError("P4")


@app.command()
def analyze(experiment_id: str) -> None:
    raise NotImplementedError("P4")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
