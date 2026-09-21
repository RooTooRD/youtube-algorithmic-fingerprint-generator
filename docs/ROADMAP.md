# Roadmap

Each phase ends at something demonstrable. Nothing downstream starts before the layer
under it can be exercised end-to-end.

## P0 — Scaffold ✅

Repo, tooling, package layout, CI. Persona / experiment / account schemas, storage
model, CLI surface, ethics boundary, ADR log.

## P1 — Identity and observation — implemented, live smoke test pending

The point where this stops being a schema and starts producing data.

- `YouTubeDriver.__aenter__` — persistent context per account, with the locale /
  timezone / geolocation / proxy envelope applied consistently.
- `interactive_login` — headed browser, human signs in, session verified and persisted.
- `assert_signed_in`, `ChallengeDetected` / `LoginRequired` detection.
- `collect()` for `home` and `watch_next`: scrape every visible recommendation with
  its rendered rank; selectors isolated in `browser/selectors.py`; degrade gracefully.
- `watch()` with real wall-clock playback.
- Alembic baseline migration; `account login` / `list` / `check` wired.

**Done when:** one provisioned account produces a complete, rank-faithful snapshot of
its home feed and one watch-next list, persisted.

## P2 — The agent loop

- `PersonaPolicy.active_at` for all four behavior modes.
- Prompt construction from persona → system prompt (versioned; the prompt text is part
  of the manifest hash).
- `ClaudeProvider` with structured `Choice` output, one retry, rank-0 fallback.
- `OllamaProvider` for cost-free replication.
- Context phase, exploration phase, step persistence, event log.

**Done when:** `yafg run demo-single-persona.yaml` completes 10 steps and the database
holds every candidate, rank, choice and justification.

## P3 — Scale and enrichment

- Experiment runner: arm matrix (personas × repetitions), bounded concurrency, one
  account per arm, per-agent rate ceiling.
- Async YouTube Data API enrichment worker (metadata, then transcripts).
- Postgres path for parallel agents; resume of interrupted runs.
- `--dry-run` arm matrix and LLM cost estimate.

**Done when:** 10 concurrent agents across 3 personas complete 50 steps without an
account collision or a rate-limit incident.

## P4 — Analysis and control surface

- Rank-weighted topical/stance scoring against `Persona.priors`.
- Cross-persona feed overlap (Jaccard at k), homogeneity, drift vs. the random
  baseline, trajectory plots over steps.
- Parquet export.
- Web dashboard (FastAPI + Svelte) for configuring and monitoring runs, matching the
  role of TRACE's Figure 2 interface.

**Done when:** a two-persona experiment produces a divergence curve with a random
baseline on the same axes.

## Open questions

- **Detection.** Does plain Playwright survive a signed-in session over days? If not,
  the fallback is Patchright/Camoufox — an ADR, not a silent swap.
- **Account longevity.** Unknown how long a synthetic profile survives. Needs a
  measured survival curve before any study design depends on it.
- **Shorts.** A separate recommender with its own feed. Worth a dedicated surface, or
  a confound to be excluded? Currently `shorts_probability: 0.0`.
- **Ordering vs. time.** Watching for real wall-clock time is the honest approach and
  also the throughput ceiling. Whether reduced dwell still registers with the
  recommender is an empirical question worth its own small experiment.
