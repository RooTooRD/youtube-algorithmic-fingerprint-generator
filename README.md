# yafg — YouTube Algorithmic Fingerprint Generator

Persona-driven agents that build **real, signed-in YouTube watch histories** so you can
measure how the recommender treats different people differently.

You declare a persona — nationality, age, sex, background, religion, education,
interests, viewing habits — and the framework drives a logged-in browser profile
through a YouTube session on that persona's behalf, logging every recommendation it
was shown, at what rank, and why it picked what it picked. Run several personas
against an identical starting state and the difference between their feeds *is* the
measurement.

Prior art: [TRACE (Dessain, Timmers & Vande Kerckhove, WSDM '26)](https://doi.org/10.1145/3773966.3779410),
which established the two-phase LLM-persona audit protocol this project follows.
The deliberate delta here is **signed-in identity**: TRACE audits signed-out
personalization, `yafg` audits what a platform does once it believes it knows who you are.

> **Status: P3 — scale/resume/enrichment implemented; live multi-agent validation pending.**
> The runner expands single-persona experiments into persona × repetition arm matrices,
> schedules them with bounded concurrency, checkpoints safe resume state, supports PostgreSQL
> for parallel agents, and backfills YouTube metadata with optional public captions. See
> [ROADMAP](docs/ROADMAP.md).

---

## The hard boundary

**This project never creates accounts and never defeats anti-bot challenges.**

- Account provisioning is manual. `yafg account login <label>` opens a *headed*
  browser and waits for a human to sign in. The code types no credential and reads no
  password.
- A CAPTCHA, consent wall or "unusual activity" interstitial raises
  `ChallengeDetected`, writes an event, and **halts the run**. There is no solver, and
  a PR adding one will be rejected.
- Commenting is not implementable: `Interactions.comment` is typed `Literal[False]`.
  Reading your own feed is measurement; authoring public content under a synthetic
  identity is manipulation of a commons.
- Likes and subscribes are off by default and require *both* the experiment YAML and
  the global `YAFG_ALLOW_INTERACTIONS` env flag.

Automating a signed-in account is against YouTube's Terms of Service. That is a real
cost you are choosing to pay for a research result, not a detail to route around. Read
[docs/ETHICS.md](docs/ETHICS.md) before you provision anything, and do not point this
at accounts belonging to real people.

---

## How it works

```
persona (the variable)  ─┐
account (the identity)  ─┼─► agent loop ─► observations ─► analysis
context (the baseline)  ─┘
```

**Phase 1 — context.** Every agent watches an identical warm-up list. This pins a
shared algorithmic starting point, so anything that diverges afterwards is
attributable to the persona and not to cold-start noise.

**Phase 2 — exploration.** For N steps:

| stage | owner | what happens |
|---|---|---|
| observe | `browser/` | scrape every visible recommendation **with its rendered rank** |
| decide | `llm/` | model picks one candidate and justifies it — or random, for the baseline |
| act | `browser/` | actually watch it, for real wall-clock time |
| log | `store/` | persist the whole candidate set, the choice, the rank, the reason |

Behavior modes, per TRACE: `single` (one stable persona per run; multiple persona refs
expand into separate arms), `mixed` (weighted draw each step — one user, plural interests),
`sequential` (preferences evolving over time), and `random` (no model-visible persona; the
control that separates algorithmic effects from behavioural ones).

## Layout

```
src/yafg/
  personas/    persona schema, registry, prompt construction
  identity/    account envelope: profile dir, locale, tz, proxy, session health
  browser/     Playwright driver + surface scrapers (home, watch-next, search)
  agent/       the loop and the persona policies
  llm/         provider port; Claude and Ollama implementations
  store/       append-only SQLAlchemy model (SQLite or Postgres)
  enrich/      async YouTube Data API metadata + optional public-caption backfill
  experiment/  experiment schema, manifest resolution, P3 matrix scheduler/resume
  analysis/    rank-weighted metrics, overlap, drift, exporters
configs/       personas, contexts and experiments as versioned YAML
```

## Quickstart

```bash
mise install && mise run install
```

```bash
cp .env.example .env  # add ANTHROPIC_API_KEY, YOUTUBE_API_KEY
uv run alembic upgrade head
```

P2's local `create_all` bootstrap did not record an Alembic revision. When upgrading a
P2 database, run `uv run alembic current` first. If it prints no revision, stamp the P2
schema before applying P3; do not stamp a database that already reports a revision:

```bash
uv run alembic stamp 8f2c3a1d9b7e
uv run alembic upgrade head
```

```bash
uv run yafg persona lint
```

```bash
uv run yafg account login amina-01
```

```bash
uv run yafg run configs/experiments/demo-single-persona.yaml --dry-run
```

The dry run resolves the context/personas, validates every provisioned account, prints
the canonical manifest hash, full arm matrix, bounded-concurrency plan, and an LLM
token/cost envelope. It makes no browser or LLM call. Dollar estimates are shown only
when `YAFG_LLM_INPUT_USD_PER_MILLION` and `YAFG_LLM_OUTPUT_USD_PER_MILLION` are set.
For a visible first smoke test:

```bash
uv run yafg run configs/experiments/demo-single-persona.yaml --headed
```

Set `llm.provider: ollama` in an experiment to use the local Ollama chat API instead of
Claude. Random-baseline mode never invokes an LLM at all. Parallel live execution
(`concurrency > 1`) requires `postgresql+asyncpg://...`; SQLite remains the serial/local
default. An interrupted matrix can be continued with `--resume`, but only from a durable
checkpoint — the runner refuses ambiguous replay. Leases are attempt-scoped, so concurrent
resume commands cannot open the same profile. After confirming no process still owns an
orphaned lease, release it with the exact token shown by `yafg account list`:

```bash
uv run yafg account unlock amina-01 --lease-id <token>
```

After runs have produced video IDs, metadata enrichment is out-of-band:

```bash
uv run yafg enrich --limit 500
uv run yafg enrich --limit 100 --transcripts  # explicit best-effort public captions
```

The official Data API is used for video metadata. Transcript retrieval is intentionally
separate and opt-in because arbitrary caption downloads are not available through an
API-key-only Data API request. Public-caption requests default to 120 per hour and cannot
be configured above 240; a challenge or rate limit halts the transcript pass.

## Reproducibility

The stored `manifest_hash` is the SHA-256 of a canonical resolved manifest: experiment,
context, referenced persona snapshots in declared order, prompt version **and template
text**, implementation version, and the declared behavior seed. Editing a referenced
config or the prompt therefore changes experimental identity. Each repetition derives a
stable behavior seed from the declared seed; matching repetitions across single-persona
arms receive the same seed. Partial runs are kept — a run that dies at step 31 of 50 is
data, not garbage — and P3 checkpoints the RNG/session/history state after each durable
step so safe interruptions can resume without replaying account history.

## License

MIT. See [LICENSE](LICENSE).
