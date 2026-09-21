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

> **Status: P1 — identity and observation implemented.** Persistent account profiles,
> session health checks, home/watch-next observation, real-time watching and the P1 CLI
> are implemented. A live signed-in smoke test is still required in your environment;
> the agent/LLM loop begins in P2. See [ROADMAP](docs/ROADMAP.md).

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

Behavior modes, per TRACE: `single` (stable identity), `mixed` (weighted draw each
step — one user, plural interests), `sequential` (preferences evolving over time),
`random` (no persona; the control that separates algorithmic effects from behavioural
ones).

## Layout

```
src/yafg/
  personas/    persona schema, registry, prompt construction
  identity/    account envelope: profile dir, locale, tz, proxy, session health
  browser/     Playwright driver + surface scrapers (home, watch-next, search)
  agent/       the loop and the persona policies
  llm/         provider port; Claude and Ollama implementations
  store/       append-only SQLAlchemy model (SQLite or Postgres)
  enrich/      async YouTube Data API metadata backfill
  experiment/  experiment schema, arm matrix, runner
  analysis/    rank-weighted metrics, overlap, drift, exporters
configs/       personas, contexts and experiments as versioned YAML
```

## Quickstart

```bash
mise install && mise run install
```

```bash
cp .env.example .env  # add ANTHROPIC_API_KEY, YOUTUBE_API_KEY
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

## Reproducibility

Personas and experiments are snapshotted into the run row at start, and the experiment
YAML is hashed into `manifest_hash`. Editing a persona file can never retroactively
change what a finished run meant. Partial runs are kept — a run that dies at step 31
of 50 is data, not garbage.

## License

MIT. See [LICENSE](LICENSE).
