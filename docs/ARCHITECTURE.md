# Architecture

## Boundaries

Four ports; nothing crosses them implicitly.

| Port | Module | Contract |
|---|---|---|
| `YouTubeDriver` | `browser/driver.py` | All platform knowledge. Every side effect on YouTube passes through here, which is what makes the interaction budget enforceable in one place. |
| `LLMProvider` | `llm/base.py` | Given a persona prompt and a ranked candidate list, return one rank + a justification. Nothing else. |
| store | `store/evidence.py` | Evidence port; SQLAlchemy adapter. No code may UPDATE an observation. |
| enricher | `enrich/` | Async, out-of-band. The browsing loop never blocks on metadata. |

The agent loop (`agent/loop.py`) owns the *protocol* and depends only on those ports.
It has no selectors, no API keys and no SQL. That is the layer worth keeping stable
while YouTube's DOM and the model lineup both churn underneath it.

## The three-object separation

```
Persona   the experimental variable    (varies between arms)
Account   the identity material        (pinned 1:1 to a study arm)
Context   the shared warm-up exposure  (identical across all arms)
```

Collapsing any two makes a result unattributable. See ADR 0003.

## Data flow

```
configs/*.yaml ──► resolve + validate ──► snapshot into Run row
                                             │
                        ┌────────────────────┘
                        ▼
   phase 1: watch context list  (persona inert)
                        │
                        ▼
   phase 2, ×N steps:
        collect(surface) ─► [Candidate × rank]
                              │
                              ├─► Observation rows  (ALL of them, ranked)
                              ▼
                          LLMProvider.choose
                              │
                              ├─► Step row (chosen rank, justification, usage)
                              ▼
                          driver.watch
                              │
                              └─► Event rows (navigation, waits, challenges)

   later, async:  enrich worker ──► Video rows (metadata, transcripts)
   later, offline: analysis ──► parquet, metrics, plots
```

## Why every candidate is stored, not just the choice

The measurement is the *offered set* and its ordering. What the agent picked is
behaviour; what it was shown is the algorithm. Rank-weighted metrics need the full
ranked list at every step, and a study that only logged the chosen video could not
compute them retroactively.

## Failure model

- `LoginRequired` / `ChallengeDetected` → run marked `failed`, steps so far kept.
- Malformed/provider-failed LLM response → provider retries once, proposes deterministic
  rank 0 with a fallback reason, then the agent validates that rank against the rendered
  set and chooses the first selectable slot if rank 0 itself is unselectable.
- Provider crash or a rendered set with no selectable video → the full candidate set is
  persisted with no chosen video before the run fails.
- Scrape parse failure on a slot → the slot is still recorded with its rank and null
  metadata. A missing title must not shift the ranks of its neighbours.
- Enrichment failure → `Video.enriched_at` stays null; retried on the next pass.

## Concurrency

P2 executes repetitions serially (`concurrency=1`), one persistent browser profile per
account and one distinct account per repetition. P3 adds bounded multi-agent concurrency.
Postgres is the store once agents run in parallel; SQLite is fine for a single researcher
and stays the default so a fresh clone runs with no services.
