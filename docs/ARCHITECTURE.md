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
- Metadata enrichment failure → `metadata_status=error`; retried on the next pass. A
  missing/deleted public resource is recorded as `not_found`. Transcript state is tracked
  separately because public captions are not guaranteed to exist or be retrievable.
- Resume only proceeds from a committed checkpoint. Evidence newer than the checkpoint
  means browser side effects may be ambiguous, so automatic replay is refused.

## Concurrency

P3 expands `single` experiments into persona × repetition runs, then schedules those runs
through an `asyncio` semaphore. Each planned run owns one distinct persistent account/profile.
Mixed/sequential/random modes remain one policy arm per repetition. PostgreSQL is required
for live `concurrency > 1`; SQLite remains the default for serial research and dry-run
planning. Failures are isolated per arm so other scheduled runs can finish. Before a browser
profile is opened, the account registry atomically acquires a non-reentrant, attempt-scoped
lease; login/check commands use the same mechanism. An orphan can only be released manually
with its exact observed token after the operator verifies no process still owns the profile.

After each durable watch (and after a decision failure that changed no account state), the
agent stores its next phase index, global evidence index, history, session counters, request
`not_before`, and JSON-serialised PRNG state in `Run.checkpoint`. Before opening a browser or
collecting a surface it persists an in-flight marker. `--resume` reuses only the latest
incomplete experiment with the exact same manifest hash and rejects in-flight operations or
evidence newer than the checkpoint rather than guessing whether a platform side effect occurred.
