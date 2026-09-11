# Working in this repo

## Non-negotiable

Read [docs/ETHICS.md](docs/ETHICS.md) first. Do not add, and do not accept, code that:

- registers accounts, types credentials, or handles passwords;
- solves, bypasses or retries past CAPTCHAs / challenge interstitials;
- authors public content (comments, posts, uploads) under a synthetic identity;
- enables likes/subscribes by default or removes the `YAFG_ALLOW_INTERACTIONS` gate;
- raises `max_requests_per_hour` above a rate a human could plausibly browse at.

These are enforced by types and validators on purpose. Loosening a validator is a
change to the project's ethics posture and needs an ADR, not a commit.

## Conventions

- `uv` + `mise`. `mise run check` = lint + typecheck + test. Run it before committing.
- Python 3.13 (see ADR 0006), ruff line-length 120, `from __future__ import annotations`
  everywhere.
- Pydantic models for anything that crosses a boundary or comes from YAML.
  `extra="forbid"` on config models — a typo in a persona file must fail loudly.
- SQLAlchemy 2.0 typed ORM. **Never UPDATE an observation.** The store is evidence.
- Selectors live only in `browser/selectors.py`. A scrape that cannot parse a slot
  still records the slot with its rank and null metadata — never drop a slot, because
  dropping one shifts every rank below it and corrupts the measurement.
- New architectural choices go in `docs/DECISIONS.md` as a numbered ADR: what was
  chosen, what it rules out, why.

## Layering

`agent/` depends on ports (`browser`, `llm`, `store`), never on implementations.
No selectors, API keys or SQL in the loop. If a change needs the loop to know which
model or which DOM it is talking to, the port is wrong.
