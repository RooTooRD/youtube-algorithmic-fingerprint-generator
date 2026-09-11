# Decisions

Append-only. Each entry records what was chosen, what it rules out, and why.

## 0001 — Signed-in accounts, not anonymous sessions

**2026-09-11**

TRACE audits signed-out personalization with disposable browser contexts. This project
binds each persona to a real, human-provisioned Google account instead.

Buys: a home feed, subscriptions, cross-session memory, and whatever profile-level
inference the platform applies once it believes it knows a user — none of which exist
signed-out. That is the research question.

Costs: ToS violation, suspension risk, a manual provisioning step that cannot be
scaled by code, and an identity envelope (locale, timezone, proxy) that must stay
pinned per account for its whole life or the account becomes a confound.

Ruled out: automated account creation and CAPTCHA solving, permanently. See
[ETHICS](ETHICS.md).

## 0002 — Playwright persistent contexts over a Selenium Grid

**2026-09-11**

TRACE used a Selenium Grid (hub + N nodes) for parallelism. With signed-in accounts,
the durable unit is the **browser profile directory**, not the node — cookies, local
storage and login state must survive between sessions and must never be copied between
accounts. `launch_persistent_context` maps to that directly; a Grid would add a hop
that the account model does not need.

Costs: no drop-in replication of TRACE's infra; parallelism is process-level and
bounded by local resources. Acceptable at the tens-of-agents scale this targets.

## 0003 — Persona, account and context are three separate objects

**2026-09-11**

The persona is the experimental variable; the account is identity material; the
context is the shared baseline exposure. Collapsing any two makes a divergence
unattributable — if persona A ran on a Belgian account and persona B on an Algerian
one, the feed difference explains nothing.

Consequence: an account is bound 1:1 to a persona for the life of a study, and reusing
one across personas invalidates its history. The `Experiment` validator refuses to
share an account between concurrent arms.

## 0004 — The LLM chooses *what*, never *how fast* or *how long*

**2026-09-11**

Model decisions are limited to selecting one candidate from a ranked list plus a
justification. Pacing, watch depth, skip probability and session structure are sampled
from `ViewingHabits`.

Rationale: those are the knobs that need to be reproducible and cheap, and a model
asked to decide them produces variance that cannot be separated from the effect under
study. A failed or malformed model response falls back to rank 0 and is logged — an
agent that cannot decide behaves like a user who takes the top result, which is a
defensible default rather than a dead run.

## 0005 — Append-only store, snapshot-on-start

**2026-09-11**

Every row is evidence: the full candidate set at every step with rendered ranks, not
just the chosen video. The resolved persona and context are snapshotted into the run
row and the experiment YAML is hashed.

Consequence: editing a persona file cannot retroactively change what a finished run
meant, and partial runs stay analysable. Nothing in the codebase may UPDATE an
observation.

## 0006 — Python 3.13, not 3.14

**2026-09-11**

Every other project in this workspace pins 3.14. This one pins 3.13 because Playwright
and greenlet wheel coverage lags a release, and the browser layer is the one place
where a source build is genuinely painful. Revisit once `playwright` publishes 3.14
wheels.
