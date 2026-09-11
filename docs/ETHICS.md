# Ethics and operating limits

This framework builds synthetic identities on a live platform. That is defensible as
research and indefensible as a growth tool. The difference is entirely in the
constraints, so the constraints are enforced in code rather than left to intent.

## Enforced in code

| Limit | Where |
|---|---|
| No account registration | `identity/` — provisioning is a headed browser and a human |
| No credential handling | Nothing in this repo accepts a password field |
| No CAPTCHA / challenge solving | `ChallengeDetected` halts the run; no solver exists |
| No public content authoring | `Interactions.comment: Literal[False]` |
| Likes/subscribes off by default | Experiment YAML **and** `YAFG_ALLOW_INTERACTIONS=1` |
| Request ceiling per agent | `Pacing.max_requests_per_hour`, default 180–240 |
| One account per arm | `Experiment` validator; accounts are never shared concurrently |

## Not enforceable in code — your responsibility

- **Terms of Service.** Automating a signed-in Google account violates YouTube's ToS.
  Accounts can and do get suspended. Never use an account that matters, never use an
  account belonging to a real person, and never use recovery details belonging to one.
- **Volume.** Ten agents is a study. Ten thousand is a bot farm regardless of what the
  YAML says. Keep the agent count proportionate to the question being asked.
- **No engagement manipulation.** Do not use this to inflate views, likes,
  subscriptions or watch time on any channel, your own included. The interaction
  budget exists to model a realistic viewer, not to move a metric.
- **Pre-registration.** For anything you intend to publish, fix the seed videos, the
  persona set and the scoring rubric *before* the first run. TRACE's lean results are
  credible because their stance-balanced seeds were chosen in advance.
- **Sensitive topics.** Auditing recommendations around religion, ethnicity or
  conflict is legitimate and is much of the point. Reporting a synthetic agent's
  trajectory as evidence about how *real* members of that group behave is not — the
  personas are probes into the algorithm, not models of people.
- **Data.** Enriched video metadata and transcripts are third-party content. Publish
  aggregates and video IDs; do not redistribute scraped corpora.

## If you are doing this institutionally

Get ethics-board sign-off before provisioning. Most IRBs treat platform audits with
synthetic profiles as non-human-subjects research, but that determination is theirs to
make, not yours, and the answer changes if any real user's data enters the pipeline.
