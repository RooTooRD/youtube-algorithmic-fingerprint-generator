"""The agent loop.

Two phases, per TRACE §3.1:

  Phase 1 — context: watch an identical warm-up list, persona inert. Establishes a
            shared algorithmic baseline so later divergence is attributable.
  Phase 2 — exploration: repeat `steps` times —
              observe  : scrape every visible recommendation with its rank
              decide   : LLM picks one candidate + justification (or random baseline)
              act      : watch it for a sampled duration
              log      : persist the full candidate set, the choice, and the reason

The loop owns no platform knowledge (that is the driver) and no model knowledge (that
is the provider). It owns the experimental protocol, and that is the thing worth
keeping stable.
"""

from __future__ import annotations

from dataclasses import dataclass

from yafg.browser.driver import YouTubeDriver
from yafg.experiment.schema import Context, Experiment
from yafg.llm.base import LLMProvider
from yafg.personas.schema import Persona


@dataclass(slots=True)
class PersonaPolicy:
    """Resolves which persona is active at step N, per behavior mode.

    single      : always the one persona
    mixed       : weighted random draw at every step (one user, plural interests)
    sequential  : switch every `switch_every` steps (preferences evolving over time)
    random      : no persona; uniform choice over candidates — the baseline that
                  isolates algorithmic effects from behavioural ones
    """

    experiment: Experiment
    personas: dict[str, Persona]

    def active_at(self, step_index: int) -> Persona | None:
        raise NotImplementedError("P2")


class AgentRun:
    def __init__(
        self,
        *,
        experiment: Experiment,
        context: Context,
        policy: PersonaPolicy,
        driver: YouTubeDriver,
        llm: LLMProvider,
        run_id: str,
    ) -> None:
        self.experiment = experiment
        self.context = context
        self.policy = policy
        self.driver = driver
        self.llm = llm
        self.run_id = run_id

    async def execute(self) -> None:
        """Run both phases to completion, or fail loudly.

        Failure policy: a ChallengeDetected or LoginRequired marks the run `failed`
        and leaves every step already written intact. Partial runs are analysable and
        are never silently discarded — a run that dies at step 31 of 50 is data.
        """
        raise NotImplementedError("P2")

    async def _run_context_phase(self) -> None:
        raise NotImplementedError("P2")

    async def _run_exploration_phase(self) -> None:
        raise NotImplementedError("P2")
