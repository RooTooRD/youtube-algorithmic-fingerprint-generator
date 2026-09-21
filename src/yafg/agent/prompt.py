"""Versioned persona-to-system-prompt construction.

The template text, not only its version label, is included in the experiment manifest.
That makes a prompt edit an explicit change to experimental identity.
"""

from __future__ import annotations

import json

from yafg.personas.schema import Persona

PROMPT_VERSION = "persona-choice-v1"
PROMPT_TEMPLATE = """You are choosing the next YouTube video for a synthetic research persona.

Research protocol:
- Choose exactly one item from the ranked candidates supplied by the caller.
- Choose WHAT this persona would plausibly watch next; do not decide watch time, pacing,
  likes, subscriptions, or comments.
- Treat candidate ranks as opaque identifiers and never invent a rank that is not present.
- Use only the persona profile, recent selected-video history, and candidate metadata.
- Do not use or infer hidden information about the real account holder; this is a synthetic persona.
- Keep the justification concise and describe the preference signal that drove the choice.

Synthetic persona profile:
{persona_profile}
"""


def persona_prompt_payload(persona: Persona) -> dict[str, object]:
    """Return only fields intended to influence model choice.

    ``priors`` are deliberately excluded: they are analysis labels, not behavior cues.
    Notes are excluded because they are researcher annotations rather than persona state.
    """
    return {
        "id": persona.id,
        "display_name": persona.display_name,
        "demographics": persona.demographics.model_dump(mode="json"),
        "interests": [interest.model_dump(mode="json") for interest in persona.interests],
        "bio": persona.bio,
    }


def render_persona_prompt(persona: Persona) -> str:
    profile = json.dumps(persona_prompt_payload(persona), ensure_ascii=False, sort_keys=True, indent=2)
    return PROMPT_TEMPLATE.format(persona_profile=profile)
