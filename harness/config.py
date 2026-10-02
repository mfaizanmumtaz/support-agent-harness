"""Every setting that changes how the agent behaves, in one validated place.

A misspelt or contradictory setting is an error, not silently ignored: an
agent that quietly runs with the wrong limits is worse than one that won't start.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- the chat model
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)
    # Support answers should be short; this also caps the cost of one reply.
    max_output_tokens: int = Field(default=800, ge=50, le=8_000)

    # --- limits per turn, so a confused model can't loop or run up a bill
    model_call_limit: int = Field(default=5, ge=1, le=20)
    search_limit: int = Field(default=3, ge=1, le=10)
    # The whole turn: every model call and search included.
    turn_timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    # --- retrieval before the model
    # Search the knowledge with the visitor's message before the first model
    # call and put the results in the prompt. Most answers then need one model
    # call instead of two (decide to search, then answer).
    presearch: bool = True
    # "how much is it?" means nothing alone: a short follow-up is searched
    # together with the visitor's previous message.
    presearch_include_previous: bool = True
    presearch_short_message_words: int = Field(default=8, ge=0, le=50)

    # --- the handoff check: a fast classifier asks, while the search runs,
    # whether this message needs a person.
    # off: not run. shadow: recorded in the trace, changes nothing.
    # enforce: a visitor asking for a person, or a business rule matching,
    # makes the agent hand off. Frustration and sensitivity are only recorded.
    handoff_check: Literal["off", "shadow", "enforce"] = "off"
    handoff_check_threshold: float = Field(default=0.85, gt=0, le=1)
    # A slow check is dropped rather than keep the visitor waiting.
    handoff_check_timeout_seconds: float = Field(default=2.5, gt=0, le=30)
    handoff_check_max_rules: int = Field(default=10, ge=0, le=30)
    handoff_check_history_messages: int = Field(default=6, ge=0, le=30)

    # --- what the model is shown
    # Earlier messages beyond this are dropped, oldest first.
    history_token_budget: int = Field(default=3_000, ge=0, le=100_000)
    max_message_chars: int = Field(default=4_000, ge=100, le=50_000)

    @model_validator(mode="after")
    def _turn_outlasts_one_call(self) -> "AgentConfig":
        if self.turn_timeout_seconds < self.timeout_seconds:
            raise ValueError("turn_timeout_seconds must be at least timeout_seconds")
        return self


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Chunks of about this size, keeping a document's own sections together.
    # Small on purpose for FAQ-style knowledge: at 400 tokens whole pages merged
    # into one vector and "can I bring my dog?" scored 0.29 against the policies
    # page (below min_score); at about one section per chunk the Pets section
    # scores 0.40 and ranks first. Long-form documents want larger chunks.
    chunk_tokens: int = Field(default=80, ge=30, le=4_000)
    # Most chunks shown to the model per search.
    top_k: int = Field(default=4, ge=1, le=50)
    # Below this cosine similarity a chunk is noise, not an answer.
    min_score: float = Field(default=0.3, ge=-1, le=1)
    # The shown chunks together stay under this; the best one always goes in.
    max_context_tokens: int = Field(default=2_000, ge=100)
