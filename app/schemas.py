"""Pydantic models for agent outputs and API payloads (CLAUDE.md rule 2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Pramana = Literal["pratyaksha", "anumana", "upamana", "shabda"]
FallacyType = Literal["asiddha", "savyabhichara", "viruddha", "satpratipaksha", "badhita"]
Mode = Literal["vada", "jalpa"]
Stage = Literal["purvapaksha", "khandana", "uttarapaksha", "siddhanta", "verdict"]
ClaimStatus = Literal["verified", "inferred", "dropped"]


# ---------------------------------------------------------------- knowledge

class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    persona_id: str
    work: str
    edition: str
    section: str
    language: str
    license: str
    url: str
    text: str
    topics: list[str] = []
    source_type: str = "primary"  # primary | account | scholarship | background


class Stance(BaseModel):
    statement: str
    source_id: str


class SupportPoint(BaseModel):
    chunk_id: str
    angle: str


class Topic(BaseModel):
    id: str
    persona_id: str
    question: str
    concept_id: str
    stance: Stance
    support: list[SupportPoint] = []
    handbook_ref: str = ""
    era: Literal["classical", "modern"] = "classical"
    scope: Literal["", "world", "india"] = ""  # modern topics: "india" = Indian public policy
    context: str = ""  # modern topics: a neutral briefing on today's situation


class WeakPoint(BaseModel):
    topic: str
    label: str = ""
    effective_pramana: Pramana
    note: str = ""


class Persona(BaseModel):
    id: str
    name: str
    era: str
    role: str
    region_id: str
    primary_works: list[str]
    core_positions: list[dict]
    weak_points: list[WeakPoint]
    style_notes: str
    difficulty: dict[Pramana, int]
    threshold: float = 55
    disclaimer: str
    primer: dict = {}
    knowledge_nodes: list[dict] = []


# ---------------------------------------------------------------- agent outputs

class Structure(BaseModel):
    pratijna: str | None = None
    hetu: str | None = None
    udaharana: str | None = None
    upanaya: str | None = None
    nigamana: str | None = None

    def present(self) -> int:
        return sum(1 for v in self.model_dump().values() if v)


class Fallacy(BaseModel):
    type: FallacyType
    span: str
    explanation: str
    fix: str


class PramanaTag(BaseModel):
    span: str
    pramana: Pramana


class Analysis(BaseModel):
    structure: Structure
    fallacies: list[Fallacy] = []
    pramanas: list[PramanaTag] = []
    tensions: list[str] = []
    method: Literal["llm", "rules"] = "rules"


class Claim(BaseModel):
    claim_id: str
    text: str
    source_ids: list[str] = []
    inferred: bool = False


class PersonaReply(BaseModel):
    text: str
    claims: list[Claim]
    method: Literal["llm", "scripted"] = "scripted"


class VerifiedClaim(BaseModel):
    claim_id: str
    text: str
    status: ClaimStatus
    source_ids: list[str]
    nli_label: Literal["entailment", "neutral", "contradiction"] | None = None
    nli_prob: float | None = None
    support: str | None = Field(default=None, description="Sentence in the cited passage that best supports the claim")
    method: Literal["nli", "lexical"] = "lexical"


class Scores(BaseModel):
    structure: float = Field(ge=0, le=1)
    evidence: float = Field(ge=0, le=1)
    pramana_fit: float = Field(ge=0, le=1)
    responsiveness: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)


class RefereeResult(BaseModel):
    scores: Scores
    reasons: dict[str, str]
    persona_in_character: bool = True
    persona_note: str = ""
    method: Literal["llm", "rules"] = "rules"


# ---------------------------------------------------------------- API payloads

class CreateDebate(BaseModel):
    persona_id: str
    topic_id: str
    mode: Mode = "jalpa"
    clan_id: Literal["nyaya", "vedanta", "charvaka", "arthashastra"] | None = None
    user_id: str = "demo"
    language: str = "en"
    stance: Literal["agree", "disagree"] | None = None  # stance gate: agree = learning (vada)

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        if self.stance == "agree":
            self.mode = "vada"


class SubmitTurn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
