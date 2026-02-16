from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from schemas_shared import AgentMeta, Attribution_level, Evidence_anchor, Test_type


# ---------------- 4) Hypotheses ----------------
class HighestValueTest(BaseModel):
    test_type: Test_type = Field(
        default="photo",
        description="Prefer 'photo' or 'measurement' since pipeline runs offline (no web).",
    )
    request: str = Field(description="Concrete test the user can do (photo/measurement/comparison).")
    expected_result_if_true: str
    expected_result_if_false: str


class Hypothesis(BaseModel):
    rank: int = Field(ge=1, le=10)
    candidate_label: str = Field(
        description="Short human label (broad). Example: 'European studio pottery cluster' or 'East Asian porcelain cluster'."
    )
    maker_or_cluster: str = Field(description="If maker is unknown, write a cluster descriptor (not a guessy maker name).")
    attribution_level: Attribution_level = Field(
        default="unknown",
        description="Granularity of this hypothesis. Avoid 'maker' unless mark literally supports it.",
    )
    region: str
    date_range: str = Field(
        description="Free text, but keep it bounded (e.g., 'c. 1920–1950', 'late 19th–early 20th c.')."
    )
    confidence_0_1: float = Field(ge=0.0, le=1.0)

    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Observable support statements (no web claims).",
        max_length=10,
    )
    supporting_anchors: List[Evidence_anchor] = Field(
        default_factory=list,
        description="Optional anchors into UPSTREAM_CONTEXT_JSON for your strongest support statements.",
        max_length=10,
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="What doesn’t fit / what would need to be true for this hypothesis to hold.",
        max_length=10,
    )
    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Key missing evidence that prevents narrowing (base_macro, signature_macro, interior, etc.).",
        max_length=8,
    )
    discriminators: List[str] = Field(
        default_factory=list,
        description="High-value discriminators that separate this from close alternatives (what to look for).",
        max_length=8,
    )
    highest_value_test: HighestValueTest


class Hypotheses_v1(AgentMeta):
    case_id: str
    hypotheses: List[Hypothesis] = Field(default_factory=list, max_length=10)
    global_limitations: List[str] = Field(
        default_factory=list,
        description="Pipeline-level limitations derived from coverage/marks (max 8).",
        max_length=8,
    )

