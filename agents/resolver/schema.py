from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from schemas_shared import AgentMeta, Attribution_level, Evidence_anchor


# ---------------- 5) Final attribution ----------------
ResolutionOutcome = Literal["attribution", "inconclusive"]

ConfidenceBand = Literal["High", "Medium", "Low", "VeryLow"]


class FinalChoice(BaseModel):
    rank: int = Field(ge=1, le=3)

    attribution: str = Field(
        description="If evidence is weak, keep this broad (region/style/cluster) rather than a specific maker guess."
    )
    attribution_level: Attribution_level = Field(
        default="unknown",
        description="Granularity of attribution. Avoid 'maker' unless marks literally support it.",
    )

    confidence_band: ConfidenceBand
    confidence_0_1: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated probability-like confidence for this specific choice, accounting for missing evidence.",
    )

    why: List[str] = Field(default_factory=list, max_length=10)
    why_not_others: List[str] = Field(default_factory=list, max_length=10)

    supporting_anchors: List[Evidence_anchor] = Field(
        default_factory=list,
        description="Anchor your strongest 'why' points to UPSTREAM_CONTEXT_JSON paths.",
        max_length=12,
    )
    contradiction_anchors: List[Evidence_anchor] = Field(
        default_factory=list,
        description="Optional anchors for key contradictions/why_not_others points.",
        max_length=10,
    )

    missing_evidence_penalties: List[str] = Field(
        default_factory=list,
        description="Specific missing or low-quality evidence items that reduce confidence (e.g., base_macro missing/blur).",
        max_length=10,
    )
    key_discriminators: List[str] = Field(
        default_factory=list,
        description="2–6 discriminators that separate this choice from close alternatives.",
        max_length=6,
    )
    what_would_change_mind: List[str] = Field(
        default_factory=list,
        description="Concrete evidence that would materially change ranking/confidence.",
        max_length=8,
    )


class NextPhoto(BaseModel):
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="1=highest ROI. Order should match priority.",
    )
    view: str = Field(
        description="Human-readable request (e.g., 'base_macro', 'interior', 'rim/top', 'left profile', 'signature_macro')."
    )
    reason: str
    how_to_shoot: str
    maps_to_missing_evidence: List[str] = Field(
        default_factory=list,
        description="Which missing_evidence_penalties/discriminators this photo addresses.",
        max_length=6,
    )


class FinalPayload(BaseModel):
    outcome: ResolutionOutcome = Field(
        default="attribution",
        description="Use 'inconclusive' when evidence does not support narrowing beyond very broad clusters.",
    )
    top_3: List[FinalChoice] = Field(
        default_factory=list,
        description="Exactly 3 ranked choices (broad if needed).",
        max_length=3,
    )
    overall_summary: str = Field(description="Short, evidence-grounded summary. No web-only claims.")
    global_limitations: List[str] = Field(
        default_factory=list,
        description="Pipeline-level limitations (coverage/marks quality) that constrain attribution.",
        max_length=8,
    )
    recommended_next_photos: List[NextPhoto] = Field(
        default_factory=list,
        description="Up to 5 ROI-ordered photo requests; do not request views already present unless retake needed due to quality.",
        max_length=5,
    )


class FinalAttribution_v1(AgentMeta):
    case_id: str
    final: FinalPayload

