from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------- Shared ----------------
class ImageInput(BaseModel):
    image_id: str
    data_ref: str  # local file path or URL (we will load local files -> data URLs)
    view_hint: Literal[
        "unknown", "front", "back", "left", "right", "top",
        "interior", "base", "detail", "rim", "foot", "signature"
    ] = "unknown"


class UserMetadata(BaseModel):
    height_in: Optional[float] = None
    weight_g: Optional[float] = None
    location_found: Optional[str] = None
    context_notes: Optional[str] = None


class AgentMeta(BaseModel):
    agent_name: str
    agent_version: str = "v1"
    timestamp_utc: str = Field(default_factory=utc_now)
    input_image_ids_used: List[str] = Field(default_factory=list)
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class CaseInput_v1(BaseModel):
    case_id: str
    images: List[ImageInput]
    user_metadata: UserMetadata = Field(default_factory=UserMetadata)


# ---------------- 1) Coverage ----------------
QualityFlag = Literal[
    "blur", "glare", "low_resolution", "cropped_object",
    "busy_background", "bad_white_balance", "perspective_distortion"
]


class MissingView(BaseModel):
    view: str
    why_it_matters: str
    how_to_shoot: str



class ViewsPresent(BaseModel):
    front: bool = False
    back: bool = False
    left: bool = False
    right: bool = False
    top: bool = False
    interior: bool = False
    base: bool = False
    base_macro: bool = False
    signature_macro: bool = False
    # add/remove fields to match the views you actually use


class CoverageReport_v1(AgentMeta):
    case_id: str
    coverage_score_0_1: float = Field(ge=0.0, le=1.0)
    quality_flags: List[QualityFlag] = Field(default_factory=list)
    views_present: ViewsPresent
    missing_high_value_views: List[MissingView] = Field(default_factory=list)


# ---------------- 2) Visual Evidence ----------------
ObjectClass = Literal["vase", "bowl", "plate", "jar", "teapot", "mug", "figurine", "tile", "unknown"]
MaterialLabel = Literal["porcelain", "stoneware", "earthenware", "terracotta", "unknown"]
ConstructionMethod = Literal["wheel_thrown", "molded", "slip_cast", "handbuilt", "unknown"]
FootType = Literal["ring_foot", "flat", "trimmed_foot", "unknown"]

Symmetry = Literal["symmetric", "asymmetric", "unknown"]
Neck = Literal["none", "short", "medium", "tall", "unknown"]
Shoulder = Literal["rounded", "angular", "sloped", "unknown"]
Rim = Literal["straight", "flared", "rolled", "everted", "unknown"]

Sheen = Literal["matte", "satin", "gloss", "mixed", "unknown"]
GlazeEffect = Literal[
    "crazing", "crystalline", "running", "pooling", "crawling",
    "pinholing", "speckling", "none_detected"
]
ApplicationMethod = Literal["sprayed", "dipped", "brushed", "poured", "unknown"]

DecorationPresence = Literal["none", "minimal", "moderate", "heavy", "unknown"]
Motif = Literal["floral", "geometric", "landscape", "figural", "abstract", "calligraphy", "unknown"]
DecorationTechnique = Literal[
    "underglaze_paint", "overglaze_enamel", "transfer",
    "incised", "relief", "slip", "unknown"
]

ConditionFlag = Literal[
    "possible_restoration", "chips", "hairlines", "staining", "kiln_flaws", "none_detected"
]


class EstimatedMaterial(BaseModel):
    label: MaterialLabel
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    rationale: List[str] = Field(default_factory=list)


class Construction(BaseModel):
    method: ConstructionMethod
    indicators: List[str] = Field(default_factory=list)
    foot_type: FootType
    seams_present: bool


class FormGeometry(BaseModel):
    symmetry: Symmetry
    neck: Neck
    shoulder: Shoulder
    rim: Rim
    proportions_notes: List[str] = Field(default_factory=list)


class GlazeSurface(BaseModel):
    sheen: Sheen
    effects: List[GlazeEffect] = Field(default_factory=list)
    color_behavior_notes: List[str] = Field(default_factory=list)
    application_method_guess: ApplicationMethod
    wear_notes: List[str] = Field(default_factory=list)


class Decoration(BaseModel):
    presence: DecorationPresence
    motif_categories: List[Motif] = Field(default_factory=list)
    technique_guess: DecorationTechnique
    layout_notes: List[str] = Field(default_factory=list)


class AgingCondition(BaseModel):
    condition_score_0_10: float = Field(ge=0.0, le=10.0)
    flags: List[ConditionFlag] = Field(default_factory=list)
    rationale: List[str] = Field(default_factory=list)


class VisualEvidencePayload(BaseModel):
    object_class: ObjectClass
    estimated_material: EstimatedMaterial
    construction: Construction
    form_geometry: FormGeometry
    glaze_surface: GlazeSurface
    decoration: Decoration
    aging_condition: AgingCondition


class VisualEvidence_v1(AgentMeta):
    case_id: str
    evidence: VisualEvidencePayload


# ---------------- 3) Marks ----------------
MarkType = Literal["impressed", "incised", "painted", "stamped", "label", "unknown"]
MarkLocation = Literal["base", "interior", "side", "unknown"]
Legibility = Literal["high", "medium", "low"]
UnderOver = Literal["under", "over", "unknown"]


class Mark(BaseModel):
    mark_type: MarkType
    location: MarkLocation
    text_or_symbols: str
    legibility: Legibility
    under_over_glaze: UnderOver
    interpretation_notes: List[str] = Field(default_factory=list)


class MarksReport_v1(AgentMeta):
    case_id: str
    marks_present: bool
    marks: List[Mark] = Field(default_factory=list)
    photo_instructions_if_unclear: List[str] = Field(default_factory=list)


# ---------------- 4) Hypotheses ----------------
class HighestValueTest(BaseModel):
    request: str
    expected_result_if_true: str
    expected_result_if_false: str


class Hypothesis(BaseModel):
    rank: int
    candidate_label: str
    maker_or_cluster: str
    region: str
    date_range: str
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    supporting_evidence: List[str]
    contradictions: List[str]
    highest_value_test: HighestValueTest


class Hypotheses_v1(AgentMeta):
    case_id: str
    hypotheses: List[Hypothesis]


# ---------------- 5) Final attribution ----------------
ConfidenceBand = Literal["High", "Medium", "Low", "VeryLow"]


class FinalChoice(BaseModel):
    rank: int
    attribution: str
    confidence_band: ConfidenceBand
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    why: List[str]
    why_not_others: List[str]
    missing_evidence_penalties: List[str]
    what_would_change_mind: List[str]


class NextPhoto(BaseModel):
    view: str
    reason: str
    how_to_shoot: str


class FinalPayload(BaseModel):
    top_3: List[FinalChoice]
    overall_summary: str
    recommended_next_photos: List[NextPhoto]


class FinalAttribution_v1(AgentMeta):
    case_id: str
    final: FinalPayload


# ---------------- 6) Value ----------------
class VenueAdjustment(BaseModel):
    venue: Literal["thrift_flip", "ebay", "regional_auction", "specialist_auction", "dealer_retail"]
    multiplier: float
    notes: str


class ValuePayload(BaseModel):
    currency: Literal["USD"] = "USD"
    range_low: float
    range_mid: float
    range_high: float
    venue_adjustments: List[VenueAdjustment]
    drivers: List[str]
    risks: List[str]


class ValueEstimate_v1(AgentMeta):
    case_id: str
    value: ValuePayload
