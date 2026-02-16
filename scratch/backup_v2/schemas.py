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
    #agent_version: str = "v1"
    #timestamp_utc: str = Field(default_factory=utc_now)

    input_image_ids_used: List[str] = Field(default_factory=list,
        description="The image IDs that were used to generate the report."
        )

    confidence_0_1: float = Field(ge=0.0, le=1.0,
        description="The confidence score of the report. 0.0 is the lowest confidence and 1.0 is the highest confidence."
        )

    notes: str = Field(default="",
        description="The notes of the report."
        )


class CaseInput_v1(BaseModel):
    case_id: str
    images: List[ImageInput]
    user_metadata: UserMetadata = Field(default_factory=UserMetadata)


# ---------------- 1) Coverage ----------------

CoverageView = Literal[
    "front",
    "back",
    "left",
    "right",
    "top",
    "interior",
    "base",
    "base_macro",
    "signature_macro",
    "detail",
    "rim",
    "foot",
    "unknown",
    ]

QualityFlag = Literal[
    "blur",
    "motion_blur",
    "glare",
    "low_resolution",
    "cropped_object",
    "busy_background",
    "bad_white_balance",
    "perspective_distortion",
    "underexposed",
    "overexposed",
    "too_far",
    ]


# class MissingView(BaseModel):
#     view: CoverageView = Field(
#         description="A view that is missing or insufficient. Prefer high-ROI views like base/base_macro/signature_macro."
#     )
#     why_it_matters: str = Field(
#         description="1-2 sentences: what identification signal this view would unlock (marks, foot, glaze pooling, etc.)."
#     )
#     how_to_shoot: str = Field(
#         description="Concrete instructions (lighting, distance, angle, focus). Mention macro + raking light for marks."
#     )


class ViewsPresent(BaseModel):
    front: bool = Field(default=False, description="True only if the object front is clearly shown and in focus.")
    back: bool = Field(default=False, description="True only if the back side is clearly shown and in focus.")
    left: bool = Field(default=False, description="True only if left profile is clearly shown and in focus.")
    right: bool = Field(default=False, description="True only if right profile is clearly shown and in focus.")
    top: bool = Field(default=False, description="True only if a top-down/rim view is clearly shown and in focus.")
    interior: bool = Field(default=False, description="True only if interior is visible enough to assess glaze/throwing.")
    base: bool = Field(default=False, description="True only if the full base is visible and reasonably sharp.")
    base_macro: bool = Field(
        default=False,
        description="True only if there is a close-up of the base/foot/mark area with readable detail.",
    )
    signature_macro: bool = Field(
        default=False,
        description="True only if any signature/chop/label is close-up and potentially legible.",
    )


class ImageAssessment(BaseModel):
    image_id: str
    view_detected: CoverageView = Field(
        description="Your best guess of what the image actually shows; use 'unknown' if unclear."
    )
    usable_for_id: bool = Field(
        description="True if the photo is sharp/close/clear enough to support identification decisions."
    )
    quality_flags: List[QualityFlag] = Field(
        default_factory=list,
        description="Only include flags that materially reduce usability for ID.",
    )
    notes: str = Field(default="", description="Short note on what is/ isn't visible and why.")


class CoverageReport_v1(AgentMeta):
    case_id: str
    coverage_score_0_1: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall view coverage quality. Be conservative; penalize missing base/base_macro/signature_macro.",
        )
    views_present: ViewsPresent = Field(
        description="Boolean checklist of high-value views present with sufficient clarity."
        )
    image_assessments: List[ImageAssessment] = Field(
        default_factory=list,
        description="One entry per image_id. Helps downstream agents trust/discount evidence.",
        )
    # missing_high_value_views: List[MissingView] = Field(
    #     default_factory=list,
    #     description="Up to 5 missing/insufficient views, ROI-ordered. If coverage is good, return empty list.",
    #     max_length=5,
    #     )

# ---------------- 2) Visual Evidence ----------------
TriState = Literal["yes", "no", "unknown"]
ObjectClass = Literal["vase", "bowl", "plate", "jar", "teapot", "mug", "figurine", "tile", "unknown"]
MaterialLabel = Literal["porcelain", "stoneware", "earthenware", "terracotta", "unknown"]
ConstructionMethod = Literal["wheel_thrown", "molded", "slip_cast", "handbuilt", "unknown"]
FootType = Literal["ring_foot", "flat", "trimmed_foot", "unknown"]

Symmetry = Literal["symmetric", "asymmetric", "unknown"]
Neck = Literal["none", "short", "medium", "tall", "unknown"]
Shoulder = Literal["rounded", "angular", "sloped", "unknown"]
Rim = Literal["straight", "flared", "rolled", "everted", "unknown"]

Sheen = Literal["matte", "satin", "gloss", "mixed", "unknown"]
GlazeEffect = Literal["crazing", "crystalline", "running", "pooling", "crawling", "pinholing", "speckling", "none_detected"]
ApplicationMethod = Literal["sprayed", "dipped", "brushed", "poured", "unknown"]

DecorationPresence = Literal["none", "minimal", "moderate", "heavy", "unknown"]
Motif = Literal["floral", "geometric", "landscape", "figural", "abstract", "calligraphy", "unknown"]
DecorationTechnique = Literal["underglaze_paint", "overglaze_enamel", "transfer", "incised", "relief", "slip", "unknown"]

ConditionFlag = Literal["possible_restoration", "chips", "hairlines", "staining", "kiln_flaws", "none_detected"]


class EstimatedMaterial(BaseModel):
    label: MaterialLabel = Field(description="Use 'unknown' if you cannot distinguish confidently.")
    confidence_0_1: float = Field(ge=0.0, le=1.0,
        description="The confidence score of the material label. 0.0 is the lowest confidence and 1.0 is the highest confidence."
        )
    rationale: List[str] = Field(
        default_factory=list,
        description="Bullet-like observations (no attribution). Prefer falsifiable cues: translucency, ring, porosity, glaze fit.",
        max_length=8,
        )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs that directly support the material call. Only include if clearly visible.",
        max_length=6,
        )


class Construction(BaseModel):
    method: ConstructionMethod = Field(description="Use 'unknown' if construction indicators are not visible.")
    indicators: List[str] = Field(
        default_factory=list,
        description="Specific, observable indicators (e.g., throwing rings, mold seam line, uniform wall thickness).",
        max_length=10,
        )
    foot_type: FootType = Field(description="Base/foot shape; use 'unknown' if base is not shown.")
    
    seams_present: TriState = Field(
        default="unknown",
        description="Mold seam lines visible? Use 'unknown' if not assessable from photos.",
        )
    throwing_rings_present: TriState = Field(
        default="unknown",
        description="Concentric throwing rings/turning marks visible? Use 'unknown' if not assessable.",
        )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs supporting construction/foot/seam assessments.",
        max_length=6,
        )


class FormGeometry(BaseModel):
    symmetry: Symmetry
    neck: Neck
    shoulder: Shoulder
    rim: Rim
    proportions_notes: List[str] = Field(
        default_factory=list,
        description="Shape/proportion observations only (no maker/style claims).",
        max_length=10,
    )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs supporting geometry assessments.",
        max_length=6,
    )


class GlazeSurface(BaseModel):
    sheen: Sheen
    effects: List[GlazeEffect] = Field(
        default_factory=list,
        description="If none visible, use ['none_detected']. Do not include 'none_detected' alongside other effects.",
        max_length=8,
    )
    color_behavior_notes: List[str] = Field(
        default_factory=list,
        description="Observable glaze/color behavior: gradients, pooling, break, thickness variation, edge effects.",
        max_length=12,
    )
    application_method_guess: ApplicationMethod = Field(description="Best guess; use 'unknown' if uncertain.")
    wear_notes: List[str] = Field(
        default_factory=list,
        description="Wear/aging notes tied to glaze: abrasions, matte wear, staining in crazing, glaze loss.",
        max_length=10,
    )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs supporting glaze/effects claims.",
        max_length=6,
    )


class Decoration(BaseModel):
    presence: DecorationPresence
    motif_categories: List[Motif] = Field(
        default_factory=list,
        description="If presence='none', leave empty. If unsure, include 'unknown' rather than guessing.",
        max_length=6,
    )
    technique_guess: DecorationTechnique
    layout_notes: List[str] = Field(
        default_factory=list,
        description="Placement/layout observations only (bands, panels, all-over, rim focus, etc.).",
        max_length=10,
    )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs supporting decoration claims.",
        max_length=6,
    )


class AgingCondition(BaseModel):
    condition_score_0_10: float = Field(
        ge=0.0,
        le=10.0,
        description="Visible-condition score only. 10=mint, 7=minor wear, 5=moderate wear, 3=notable damage, 0=severe damage.",
    )
    flags: List[ConditionFlag] = Field(
        default_factory=list,
        description="If any damage/restoration flags are present, do not include 'none_detected'. Avoid duplicates.",
        max_length=8,
    )
    rationale: List[str] = Field(
        default_factory=list,
        description="Short, observable reasons for score/flags.",
        max_length=10,
    )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs supporting condition claims.",
        max_length=6,
    )


class VisualEvidencePayload(BaseModel):
    object_class: ObjectClass
    key_observations: List[str] = Field(
        default_factory=list,
        description="High-signal, non-attribution observations that downstream should weigh (max 10).",
        max_length=10,
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="What you could not assess due to missing/unclear views (base not shown, interior not visible, glare, etc.).",
        max_length=8,
    )
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
MarkOrientation = Literal["upright","rotated_90","rotated_180","rotated_270","mirrored","unknown",]

MarkContentType = Literal["letters", "numbers", "symbol", "monogram", "mixed", "unknown"]
ScriptGuess = Literal["english", "latin", "cyrillic", "hanzi", "kana", "arabic", "other", "unknown"]


# class PhotoInstruction(BaseModel):
#     request: str = Field(description="What exact photo to take (e.g., base_macro straight-on, signature_macro, raking light series).")
#     why_it_matters: str = Field(description="1 sentence: what it would clarify (characters, depth, stroke order, etc.).")
#     how_to_shoot: str = Field(description="Concrete settings: macro, fill frame, tripod/steady, raking light angles, avoid glare.")
#     target_image_ids: List[str] = Field(
#         default_factory=list,
#         description="If applicable: image_ids that were close but insufficient (so user can retake similarly).",
#         max_length=6,
#     )


class Mark(BaseModel):
    mark_id: str = Field(description="Stable identifier like 'mark_01', 'mark_02'.")
    mark_type: MarkType
    location: MarkLocation
    orientation: MarkOrientation = Field(default="unknown")
    legibility: Legibility
    under_over_glaze: UnderOver

    content_type: MarkContentType = Field(
        default="unknown",
        description="High-level content classification (letters/numbers/symbol/monogram/mixed).",
    )
    script_guess: ScriptGuess = Field(
        default="unknown",
        description="Best guess of script family if letters are present; use 'unknown' if uncertain.",
    )

    verbatim_transcription: str = Field(
        description=(
            "Transcribe ONLY what is visible. Preserve line breaks with '\\n'. "
            "Use '?' for unclear characters and '[illegible]' for unreadable segments."
        )
    )
    alternate_readings: List[str] = Field(
        default_factory=list,
        description="Other plausible readings for ambiguous characters/segments (keep short).",
        max_length=6,
    )
    symbol_description: List[str] = Field(
        default_factory=list,
        description="If symbols/logos are present: describe shapes (e.g., 'sunburst', 'crown', 'circle with cross').",
        max_length=8,
    )

    confidence_0_1: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the transcription + classification for this mark (not attribution).",
    )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs that show this mark most clearly.",
        max_length=6,
    )
    interpretation_notes: List[str] = Field(
        default_factory=list,
        description="Non-attribution notes only (e.g., 'looks like model number', 'might be upside down', 'possibly impressed then glazed').",
        max_length=10,
    )


class MarksReport_v1(AgentMeta):
    case_id: str

    marks_present: TriState = Field(
        description="yes/no/unknown based strictly on visible evidence. Use 'unknown' if base/mark area is too unclear.",
    )
    marks_present_confidence_0_1: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in marks_present (not in any maker inference).",
    )

    marks: List[Mark] = Field(default_factory=list, description="0..N marks detected. Empty if none/unknown.")
    best_mark_for_id: Optional[str] = Field(
        default=None,
        description="mark_id of the single highest-value mark for identification, if any.",
    )

    negative_evidence_notes: List[str] = Field(
        default_factory=list,
        description="What areas were checked but showed no legible mark (e.g., 'base center overglazed, no visible stamp').",
        max_length=8,
    )
    # photo_instructions_if_unclear: List[PhotoInstruction] = Field(
    #     default_factory=list,
    #     description="If marks are unknown/low legibility, provide up to 5 ROI-ordered instructions.",
    #     max_length=5,
    # )

# ---------------- 4) Hypotheses ----------------

Hypothesis_source = Literal["coverage", "visual_evidence", "marks_report", "user_metadata", "other"]
Attribution_level = Literal[
    "maker",                 # specific maker/studio name
    "factory_or_workshop",   # factory/workshop but not a single named artist
    "region_style_cluster",  # region/style/cluster bucket
    "unknown",
]
Test_type = Literal["photo", "measurement", "comparison", "research", "other"]


class Evidence_anchor(BaseModel):
    source: Hypothesis_source
    path: str = Field(
        description="Dot-path into UPSTREAM_CONTEXT_JSON (e.g., 'marks_report.marks[0].verbatim_transcription')."
    )
    claim: str = Field(description="Short statement of what this anchor supports.")
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs that support this anchor, if applicable.",
        max_length=6,
    )


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
    maker_or_cluster: str = Field(
        description="If maker is unknown, write a cluster descriptor (not a guessy maker name)."
    )
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

# ---------------- 5) Final attribution ----------------

ResolutionOutcome = Literal["attribution", "inconclusive"]

# Reuse Attribution_level + Evidence_anchor from Hypotheses section above.

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
    overall_summary: str = Field(
        description="Short, evidence-grounded summary. No web-only claims."
    )
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
