from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from schemas_shared import AgentMeta, TriState


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
GlazeEffect = Literal["crazing", "crystalline", "running", "pooling", "crawling", "pinholing", "speckling", "none_detected"]
ApplicationMethod = Literal["sprayed", "dipped", "brushed", "poured", "unknown"]

DecorationPresence = Literal["none", "minimal", "moderate", "heavy", "unknown"]
Motif = Literal["floral", "geometric", "landscape", "figural", "abstract", "calligraphy", "unknown"]
DecorationTechnique = Literal["underglaze_paint", "overglaze_enamel", "transfer", "incised", "relief", "slip", "unknown"]

ConditionFlag = Literal["possible_restoration", "chips", "hairlines", "staining", "kiln_flaws", "none_detected"]


class EstimatedMaterial(BaseModel):
    label: MaterialLabel = Field(description="Use 'unknown' if you cannot distinguish confidently.")
    confidence_0_1: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence score of the material label. 0.0 is the lowest confidence and 1.0 is the highest confidence.",
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

