from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from schemas_shared import AgentMeta


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
    views_present: ViewsPresent = Field(description="Boolean checklist of high-value views present with sufficient clarity.")
    image_assessments: List[ImageAssessment] = Field(
        default_factory=list,
        description="One entry per image_id. Helps downstream agents trust/discount evidence.",
    )
    # missing_high_value_views intentionally omitted (currently not in schema.py in this project)

