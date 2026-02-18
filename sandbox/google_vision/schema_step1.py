"""
Pydantic schemas for Step 1 (Lens-like) visual feature extraction outputs.

These are intentionally conservative and avoid maker attribution.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class VlmSilhouetteFeatures(BaseModel):
    shape_label: str = Field(description="High-level silhouette label (e.g., 'inverted_teardrop').")
    neck_description: str = Field(default="", description="Description of neck/shoulder/rim shape.")
    descriptors: list[str] = Field(default_factory=list, description="Short silhouette descriptors.")
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    evidence_notes: list[str] = Field(default_factory=list, description="Pixel-grounded evidence notes.")


class VlmGlazeFeatures(BaseModel):
    technique_candidates: list[str] = Field(
        default_factory=list,
        description="Candidate glaze/finish technique terms (no maker/brand names required).",
    )
    finish_descriptors: list[str] = Field(
        default_factory=list,
        description="Descriptors like 'high_gloss', 'translucent', 'matte', 'crackle', etc.",
    )
    gradient_description: str = Field(
        default="",
        description="If present, describe vertical/horizontal color gradation.",
    )
    colors: list[str] = Field(default_factory=list, description="Human-readable dominant colors.")
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    evidence_notes: list[str] = Field(default_factory=list)


class VlmMotifFeatures(BaseModel):
    motif_type: Literal["botanical", "geometric", "figurative", "abstract", "none", "unknown"] = "unknown"
    motif_candidates: list[str] = Field(
        default_factory=list,
        description="Candidate motif labels (e.g., 'holly_berries', 'leaf_sprays').",
    )
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    evidence_notes: list[str] = Field(default_factory=list)


class Step1VlmFeatures(BaseModel):
    """
    Output of the multimodal model for Step 1 interpretation.
    Must not include maker attribution.
    """

    silhouette: VlmSilhouetteFeatures
    glaze: VlmGlazeFeatures
    motif: VlmMotifFeatures
    overall_confidence_0_1: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class Step1PerImage(BaseModel):
    image_path: str
    vision_features: dict[str, Any] = Field(default_factory=dict)
    derived_metrics: dict[str, Any] = Field(default_factory=dict)
    vlm_features: Optional[Step1VlmFeatures] = None

