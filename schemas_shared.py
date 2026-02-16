from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------- Shared inputs ----------------
class ImageInput(BaseModel):
    image_id: str
    data_ref: str  # local file path or URL (we will load local files -> data URLs)
    view_hint: Literal[
        "unknown",
        "front",
        "back",
        "left",
        "right",
        "top",
        "interior",
        "base",
        "detail",
        "rim",
        "foot",
        "signature",
    ] = "unknown"


class UserMetadata(BaseModel):
    height_in: Optional[float] = None
    weight_g: Optional[float] = None
    location_found: Optional[str] = None
    context_notes: Optional[str] = None


class AgentMeta(BaseModel):
    agent_name: str
    # agent_version: str = "v1"
    # timestamp_utc: str = Field(default_factory=utc_now)

    input_image_ids_used: List[str] = Field(
        default_factory=list,
        description="The image IDs that were used to generate the report.",
    )
    confidence_0_1: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence score of the report. 0.0 is the lowest confidence and 1.0 is the highest confidence.",
    )
    notes: str = Field(default="", description="The notes of the report.")


class CaseInput_v1(BaseModel):
    case_id: str
    images: List[ImageInput]
    user_metadata: UserMetadata = Field(default_factory=UserMetadata)


# ---------------- Shared enums / helpers ----------------
TriState = Literal["yes", "no", "unknown"]


# ---------------- Cross-agent hypothesis/resolution types ----------------
Hypothesis_source = Literal["coverage", "visual_evidence", "marks_report", "user_metadata", "other"]

Attribution_level = Literal[
    "maker",  # specific maker/studio name
    "factory_or_workshop",  # factory/workshop but not a single named artist
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

