from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from schemas_shared import AgentMeta, TriState


# ---------------- 3) Marks ----------------
MarkType = Literal["impressed", "incised", "painted", "stamped", "label", "unknown"]
MarkLocation = Literal["base", "interior", "side", "unknown"]
Legibility = Literal["high", "medium", "low"]
UnderOver = Literal["under", "over", "unknown"]
MarkOrientation = Literal["upright", "rotated_90", "rotated_180", "rotated_270", "mirrored", "unknown"]

MarkContentType = Literal["letters", "numbers", "symbol", "monogram", "mixed", "unknown"]
ScriptGuess = Literal["english", "latin", "cyrillic", "hanzi", "kana", "arabic", "other", "unknown"]


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
    # photo_instructions_if_unclear intentionally omitted (currently not in schema.py in this project)

