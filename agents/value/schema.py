from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel

from schemas_shared import AgentMeta


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

