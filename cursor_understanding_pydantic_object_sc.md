# Understanding Pydantic object schemas in agentic frameworks
_Exported on 2/10/2026 at 11:55:34 CST from Cursor (2.4.22)_

---

**User**

Review and get familiar with the following scripts which are an agentic framework for analyzing and identifying ceramics.
ceramic_id.py - main script that runs the pipeline, loads data, calls each agent.
schemas.py - contains the structured output definitions for each agent.
prompts.py - contains the prompts which are used by each agent. Each agent has a narrow task and will ingest the original input data and for certain agents also take in the output information from the previous agent. 

Can you give me a crash course on:
-What the Pydantic object schema classes do in the context of this script and agentic frameworks in general. 
-How a model interprets and decides to fill out these object schemas when supplied in the api call with the prompt.
-How is the structured output from one agent, which is set by its own object schema, interpretted by the next model?
-Can better object schemas with more information improve overall performance?
If so, how is that done? Should additional details be added in the "description" variable for each pydantic 'Field' function?
-For each agents prompt and object schema, should the prompts be updated with additional information concerning the respective schema objects?



---

**Cursor**

I’ll skim `requirements.txt` plus the three scripts to see exactly how schemas are defined, how they’re passed into the model call, and how each agent’s output is handed to the next—then I’ll answer each bullet with code-referenced specifics and practical guidance on improving schema quality.

### What the Pydantic schema classes do (in *this* pipeline, and generally)

In your code, each “agent” is **a model call constrained to return one specific Pydantic model** via OpenAI Structured Outputs (`client.responses.parse(..., text_format=YourPydanticModel)`), then you persist `model_dump()` into `results`:

```87:176:c:\projects\LLM\ceramics_agent\ceramic_id.py
def parse_with_schema(
    client: OpenAI,
    model: str,
    system_prompt: str,
    case: CaseInput_v1,
    text_format: Type[Any],
    max_output_tokens: int = 1200,
        ) -> Any:

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        text_format=text_format,
        max_output_tokens=max_output_tokens,
        )

    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("Model did not return a parsed structured output (possible refusal).")
    return parsed
```

So the Pydantic classes in `schemas.py` are doing four jobs:

- **Define the output contract** for each agent (required/optional fields, nested structure).
- **Constrain the model’s output space** via enums (`Literal[...]`), numeric bounds (`Field(ge=..., le=...)`), and defaults (e.g. `unknown`, empty lists).
- **Validate + parse** the returned JSON into Python objects (and then serialize back to JSON via `.model_dump()`).
- **Stabilize inter-agent interfaces**: downstream agents can rely on keys like `visual_evidence.evidence.glaze_surface.effects` always existing with known types.

You also use a shared base `AgentMeta` to enforce common metadata across all agent outputs (name/version/timestamp/confidence/etc.):

```29:36:c:\projects\LLM\ceramics_agent\schemas.py
class AgentMeta(BaseModel):
    agent_name: str
    agent_version: str = "v1"
    timestamp_utc: str = Field(default_factory=utc_now)
    input_image_ids_used: List[str] = Field(default_factory=list)
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    notes: str = ""
```

### How the model “decides” to fill schemas when you supply them in the API call

Mechanically, `text_format=CoverageReport_v1` (etc.) causes the API to provide the model a **JSON-schema-like target** derived from the Pydantic model, and the decoding is **constrained** to produce a valid instance (or an explicit refusal). Conceptually, the model is doing:

- **Interpret prompt + inputs (images + text)**.
- **Map evidence → fields** under the constraints you’ve set:
  - If a field is `Literal[...]`, it must choose one of those tokens (you often include `"unknown"` as a safe sink, e.g. `MaterialLabel`, `ConstructionMethod`, etc.).
  - If a field has bounds, it must produce a number inside the range (e.g. `confidence_0_1`, `condition_score_0_10`).
  - If a field is required (no default, no `Optional[...]`), it must fill it somehow—even if that means guessing—unless you design a safe default/“unknown” path.

So **schema design strongly affects behavior**: required fields without an “unknown/insufficient_evidence” option tend to force brittle guesses; enums without a good fallback tend to force misclassification.

### How structured output from one agent is interpreted by the next model (in your code)

In your pipeline, the next agent does **not** consume the previous agent’s Pydantic object directly as a typed input. Instead, you:

1. Convert upstream outputs to plain dict JSON (`model_dump()` already happened into `results`).
2. Inject that JSON into the *next* call as a **text blob** labeled `UPSTREAM_CONTEXT_JSON`.

Example for hypotheses:

```193:225:c:\projects\LLM\ceramics_agent\ceramic_id.py
hyp_context = {
    "coverage": results["coverage"],
    "visual_evidence": results["visual_evidence"],
    "marks_report": results["marks_report"],
}

content = [
    {"type": "input_text", "text": "UPSTREAM_CONTEXT_JSON:\n" + json.dumps(hyp_context, ensure_ascii=False)},
    *content,
]
hyp_response = client.responses.parse(
    model=model_reason,
    input=[
        {"role": "system", "content": hyp_system},
        {"role": "user", "content": content},
    ],
    text_format=Hypotheses_v1,
)
```

That means downstream interpretation is “LLM reads JSON-as-text”:
- It **must infer the meaning of keys** from the prompt plus the JSON structure.
- There’s no automatic guarantee it will respect every upstream nuance unless you **explicitly teach the prompt what fields matter**.

Your prompts *do* describe the intended dependency at a high level (e.g., “Using evidence + marks…”):

```35:45:c:\projects\LLM\ceramics_agent\prompts.py
HYPOTHESES_PROMPT = """You are the Hypothesis Generator.

Task:
Using evidence + marks (if any), propose up to 10 plausible attributions/clusters in Hypotheses_v1.
...
"""
```

### Can better object schemas improve performance? Yes—here’s how (and what “better” means)

**Yes**, often materially, because schemas are doing more than formatting—they shape the model’s search space and reduce ambiguity.

What tends to improve overall quality:

- **Add “escape hatches” to avoid forced guessing**
  - You already do this well with many `"unknown"` enums.
  - For any required complex object where uncertainty is common, consider either:
    - making it `Optional[...]`, or
    - adding explicit flags like `insufficient_evidence: bool` + `insufficient_evidence_reason: str`.

- **Prefer low-entropy, well-defined categories + “rationale”**
  - You already include `rationale: List[str]` in multiple places (good).
  - If you find the model overfits categories, broaden enums or add “other” with a free-text companion: `other_notes`.

- **Anchor fields to observations / images**
  - A high-leverage pattern: add `supporting_image_ids: List[str]` on key claims (marks, construction seams, glaze effects). This reduces hallucinated details because the model “pays” by pointing to an image.

- **Make confidence semantics consistent**
  - You have `confidence_0_1` in `AgentMeta` and in some nested objects. If you want calibration, define what 0.2 vs 0.8 *means* somewhere (schema description + prompt).

#### Should you add more detail in `Field(description=...)`?
**Often yes.** Field descriptions become part of the JSON schema that the model is guided by, and they’re a good place for **short operational definitions** and **decision boundaries**.

Good uses of `description`:
- “Set `views_present.base_macro` true only if the base fills >30% of frame and is in focus.”
- “`condition_score_0_10`: 10 = mint/no visible flaws; 5 = moderate wear; 0 = severe damage; base only on visible evidence.”
- “`crazing`: network of fine crack lines in glaze; do not confuse with scratches.”

Avoid:
- long narrative
- duplicating the whole prompt
- subjective fluff

### Should prompts be updated to include additional schema guidance?
**Yes, selectively—only for fields that are commonly misfilled or semantically tricky.**

Your current prompts are short and role-focused (good), but they mostly say “Fill X schema” without explaining tricky fields. A pragmatic approach:

- Keep the prompt role + rules as-is.
- Add a short **“Schema notes (only the tricky parts)”** section:
  - definitions for ambiguous enum choices
  - how to use `"unknown"`
  - how to populate `agent_name` / `confidence_0_1` consistently
  - constraints like max counts (you already have some, e.g. missing views up to 5)

One concrete thing I noticed: `AgentMeta` requires `agent_name` and `confidence_0_1`, but your prompts don’t explicitly tell the model what to set for `agent_name` (it will still fill *something*, but consistency may suffer). That’s a classic case where a **1-line prompt instruction** or a **default value in the schema** reduces variance.

If you want, I can also point out (agent-by-agent) which fields are most likely to be noisy given your current schema/prompt pairing, and where adding `description` vs adding a short prompt note usually pays off most.

---

**User**

Yes let's go through the schema and prompts agent-by-agent. 
Follow this order:
-Provide the updated code for object schema, prompt, and any other changes with the recommended fixes or improvements that I can copy and paste into the script.
-Explain what has changed and how it improves the single agent call and or system as a whole.

Let's start with the Coverage model.

---

**Cursor**

### Updated `schemas.py` (Coverage section only — copy/paste replacement)

```python
# ---------------- 1) Coverage ----------------
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

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


class MissingView(BaseModel):
    view: CoverageView = Field(
        description="A view that is missing or insufficient. Prefer high-ROI views like base/base_macro/signature_macro."
    )
    why_it_matters: str = Field(
        description="1-2 sentences: what identification signal this view would unlock (marks, foot, glaze pooling, etc.)."
    )
    how_to_shoot: str = Field(
        description="Concrete instructions (lighting, distance, angle, focus). Mention macro + raking light for marks."
    )


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
    missing_high_value_views: List[MissingView] = Field(
        default_factory=list,
        description="Up to 5 missing/insufficient views, ROI-ordered. If coverage is good, return empty list.",
        max_length=5,
    )
```

### Updated `prompts.py` (`COVERAGE_PROMPT` — copy/paste replacement)

```python
COVERAGE_PROMPT = """You are the Image QA + View Coverage Auditor. Do NOT identify pottery.

Task:
Assess view coverage and photo quality for ceramics identification.
Fill the CoverageReport_v1 schema.

Hard rules:
- Do NOT name makers, studios, styles, eras, or attributions.
- Be conservative: if unsure, mark views_present.* as false, and use view_detected='unknown'.
- Evaluate what the image ACTUALLY shows (do not trust view_hint blindly).
- Prefer actionable missing views: base, base_macro, signature_macro, interior, rim/top, profile.

Schema-specific instructions:
- agent_name: set to 'coverage'
- confidence_0_1:
  - ~0.9 = clear, unambiguous views + consistent assessments
  - ~0.5 = mixed/partial evidence
  - ~0.2 = very unclear/low-quality
- image_assessments: include one entry per image_id.
- coverage_score_0_1: penalize missing base/base_macro/signature_macro heavily.

missing_high_value_views:
- Up to 5, ordered by ROI.
- Include concrete how_to_shoot (macro mode, raking light, fill frame, perpendicular angle, steady camera).
"""
```

### Other recommended change (optional but high-leverage)

In `ceramic_id.py`, you currently “patch” only `input_image_ids_used` post-hoc, but you don’t enforce consistent `agent_name`. Adding a tiny post-parse patch (per agent) reduces downstream variance in `UPSTREAM_CONTEXT_JSON` and makes debugging easier. For Coverage specifically, you’d ensure `agent_name == "coverage"` even if the model outputs something else.

### What changed, and why it helps

- **Added `ImageAssessment` per `image_id`**: downstream agents (and you) can see *which* images are actually usable and what view they depict. This reduces cascade errors where a later agent assumes a base view exists because a blurry base-ish photo was present.
- **Made `MissingView.view` a constrained enum (`CoverageView`)**: prevents free-text drift (“bottom”, “underside”, “base closeup”) and makes missing-view recommendations more machine-actionable.
- **Expanded `QualityFlag`**: captures common failure modes (exposure, distance) that directly affect whether marks/evidence extraction should be trusted.
- **Added `Field(description=...)` where semantics are easy to misinterpret**: improves consistency in booleans like `base_macro` (your pipeline logic depends on these for `should_run_marks`), and helps the model choose conservative truth values.
- **Prompt now pins `agent_name` + confidence calibration + per-image requirement**: reduces schema-compliance “randomness” and makes `confidence_0_1` comparable across runs.

If you paste your current `case_1234_no_base.json`/`case_1234.json` outputs, I can point out exactly which of these new fields would have prevented (or made explicit) the most important downstream mistakes for your dataset.

