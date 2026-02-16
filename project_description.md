# Project Description: Ceramics Identification Agent Pipeline

## 1) High-level goal
This project runs a multi-agent LLM pipeline to **analyze a set of ceramic object images** (plus optional user metadata like height/weight) and produce:
- a structured visual evidence report,
- optional transcription of marks,
- a ranked set of hypotheses,
- a calibrated Top-3 final attribution (often at “cluster/region/style” granularity when evidence is weak),
- and an optional conservative value range.

The pipeline uses **OpenAI Structured Outputs** (Pydantic models) to enforce consistent JSON outputs per stage.

## 2) Pipeline agent architecture (what runs, in order)
Entry point: `ceramic_id.py`.

The pipeline builds multimodal inputs (text + images as data URLs) from a `CaseInput_v1` and then runs these agents:

1. **Coverage** (`agents/coverage/`)
   - **Purpose**: image QA + view coverage audit (what views exist, whether images are usable).
   - **Output schema**: `CoverageReport_v1` with `views_present` and per-image `image_assessments`.

2. **Evidence** (`agents/evidence/`)
   - **Purpose**: extract “forensic” visual evidence (material, construction, form geometry, glaze surface, decoration, condition).
   - **Output schema**: `VisualEvidence_v1`.
   - **Consumes upstream context**: coverage is prepended as `UPSTREAM_CONTEXT_JSON` so evidence can discount unusable images.

3. **Marks** (`agents/marks/`) (conditional)
   - **Purpose**: detect/transcribe marks (base/signature/labels) without inferring attribution.
   - **Output schema**: `MarksReport_v1`.
   - **Runs only if** `should_run_marks(coverage)` indicates base/base-macro/signature-macro views are present.
   - **Consumes upstream context**: coverage is prepended as `UPSTREAM_CONTEXT_JSON` to prioritize usable base/mark photos.

4. **Hypotheses** (`agents/hypotheses/`)
   - **Purpose**: generate up to 10 plausible attributions/clusters using upstream evidence + marks.
   - **Output schema**: `Hypotheses_v1` with per-hypothesis discriminators/tests and optional anchors into upstream JSON.
   - **Consumes upstream context**: coverage + visual evidence + marks report.

5. **Resolver** (`agents/resolver/`)
   - **Purpose**: contradiction-aware pruning and ranking into a calibrated Top 3; proposes next photos.
   - **Output schema**: `FinalAttribution_v1`.
   - **Consumes upstream context**: coverage + visual evidence + marks report + hypotheses.

6. **Value** (`agents/value/`) (optional)
   - **Purpose**: conservative internal-only USD range; no web claims.
   - **Output schema**: `ValueEstimate_v1`.
   - **Consumes upstream context**: final attribution + visual evidence + user metadata.

### How upstream context is passed
Most downstream agents receive a JSON blob prepended to the user message as:

`UPSTREAM_CONTEXT_JSON:\n{...}`

This is handled in `parse_with_schema(..., upstream_context=...)` in `ceramic_id.py`.

## 3) Useful conventions / collaboration notes

### Code organization (current structure)
- **Shared types** live in `schemas_shared.py` (e.g., `CaseInput_v1`, `AgentMeta`, shared enums like `TriState`).
- **Agent-local prompt + schema** live together under `agents/<agent_name>/prompt.py` and `agents/<agent_name>/schema.py`.
- `schemas.py` and `prompts.py` are **compatibility re-export modules** (thin shims).

### Structured output contracts
- Each agent returns a Pydantic model instance via `client.responses.parse(..., text_format=<PydanticModel>)`.
- The pipeline persists results via `.model_dump()` into a single JSON result document (`case_<id>.json` and optionally `dev_data/` snapshots).

### Prompting rules that reduce cascade errors
- Early agents (Coverage/Evidence/Marks) are instructed **not to attribute makers/eras**; they focus on measurable/observable cues.
- Downstream reasoning (Hypotheses/Resolver) should remain **conservative** when marks are absent or coverage is incomplete.

### Environment/runtime notes
- Dependencies are pinned in `requirements.txt` (Pydantic v2, OpenAI SDK v2).
- The pipeline expects `OPENAI_API_KEY` in environment (loaded via `python-dotenv`).
- Images are passed as **data URLs** built from local paths, or as remote URLs if `data_ref` is already `http...`.

