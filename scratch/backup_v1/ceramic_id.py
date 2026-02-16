from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import ssl
import urllib3
import httpx
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from dotenv import load_dotenv
from openai import OpenAI

from schemas import (
    CaseInput_v1, CoverageReport_v1, VisualEvidence_v1, MarksReport_v1,
    Hypotheses_v1, FinalAttribution_v1, ValueEstimate_v1
)
from prompts import (
    COVERAGE_PROMPT, EVIDENCE_PROMPT, MARKS_PROMPT,
    HYPOTHESES_PROMPT, RESOLVER_PROMPT, VALUE_PROMPT
)

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
http_client = httpx.Client(verify=False)

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def image_to_data_url(path: Path) -> str:
    mime = guess_mime(path)
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_multimodal_content(case: CaseInput_v1) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build the `content` list for a single user message:
    - includes metadata text
    - includes per-image view_hint text
    - includes images as data URLs
    """
    content: List[Dict[str, Any]] = []
    used: List[str] = []

    content.append({
        "type": "input_text",
        "text": f"Case ID: {case.case_id}\nUser metadata: {case.user_metadata.model_dump()}"
    })

    for img in case.images:
        if not img.data_ref.startswith("http"):
            p = Path(img.data_ref)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {img.data_ref}")
            image_url = image_to_data_url(p)
        else:
            image_url = img.data_ref
        content.append({
            "type": "input_text",
            "text": f"image_id={img.image_id} view_hint={img.view_hint}"
        })
        content.append({
            "type": "input_image",
            "image_url": image_url
        })
        used.append(img.image_id)

    return content, used


def parse_with_schema(
    client: OpenAI,
    model: str,
    system_prompt: str,
    case: CaseInput_v1,
    text_format: Type[Any],
    max_output_tokens: int = 1200,
        ) -> Any:
    
    """
        Calls Responses API using Structured Outputs:
        client.responses.parse(..., text_format=YourPydanticModel)
        Returns response.output_parsed (already a parsed Pydantic instance or dict-like).
    """

    content, used = build_multimodal_content(case)

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
        # Structured Outputs supports explicit refusals; output_parsed can be None.
        # You can also inspect response.output for refusal content if needed.
        raise RuntimeError("Model did not return a parsed structured output (possible refusal).")

    # Ensure agent meta has the image ids used when useful (optional).
    # If the model already filled input_image_ids_used, keep it. Otherwise, patch.
    try:
        if getattr(parsed, "input_image_ids_used", None) == []:
            parsed.input_image_ids_used = used  # type: ignore[attr-defined]
    except Exception:
        pass

    return parsed


# def should_run_marks(coverage: CoverageReport_v1) -> bool:
#     vp = coverage.views_present or {}
#     return bool(vp.get("base", False) or vp.get("base_macro", False) or vp.get("signature_macro", False))

def should_run_marks(coverage: CoverageReport_v1) -> bool:
    vp = coverage.views_present
    return bool(
        getattr(vp, "base", False)
        or getattr(vp, "base_macro", False)
        or getattr(vp, "signature_macro", False)
    )

def run_pipeline(
            client: OpenAI,
            case: CaseInput_v1,
            model_vision: str,
            model_reason: str,
            include_value: bool,
                ) -> Dict[str, Any]:
    

    results: Dict[str, Any] = {"case_id": case.case_id}

    # 1) Coverage (cheap model is fine; but must support structured outputs)
    print("Coverage")
    coverage: CoverageReport_v1 = parse_with_schema(
        client=client,
        model=model_vision,
        system_prompt=COVERAGE_PROMPT,
        case=case,
        text_format=CoverageReport_v1,
        max_output_tokens=900,
        )
    results["coverage"] = coverage.model_dump()

    # 2) Evidence (vision model)
    print("Evidence")
    evidence: VisualEvidence_v1 = parse_with_schema(
        client=client,
        model=model_vision,
        system_prompt=EVIDENCE_PROMPT,
        case=case,
        text_format=VisualEvidence_v1,
        max_output_tokens=1600,
    )
    results["visual_evidence"] = evidence.model_dump()

    # 3) Marks (conditional)
    print("Marks")
    if should_run_marks(coverage):
        marks: MarksReport_v1 = parse_with_schema(
            client=client,
            model=model_vision,
            system_prompt=MARKS_PROMPT,
            case=case,
            text_format=MarksReport_v1,
            max_output_tokens=1200,
        )
        results["marks_report"] = marks.model_dump()
    else:
        results["marks_report"] = {"marks_present": False, "marks": []}

    # 4) Hypotheses (reasoning model)
    # Note: The schema enforces structure; the model still needs context.
    # Provide the upstream results as part of the system prompt? Better: add them as user text.
    # We’ll inject them as an additional user message (text-only).
    print("Hypotheses")
    hyp_system = HYPOTHESES_PROMPT
    hyp_context = {
        "coverage": results["coverage"],
        "visual_evidence": results["visual_evidence"],
        "marks_report": results["marks_report"],
    }

    # Make a temporary "case-like" wrapper: easiest is to add a context-only run by extending messages.
    # We’ll do it by calling parse directly here with extra user text before the images.
    content, _ = build_multimodal_content(case)
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
        max_output_tokens=1800,
    )
    hypotheses = hyp_response.output_parsed
    if hypotheses is None:
        raise RuntimeError("Hypotheses agent did not return structured output (possible refusal).")
    results["hypotheses"] = hypotheses.model_dump()

    # 5) Resolver (reasoning model)
    print("Resolver")
    res_system = RESOLVER_PROMPT
    res_context = {
        "coverage": results["coverage"],
        "visual_evidence": results["visual_evidence"],
        "marks_report": results["marks_report"],
        "hypotheses": results["hypotheses"],
    }
    content, _ = build_multimodal_content(case)
    content = [
        {"type": "input_text", "text": "UPSTREAM_CONTEXT_JSON:\n" + json.dumps(res_context, ensure_ascii=False)},
        *content,
    ]
    res_response = client.responses.parse(
        model=model_reason,
        input=[
            {"role": "system", "content": res_system},
            {"role": "user", "content": content},
        ],
        text_format=FinalAttribution_v1,
        max_output_tokens=1800,
    )
    final = res_response.output_parsed
    if final is None:
        raise RuntimeError("Resolver did not return structured output (possible refusal).")
    results["final_attribution"] = final.model_dump()

    # 6) Value (optional)
    if include_value:
        print("Value")
        val_system = VALUE_PROMPT
        val_context = {
            "final_attribution": results["final_attribution"],
            "visual_evidence": results["visual_evidence"],
            "user_metadata": case.user_metadata.model_dump(),
        }
        content, _ = build_multimodal_content(case)
        content = [
            {"type": "input_text", "text": "UPSTREAM_CONTEXT_JSON:\n" + json.dumps(val_context, ensure_ascii=False)},
            *content,
        ]
        val_response = client.responses.parse(
            model=model_reason,
            input=[
                {"role": "system", "content": val_system},
                {"role": "user", "content": content},
            ],
            text_format=ValueEstimate_v1,
            max_output_tokens=1200,
        )
        value = val_response.output_parsed
        if value is None:
            raise RuntimeError("Value agent did not return structured output (possible refusal).")
        results["value_estimate"] = value.model_dump()

    return results


def load_case_from_dir(case_id: str, img_dir: Path, height_in: Optional[float]) -> CaseInput_v1:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    if not paths:
        raise ValueError(f"No images found in {img_dir}")

    images = []
    for i, p in enumerate(paths):
        images.append({
            "image_id": f"img_{i:02d}",
            "data_ref": str(p),
            "view_hint": "unknown"
        })

    return CaseInput_v1.model_validate({
        "case_id": case_id,
        "images": images,
        "user_metadata": {"height_in": height_in}
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--height-in", type=float, default=None)
    parser.add_argument("--out", default=None)

    # Structured outputs support starts with GPT-4o snapshots and later. :contentReference[oaicite:2]{index=2}
    parser.add_argument("--model-vision", default="gpt-4o-2024-08-06")
    parser.add_argument("--model-reason", default="gpt-4o-2024-08-06")

    parser.add_argument("--value", action="store_true")
    args = parser.parse_args()

    client = OpenAI(api_key=OPENAI_API_KEY,http_client=http_client)

    case = load_case_from_dir(args.case_id, Path(args.img_dir), args.height_in)

    results = run_pipeline(
        client=client,
        case=case,
        model_vision=args.model_vision,
        model_reason=args.model_reason,
        include_value=args.value,
    )

    out_path = Path(args.out) if args.out else Path(f"case_{args.case_id}.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")



def load_case_from_paths(case_id: str, img_paths: list[str], height_in: Optional[float]) -> CaseInput_v1:
    images = []
    for i, p in enumerate(img_paths):
        images.append({
            "image_id": f"img_{i:02d}",
            "data_ref": p,
            "view_hint": "unknown"
            })

    return CaseInput_v1.model_validate({
        "case_id": case_id,
        "images": images,
        "user_metadata": {"height_in": height_in}
        })

def manual_main(case_id : str, img_dir: Path, height_in: Optional[float], out: Optional[Path], model_vision: str, model_reason: str, value: bool):
    
    
    client = OpenAI(api_key=OPENAI_API_KEY,http_client=http_client)
    
    case = load_case_from_dir(case_id, img_dir, height_in)
    
    results = run_pipeline(
        client=client,
        case=case,
        model_vision=model_vision,
        model_reason=model_reason,
        include_value=value,
        )
    
    out_path = out if out else Path(f"case_{case_id}.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"Wrote: {out_path}")


def download_images(case_id: str, image_urls: list[str]) -> Path:
    '''
    Download images from the internet and save them to a local directory.
    Args:
        case_id: The case ID.
        image_urls: A list of image URLs.
    Returns:
        img_dir: The local directory where the images were saved.
    '''
    image_directory = rf".\images\{case_id}"
        
    img_dir = Path(image_directory)
    img_dir.mkdir(parents=True, exist_ok=True)

    for image_url in image_urls:
        image_path = os.path.join(image_directory, os.path.basename(image_url))
        response = requests.get(image_url, verify=False)
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        

    return img_dir

if __name__ == "__main__":

    print('breakpoint')

    if len(sys.argv) > 1:

        main()

    else:
        
        case_id="12345"
        image_urls=[
            r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32728000.jpg",
            r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32727999.jpg",
            r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32728001.jpg",
            r"https://946e583539399c301dc7-100ffa5b52865b8ec92e09e9de9f4d02.ssl.cf2.rackcdn.com/62930/32727998.jpg",  # Bottom Image.

            ]
        img_dir = download_images(case_id, image_urls)
        height_in=6
        out=None
        model_vision = "gpt-4o-2024-08-06"
        model_reason = "gpt-4o-2024-08-06"
        value=True
        manual_main(case_id, img_dir, height_in, out, model_vision, model_reason, value)
