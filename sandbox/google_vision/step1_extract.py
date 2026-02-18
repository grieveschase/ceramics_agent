"""
Step 1 (Lens-like): Visual feature extraction.

This script is the orchestrator for Step 1, starting with Cloud Vision REST signals:
  - LABEL_DETECTION
  - OBJECT_LOCALIZATION
  - IMAGE_PROPERTIES
  - CROP_HINTS

Later steps (derived metrics + VLM synthesis + aggregation) will build on the JSON
artifact produced here.

Refs:
  - Vision REST annotate: https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate
  - Crop hints params: https://cloud.google.com/vision/docs/reference/rest/v1/ImageContext
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from PIL import Image as PilImage
from pydantic import ValidationError

from schema_step1 import Step1VlmFeatures
from vertex_gemini_client import (
    chat_completion_with_image,
    infer_project_id_from_service_account_key,
    make_vertex_openai_client,
)

load_dotenv()

VISION_ANNOTATE_URL = "https://vision.googleapis.com/v1/images:annotate"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Vision synchronous annotate guidance: keep request batches small (commonly <= 16).
DEFAULT_BATCH_SIZE = 16
DEFAULT_VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
DEFAULT_VERTEX_MODEL = os.getenv("VERTEX_GEMINI_MODEL", "google/gemini-2.0-flash-001")


def project_images_dir() -> Path:
    """Project root is parent of sandbox; images dir is project_root/images."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root / "images"


def collect_image_paths(img_dir: Path) -> list[Path]:
    """Sorted list of image paths under img_dir (non-recursive)."""
    return [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]


def get_access_token(credentials_path: str) -> str:
    """Return a valid Bearer token for Google APIs using the service account key."""
    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    return creds.token


def _poly_to_bbox_norm(vertices: list[dict[str, Any]]) -> Optional[dict[str, float]]:
    """
    Convert a Vision polygon (normalizedVertices) to a normalized bounding box.
    Returns None if vertices are missing.
    """
    if not vertices:
        return None
    xs = [float(v.get("x", 0.0)) for v in vertices]
    ys = [float(v.get("y", 0.0)) for v in vertices]
    return {
        "x_min": max(0.0, min(xs)),
        "y_min": max(0.0, min(ys)),
        "x_max": min(1.0, max(xs)),
        "y_max": min(1.0, max(ys)),
    }


def _crop_hint_vertices_to_bbox_px(vertices: list[dict[str, Any]]) -> Optional[dict[str, int]]:
    """Convert crop hint pixel vertices into a pixel bounding box."""
    if not vertices:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for v in vertices:
        if v.get("x") is None or v.get("y") is None:
            continue
        xs.append(int(v["x"]))
        ys.append(int(v["y"]))
    if not xs or not ys:
        return None
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def _clamp_crop_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _bbox_norm_to_crop_box_px(bbox_norm: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    left = int(round(bbox_norm["x_min"] * width))
    right = int(round(bbox_norm["x_max"] * width))
    top = int(round(bbox_norm["y_min"] * height))
    bottom = int(round(bbox_norm["y_max"] * height))
    return _clamp_crop_box((left, top, right, bottom), width=width, height=height)


def _compute_top_bottom_color_delta(img: PilImage.Image, crop_box: tuple[int, int, int, int]) -> dict[str, Any]:
    """
    Compute mean RGB of top 20% and bottom 20% of the cropped region.
    Returns RGB means, luminance means, and deltas.
    """
    cropped = img.crop(crop_box).convert("RGB")
    arr = np.asarray(cropped, dtype=np.float32)  # (h, w, 3)
    h = arr.shape[0]
    if h < 10:
        # Too small to meaningfully split; fall back to whole crop.
        top = arr
        bottom = arr
    else:
        band = max(1, int(round(h * 0.2)))
        top = arr[:band, :, :]
        bottom = arr[-band:, :, :]

    top_mean = top.reshape(-1, 3).mean(axis=0)
    bottom_mean = bottom.reshape(-1, 3).mean(axis=0)

    # Relative luminance proxy in sRGB (not gamma-corrected; good enough for heuristic).
    top_lum = float(0.2126 * top_mean[0] + 0.7152 * top_mean[1] + 0.0722 * top_mean[2])
    bottom_lum = float(0.2126 * bottom_mean[0] + 0.7152 * bottom_mean[1] + 0.0722 * bottom_mean[2])
    lum_delta = top_lum - bottom_lum

    rgb_delta = float(np.linalg.norm(top_mean - bottom_mean))
    gradient_direction = "top_darker" if lum_delta < 0 else "bottom_darker" if lum_delta > 0 else "flat"

    return {
        "top_mean_rgb": {"r": float(top_mean[0]), "g": float(top_mean[1]), "b": float(top_mean[2])},
        "bottom_mean_rgb": {"r": float(bottom_mean[0]), "g": float(bottom_mean[1]), "b": float(bottom_mean[2])},
        "delta_rgb_l2": rgb_delta,
        "top_luminance": top_lum,
        "bottom_luminance": bottom_lum,
        "luminance_delta_top_minus_bottom": lum_delta,
        "gradient_direction": gradient_direction,
    }


def _select_primary_object_bbox_norm(localized_objects: list[dict[str, Any]]) -> Optional[dict[str, float]]:
    """Pick the highest-score object bbox_norm (prefer vase-like names if available)."""
    if not localized_objects:
        return None

    preferred = {"vase", "urn", "pottery", "ceramic", "artifact"}
    best = None
    best_key = (-1.0, -1.0)  # (preferred_flag, score)
    for obj in localized_objects:
        bbox = obj.get("bbox_norm")
        if not bbox:
            continue
        score = float(obj.get("score", 0.0))
        name = str(obj.get("name", "")).strip().lower()
        pref_flag = 1.0 if name in preferred else 0.0
        key = (pref_flag, score)
        if key > best_key:
            best_key = key
            best = bbox
    return best


def _normalize_vision_response(path: Path, rest_item: dict[str, Any]) -> dict[str, Any]:
    """Normalize key Vision outputs into a stable, compact structure."""
    normalized: dict[str, Any] = {
        "image_path": str(path),
        "vision_features": {
            "labels": [],
            "localized_objects": [],
            "image_properties": {"dominant_colors": []},
            "crop_hints": [],
        },
        "derived_metrics": {},
        "vision_raw": rest_item,
    }

    if "error" in rest_item:
        normalized["vision_features"]["error"] = rest_item["error"].get("message", str(rest_item["error"]))
        return normalized

    # Labels
    for ann in rest_item.get("labelAnnotations") or []:
        normalized["vision_features"]["labels"].append(
            {
                "description": ann.get("description", ""),
                "mid": ann.get("mid"),
                "score": float(ann.get("score", 0.0)),
                "topicality": float(ann.get("topicality", 0.0)) if ann.get("topicality") is not None else None,
            }
        )

    # Localized objects
    for obj in rest_item.get("localizedObjectAnnotations") or []:
        verts = obj.get("boundingPoly", {}).get("normalizedVertices") or []
        normalized["vision_features"]["localized_objects"].append(
            {
                "name": obj.get("name", ""),
                "mid": obj.get("mid"),
                "score": float(obj.get("score", 0.0)),
                "bounds": [{"x": float(v.get("x", 0.0)), "y": float(v.get("y", 0.0))} for v in verts],
                "bbox_norm": _poly_to_bbox_norm(verts),
            }
        )

    # Image properties (dominant colors)
    img_props = rest_item.get("imagePropertiesAnnotation") or {}
    dominant = (img_props.get("dominantColors") or {}).get("colors") or []
    for c in dominant:
        color = c.get("color") or {}
        normalized["vision_features"]["image_properties"]["dominant_colors"].append(
            {
                "rgb": {
                    "r": float(color.get("red", 0.0)),
                    "g": float(color.get("green", 0.0)),
                    "b": float(color.get("blue", 0.0)),
                },
                "score": float(c.get("score", 0.0)),
                "pixel_fraction": float(c.get("pixelFraction", 0.0)),
            }
        )

    # Crop hints
    crop = rest_item.get("cropHintsAnnotation") or {}
    hints = crop.get("cropHints") or []
    for hint in hints:
        vertices = (hint.get("boundingPoly") or {}).get("vertices") or []
        # Crop hints are pixel vertices (not normalized). Keep as-is + derive bbox if possible later.
        normalized["vision_features"]["crop_hints"].append(
            {
                "confidence": float(hint.get("confidence", 0.0)),
                "importance_fraction": float(hint.get("importanceFraction", 0.0)),
                "vertices": [{"x": v.get("x"), "y": v.get("y")} for v in vertices],
            }
        )

    # ----- Derived metrics (lightweight, deterministic) -----
    try:
        with PilImage.open(path) as img:
            width, height = img.size
            localized_objects = normalized["vision_features"]["localized_objects"]
            primary_bbox_norm = _select_primary_object_bbox_norm(localized_objects)

            crop_box_source = "full_image"
            crop_box_px = (0, 0, width, height)

            if primary_bbox_norm:
                crop_box_source = "object_localization"
                crop_box_px = _bbox_norm_to_crop_box_px(primary_bbox_norm, width=width, height=height)
            else:
                # Fall back to the top crop hint (pixel vertices)
                crop_hints = normalized["vision_features"]["crop_hints"]
                if crop_hints:
                    hint_bbox = _crop_hint_vertices_to_bbox_px(crop_hints[0].get("vertices") or [])
                    if hint_bbox:
                        crop_box_source = "crop_hint"
                        crop_box_px = _clamp_crop_box(
                            (hint_bbox["x_min"], hint_bbox["y_min"], hint_bbox["x_max"], hint_bbox["y_max"]),
                            width=width,
                            height=height,
                        )

            left, top, right, bottom = crop_box_px
            crop_w = max(1, right - left)
            crop_h = max(1, bottom - top)
            aspect_ratio_h_over_w = float(crop_h / crop_w)

            color_delta = _compute_top_bottom_color_delta(img, crop_box=crop_box_px)

            normalized["derived_metrics"] = {
                "image_size_px": {"width": int(width), "height": int(height)},
                "primary_bbox_source": crop_box_source,
                "primary_crop_box_px": {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)},
                "bbox_aspect_ratio_h_over_w": aspect_ratio_h_over_w,
                "top_bottom_color_delta": color_delta,
            }
    except Exception as e:
        normalized["derived_metrics"] = {"error": f"derived_metrics_failed: {type(e).__name__}: {e}"}

    return normalized


def _build_image_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a model response that may contain extra text.
    Conservative: takes substring from first '{' to last '}'.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in VLM response.")
    return json.loads(text[start : end + 1])


def _build_step1_vlm_prompt(*, vision_features: dict[str, Any], derived_metrics: dict[str, Any]) -> str:
    """
    Prompt for Step 1 only: silhouette, glaze/finish, motif.
    No maker attribution.
    """
    labels = [l.get("description") for l in (vision_features.get("labels") or [])][:15]
    dominant = (vision_features.get("image_properties") or {}).get("dominant_colors") or []
    dominant_colors = [
        d.get("rgb") for d in sorted(dominant, key=lambda x: float(x.get("pixel_fraction", 0.0)), reverse=True)[:5]
    ]
    color_delta = (derived_metrics.get("top_bottom_color_delta") or {})
    aspect = derived_metrics.get("bbox_aspect_ratio_h_over_w")

    return (
        "You are performing Step 1: Visual Feature Extraction for a ceramic object image.\n"
        "DO NOT guess or mention maker names, factories, countries, regions, dates, or attributions.\n"
        "Focus only on observable visual cues: silhouette/form, glaze/finish, and decorative motif.\n"
        "\n"
        "Context signals (from a vision API + simple measurements):\n"
        f"- vision_labels_top: {labels}\n"
        f"- dominant_colors_rgb_top: {dominant_colors}\n"
        f"- bbox_aspect_ratio_h_over_w: {aspect}\n"
        f"- top_bottom_color_delta: {color_delta}\n"
        "\n"
        "Return ONLY valid JSON matching this schema (no markdown, no extra keys):\n"
        "{\n"
        '  "silhouette": {\n'
        '    "shape_label": "string",\n'
        '    "neck_description": "string",\n'
        '    "descriptors": ["string", "..."],\n'
        '    "confidence_0_1": 0.0,\n'
        '    "evidence_notes": ["string", "..."]\n'
        "  },\n"
        '  "glaze": {\n'
        '    "technique_candidates": ["string", "..."],\n'
        '    "finish_descriptors": ["string", "..."],\n'
        '    "gradient_description": "string",\n'
        '    "colors": ["string", "..."],\n'
        '    "confidence_0_1": 0.0,\n'
        '    "evidence_notes": ["string", "..."]\n'
        "  },\n"
        '  "motif": {\n'
        '    "motif_type": "botanical|geometric|figurative|abstract|none|unknown",\n'
        '    "motif_candidates": ["string", "..."],\n'
        '    "confidence_0_1": 0.0,\n'
        '    "evidence_notes": ["string", "..."]\n'
        "  },\n"
        '  "overall_confidence_0_1": 0.0,\n'
        '  "limitations": ["string", "..."],\n'
        '  "debug": {}\n'
        "}\n"
    )


def _build_annotate_request(image_path: Path, crop_hint_aspect_ratios: list[float]) -> dict[str, Any]:
    content_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "image": {"content": content_b64},
        "features": [
            {"type": "LABEL_DETECTION", "maxResults": 50},
            {"type": "OBJECT_LOCALIZATION", "maxResults": 20},
            {"type": "IMAGE_PROPERTIES", "maxResults": 10},
            {"type": "CROP_HINTS", "maxResults": 5},
        ],
        "imageContext": {
            "cropHintsParams": {"aspectRatios": crop_hint_aspect_ratios},
        },
    }


def vision_annotate_batch(
    image_paths: list[Path],
    *,
    credentials_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    crop_hint_aspect_ratios: Optional[list[float]] = None,
    verify_ssl: bool = False,
    vertex_location: str = DEFAULT_VERTEX_LOCATION,
    vertex_model: str = DEFAULT_VERTEX_MODEL,
    run_vlm: bool = True,
) -> list[dict[str, Any]]:
    """
    Call Vision REST images:annotate in batches and return normalized outputs.
    """
    if crop_hint_aspect_ratios is None:
        crop_hint_aspect_ratios = [1.0, 4 / 3, 16 / 9]

    token = get_access_token(credentials_path)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    normalized_results: list[dict[str, Any]] = []

    # Optional VLM client (Vertex AI Gemini via OpenAI-compatible endpoint)
    vertex_client = None
    if run_vlm:
        project_id = infer_project_id_from_service_account_key(credentials_path)
        if not project_id:
            raise RuntimeError("Could not infer project_id from service account key; set it in the key file.")
        vertex_token = token  # reuse same access token scope
        vertex_client = make_vertex_openai_client(
            project_id=project_id,
            location=vertex_location,
            access_token=vertex_token,
            verify_ssl=verify_ssl,
        )

    with httpx.Client(verify=verify_ssl, timeout=60.0) as http_client:
        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i : i + batch_size]
            body = {
                "requests": [
                    _build_annotate_request(p, crop_hint_aspect_ratios=crop_hint_aspect_ratios)
                    for p in chunk_paths
                ]
            }
            resp = http_client.post(VISION_ANNOTATE_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            responses = data.get("responses") or []
            for path, rest_item in zip(chunk_paths, responses):
                per_image = _normalize_vision_response(path, rest_item)

                if run_vlm and vertex_client is not None and "error" not in per_image.get("vision_features", {}):
                    try:
                        prompt = _build_step1_vlm_prompt(
                            vision_features=per_image.get("vision_features", {}),
                            derived_metrics=per_image.get("derived_metrics", {}),
                        )
                        image_data_url = _build_image_data_url(path)
                        raw_text = chat_completion_with_image(
                            client=vertex_client,
                            model=vertex_model,
                            prompt_text=prompt,
                            image_url=image_data_url,
                            max_output_tokens=900,
                        )
                        parsed = _extract_json_object(raw_text)
                        per_image["vlm_features"] = Step1VlmFeatures.model_validate(parsed).model_dump()
                    except (ValidationError, ValueError) as e:
                        per_image["vlm_features_error"] = f"{type(e).__name__}: {e}"
                    except Exception as e:
                        per_image["vlm_features_error"] = f"VLM_call_failed: {type(e).__name__}: {e}"

                normalized_results.append(per_image)

    return normalized_results


def _most_common_weighted(items: list[tuple[str, float]]) -> Optional[str]:
    """
    Return the item with max total weight. items are (value, weight).
    """
    totals: dict[str, float] = {}
    for v, w in items:
        if not v:
            continue
        totals[v] = totals.get(v, 0.0) + float(w)
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def aggregate_case_results(per_image_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate multi-image Step 1 results into a case-level summary.
    Conservative: prefers consensus and surfaces conflicts/limitations.
    """
    silhouette_votes: list[tuple[str, float]] = []
    motif_type_votes: list[tuple[str, float]] = []
    technique_votes: list[tuple[str, float]] = []
    finish_votes: list[tuple[str, float]] = []
    limitations: list[str] = []
    overall_confs: list[float] = []

    best_silhouette = {"image_path": None, "confidence": -1.0}
    best_motif = {"image_path": None, "confidence": -1.0}
    best_glaze = {"image_path": None, "confidence": -1.0}

    gradient_dirs: dict[str, int] = {"top_darker": 0, "bottom_darker": 0, "flat": 0, "unknown": 0}

    for item in per_image_results:
        vlm = item.get("vlm_features") or {}
        if not vlm:
            continue

        overall = vlm.get("overall_confidence_0_1")
        if isinstance(overall, (int, float)):
            overall_confs.append(float(overall))

        lim = vlm.get("limitations") or []
        limitations.extend([str(x) for x in lim if str(x).strip()])

        sil = vlm.get("silhouette") or {}
        sil_label = str(sil.get("shape_label") or "").strip()
        sil_conf = float(sil.get("confidence_0_1", 0.0) or 0.0)
        if sil_label:
            silhouette_votes.append((sil_label, sil_conf))
        if sil_conf > best_silhouette["confidence"]:
            best_silhouette = {"image_path": item.get("image_path"), "confidence": sil_conf}

        glaze = vlm.get("glaze") or {}
        glaze_conf = float(glaze.get("confidence_0_1", 0.0) or 0.0)
        for t in glaze.get("technique_candidates") or []:
            technique_votes.append((str(t), glaze_conf))
        for f in glaze.get("finish_descriptors") or []:
            finish_votes.append((str(f), glaze_conf))
        if glaze_conf > best_glaze["confidence"]:
            best_glaze = {"image_path": item.get("image_path"), "confidence": glaze_conf}

        motif = vlm.get("motif") or {}
        motif_type = str(motif.get("motif_type") or "").strip()
        motif_conf = float(motif.get("confidence_0_1", 0.0) or 0.0)
        if motif_type:
            motif_type_votes.append((motif_type, motif_conf))
        if motif_conf > best_motif["confidence"]:
            best_motif = {"image_path": item.get("image_path"), "confidence": motif_conf}

        derived = item.get("derived_metrics") or {}
        td = (derived.get("top_bottom_color_delta") or {}).get("gradient_direction")
        if isinstance(td, str) and td in gradient_dirs:
            gradient_dirs[td] += 1
        else:
            gradient_dirs["unknown"] += 1

    silhouette_summary = _most_common_weighted(silhouette_votes)
    motif_type_summary = _most_common_weighted(motif_type_votes)
    glaze_technique_summary = _most_common_weighted(technique_votes)
    glaze_finish_summary = _most_common_weighted(finish_votes)

    conflicts: list[str] = []
    # Simple conflict detection: multiple gradient directions observed.
    nonzero_grad = [k for k, v in gradient_dirs.items() if v and k not in {"unknown"}]
    if len(nonzero_grad) > 1:
        conflicts.append(f"gradient_direction_conflict: {gradient_dirs}")

    case_conf = float(sum(overall_confs) / len(overall_confs)) if overall_confs else 0.0

    return {
        "schema_version": "step1_features_v1",
        "images": per_image_results,
        "case_summary": {
            "silhouette_summary": silhouette_summary,
            "glaze_summary": {
                "technique": glaze_technique_summary,
                "finish": glaze_finish_summary,
            },
            "motif_summary": {"motif_type": motif_type_summary},
            "best_images": {
                "silhouette": best_silhouette,
                "glaze": best_glaze,
                "motif": best_motif,
            },
            "confidence_0_1": case_conf,
            "conflicts": conflicts,
            "limitations": sorted(set([x for x in limitations if x])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: extract Vision signals (REST).")
    parser.add_argument("--img-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--credentials", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--vertex-location", default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument("--vertex-model", default=DEFAULT_VERTEX_MODEL)
    parser.add_argument("--no-vlm", action="store_true", help="Disable Vertex Gemini VLM interpretation step.")
    args = parser.parse_args()

    img_dir = (args.img_dir if args.img_dir is not None else project_images_dir()).resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")

    credentials_path = args.credentials if args.credentials is not None else os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise SystemExit("Set GOOGLE_APPLICATION_CREDENTIALS or pass --credentials PATH.")
    credentials_path = Path(credentials_path).resolve()
    if not credentials_path.is_file():
        raise SystemExit(f"Credentials file not found: {credentials_path}")

    image_paths = collect_image_paths(img_dir)
    if not image_paths:
        raise SystemExit(f"No images found in: {img_dir}")

    results = vision_annotate_batch(
        image_paths,
        credentials_path=str(credentials_path),
        batch_size=max(1, int(args.batch_size)),
        verify_ssl=False,
        vertex_location=str(args.vertex_location),
        vertex_model=str(args.vertex_model),
        run_vlm=not bool(args.no_vlm),
    )

    aggregated = aggregate_case_results(results)

    out_path = args.out if args.out is not None else Path("step1_features_v1.json")
    out_path.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

