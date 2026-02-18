"""
Detection step: visual feature extraction using Google Cloud Vision API (REST).
Loads images from the project's images/ directory (or --img-dir), runs
LABEL_DETECTION and OBJECT_LOCALIZATION via REST, and outputs a structured summary.
Uses httpx with verify=False to avoid SSL cert issues in locked-down environments.

Usage:
  python sandbox/google_vision/detection_example.py [--img-dir PATH] [--out PATH]
  Default img-dir: project_root/images (all image files under that dir).

Requires: pip install -r sandbox/google_vision/requirements.txt
Auth: GOOGLE_APPLICATION_CREDENTIALS env pointing to a service account JSON key file.
Ref: https://cloud.google.com/vision/docs/request
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.auth.transport.requests import Request

load_dotenv()

VISION_ANNOTATE_URL = "https://vision.googleapis.com/v1/images:annotate"
BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def project_images_dir() -> Path:
    """Project root is parent of sandbox; images dir is project_root/images."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root / "images"


def collect_image_paths(img_dir: Path) -> list[Path]:
    """Sorted list of image paths under img_dir (non-recursive)."""
    paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return paths


def build_rest_requests(image_paths: list[Path]) -> list[tuple[Path, dict]]:
    """Build Vision REST API request bodies: (path, request_dict) per image."""
    requests_with_paths: list[tuple[Path, dict]] = []
    for path in image_paths:
        content_b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        request = {
            "image": {"content": content_b64},
            "features": [
                {"type": "LABEL_DETECTION", "maxResults": 50},
                {"type": "OBJECT_LOCALIZATION", "maxResults": 20},
            ],
        }
        requests_with_paths.append((path, request))
    return requests_with_paths


def extract_features_from_rest_response(path: Path, rest_item: dict) -> dict:
    """Turn one REST response item into the same structured feature summary as before."""
    out = {
        "image_path": str(path),
        "labels": [],
        "localized_objects": [],
    }
    if "error" in rest_item:
        out["error"] = rest_item["error"].get("message", str(rest_item["error"]))
        return out

    for ann in rest_item.get("labelAnnotations") or []:
        out["labels"].append({
            "description": ann.get("description", ""),
            "score": round(ann.get("score", 0), 4),
            "topicality": round(ann["topicality"], 4) if ann.get("topicality") is not None else None,
        })
    for obj in rest_item.get("localizedObjectAnnotations") or []:
        verts = obj.get("boundingPoly", {}).get("normalizedVertices") or []
        out["localized_objects"].append({
            "name": obj.get("name", ""),
            "score": round(obj.get("score", 0), 4),
            "bounds": [{"x": v.get("x", 0), "y": v.get("y", 0)} for v in verts],
        })
    return out


def get_access_token(credentials_path: str) -> str:
    """Return a valid Bearer token for the Vision API using the service account key."""
    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    return creds.token


def run_detection(
    img_dir: Path,
    credentials_path: str,
    http_client: httpx.Client,
) -> list[dict]:
    """Run detection on all images in img_dir via Vision REST API; return per-image feature dicts."""
    image_paths = collect_image_paths(img_dir)
    if not image_paths:
        return []

    token = get_access_token(credentials_path)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    requests_with_paths = build_rest_requests(image_paths)
    all_results: list[dict] = []

    for i in range(0, len(requests_with_paths), BATCH_SIZE):
        chunk = requests_with_paths[i : i + BATCH_SIZE]
        paths_chunk = [p for p, _ in chunk]
        body = {"requests": [r for _, r in chunk]}
        response = http_client.post(VISION_ANNOTATE_URL, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        responses_list = data.get("responses") or []
        for path, rest_item in zip(paths_chunk, responses_list):
            all_results.append(extract_features_from_rest_response(path, rest_item))

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual feature extraction (Detection) via Google Cloud Vision API (REST)."
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=None,
        help="Directory containing images (default: project_root/images)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for detection results",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=None,
        help="Path to service account JSON key (default: GOOGLE_APPLICATION_CREDENTIALS env)",
    )
    args = parser.parse_args()

    img_dir = args.img_dir if args.img_dir is not None else project_images_dir()
    img_dir = img_dir.resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")

    credentials_path = args.credentials
    if credentials_path is None:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise SystemExit(
            "Set GOOGLE_APPLICATION_CREDENTIALS or pass --credentials PATH to a service account JSON key."
        )
    credentials_path = Path(credentials_path).resolve()
    if not credentials_path.is_file():
        raise SystemExit(f"Credentials file not found: {credentials_path}")

    with httpx.Client(verify=False) as http_client:
        results = run_detection(img_dir, str(credentials_path), http_client)

    if not results:
        print("No images found.")
        return

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.out is not None:
        args.out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
