"""
Vertex AI Gemini (OpenAI-compatible) helper.

Uses the OpenAI Python SDK against Vertex's OpenAI-compatible endpoint:
  https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi

Ref:
  https://docs.cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-chat-completions-non-streaming-image
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import httpx
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from openai import OpenAI


def infer_project_id_from_service_account_key(credentials_path: str) -> Optional[str]:
    try:
        data = json.loads(Path(credentials_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    project_id = data.get("project_id")
    return str(project_id) if project_id else None


def get_access_token(credentials_path: str) -> str:
    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    if not creds.token:
        raise RuntimeError("Failed to obtain access token for Vertex AI.")
    return creds.token


def make_vertex_openai_client(
    *,
    project_id: str,
    location: str,
    access_token: str,
    verify_ssl: bool = False,
) -> OpenAI:
    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi"
    )
    http_client = httpx.Client(verify=verify_ssl, timeout=60.0)
    return OpenAI(base_url=base_url, api_key=access_token, http_client=http_client)


def chat_completion_with_image(
    *,
    client: OpenAI,
    model: str,
    prompt_text: str,
    image_url: str,
    max_output_tokens: int = 800,
) -> str:
    """
    Send a single-image multimodal chat completion to Vertex Gemini.

    Vertex docs show `image_url` as a string (e.g. gs://...). We also support a data URL string.
    If the simplified form fails, we retry with OpenAI's dict form: {"url": ...}.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": image_url},
            ],
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        # Retry with OpenAI's typical image_url object format.
        messages_retry = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages_retry,
            max_tokens=max_output_tokens,
        )
        return resp.choices[0].message.content or ""

