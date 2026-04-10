#!/usr/bin/env python3
"""Shared client/model resolution for local Waldwicht test scripts."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8432")
BASE_URL = f"http://{HOST}:{PORT}/v1"
REQUEST_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "600"))
TEST_ENABLE_THINKING = os.environ.get("TEST_ENABLE_THINKING", "").lower() in {"1", "true", "yes", "on"}


def create_client() -> OpenAI:
    return OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        timeout=REQUEST_TIMEOUT,
    )


def resolve_model(client: OpenAI) -> tuple[str, list[str]]:
    try:
        models = client.models.list().data
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach local server at {BASE_URL}. Start it with `make start MODEL=<model>` and retry."
        ) from exc

    if not models:
        raise RuntimeError(f"No models returned by local server at {BASE_URL}.")

    model_ids = [model.id for model in models]
    requested_model = os.environ.get("MODEL")
    if requested_model:
        if requested_model not in model_ids:
            available = ", ".join(model_ids)
            raise RuntimeError(
                f"Requested MODEL={requested_model} is not exposed by {BASE_URL}. Available models: {available}"
            )
        return requested_model, model_ids

    for model_id in model_ids:
        if "Waldwicht" in model_id:
            return model_id, model_ids

    return model_ids[0], model_ids


def apply_chat_defaults(kwargs: dict | None = None) -> dict:
    """Inject stable per-request defaults for local integration tests."""
    merged = dict(kwargs or {})
    extra_body = dict(merged.pop("extra_body", {}) or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})

    if not TEST_ENABLE_THINKING:
        chat_template_kwargs.setdefault("enable_thinking", False)

    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        merged["extra_body"] = extra_body

    return merged


def require_content(response) -> str:
    """Return assistant content or raise a readable error.

    Some server configurations may emit reasoning without final content when a
    request truncates early. Tests should fail with a clear message in that case.
    """
    choice = response.choices[0]
    message = choice.message
    content = message.content
    if content:
        return content

    reasoning = getattr(message, "reasoning", None) or ""
    raise ValueError(
        f"No final content returned (finish_reason={choice.finish_reason}, reasoning_chars={len(reasoning)})"
    )