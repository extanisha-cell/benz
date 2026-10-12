"""OpenRouter chat-completions client with the same interface as GeminiClient.

Standard library only. The request shape follows https://openrouter.ai/docs/api-reference/overview:
  POST {API_BASE}/chat/completions        header: Authorization: Bearer <key>
  messages: a system message, then one user message whose content is a list of
            {"type": "text"} and {"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}} parts
  temperature, max_tokens, seed, reasoning {"effort": ...}, provider {order, allow_fallbacks, data_collection, ...}

OpenRouter forwards the request to a model provider (for Gemini, Google), so this is not byte-for-byte what
Google's own API receives: thinking level goes through `reasoning.effort`, there is no media-resolution
setting, and OpenRouter chooses the provider unless `route_to` pins one. Every record keeps the model name
and provider the response reports, so a change of provider shows up in the report.
"""
from __future__ import annotations

import base64
import json

from .gemini import (
    GeminiClient,
    GeminiError,
    GeminiResult,
    GenSettings,
    ImagePart,
    TextPart,
    redact,
    urllib_get,
)

API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- request / response


def _part_json(part) -> dict:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        data = base64.b64encode(part.data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{part.mime};base64,{data}"}}
    raise TypeError(f"unknown part type: {type(part).__name__}")


def build_request(system_prompt: str, parts: list, settings: GenSettings, run_idx: int = 0) -> dict:
    body: dict = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [_part_json(p) for p in parts]},
        ],
    }
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    if settings.max_output_tokens is not None:
        body["max_tokens"] = settings.max_output_tokens
    if settings.seed_base is not None:
        body["seed"] = settings.seed_base + run_idx
    if settings.thinking_level:
        body["reasoning"] = {"effort": settings.thinking_level}
    provider: dict = {}
    if settings.route_to:
        provider.update(order=list(settings.route_to), allow_fallbacks=False)
    if settings.data_collection:
        provider["data_collection"] = settings.data_collection
    if any(k in body for k in ("temperature", "max_tokens", "seed", "reasoning")):
        provider["require_parameters"] = True  # fail loudly instead of a provider silently ignoring a setting
    if provider:
        body["provider"] = provider
    return body


def _error_from(err: dict) -> GeminiError:
    code = err.get("code")
    status = code if isinstance(code, int) else None
    message = str(err.get("message", "")).strip()[:500] or "no message"
    return GeminiError(f"provider error {code}: {message}", status=status, retryable=status in RETRYABLE_CODES)


def _usage(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    usage = {k: raw[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
             if isinstance(raw.get(k), (int, float)) and not isinstance(raw.get(k), bool)}
    reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if isinstance(reasoning, (int, float)) and not isinstance(reasoning, bool):
        usage["reasoning_tokens"] = reasoning
    return usage


def parse_response(obj: dict) -> GeminiResult:
    """Raises GeminiError when the body reports a provider error (even inside a 200 response)."""
    choices = obj.get("choices") or []
    if isinstance(obj.get("error"), dict) and not choices:
        raise _error_from(obj["error"])
    model = obj.get("model")
    provider = obj.get("provider")
    result = GeminiResult(
        model_version=f"{model} @ {provider}" if model and isinstance(provider, str) and provider else model,
        response_id=obj.get("id"),
        usage=_usage(obj.get("usage")),
    )
    if choices:
        choice = choices[0]
        if isinstance(choice.get("error"), dict):
            raise _error_from(choice["error"])
        result.finish_reason = choice.get("finish_reason")
        if result.finish_reason == "error":
            raise GeminiError("the provider ended the response with finish_reason=error", retryable=True)
        if result.finish_reason == "content_filter":
            result.block_reason = "content_filter"
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        result.text = content if isinstance(content, str) else ""
    return result


# --------------------------------------------------------------------------- client + model discovery


class OpenRouterClient(GeminiClient):
    def __init__(self, api_key: str, settings: GenSettings | None = None, **kwargs):
        kwargs.setdefault("api_base", API_BASE)
        super().__init__(api_key, settings, **kwargs)
        if self.settings.backend != "openrouter":
            raise ValueError("OpenRouterClient needs settings with backend='openrouter'")

    def _prepare(self, system_prompt: str, parts: list, run_idx: int) -> tuple[str, dict, bytes]:
        body = build_request(system_prompt, parts, self.settings, run_idx)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._key}",
            "X-Title": "chemverify",
        }
        return f"{self._api_base}/chat/completions", headers, json.dumps(body).encode("utf-8")

    def _parse(self, obj: dict) -> GeminiResult:
        return parse_response(obj)


def list_models(api_key: str, *, api_base: str = API_BASE, http_get=urllib_get, timeout: float = 60) -> list[dict]:
    """Models OpenRouter offers, in the same shape the Gemini lister returns."""
    status, data = http_get(f"{api_base.rstrip('/')}/models", {"Authorization": f"Bearer {api_key}"}, timeout)
    if status != 200:
        message = ""
        try:
            message = str(json.loads(data).get("error", {}).get("message", ""))
        except (ValueError, AttributeError):
            pass
        raise GeminiError(redact(f"model listing failed: HTTP {status}: {message[:300] or 'no message'}", api_key), status=status)
    models = []
    for model in json.loads(data).get("data", []):
        modalities = (model.get("architecture") or {}).get("input_modalities") or []
        models.append({
            "name": str(model.get("id", "")),
            "display_name": model.get("name"),
            "methods": ["chat.completions"] + (["image-input"] if "image" in modalities else []),
        })
    return models
