"""Minimal Gemini REST client (generateContent): retries, rate limiting, key redaction.

Standard library only. The request shape follows the Gemini 3 developer guide:
  POST {API_BASE}/models/{model}:generateContent        header: x-goog-api-key
  systemInstruction, contents[].parts[] (text | inlineData), generationConfig
  generationConfig.thinkingConfig.thinkingLevel          (default "high" on Gemini 3.1 Pro)
  per-part mediaResolution {"level": "media_resolution_high"}
Google recommends leaving temperature at its default (1.0) for Gemini 3, so it is omitted unless set.
Every setting that is left as None is simply not sent, so a run reflects the API defaults.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

DEFAULT_MODEL = "gemini-3.1-pro-preview"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
THINKING_LEVELS = ("minimal", "low", "medium", "high")
MEDIA_RESOLUTIONS = ("low", "medium", "high", "ultra_high")
KEY_VARS = ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
BACKENDS = ("gemini", "openrouter")
DATA_COLLECTION = ("allow", "deny")
# Deliberately outside the project: the project lives in OneDrive, which would sync a key to the cloud.
ENV_FILE = Path.home() / ".chemverify.env"


class GeminiError(Exception):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass(frozen=True)
class GenSettings:
    model: str = DEFAULT_MODEL
    temperature: float | None = None
    thinking_level: str | None = None
    media_resolution: str | None = None
    max_output_tokens: int | None = None
    seed_base: int | None = None
    response_mime_type: str | None = None
    backend: str = "gemini"  # who we call: Google directly, or OpenRouter (which forwards to Google)
    route_to: tuple[str, ...] | None = None  # OpenRouter only: provider slugs, no fallback to others
    data_collection: str | None = None  # OpenRouter only: "deny" = only providers that do not keep or train on data

    def __post_init__(self) -> None:
        if self.thinking_level is not None and self.thinking_level not in THINKING_LEVELS:
            raise ValueError(f"thinking_level must be one of {THINKING_LEVELS}")
        if self.media_resolution is not None and self.media_resolution not in MEDIA_RESOLUTIONS:
            raise ValueError(f"media_resolution must be one of {MEDIA_RESOLUTIONS}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.data_collection is not None and self.data_collection not in DATA_COLLECTION:
            raise ValueError(f"data_collection must be one of {DATA_COLLECTION}")
        if self.backend == "openrouter" and self.media_resolution is not None:
            raise ValueError("media_resolution cannot be set through OpenRouter; its API has no such setting")
        if self.backend != "openrouter" and (self.route_to or self.data_collection):
            raise ValueError("route_to and data_collection only apply to the openrouter backend")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    mime: str


@dataclass
class GeminiResult:
    text: str = ""
    finish_reason: str | None = None
    block_reason: str | None = None
    usage: dict = field(default_factory=dict)
    model_version: str | None = None
    response_id: str | None = None


# --------------------------------------------------------------------------- request / response


def _part_json(part, settings: GenSettings) -> dict:
    if isinstance(part, TextPart):
        return {"text": part.text}
    if isinstance(part, ImagePart):
        out: dict = {"inlineData": {"mimeType": part.mime, "data": base64.b64encode(part.data).decode("ascii")}}
        if settings.media_resolution:
            out["mediaResolution"] = {"level": f"media_resolution_{settings.media_resolution}"}
        return out
    raise TypeError(f"unknown part type: {type(part).__name__}")


def build_request(system_prompt: str, parts: list, settings: GenSettings, run_idx: int = 0) -> dict:
    gen: dict = {}
    if settings.temperature is not None:
        gen["temperature"] = settings.temperature
    if settings.max_output_tokens is not None:
        gen["maxOutputTokens"] = settings.max_output_tokens
    if settings.seed_base is not None:
        gen["seed"] = settings.seed_base + run_idx
    if settings.response_mime_type:
        gen["responseMimeType"] = settings.response_mime_type
    if settings.thinking_level:
        gen["thinkingConfig"] = {"thinkingLevel": settings.thinking_level}
    body: dict = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [_part_json(p, settings) for p in parts]}],
    }
    if gen:
        body["generationConfig"] = gen
    return body


def parse_response(obj: dict) -> GeminiResult:
    result = GeminiResult(
        model_version=obj.get("modelVersion"),
        response_id=obj.get("responseId"),
        usage={k: v for k, v in (obj.get("usageMetadata") or {}).items() if isinstance(v, (int, float))},
    )
    result.block_reason = (obj.get("promptFeedback") or {}).get("blockReason")
    candidates = obj.get("candidates") or []
    if candidates:
        cand = candidates[0]
        result.finish_reason = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        result.text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    return result


# --------------------------------------------------------------------------- transport


def urllib_post(url: str, headers: dict, payload: bytes, timeout: float) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def urllib_get(url: str, headers: dict, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def redact(text: str, key: str | None) -> str:
    return text.replace(key, "***") if key else text


def _retry_after(headers: dict, detail: dict) -> float | None:
    lowered = {k.lower(): v for k, v in headers.items()}
    value = lowered.get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    for item in detail.get("details") or []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            match = re.match(r"^([\d.]+)s$", str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


def _http_error(status: int, headers: dict, data: bytes) -> GeminiError:
    detail: dict = {}
    try:
        parsed = json.loads(data.decode("utf-8", "replace"))
        detail = parsed.get("error", {}) if isinstance(parsed, dict) else {}
        message = str(detail.get("message", ""))
    except ValueError:
        message = data[:300].decode("utf-8", "replace")
    blob = f"{message} {json.dumps(detail)[:600]}"
    daily = status == 429 and ("PerDay" in blob or "per day" in blob.lower())
    retryable = status in (408, 429, 500, 502, 503, 504) and not daily
    text = f"HTTP {status}: {message[:500] or 'no message'}"
    if daily:
        text += " (daily quota exhausted; not retrying)"
    return GeminiError(text, status=status, retryable=retryable, retry_after=_retry_after(headers, detail))


class RateLimiter:
    """Spaces calls at most `rpm` per minute across threads. rpm=None disables it."""

    def __init__(self, rpm: float | None, clock=time.monotonic, sleep=time.sleep):
        self._interval = 60.0 / rpm if rpm else 0.0
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self._interval:
            return
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self._interval
        if slot > now:
            self._sleep(slot - now)


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        settings: GenSettings | None = None,
        *,
        transport: Callable[[str, dict, bytes, float], tuple[int, dict, bytes]] = urllib_post,
        api_base: str = API_BASE,
        max_retries: int = 6,
        timeout_s: float = 300,
        rpm: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        if not api_key:
            raise ValueError("an API key is required")
        self._key = api_key
        self.settings = settings or GenSettings()
        self._transport = transport
        self._api_base = api_base.rstrip("/")
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._limiter = RateLimiter(rpm, sleep=sleep)
        self._sleep = sleep
        self._rng = rng or random.Random()

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self.settings.model!r})"

    def _backoff(self, attempt: int) -> float:
        return min(60.0, 2.0 ** (attempt + 1)) * (0.75 + 0.5 * self._rng.random())

    # The three hooks below are all a backend has to override; retries, rate limiting and redaction are shared.
    def _prepare(self, system_prompt: str, parts: list, run_idx: int) -> tuple[str, dict, bytes]:
        body = build_request(system_prompt, parts, self.settings, run_idx)
        url = f"{self._api_base}/models/{self.settings.model}:generateContent"
        return url, {"Content-Type": "application/json", "x-goog-api-key": self._key}, json.dumps(body).encode("utf-8")

    def _parse(self, obj: dict) -> GeminiResult:
        return parse_response(obj)

    def generate(self, system_prompt: str, parts: list, run_idx: int = 0) -> GeminiResult:
        url, headers, payload = self._prepare(system_prompt, parts, run_idx)
        last: GeminiError | None = None
        for attempt in range(self._max_retries + 1):
            self._limiter.wait()
            try:
                status, resp_headers, data = self._transport(url, headers, payload, self._timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = GeminiError(redact(f"network error: {type(exc).__name__}: {exc}", self._key), retryable=True)
                delay = self._backoff(attempt)
            else:
                if status == 200:
                    try:
                        return self._parse(json.loads(data))
                    except GeminiError as exc:  # a 200 whose body reports a provider error
                        err = GeminiError(redact(str(exc), self._key), status=exc.status, retryable=exc.retryable, retry_after=exc.retry_after)
                        if not err.retryable:
                            raise err
                        last = err
                        delay = self._backoff(attempt)
                    except ValueError:
                        last = GeminiError("response was not valid JSON", status=200, retryable=True)
                        delay = self._backoff(attempt)
                else:
                    err = _http_error(status, resp_headers, data)
                    err = GeminiError(redact(str(err), self._key), status=err.status, retryable=err.retryable, retry_after=err.retry_after)
                    if not err.retryable:
                        raise err
                    last = err
                    delay = err.retry_after if err.retry_after is not None else self._backoff(attempt)
            if attempt == self._max_retries:
                break
            self._sleep(min(delay, 120.0))
        assert last is not None
        raise last


# --------------------------------------------------------------------------- key + model discovery


def read_env_file(path) -> dict[str, str]:
    """NAME=value lines. Blank lines and '#' comments are skipped; quotes and spaces around a value are stripped."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        else:
            value = value.split(" #")[0].strip()
        values[name] = value
    return values


def detect_backend(key: str | None, source: str = "") -> str:
    """Which service a key belongs to: OpenRouter keys start with 'sk-or-'; so does anything stored as OPENROUTER_API_KEY."""
    if (key and key.startswith("sk-or-")) or "OPENROUTER_API_KEY" in source:
        return "openrouter"
    return "gemini"


def key_shape_warning(key: str, backend: str = "gemini") -> str | None:
    """A hint when a key does not look like one for the backend in use (never includes any part of the key)."""
    if backend == "openrouter":
        if key.startswith("sk-or-"):
            return None
        return f"this key has {len(key)} characters and does not start with 'sk-or-'; OpenRouter keys do, so OpenRouter may reject it"
    if len(key) == 39 and key.startswith("AIza"):
        return None
    return (f"this key has {len(key)} characters and {'starts' if key.startswith('AIza') else 'does not start'} with 'AIza'; "
            "a Gemini API key from Google AI Studio is usually 39 characters starting with 'AIza', so Google may reject it")


def get_api_key(env: dict | None = None, env_file=None) -> tuple[str | None, str]:
    """Return (key, source). Order: the key file (default ~/.chemverify.env), the process environment,
    then the Windows *user* environment, so a freshly set variable works without restarting VS Code.
    The key file wins so that a stale variable never blocks a fixed key. The key is never printed or stored.
    When `env` is given (tests), only that mapping, and `env_file` if given, are consulted."""
    use_registry = env is None
    if env_file is None and env is None:
        env_file = ENV_FILE
    if env_file is not None:
        file_values = read_env_file(env_file)
        for name in KEY_VARS:
            if file_values.get(name):
                return file_values[name], f"key file {Path(env_file).name}:{name}"
    env = os.environ if env is None else env
    for name in KEY_VARS:
        value = (env.get(name) or "").strip()
        if value:
            return value, f"environment:{name}"
    if use_registry and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                for name in KEY_VARS:
                    try:
                        value, _ = winreg.QueryValueEx(handle, name)
                    except FileNotFoundError:
                        continue
                    if isinstance(value, str) and value.strip():
                        return value.strip(), f"windows-user-environment:{name}"
        except OSError:
            pass
    return None, "not set"


def list_models(api_key: str, *, api_base: str = API_BASE, http_get=urllib_get, timeout: float = 60) -> list[dict]:
    """Models visible to this key, with the methods each supports."""
    found: list[dict] = []
    token = None
    while True:
        url = f"{api_base.rstrip('/')}/models?pageSize=200"
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        status, data = http_get(url, {"x-goog-api-key": api_key}, timeout)
        if status != 200:
            reason = redact(str(_http_error(status, {}, data)), api_key)
            raise GeminiError(f"model listing failed: {reason}", status=status)
        obj = json.loads(data)
        for model in obj.get("models", []):
            name = str(model.get("name", ""))
            found.append(
                {
                    "name": name[7:] if name.startswith("models/") else name,
                    "display_name": model.get("displayName"),
                    "methods": model.get("supportedGenerationMethods", []),
                }
            )
        token = obj.get("nextPageToken")
        if not token:
            return found
