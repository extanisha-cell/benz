import base64
import json
import unittest

from chemverify import gemini, openrouter
from chemverify.gemini import GeminiError, GenSettings, ImagePart, TextPart
from chemverify.openrouter import OpenRouterClient

PARTS = [TextPart("Image A:"), ImagePart(b"\x89PNG-bytes", "image/png")]
KEY = "sk-or-v1-SECRET-KEY-123"


def settings(**kw) -> GenSettings:
    return GenSettings(model=openrouter.DEFAULT_MODEL, backend="openrouter", **kw)


def ok_body(**over) -> dict:
    body = {
        "id": "gen-1", "model": "google/gemini-3.1-pro-preview", "provider": "Google AI Studio",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "1\nSame compound."}}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020, "cost": 0.004,
                  "completion_tokens_details": {"reasoning_tokens": 100}},
    }
    body.update(over)
    return body


def resp(status, body, headers=None):
    return status, headers or {}, json.dumps(body).encode() if not isinstance(body, bytes) else body


class Seq:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), payload))
        return self.responses.pop(0)


def client(transport, **kw):
    sleeps = []
    return OpenRouterClient(KEY, kw.pop("settings", settings()), transport=transport, sleep=sleeps.append, **kw), sleeps


class RequestTests(unittest.TestCase):
    def test_minimal_request(self):
        body = openrouter.build_request("SYSTEM", PARTS, settings())
        self.assertEqual(body["model"], "google/gemini-3.1-pro-preview")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYSTEM"})
        user = body["messages"][1]
        self.assertEqual(user["role"], "user")
        self.assertEqual(user["content"][0], {"type": "text", "text": "Image A:"})
        url = user["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), b"\x89PNG-bytes")
        for field in ("temperature", "max_tokens", "seed", "reasoning", "provider"):
            self.assertNotIn(field, body)  # unset settings are not sent, so the API defaults apply

    def test_settings_map_to_openrouter_fields(self):
        s = settings(temperature=1.0, thinking_level="low", max_output_tokens=500, seed_base=10)
        body = openrouter.build_request("S", PARTS, s, run_idx=3)
        self.assertEqual((body["temperature"], body["max_tokens"], body["seed"]), (1.0, 500, 13))
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["provider"], {"require_parameters": True})  # never silently ignore a setting

    def test_routing_and_data_collection(self):
        s = settings(route_to=("google-vertex",), data_collection="deny")
        provider = openrouter.build_request("S", PARTS, s)["provider"]
        self.assertEqual(provider, {"order": ["google-vertex"], "allow_fallbacks": False, "data_collection": "deny"})

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            settings(media_resolution="high")  # OpenRouter has no such setting
        with self.assertRaises(ValueError):
            GenSettings(route_to=("x",))  # routing only makes sense on openrouter
        with self.assertRaises(ValueError):
            settings(data_collection="maybe")
        with self.assertRaises(ValueError):
            GenSettings(backend="elsewhere")

    def test_backend_is_part_of_the_fingerprint(self):
        self.assertNotEqual(GenSettings().fingerprint(), GenSettings(backend="openrouter").fingerprint())


class ParseTests(unittest.TestCase):
    def test_ok_response(self):
        r = openrouter.parse_response(ok_body())
        self.assertEqual(r.text, "1\nSame compound.")
        self.assertEqual(r.finish_reason, "stop")
        self.assertIsNone(r.block_reason)
        self.assertEqual(r.model_version, "google/gemini-3.1-pro-preview @ Google AI Studio")
        self.assertEqual(r.response_id, "gen-1")
        self.assertEqual(r.usage, {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020, "cost": 0.004, "reasoning_tokens": 100})

    def test_provider_missing_leaves_plain_model_name(self):
        body = ok_body()
        del body["provider"]
        self.assertEqual(openrouter.parse_response(body).model_version, "google/gemini-3.1-pro-preview")

    def test_content_filter_is_a_block(self):
        body = ok_body(choices=[{"finish_reason": "content_filter", "message": {"content": None}}])
        r = openrouter.parse_response(body)
        self.assertEqual(r.block_reason, "content_filter")
        self.assertEqual(r.text, "")

    def test_null_and_list_content(self):
        self.assertEqual(openrouter.parse_response(ok_body(choices=[{"finish_reason": "length", "message": {"content": None}}])).text, "")
        parts = [{"type": "text", "text": "0\n"}, {"type": "text", "text": "Differs."}]
        self.assertEqual(openrouter.parse_response(ok_body(choices=[{"finish_reason": "stop", "message": {"content": parts}}])).text, "0\nDiffers.")

    def test_errors_inside_a_200_body(self):
        with self.assertRaises(GeminiError) as ctx:
            openrouter.parse_response({"error": {"code": 502, "message": "upstream down"}})
        self.assertTrue(ctx.exception.retryable)
        with self.assertRaises(GeminiError) as ctx:
            openrouter.parse_response(ok_body(choices=[{"finish_reason": "error", "error": {"code": 400, "message": "bad input"}}]))
        self.assertFalse(ctx.exception.retryable)
        with self.assertRaises(GeminiError) as ctx:
            openrouter.parse_response(ok_body(choices=[{"finish_reason": "error", "message": {"content": ""}}]))
        self.assertTrue(ctx.exception.retryable)


class ClientTests(unittest.TestCase):
    def test_request_goes_to_chat_completions_with_bearer_auth(self):
        t = Seq(resp(200, ok_body()))
        c, _ = client(t)
        result = c.generate("S", PARTS)
        url, headers, payload = t.calls[0]
        self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(headers["Authorization"], f"Bearer {KEY}")
        self.assertNotIn("x-goog-api-key", headers)
        self.assertEqual(json.loads(payload)["messages"][0]["content"], "S")
        self.assertEqual(result.text, "1\nSame compound.")

    def test_retries_rate_limit_then_succeeds(self):
        t = Seq(resp(429, {"error": {"code": 429, "message": "slow down"}}, {"Retry-After": "7"}), resp(200, ok_body()))
        c, sleeps = client(t)
        self.assertEqual(c.generate("S", PARTS).text, "1\nSame compound.")
        self.assertEqual(sleeps, [7.0])

    def test_retries_a_provider_error_reported_inside_a_200(self):
        t = Seq(resp(200, {"error": {"code": 503, "message": "overloaded"}}), resp(200, ok_body()))
        c, sleeps = client(t)
        self.assertEqual(c.generate("S", PARTS).text, "1\nSame compound.")
        self.assertEqual(len(sleeps), 1)

    def test_non_retryable_error_surfaces_the_message_but_never_the_key(self):
        t = Seq(resp(402, {"error": {"code": 402, "message": f"Insufficient credits for {KEY}"}}))
        c, _ = client(t)
        with self.assertRaises(GeminiError) as ctx:
            c.generate("S", PARTS)
        text = str(ctx.exception)
        self.assertIn("Insufficient credits", text)
        self.assertIn("HTTP 402", text)
        self.assertNotIn(KEY, text)
        self.assertEqual(len(t.calls), 1)

    def test_error_inside_200_is_redacted_too(self):
        t = Seq(resp(200, {"error": {"code": 400, "message": f"bad key {KEY}"}}))
        c, _ = client(t)
        with self.assertRaises(GeminiError) as ctx:
            c.generate("S", PARTS)
        self.assertNotIn(KEY, str(ctx.exception))

    def test_repr_hides_key(self):
        c, _ = client(Seq())
        self.assertNotIn(KEY, repr(c))
        self.assertIn("OpenRouterClient", repr(c))

    def test_wrong_backend_settings_rejected(self):
        with self.assertRaises(ValueError):
            OpenRouterClient(KEY, GenSettings(), transport=Seq())


class ListModelsTests(unittest.TestCase):
    def test_lists_in_the_common_shape(self):
        data = json.dumps({"data": [
            {"id": "google/gemini-3.1-pro-preview", "name": "Google: Gemini 3.1 Pro Preview", "architecture": {"input_modalities": ["text", "image"]}},
            {"id": "other/text-only", "name": "Text", "architecture": {"input_modalities": ["text"]}},
        ]}).encode()
        seen = {}

        def get(url, headers, timeout):
            seen.update(url=url, headers=headers)
            return 200, data

        models = openrouter.list_models(KEY, http_get=get)
        self.assertEqual(seen["url"], "https://openrouter.ai/api/v1/models")
        self.assertEqual([m["name"] for m in models], ["google/gemini-3.1-pro-preview", "other/text-only"])
        self.assertIn("image-input", models[0]["methods"])
        self.assertNotIn("image-input", models[1]["methods"])

    def test_error_message_shown_and_key_redacted(self):
        body = json.dumps({"error": {"message": f"No auth credentials found for {KEY}"}}).encode()
        with self.assertRaises(GeminiError) as ctx:
            openrouter.list_models(KEY, http_get=lambda u, h, t: (401, body))
        self.assertIn("No auth credentials", str(ctx.exception))
        self.assertNotIn(KEY, str(ctx.exception))


class BackendDetectionTests(unittest.TestCase):
    def test_detect_backend(self):
        self.assertEqual(gemini.detect_backend("sk-or-v1-abc", "environment:GEMINI_API_KEY"), "openrouter")
        self.assertEqual(gemini.detect_backend("whatever", "key file .chemverify.env:OPENROUTER_API_KEY"), "openrouter")
        self.assertEqual(gemini.detect_backend("AIzaXXXX", "key file .chemverify.env:GEMINI_API_KEY"), "gemini")
        self.assertEqual(gemini.detect_backend(None, "not set"), "gemini")

    def test_key_shape_warning_per_backend(self):
        self.assertIsNone(gemini.key_shape_warning("sk-or-v1-" + "a" * 64, "openrouter"))
        self.assertIn("does not start with 'sk-or-'", gemini.key_shape_warning("AIza" + "x" * 35, "openrouter"))
        self.assertIn("73 characters", gemini.key_shape_warning("sk-or-v1-" + "a" * 64, "gemini"))

    def test_openrouter_var_is_read_first(self):
        env = {"GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": "o"}
        self.assertEqual(gemini.get_api_key(env), ("o", "environment:OPENROUTER_API_KEY"))


if __name__ == "__main__":
    unittest.main()
