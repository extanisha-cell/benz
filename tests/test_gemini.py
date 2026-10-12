import base64
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from chemverify import gemini
from chemverify.gemini import GenSettings, GeminiClient, GeminiError, ImagePart, TextPart

OK = json.dumps({
    "candidates": [{"content": {"parts": [{"text": "1\nSame."}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 50, "trafficType": "ON_DEMAND"},
    "modelVersion": "gemini-3.1-pro-preview", "responseId": "r1",
}).encode()


class Seq:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), payload))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(transport, **kw):
    sleeps = []
    c = GeminiClient("SECRET-KEY-123", GenSettings(), transport=transport, sleep=sleeps.append, **kw)
    return c, sleeps


PARTS = [TextPart("Image A:"), ImagePart(b"\x89PNG-bytes", "image/png")]


class RequestTests(unittest.TestCase):
    def test_minimal_request_sends_no_generation_config(self):
        body = gemini.build_request("SYSTEM", PARTS, GenSettings())
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "SYSTEM"}]})
        self.assertNotIn("generationConfig", body)
        parts = body["contents"][0]["parts"]
        self.assertEqual(parts[0], {"text": "Image A:"})
        self.assertEqual(base64.b64decode(parts[1]["inlineData"]["data"]), b"\x89PNG-bytes")
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
        self.assertNotIn("mediaResolution", parts[1])
        self.assertEqual(body["contents"][0]["role"], "user")

    def test_settings_map_to_documented_fields(self):
        s = GenSettings(temperature=1.0, thinking_level="low", media_resolution="high", max_output_tokens=500, seed_base=10)
        body = gemini.build_request("S", PARTS, s, run_idx=3)
        self.assertEqual(body["generationConfig"], {
            "temperature": 1.0, "maxOutputTokens": 500, "seed": 13, "thinkingConfig": {"thinkingLevel": "low"}})
        self.assertEqual(body["contents"][0]["parts"][1]["mediaResolution"], {"level": "media_resolution_high"})

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            GenSettings(thinking_level="extreme")
        with self.assertRaises(ValueError):
            GenSettings(media_resolution="huge")

    def test_fingerprint_changes_with_settings(self):
        self.assertNotEqual(GenSettings().fingerprint(), GenSettings(thinking_level="low").fingerprint())
        self.assertEqual(GenSettings().fingerprint(), GenSettings().fingerprint())


class ResponseTests(unittest.TestCase):
    def test_thought_parts_are_excluded_and_usage_filtered(self):
        obj = {"candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}, {"text": "1"}]}, "finishReason": "STOP"}],
               "usageMetadata": {"promptTokenCount": 5, "x": "str"}}
        r = gemini.parse_response(obj)
        self.assertEqual(r.text, "1")
        self.assertEqual(r.usage, {"promptTokenCount": 5})

    def test_blocked_and_empty(self):
        r = gemini.parse_response({"promptFeedback": {"blockReason": "SAFETY"}})
        self.assertEqual((r.text, r.block_reason), ("", "SAFETY"))
        r = gemini.parse_response({"candidates": [{"finishReason": "MAX_TOKENS"}]})
        self.assertEqual((r.text, r.finish_reason), ("", "MAX_TOKENS"))


class ClientTests(unittest.TestCase):
    def test_success_sends_key_in_header_only(self):
        t = Seq((200, {}, OK))
        c, sleeps = client(t)
        r = c.generate("SYS", PARTS)
        self.assertEqual(r.text, "1\nSame.")
        url, headers, payload = t.calls[0]
        self.assertTrue(url.endswith("/models/gemini-3.1-pro-preview:generateContent"))
        self.assertEqual(headers["x-goog-api-key"], "SECRET-KEY-123")
        self.assertNotIn(b"SECRET-KEY-123", payload)
        self.assertNotIn("SECRET-KEY-123", url)
        self.assertEqual(sleeps, [])
        self.assertNotIn("SECRET", repr(c))

    def test_retries_503_then_succeeds(self):
        t = Seq((503, {}, b'{"error":{"message":"overloaded"}}'), (200, {}, OK))
        c, sleeps = client(t)
        self.assertEqual(c.generate("S", PARTS).text, "1\nSame.")
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(len(sleeps), 1)

    def test_retry_after_header_and_retryinfo_body(self):
        t = Seq((429, {"Retry-After": "7"}, b"{}"), (200, {}, OK))
        c, sleeps = client(t)
        c.generate("S", PARTS)
        self.assertEqual(sleeps, [7.0])
        body = json.dumps({"error": {"message": "slow down", "details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]}}).encode()
        t = Seq((429, {}, body), (200, {}, OK))
        c, sleeps = client(t)
        c.generate("S", PARTS)
        self.assertEqual(sleeps, [12.0])

    def test_client_errors_are_not_retried(self):
        t = Seq((400, {}, b'{"error":{"message":"bad image"}}'))
        c, sleeps = client(t)
        with self.assertRaises(GeminiError) as ctx:
            c.generate("S", PARTS)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(len(t.calls), 1)
        self.assertIn("bad image", str(ctx.exception))

    def test_daily_quota_is_not_retried(self):
        body = json.dumps({"error": {"message": "Quota exceeded", "details": [{"violations": [{"quotaId": "RequestsPerDay"}]}]}}).encode()
        t = Seq((429, {}, body))
        c, _ = client(t)
        with self.assertRaises(GeminiError) as ctx:
            c.generate("S", PARTS)
        self.assertEqual(len(t.calls), 1)
        self.assertIn("daily quota", str(ctx.exception))

    def test_gives_up_after_max_retries(self):
        t = Seq(*[(500, {}, b"{}")] * 3)
        c, sleeps = client(t, max_retries=2)
        with self.assertRaises(GeminiError):
            c.generate("S", PARTS)
        self.assertEqual(len(t.calls), 3)
        self.assertEqual(len(sleeps), 2)

    def test_network_errors_are_retried(self):
        t = Seq(urllib.error.URLError("reset"), TimeoutError("slow"), (200, {}, OK))
        c, _ = client(t)
        self.assertEqual(c.generate("S", PARTS).text, "1\nSame.")

    def test_key_never_appears_in_errors(self):
        t = Seq((400, {}, b'{"error":{"message":"invalid key SECRET-KEY-123"}}'))
        c, _ = client(t)
        with self.assertRaises(GeminiError) as ctx:
            c.generate("S", PARTS)
        self.assertNotIn("SECRET-KEY-123", str(ctx.exception))

    def test_invalid_json_body_is_retried(self):
        t = Seq((200, {}, b"<html>"), (200, {}, OK))
        c, _ = client(t)
        self.assertEqual(c.generate("S", PARTS).text, "1\nSame.")

    def test_empty_key_rejected(self):
        with self.assertRaises(ValueError):
            GeminiClient("")


class HelperTests(unittest.TestCase):
    def test_rate_limiter_spaces_calls(self):
        sleeps = []
        limiter = gemini.RateLimiter(60, clock=lambda: 0.0, sleep=sleeps.append)
        for _ in range(3):
            limiter.wait()
        self.assertEqual(sleeps, [1.0, 2.0])
        gemini.RateLimiter(None).wait()

    def test_get_api_key_from_env(self):
        self.assertEqual(gemini.get_api_key({"GEMINI_API_KEY": " abc "}), ("abc", "environment:GEMINI_API_KEY"))
        self.assertEqual(gemini.get_api_key({"GOOGLE_API_KEY": "g"})[0], "g")
        self.assertEqual(gemini.get_api_key({}), (None, "not set"))

    def write_env_file(self, text: str) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="chemverify_test_"))
        self.addCleanup(lambda: [p.unlink() for p in directory.iterdir()] or directory.rmdir())
        path = directory / ".chemverify.env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_key_file_parsing(self):
        path = self.write_env_file('# comment\n\nexport GEMINI_API_KEY = "AIzaSECRET"   \nOTHER=x\n')
        self.assertEqual(gemini.read_env_file(path), {"GEMINI_API_KEY": "AIzaSECRET", "OTHER": "x"})
        self.assertEqual(gemini.read_env_file(self.write_env_file("GEMINI_API_KEY=abc # note")), {"GEMINI_API_KEY": "abc"})
        self.assertEqual(gemini.read_env_file(path.parent / "missing.env"), {})

    def test_key_file_wins_over_environment_and_never_leaks_into_the_source_label(self):
        path = self.write_env_file("GEMINI_API_KEY=from-file\n")
        key, source = gemini.get_api_key({"GEMINI_API_KEY": "stale"}, env_file=path)
        self.assertEqual(key, "from-file")
        self.assertEqual(source, "key file .chemverify.env:GEMINI_API_KEY")
        self.assertNotIn("from-file", source)

    def test_empty_key_file_falls_through_to_the_environment(self):
        path = self.write_env_file("GEMINI_API_KEY=\n")
        self.assertEqual(gemini.get_api_key({"GEMINI_API_KEY": "stale"}, env_file=path)[0], "stale")
        self.assertEqual(gemini.get_api_key({}, env_file=path), (None, "not set"))

    def test_key_shape_warning_describes_the_key_without_quoting_it(self):
        good = "AIza" + "x" * 35
        self.assertIsNone(gemini.key_shape_warning(good))
        warning = gemini.key_shape_warning("sk-" + "y" * 70)
        self.assertIn("73 characters", warning)
        self.assertIn("does not start", warning)
        self.assertNotIn("yyy", warning)

    def test_list_models_paginates(self):
        pages = [(200, json.dumps({"models": [{"name": "models/a", "supportedGenerationMethods": ["generateContent"]}], "nextPageToken": "t"}).encode()),
                 (200, json.dumps({"models": [{"name": "models/b"}]}).encode())]
        urls = []

        def get(url, headers, timeout):
            urls.append(url)
            return pages.pop(0)

        models = gemini.list_models("k", http_get=get)
        self.assertEqual([m["name"] for m in models], ["a", "b"])
        self.assertIn("pageToken=t", urls[1])
        self.assertNotIn("key=", urls[0])

    def test_list_models_error_shows_googles_message_but_never_the_key(self):
        body = json.dumps({"error": {"code": 400, "message": "API key not valid (secret-key-123). Please pass a valid API key."}}).encode()
        with self.assertRaises(gemini.GeminiError) as ctx:
            gemini.list_models("secret-key-123", http_get=lambda url, headers, timeout: (400, body))
        text = str(ctx.exception)
        self.assertIn("API key not valid", text)
        self.assertIn("HTTP 400", text)
        self.assertNotIn("secret-key-123", text)


if __name__ == "__main__":
    unittest.main()
