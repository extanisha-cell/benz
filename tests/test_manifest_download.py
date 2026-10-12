import json
import tempfile
import unittest
from pathlib import Path

from chemverify.download import DownloadError, download_all
from chemverify.manifest import Row, check_host, download_url, read_rows, sanitize_url

from .test_imageinfo import make_png

S3 = ("https://annotationprod.s3.ap-south-1.amazonaws.com/Script-Uploads/x_molecule_434.png"
      "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=secret123&X-Amz-Expires=604800")
DRIVE = "https://drive.google.com/file/d/ABC_def-123/view?usp=drivesdk"
HEADER = "IDs,Samples,Wrongly annotated,Correct version\n"


def write_csv(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return Path(handle.name)


class ManifestTests(unittest.TestCase):
    def test_read_rows_and_sanitising(self):
        rows = read_rows(write_csv(HEADER + f"m1,{S3},{DRIVE},{DRIVE}\n"))
        self.assertEqual(rows[0].id, "m1")
        clean = sanitize_url(rows[0].ref)
        self.assertNotIn("secret123", clean)
        self.assertNotIn("?", clean)
        self.assertEqual(sanitize_url(DRIVE), "https://drive.google.com/file/d/ABC_def-123/view")
        self.assertEqual(download_url(DRIVE), "https://drive.google.com/uc?export=download&id=ABC_def-123")
        self.assertEqual(download_url(S3), S3)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            read_rows(write_csv(HEADER + f"../evil,{DRIVE},{DRIVE},{DRIVE}\n"))
        with self.assertRaises(ValueError):
            read_rows(write_csv(HEADER + f"m1,{DRIVE},{DRIVE},{DRIVE}\nm1,{DRIVE},{DRIVE},{DRIVE}\n"))
        with self.assertRaises(ValueError):
            read_rows(write_csv(HEADER + f"m1,http://drive.google.com/file/d/x/view,{DRIVE},{DRIVE}\n"))
        with self.assertRaises(ValueError):
            read_rows(write_csv("IDs,Samples\nm1,https://x\n"))

    def test_host_allow_list(self):
        check_host(DRIVE)
        check_host(S3)
        for bad in ("https://evil.example.com/a.png", "file:///etc/passwd", "http://drive.google.com/x",
                    "https://drive.google.com.evil.com/x"):
            with self.assertRaises(ValueError, msg=bad):
                check_host(bad)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.images = Path(self.tmp.name) / "data" / "images"
        self.addCleanup(self.tmp.cleanup)
        self.rows = [Row("m1", S3, DRIVE, DRIVE)]

    def test_downloads_validates_and_is_idempotent(self):
        calls = []

        def fetch(url):
            calls.append(url)
            return make_png(), "image/png"

        manifest, problems = download_all(self.rows, self.images, fetch=fetch, log=lambda m: None)
        self.assertEqual(problems, [])
        self.assertTrue(manifest["molecules"][0]["complete"])
        self.assertEqual(len(calls), 3)
        self.assertTrue((self.images / "m1" / "ref.png").exists())
        self.assertNotIn("secret123", json.dumps(manifest))
        self.assertEqual(manifest["molecules"][0]["images"]["ref"]["file"], "images/m1/ref.png")

        manifest2, _ = download_all(self.rows, self.images, fetch=fetch, log=lambda m: None)
        self.assertEqual(len(calls), 3, "second run must reuse the files on disk")
        self.assertTrue(manifest2["molecules"][0]["complete"])

    def test_html_and_non_image_and_bad_host_are_reported(self):
        def fetch(url):
            if "ABC" in url:
                return b"<html>sign in</html>", "text/html; charset=utf-8"
            return b"\x00\x01 not an image", "application/octet-stream"

        rows = [Row("m1", S3, DRIVE, DRIVE), Row("m2", "https://evil.example.com/a.png", DRIVE, DRIVE)]
        manifest, problems = download_all(rows, self.images, fetch=fetch, log=lambda m: None)
        self.assertFalse(manifest["molecules"][0]["complete"])
        text = "\n".join(problems)
        self.assertIn("HTML page", text)
        self.assertIn("not a recognised image", text)
        self.assertIn("allow-list", text)
        self.assertNotIn("secret123", text)

    def test_stale_file_with_other_extension_is_replaced(self):
        folder = self.images / "m1"
        folder.mkdir(parents=True)
        (folder / "ref.jpg").write_bytes(b"junk")
        manifest, problems = download_all(self.rows, self.images, fetch=lambda u: (make_png(), "image/png"), log=lambda m: None)
        self.assertEqual(problems, [])
        self.assertFalse((folder / "ref.jpg").exists())
        self.assertTrue((folder / "ref.png").exists())


if __name__ == "__main__":
    unittest.main()
