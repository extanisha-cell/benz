import struct
import unittest
import zlib

from chemverify.imageinfo import sniff


def make_png(width=3, height=2, colour_type=2, trns=False, iend=True) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    channels = {2: 3, 6: 4, 0: 1}[colour_type]
    raw = b"".join(b"\x00" + b"\x00" * (width * channels) for _ in range(height))
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0))
    if trns:
        out += chunk(b"tRNS", b"\x00\x00")
    out += chunk(b"IDAT", zlib.compress(raw))
    if iend:
        out += chunk(b"IEND", b"")
    return out


def make_jpeg(width=9, height=7, eoi=True) -> bytes:
    sof = b"\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", height, width) + b"\x01\x01\x11\x00"
    return b"\xff\xd8" + sof + (b"\xff\xd9" if eoi else b"")


class SniffTests(unittest.TestCase):
    def test_png_basic(self):
        info = sniff(make_png())
        self.assertEqual((info.format, info.mime, info.width, info.height), ("png", "image/png", 3, 2))
        self.assertFalse(info.has_alpha)
        self.assertTrue(info.complete)
        self.assertTrue(info.gemini_supported)
        self.assertEqual(info.extension, "png")

    def test_png_alpha_from_colour_type_and_trns(self):
        self.assertTrue(sniff(make_png(colour_type=6)).has_alpha)
        self.assertTrue(sniff(make_png(trns=True)).has_alpha)

    def test_png_truncated(self):
        self.assertFalse(sniff(make_png(iend=False)).complete)

    def test_jpeg(self):
        info = sniff(make_jpeg())
        self.assertEqual((info.format, info.width, info.height, info.complete), ("jpeg", 9, 7, True))
        self.assertEqual(info.extension, "jpg")
        self.assertFalse(sniff(make_jpeg(eoi=False)).complete)

    def test_jpeg_with_trailing_camera_data_is_complete(self):
        # Samsung-style trailer after the end-of-image marker must not look like truncation.
        info = sniff(make_jpeg() + b"\x00\x00\x00\x66SEFT" * 20)
        self.assertEqual((info.width, info.height, info.complete), (9, 7, True))

    def test_jpeg_scan_data_with_stuffed_bytes(self):
        sos = b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00"
        scan = b"\x12\xff\x00\x34\xff\xd0\x56"  # stuffed FF00 and a restart marker are not segment markers
        body = make_jpeg(eoi=False) + sos + scan
        self.assertFalse(sniff(body).complete, "no end-of-image marker: truncated")
        self.assertTrue(sniff(body + b"\xff\xd9").complete)

    def test_jpeg_cut_off_mid_segment(self):
        self.assertFalse(sniff(make_jpeg()[:6]).complete)

    def test_webp_vp8x(self):
        body = b"VP8X" + (10).to_bytes(4, "little") + bytes([0x10, 0, 0, 0]) + (99).to_bytes(3, "little") + (49).to_bytes(3, "little")
        data = b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WEBP" + body
        info = sniff(data)
        self.assertEqual((info.format, info.width, info.height, info.has_alpha, info.complete), ("webp", 100, 50, True, True))

    def test_gif_is_not_accepted_by_gemini(self):
        info = sniff(b"GIF89a" + struct.pack("<HH", 5, 4) + b"\x00\x00\x00;")
        self.assertEqual((info.format, info.width, info.height), ("gif", 5, 4))
        self.assertFalse(info.gemini_supported)

    def test_unrecognised(self):
        self.assertIsNone(sniff(b"<html>not an image</html>"))
        self.assertIsNone(sniff(b""))


if __name__ == "__main__":
    unittest.main()
