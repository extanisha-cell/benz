"""Dependency-free image sniffing: format, size, alpha and truncation checks.

Only headers/chunks are read and pixel data is never decoded, so no image library is
needed and the module is safe to run under ``python -I``.
"""
from __future__ import annotations

import struct
from dataclasses import asdict, dataclass

GEMINI_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"})
_EXTENSIONS = {"png": "png", "jpeg": "jpg", "webp": "webp", "gif": "gif", "heic": "heic", "heif": "heif"}
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_SOF_MARKERS = frozenset(set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC})
_HEIC_BRANDS = (b"heic", b"heix", b"hevc", b"hevx")
_HEIF_BRANDS = (b"mif1", b"msf1", b"heif")


@dataclass(frozen=True)
class ImageInfo:
    format: str
    mime: str
    width: int | None
    height: int | None
    has_alpha: bool
    complete: bool
    gemini_supported: bool

    @property
    def extension(self) -> str:
        return _EXTENSIONS.get(self.format, "bin")

    def as_dict(self) -> dict:
        return asdict(self)


def sniff(data: bytes) -> ImageInfo | None:
    """Identify an image from its leading bytes; None when it is not a recognised image."""
    if data.startswith(_PNG_SIG):
        return _png(data)
    if data[:2] == b"\xff\xd8":
        return _jpeg(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp(data)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return _gif(data)
    if data[4:8] == b"ftyp":
        return _heif(data)
    return None


def _png(data: bytes) -> ImageInfo | None:
    width = height = None
    has_alpha = complete = False
    pos, n = 8, len(data)
    while pos + 8 <= n:
        length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if ctype == b"IHDR" and len(body) >= 13:
            width, height = struct.unpack(">II", body[:8])
            has_alpha = has_alpha or body[9] in (4, 6)
        elif ctype == b"tRNS":
            has_alpha = True
        elif ctype == b"IEND":
            complete = True
            break
        pos += 12 + length
    if width is None:
        return None
    return ImageInfo("png", "image/png", width, height, has_alpha, complete, True)


def _jpeg(data: bytes) -> ImageInfo:
    """Walk the segments to the first real end-of-image marker. Camera apps often append extra
    data after it (Samsung 'SEFT' trailers, motion-photo video), which must not count as truncation."""
    width = height = None
    complete = False
    pos, n = 2, len(data)
    while True:
        pos = data.find(b"\xff", pos)  # inside entropy-coded data this skips to the next 0xFF
        if pos < 0 or pos + 1 >= n:
            break
        marker = data[pos + 1]
        if marker in (0x00, 0xFF) or marker == 0x01 or 0xD0 <= marker <= 0xD8:
            pos += 1 if marker == 0xFF else 2  # byte stuffing, fill byte, restart or standalone marker
            continue
        if marker == 0xD9:
            complete = True
            break
        if pos + 4 > n:
            break
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        if marker in _SOF_MARKERS and width is None and pos + 9 <= n:
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
        pos += 2 + max(length, 2)
    return ImageInfo("jpeg", "image/jpeg", width, height, False, complete, True)


def _webp(data: bytes) -> ImageInfo:
    kind = data[12:16]
    width = height = None
    has_alpha = False
    if kind == b"VP8X" and len(data) >= 30:
        has_alpha = bool(data[20] & 0x10)
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
    elif kind == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
    elif kind == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        has_alpha = bool((bits >> 28) & 1)
    complete = len(data) >= int.from_bytes(data[4:8], "little") + 8
    return ImageInfo("webp", "image/webp", width, height, has_alpha, complete, True)


def _gif(data: bytes) -> ImageInfo:
    width, height = struct.unpack("<HH", data[6:10]) if len(data) >= 10 else (None, None)
    complete = data.rstrip(b"\x00").endswith(b";")
    return ImageInfo("gif", "image/gif", width, height, False, complete, False)


def _heif(data: bytes) -> ImageInfo | None:
    brand = data[8:12]
    if brand in _HEIC_BRANDS:
        return ImageInfo("heic", "image/heic", None, None, False, True, True)
    if brand in _HEIF_BRANDS:
        return ImageInfo("heif", "image/heif", None, None, False, True, True)
    return None
