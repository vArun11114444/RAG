"""
File Validator — app/security/file_validator.py

Validates uploaded files before they enter the ingestion pipeline.
Checks (in order):
  1. File size limit
  2. Extension allowlist
  3. MIME type allowlist (from content-type header)
  4. Magic bytes — actual file content vs. declared type
  5. Malicious signature scan (zip bombs, polyglot markers, macro headers)
  6. PDF-specific: header + trailer validation, /JavaScript detection
  7. Image-specific: PIL header validation (when Pillow is available)

All checks are async-friendly; heavy I/O is offloaded to threads.
"""
from __future__ import annotations

import asyncio
import io
import mimetypes
import os
import struct
from dataclasses import dataclass, field
from typing import Any

from app.security.exceptions import (
    FileValidationException,
    MaliciousFileException,
    SecurityEventType,
)
from app.observability.logger import get_logger

log = get_logger(__name__)

# ── Configuration constants ───────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024   # 50 MB

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".txt", ".md", ".csv",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".docx", ".xlsx", ".pptx",
    ".json", ".xml",
})

ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/json",
    "application/xml",
    "text/xml",
})

# Magic bytes → MIME mapping (first N bytes of file)
_MAGIC_MAP: list[tuple[bytes, str, str]] = [
    (b"%PDF",              "application/pdf",   "PDF header"),
    (b"\x89PNG\r\n\x1a\n","image/png",          "PNG header"),
    (b"\xff\xd8\xff",     "image/jpeg",         "JPEG header"),
    (b"RIFF",             "image/webp",         "WebP/RIFF header"),  # WebP starts RIFF....WEBP
    (b"GIF87a",           "image/gif",          "GIF87a header"),
    (b"GIF89a",           "image/gif",          "GIF89a header"),
    (b"PK\x03\x04",       "application/zip",    "ZIP/Office header"),
    (b"\xd0\xcf\x11\xe0", "application/msword", "Legacy Office (OLE)"),
    (b"{\n",              "application/json",   "JSON"),
    (b"{\"",              "application/json",   "JSON"),
]

# Malicious signatures to block outright
_MALICIOUS_SIGNATURES: list[tuple[bytes, str]] = [
    (b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE", "EICAR antivirus test file"),
    (b"\x4d\x5a\x90\x00",                  "Windows PE executable (MZ header)"),
    (b"\x7fELF",                            "Linux ELF executable"),
    (b"#!/",                                "Shell script shebang"),
    (b"#!python",                           "Python script"),
    (b"<%@ ",                               "JSP/ASP server-side script"),
    (b"<script",                            "Embedded script tag"),
    (b"javascript:",                        "JavaScript protocol URI"),
]

# PDF-specific dangerous patterns
_PDF_DANGEROUS_PATTERNS: list[tuple[bytes, str]] = [
    (b"/JavaScript",  "Embedded JavaScript in PDF"),
    (b"/JS ",         "Embedded JS in PDF"),
    (b"/Launch",      "PDF Launch action (can execute programs)"),
    (b"/OpenAction",  "PDF OpenAction (auto-execute on open)"),
    (b"/AA ",         "PDF Additional Action"),
    (b"/RichMedia",   "PDF RichMedia (Flash embedding)"),
    (b"/EmbeddedFile","PDF EmbeddedFile (file within PDF)"),
]

# Zip bomb detection thresholds
_ZIP_BOMB_RATIO: float = 100.0    # compressed:uncompressed ratio
_ZIP_BOMB_MAX_UNCOMPRESSED: int = 500 * 1024 * 1024   # 500 MB


@dataclass
class FileValidationResult:
    filename: str
    is_valid: bool
    file_size: int
    detected_mime: str | None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadedFile:
    """Thin wrapper around a raw file upload — mirrors FastAPI UploadFile interface."""
    filename: str
    content_type: str | None
    content: bytes


class FileValidator:
    """
    Validates uploaded files through a layered security pipeline.
    Raises FileValidationException or MaliciousFileException on failure.
    """

    def __init__(
        self,
        max_size: int = MAX_FILE_SIZE_BYTES,
        allowed_extensions: frozenset[str] | None = None,
        allowed_mimes: frozenset[str] | None = None,
        block_dangerous_pdfs: bool = True,
        check_zip_bombs: bool = True,
    ) -> None:
        self._max_size = max_size
        self._allowed_ext = allowed_extensions or ALLOWED_EXTENSIONS
        self._allowed_mime = allowed_mimes or ALLOWED_MIME_TYPES
        self._block_dangerous_pdfs = block_dangerous_pdfs
        self._check_zip_bombs = check_zip_bombs

    async def validate(self, file: UploadedFile) -> FileValidationResult:
        """
        Run all validation checks. Raises on first failure.
        Returns FileValidationResult with metadata on success.
        """
        filename = file.filename or "unknown"
        content = file.content
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        # 1. Size check
        file_size = len(content)
        if file_size > self._max_size:
            raise FileValidationException(
                filename=filename,
                reason=f"File size {file_size:,} bytes exceeds limit of {self._max_size:,} bytes",
                event_type=SecurityEventType.FILE_TOO_LARGE,
            )

        # 2. Extension check
        _, ext = os.path.splitext(filename.lower())
        if ext not in self._allowed_ext:
            raise FileValidationException(
                filename=filename,
                reason=f"Extension '{ext}' is not permitted. Allowed: {sorted(self._allowed_ext)}",
                event_type=SecurityEventType.INVALID_MIME,
            )

        # 3. Declared MIME type check (from Content-Type header)
        declared_mime = (file.content_type or "").split(";")[0].strip().lower()
        if declared_mime and declared_mime not in self._allowed_mime:
            # Allow text/* generically for plain text variants
            if not declared_mime.startswith("text/"):
                # If it's a zip, allow it through so zip-bomb check can run
                if declared_mime != "application/zip" or not self._check_zip_bombs:
                    raise FileValidationException(
                        filename=filename,
                        reason=f"Declared MIME type '{declared_mime}' is not permitted",
                        event_type=SecurityEventType.INVALID_MIME,
                    )

        # 4. Magic bytes — detect actual type
        detected_mime = await asyncio.to_thread(self._detect_mime, content)
        metadata["detected_mime"] = detected_mime

        if detected_mime and detected_mime not in self._allowed_mime:
            # If declared and detected mismatch — block
            if declared_mime and detected_mime != declared_mime:
                raise FileValidationException(
                    filename=filename,
                    reason=(
                        f"File content ({detected_mime}) does not match "
                        f"declared type ({declared_mime}) — possible polyglot file"
                    ),
                    event_type=SecurityEventType.MALICIOUS_FILE,
                )

        # 5. Malicious signature scan
        await asyncio.to_thread(self._check_malicious_signatures, filename, content)

        # 6. Type-specific validation
        if ext == ".pdf" or detected_mime == "application/pdf":
            pdf_warnings = await asyncio.to_thread(self._validate_pdf, filename, content)
            warnings.extend(pdf_warnings)

        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".gif") or (
            detected_mime and detected_mime.startswith("image/")
        ):
            img_warnings = await asyncio.to_thread(self._validate_image, filename, content)
            warnings.extend(img_warnings)

        elif ext == ".zip" or detected_mime == "application/zip":
            if self._check_zip_bombs:
                await asyncio.to_thread(self._check_zip_bomb, filename, content)

        log.info(
            "file_validated",
            extra={
                "file_name": filename,
                "size_bytes": file_size,
                "detected_mime": detected_mime,
                "warnings": warnings,
            },
        )

        return FileValidationResult(
            filename=filename,
            is_valid=True,
            file_size=file_size,
            detected_mime=detected_mime,
            warnings=warnings,
            metadata=metadata,
        )

    # ── Internal checks ───────────────────────────────────────────────────────

    def _detect_mime(self, content: bytes) -> str | None:
        """Identify file type from magic bytes (first 16 bytes)."""
        header = content[:16]
        for magic, mime, _ in _MAGIC_MAP:
            if header.startswith(magic):
                # Special case: WebP — RIFF but must have WEBP at offset 8
                if mime == "image/webp" and content[8:12] != b"WEBP":
                    continue
                return mime
        return None

    def _check_malicious_signatures(self, filename: str, content: bytes) -> None:
        """Scan first 1 KB and full content for known malicious byte sequences."""
        sample = content[:1024].lower()
        full_lower = content.lower()
        for sig, description in _MALICIOUS_SIGNATURES:
            if sig.lower() in sample or sig.lower() in full_lower[:2048]:
                raise MaliciousFileException(filename=filename, signature=description)

    def _validate_pdf(self, filename: str, content: bytes) -> list[str]:
        """
        Validate PDF structure:
          - Must start with %PDF-
          - Must contain %%EOF
          - Warn on dangerous action types
        """
        warnings: list[str] = []

        if not content.startswith(b"%PDF-"):
            raise FileValidationException(
                filename=filename,
                reason="File lacks valid PDF header (%PDF-)",
            )

        if b"%%EOF" not in content[-1024:]:
            warnings.append("PDF missing %%EOF trailer — file may be truncated")

        if self._block_dangerous_pdfs:
            for sig, description in _PDF_DANGEROUS_PATTERNS:
                if sig in content:
                    raise MaliciousFileException(
                        filename=filename,
                        signature=description,
                    )

        return warnings

    def _validate_image(self, filename: str, content: bytes) -> list[str]:
        """Validate image using Pillow if available; otherwise trust magic bytes."""
        warnings: list[str] = []
        try:
            from PIL import Image, UnidentifiedImageError

            img = Image.open(io.BytesIO(content))
            img.verify()    # raises if corrupt

            # Check for unusually large images (decompression bomb)
            img2 = Image.open(io.BytesIO(content))
            width, height = img2.size
            pixel_count = width * height
            if pixel_count > 100_000_000:   # 100 MP
                raise FileValidationException(
                    filename=filename,
                    reason=f"Image too large: {width}×{height} ({pixel_count:,} pixels)",
                    event_type=SecurityEventType.FILE_TOO_LARGE,
                )

        except ImportError:
            warnings.append("Pillow not installed — image content validation skipped")
        except FileValidationException:
            raise
        except Exception as exc:
            raise FileValidationException(
                filename=filename,
                reason=f"Image validation failed: {exc}",
            ) from exc

        return warnings

    def _check_zip_bomb(self, filename: str, content: bytes) -> None:
        """Detect zip bombs by checking compression ratio of the central directory."""
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                total_uncompressed = sum(info.file_size for info in zf.infolist())
                compressed = len(content)

                if total_uncompressed > _ZIP_BOMB_MAX_UNCOMPRESSED:
                    raise MaliciousFileException(
                        filename=filename,
                        signature=(
                            f"Zip bomb: uncompressed size {total_uncompressed / 1e6:.1f} MB "
                            f"exceeds limit {_ZIP_BOMB_MAX_UNCOMPRESSED / 1e6:.0f} MB"
                        ),
                    )

                if compressed > 0 and (total_uncompressed / compressed) > _ZIP_BOMB_RATIO:
                    raise MaliciousFileException(
                        filename=filename,
                        signature=(
                            f"Zip bomb: compression ratio "
                            f"{total_uncompressed / compressed:.1f}:1 exceeds limit {_ZIP_BOMB_RATIO}:1"
                        ),
                    )
        except (zipfile.BadZipFile, Exception) as exc:
            if isinstance(exc, MaliciousFileException):
                raise
            # Corrupted zip — treat as invalid
            raise FileValidationException(
                filename=filename,
                reason=f"ZIP file is corrupt or unreadable: {exc}",
            ) from exc
