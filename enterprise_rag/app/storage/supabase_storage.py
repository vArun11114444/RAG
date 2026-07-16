"""
app/storage/supabase_storage.py

Supabase Storage abstraction layer.

Replaces local filesystem PDF storage with Supabase Storage buckets.
The ingestion pipeline interface is unchanged — callers receive a public URL
instead of a local file path. All downstream logic (OCR, chunking, embedding)
continues to work by downloading the file from the URL.

Environment variables:
    SUPABASE_URL    = https://<project-ref>.supabase.co
    SUPABASE_KEY    = your-service-role-key
    SUPABASE_BUCKET = rag-documents   (bucket name in Supabase Storage)
"""
from __future__ import annotations

import asyncio
import mimetypes
import uuid
from pathlib import Path
from typing import BinaryIO

from app.config import get_settings
from app.observability.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class SupabaseStorageService:
    """
    Upload files to Supabase Storage and return public URLs.

    Usage:
        storage = SupabaseStorageService()
        url = await storage.upload(file_bytes, filename="report.pdf")
        # url = https://<project>.supabase.co/storage/v1/object/public/rag-documents/report.pdf
    """

    def __init__(self) -> None:
        self._client = None
        self._bucket = settings.SUPABASE_BUCKET

    def _get_client(self):
        """Lazy-initialise Supabase client."""
        if self._client is None:
            from supabase import create_client
            self._client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY,
            )
        return self._client

    async def upload(
        self,
        file_data: bytes,
        filename: str,
        content_type: str | None = None,
        folder: str = "uploads",
    ) -> str:
        """
        Upload file bytes to Supabase Storage.

        Args:
            file_data:    Raw file bytes.
            filename:     Original filename (used to preserve extension).
            content_type: MIME type. Auto-detected if not provided.
            folder:       Subfolder within the bucket (default: "uploads").

        Returns:
            Public URL string for the uploaded file.
        """
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set for cloud file storage."
            )

        # Generate a unique storage path to prevent collisions
        ext = Path(filename).suffix
        unique_name = f"{folder}/{uuid.uuid4().hex}{ext}"

        # Auto-detect MIME type if not provided
        if content_type is None:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"

        def _upload():
            client = self._get_client()
            response = client.storage.from_(self._bucket).upload(
                path=unique_name,
                file=file_data,
                file_options={"content-type": content_type, "upsert": "false"},
            )
            return response

        try:
            await asyncio.to_thread(_upload)
        except Exception as exc:
            log.error("supabase_upload_failed", extra={
                "filename": filename, "error": str(exc)
            })
            raise RuntimeError(f"File upload to Supabase failed: {exc}") from exc

        # Build public URL
        public_url = (
            f"{settings.SUPABASE_URL.rstrip('/')}"
            f"/storage/v1/object/public/{self._bucket}/{unique_name}"
        )

        log.info("supabase_upload_success", extra={
            "filename": filename,
            "path": unique_name,
            "url": public_url,
        })
        return public_url

    async def upload_from_path(self, local_path: str, folder: str = "uploads") -> str:
        """
        Upload a local file by path. Convenience wrapper around upload().
        Replaces the existing pattern of storing files to disk.
        """
        path = Path(local_path)
        file_data = await asyncio.to_thread(path.read_bytes)
        content_type, _ = mimetypes.guess_type(local_path)
        return await self.upload(
            file_data=file_data,
            filename=path.name,
            content_type=content_type,
            folder=folder,
        )

    async def delete(self, public_url: str) -> None:
        """Delete a file given its public URL."""
        # Extract path from URL
        marker = f"/object/public/{self._bucket}/"
        if marker not in public_url:
            return
        storage_path = public_url.split(marker, 1)[1]

        def _delete():
            self._get_client().storage.from_(self._bucket).remove([storage_path])

        try:
            await asyncio.to_thread(_delete)
            log.info("supabase_delete_success", extra={"path": storage_path})
        except Exception as exc:
            log.warning("supabase_delete_failed", extra={"error": str(exc)})

    async def get_signed_url(self, public_url: str, expires_in: int = 3600) -> str:
        """
        Return a time-limited signed URL for a private bucket.
        (If your bucket is public, you can use the public URL directly.)
        """
        marker = f"/object/public/{self._bucket}/"
        storage_path = public_url.split(marker, 1)[1] if marker in public_url else public_url

        def _sign():
            response = self._get_client().storage.from_(self._bucket).create_signed_url(
                path=storage_path,
                expires_in=expires_in,
            )
            return response.get("signedURL", public_url)

        return await asyncio.to_thread(_sign)
