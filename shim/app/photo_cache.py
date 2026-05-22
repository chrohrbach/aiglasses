"""Persist Rokid-served camera frames to a local volume.

Rokid uploads camera frames to its own CDN and hands us short-lived URLs in
the `image_url` field. Two problems with that:

  1. The model fetches the image but we lose all trace afterwards.
  2. Downstream tools (gmail attachment, knowledge ingest, vision recall in a
     later turn) can't reach the Rokid URL — either it expires or it's
     auth-gated.

This module rewrites image URLs on the way in: each incoming Rokid URL is
downloaded once into `/app/data/photos/<sha>.<ext>` and the in-memory
request is mutated so the model sees `<PHOTOS_PUBLIC_URL>/photos/<sha>.<ext>`
instead. The local files are served by FastAPI under `/photos/{name}`.
"""

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path

import httpx

from .rokid_types import RokidRequest

logger = logging.getLogger(__name__)

PHOTO_DIR = Path(os.environ.get("PHOTO_CACHE_DIR", "/app/data/photos"))
PHOTOS_PUBLIC_URL = os.environ.get("PHOTOS_PUBLIC_URL", "http://rokid-shim:8000").rstrip("/")
PHOTO_RETENTION_HOURS = int(os.environ.get("PHOTO_RETENTION_HOURS", "48"))
PHOTO_MAX_BYTES = int(os.environ.get("PHOTO_MAX_BYTES", str(10 * 1024 * 1024)))

PHOTO_DIR.mkdir(parents=True, exist_ok=True)

_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/heic": "heic",
    "image/heif": "heif",
}


def _ext_from_url_or_mime(url: str, content_type: str | None) -> str:
    if content_type:
        ext = _MIME_TO_EXT.get(content_type.split(";")[0].strip().lower())
        if ext:
            return ext
    # Fall back to URL suffix
    for cand in ("jpg", "jpeg", "png", "gif", "webp", "heic", "heif"):
        if f".{cand}" in url.lower():
            return "jpg" if cand == "jpeg" else cand
    return "bin"


async def _cache_one(url: str) -> str | None:
    """Download a single image URL, return the new public URL (or None on fail)."""
    sha = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    # Reuse if already cached
    for existing in PHOTO_DIR.glob(f"{sha}.*"):
        return f"{PHOTOS_PUBLIC_URL}/photos/{existing.name}"

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as c:
            resp = await c.get(url)
        resp.raise_for_status()
        if len(resp.content) > PHOTO_MAX_BYTES:
            logger.warning("photo %s exceeds max bytes (%d), skipped", sha, len(resp.content))
            return None
        ext = _ext_from_url_or_mime(url, resp.headers.get("content-type"))
        path = PHOTO_DIR / f"{sha}.{ext}"
        path.write_bytes(resp.content)
        return f"{PHOTOS_PUBLIC_URL}/photos/{path.name}"
    except Exception as e:
        logger.warning("failed to cache image %s: %s", url, e)
        return None


async def cache_request_images(req: RokidRequest) -> None:
    """Walk the request, replacing image_url fields with cached local URLs."""
    image_items = [m for m in req.message if m.type == "image" and m.image_url]
    if not image_items:
        return
    results = await asyncio.gather(*(_cache_one(item.image_url) for item in image_items))
    for item, new_url in zip(image_items, results, strict=True):
        if new_url:
            item.image_url = new_url


def cleanup_old(now: float | None = None) -> int:
    """Delete photos older than PHOTO_RETENTION_HOURS. Returns count removed."""
    cutoff = (now or time.time()) - PHOTO_RETENTION_HOURS * 3600
    removed = 0
    for p in PHOTO_DIR.glob("*"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def get_path(name: str) -> Path | None:
    """Resolve a /photos/{name} lookup safely (no path traversal)."""
    if "/" in name or "\\" in name or name.startswith(".") or len(name) > 80:
        return None
    p = PHOTO_DIR / name
    if not p.is_file():
        return None
    return p
