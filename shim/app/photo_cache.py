"""Persist Rokid-served camera frames AND inline them as base64 for the model.

Rokid uploads camera frames to its own CDN and hands us short-lived URLs.
Two problems we solve:

  1. The Rokid CDN URL expires quickly and may not be reachable from
     cloud LLM backends (OpenRouter, etc.) — so we always inline the
     bytes as a `data:image/...;base64,...` URL in the request the model
     sees. Works with every OpenAI-compat backend, no network hop needed.
  2. Downstream tools (gmail attach, knowledge ingest, vision recall in a
     later turn) need the photo after the SSE turn ends — so we also keep
     a copy on disk at `/app/data/photos/<sha>.<ext>` and expose it via
     `GET /photos/{name}`. Tools running in-cluster can fetch from there.

In short: base64 for the immediate model call, disk for everything after.
"""

import asyncio
import base64
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
# When true (default), the rewritten image_url is a `data:` URL with the bytes
# inline — guarantees the model can read it regardless of network topology.
# Set to false to fall back to the public-URL mode (only works if the model
# backend can reach PHOTOS_PUBLIC_URL).
PHOTO_INLINE_BASE64 = os.environ.get("PHOTO_INLINE_BASE64", "true").lower() == "true"

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


_EXT_TO_MIME = {v: k for k, v in _MIME_TO_EXT.items()}


def _public_url_for(name: str) -> str:
    return f"{PHOTOS_PUBLIC_URL}/photos/{name}"


def _data_url_from_path(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    mime = _EXT_TO_MIME.get(ext, "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def _cache_one(url: str) -> str | None:
    """Cache an image to disk and return the URL the model should fetch.

    The returned URL is either a `data:...;base64,...` (default — works for
    any model backend) or a public `/photos/<name>` URL (when
    PHOTO_INLINE_BASE64 is false).
    """
    sha = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]

    # Reuse if already cached
    for existing in PHOTO_DIR.glob(f"{sha}.*"):
        if PHOTO_INLINE_BASE64:
            return _data_url_from_path(existing)
        return _public_url_for(existing.name)

    # data: URLs we received directly — just hand them back, nothing to cache
    if url.startswith("data:"):
        return url

    try:
        # Many CDNs (Wikimedia, some commercial ones) reject the default httpx UA.
        # A boring browser UA gets through everywhere without raising flags.
        headers = {"User-Agent": "Mozilla/5.0 (compatible; rokid-shim/1.0)"}
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as c:
            resp = await c.get(url)
        resp.raise_for_status()
        if len(resp.content) > PHOTO_MAX_BYTES:
            logger.warning("photo %s exceeds max bytes (%d), skipped", sha, len(resp.content))
            return None
        ext = _ext_from_url_or_mime(url, resp.headers.get("content-type"))
        path = PHOTO_DIR / f"{sha}.{ext}"
        path.write_bytes(resp.content)
        if PHOTO_INLINE_BASE64:
            return _data_url_from_path(path)
        return _public_url_for(path.name)
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
