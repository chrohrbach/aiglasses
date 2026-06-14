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


def _archive_url_for_sha(sha: str) -> str | None:
    """Stable public /photos/<name> URL for a cached photo, if a disk copy exists."""
    for existing in PHOTO_DIR.glob(f"{sha}.*"):
        return _public_url_for(existing.name)
    return None


def _data_url_from_path(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    mime = _EXT_TO_MIME.get(ext, "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


_in_memory_cache: dict[str, str] = {}


def _write_to_disk_bg(path: Path, content: bytes) -> None:
    try:
        path.write_bytes(content)
        logger.info("Saved image to disk in background: %s", path)
    except Exception as e:
        logger.warning("Failed to write image to disk in background: %s", e)


async def _cache_one(url: str) -> tuple[str | None, str | None]:
    """Cache an image and return ``(model_url, archive_url)``.

    ``model_url`` is what the model should fetch — either a
    ``data:...;base64,...`` URL (default, works for any backend) or a public
    ``/photos/<name>`` URL (when PHOTO_INLINE_BASE64 is false). It is ``None``
    when caching fails.

    ``archive_url`` is the stable public ``/photos/<name>`` URL backed by the
    durable disk copy, suitable for handing to ``attach_asset`` (mcp-memory
    fetches it server-side). It is ``None`` when there is no disk copy (e.g. a
    ``data:`` URL supplied directly by the caller).
    """
    sha = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]

    # 1. Check in-memory cache first (super fast, no disk I/O)
    if sha in _in_memory_cache:
        cached_url = _in_memory_cache[sha]
        if PHOTO_INLINE_BASE64 and cached_url.startswith("data:"):
            return cached_url, _archive_url_for_sha(sha)
        elif not PHOTO_INLINE_BASE64 and not cached_url.startswith("data:"):
            return cached_url, _archive_url_for_sha(sha)

    # 2. Check if already on disk
    for existing in PHOTO_DIR.glob(f"{sha}.*"):
        archive_url = _public_url_for(existing.name)
        if PHOTO_INLINE_BASE64:
            data_url = _data_url_from_path(existing)
            _in_memory_cache[sha] = data_url
            return data_url, archive_url
        _in_memory_cache[sha] = archive_url
        return archive_url, archive_url

    # data: URLs we received directly — just hand them back, nothing to cache
    # and no stable archive URL (we never wrote it to disk).
    if url.startswith("data:"):
        return url, None

    try:
        # Many CDNs (Wikimedia, some commercial ones) reject the default httpx UA.
        # A boring browser UA gets through everywhere without raising flags.
        headers = {"User-Agent": "Mozilla/5.0 (compatible; rokid-shim/1.0)"}
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as c:
            resp = await c.get(url)
        resp.raise_for_status()
        if len(resp.content) > PHOTO_MAX_BYTES:
            logger.warning("photo %s exceeds max bytes (%d), skipped", sha, len(resp.content))
            return None, None
        ext = _ext_from_url_or_mime(url, resp.headers.get("content-type"))
        path = PHOTO_DIR / f"{sha}.{ext}"

        # Write the disk copy synchronously (in a thread) BEFORE advertising the
        # archive URL, so mcp-memory can fetch it immediately when the model
        # calls attach_asset within the same turn.
        await asyncio.to_thread(_write_to_disk_bg, path, resp.content)
        archive_url = _public_url_for(path.name)

        if PHOTO_INLINE_BASE64:
            mime = _EXT_TO_MIME.get(ext.lower(), "application/octet-stream")
            b64 = base64.b64encode(resp.content).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"
            _in_memory_cache[sha] = data_url
            return data_url, archive_url
        _in_memory_cache[sha] = archive_url
        return archive_url, archive_url
    except Exception as e:
        logger.warning("failed to cache image %s: %s", url, e)
        return None, None


async def cache_request_images(req: RokidRequest) -> list[str]:
    """Walk the request, replacing image_url fields with cached local URLs.

    Returns the list of stable public archive URLs
    (``${PHOTOS_PUBLIC_URL}/photos/<sha>.<ext>``) for the images that were
    persisted to disk — callers may advertise these to the model so it can pass
    them verbatim to ``attach_asset``. Callers that ignore the return value keep
    working unchanged; the in-request ``image_url`` rewrite is unaffected.
    """
    image_items = [m for m in req.message if m.type == "image" and m.image_url]
    if not image_items:
        return []
    results = await asyncio.gather(*(_cache_one(item.image_url) for item in image_items))
    archive_urls: list[str] = []
    for item, (model_url, archive_url) in zip(image_items, results, strict=True):
        if model_url:
            item.image_url = model_url
        if archive_url:
            archive_urls.append(archive_url)
    return archive_urls


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
    if len(name) > 80:
        return None
    try:
        resolved_dir = PHOTO_DIR.resolve()
        p = (PHOTO_DIR / name).resolve()
        if not p.is_file():
            return None
        # Ensure the resolved file path is inside the resolved directory
        # Using commonpath is extremely robust on both Windows and Linux
        if os.path.commonpath([resolved_dir, p]) != str(resolved_dir):
            return None
        return p
    except Exception:
        return None
