import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image


def row_to_full_text(row: sqlite3.Row) -> str:
    parts = [row["title"], row["description"], row["city"], row["canton"], row["object_category"]]
    return " ".join(p for p in parts if p)


def parse_features(features_json: str | None) -> list[str]:
    if not features_json:
        return []
    try:
        return json.loads(features_json)
    except (json.JSONDecodeError, TypeError):
        return []


def fetch_hero_image(listing: dict, request_timeout_s: int) -> bytes | None:
    url = listing.get("hero_image_url")
    if not url:
        images_json = listing.get("images_json")
        url = _first_url_from_json(images_json)
    if not url:
        return None
    return _download_image(url, request_timeout_s)


def _download_image(url: str, request_timeout_s: int) -> bytes | None:
    if url.startswith("/"):
        return _read_local_image(url)
    try:
        resp = requests.get(url, timeout=request_timeout_s)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        from app.ingestion.components.logger import IngestionLogger
        IngestionLogger.get().warning("Image download failed for %s: %s", url, exc)
        return None


def _read_local_image(path: str) -> bytes | None:
    local = Path(__file__).parents[3] / "raw_data" / "images" / Path(path).name
    if local.exists():
        return local.read_bytes()
    return None


def _first_url_from_json(images_json: str | None) -> str | None:
    if not images_json:
        return None
    try:
        parsed = json.loads(images_json)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        for item in parsed.get("images", []) or []:
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"])
            if isinstance(item, str) and item:
                return item
    if isinstance(parsed, list) and parsed:
        return str(parsed[0]) if parsed[0] else None
    return None

def resize_and_crop_image(image_bytes: bytes, width: int, height: int) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = max(width / img.width, height / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    left = (img.width - width) // 2
    top = (img.height - height) // 2
    img = img.crop((left, top, left + width, top + height))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def download_images_batch(
    listings: list[dict],
    max_workers: int = 8,
    request_timeout_s: int = 30,
    target_width: int | None = None,
    target_height: int | None = None,
) -> None:
    def _download_one(listing: dict) -> None:
        image_bytes = fetch_hero_image(listing, request_timeout_s)
        if image_bytes and target_width and target_height:
            try:
                image_bytes = resize_and_crop_image(image_bytes, target_width, target_height)
            except Exception:
                pass
        listing["_image_bytes"] = image_bytes

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_download_one, listings))


def parse_images_json(images_json: str):
    if not images_json:
        return []
    try:
        images = json.loads(images_json)
        return [i["url"] for i in images["images"]]
    except (json.JSONDecodeError, TypeError):
        return []
