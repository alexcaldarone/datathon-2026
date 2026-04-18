import json
import sqlite3
from pathlib import Path

import requests


_REQUEST_TIMEOUT: int = 10


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


def fetch_hero_image(listing: dict) -> bytes | None:
    url = listing.get("hero_image_url")
    if not url:
        images_json = listing.get("images_json")
        url = _first_url_from_json(images_json)
    if not url:
        return None
    return _download_image(url)


def _download_image(url: str) -> bytes | None:
    if url.startswith("/"):
        return _read_local_image(url)
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"Image download failed for {url}: {exc}")
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
