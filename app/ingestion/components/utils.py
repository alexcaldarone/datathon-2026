import json
import sqlite3


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
