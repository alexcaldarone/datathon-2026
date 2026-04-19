from __future__ import annotations

import json
import os
import re
import time

import boto3

_NON_ENGLISH_MARKERS = {
    "wohnung", "zimmer", "miete", "strasse", "nähe", "küche", "schlafzimmer",
    "appartement", "chambre", "loyer", "cuisine", "près", "étage",
    "appartamento", "camera", "affitto", "cucina", "vicino", "piano",
    "zürich", "bern", "luzern", "genève", "lausanne",
}


def is_english(text: str) -> bool:
    tokens = set(re.findall(r"[a-zäöüéèàâêîôû]+", text.lower()))
    hits = tokens & _NON_ENGLISH_MARKERS
    return len(hits) < 3


def translate_to_english(text: str, model_id: str, retries: int = 3) -> str:
    client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_DEFAULT_REGION"])
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Translate the following real estate search query to English. "
                    "Preserve all factual details. Return only the translation.\n\n"
                    f"{text}"
                ),
            }
        ],
    })
    for attempt in range(retries):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            raw = json.loads(resp["body"].read())
            return raw["content"][0]["text"].strip()
        except Exception as exc:
            if attempt == retries - 1:
                return text
            time.sleep(2 ** attempt)
    return text
