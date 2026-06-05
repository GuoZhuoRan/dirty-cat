"""Hume AI voice emotion detection via streaming WebSocket API."""

from __future__ import annotations

import os
import tempfile
from typing import Any

from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.expression_measurement.stream.stream.types import Config

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

EMOTION_TO_CAT_STATE: dict[str, str] = {
    "Tiredness": "idle", "Calmness": "idle",
    "Sadness": "care", "Anxiety": "care", "Stress": "care",
    "Anger": "care", "Fear": "care", "Distress": "care",
    "Joy": "approval", "Excitement": "approval", "Satisfaction": "approval",
    "Amusement": "approval", "Pride": "approval",
    "Contempt": "roast", "Boredom": "roast", "Disgust": "roast",
}

ENERGY_HIGH = {"Joy", "Excitement", "Amusement", "Enthusiasm"}
ENERGY_LOW  = {"Tiredness", "Sadness", "Anxiety", "Stress", "Boredom"}


async def analyze_voice_emotion(audio_bytes: bytes, mime_type: str = "audio/webm") -> dict[str, Any]:
    api_key = os.getenv("HUME_API_KEY", "")
    if not api_key:
        raise RuntimeError("HUME_API_KEY is not set")

    ext = "webm" if "webm" in mime_type else "wav"
    client = AsyncHumeClient(api_key=api_key)

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        async with client.expression_measurement.stream.connect() as socket:
            response = await socket.send_file(tmp_path, config=Config(prosody={}))
    finally:
        os.unlink(tmp_path)

    return _parse(response)


def _parse(response: Any) -> dict[str, Any]:
    scores: dict[str, float] = {}

    prosody = getattr(response, "prosody", None)
    if prosody is None:
        return {"top_emotions": [], "cat_state": "roast", "energy_level": "medium"}

    for prediction in getattr(prosody, "predictions", []) or []:
        for emotion in getattr(prediction, "emotions", []) or []:
            name = getattr(emotion, "name", None)
            score = getattr(emotion, "score", 0.0)
            if name:
                scores[name] = scores.get(name, 0.0) + score

    if not scores:
        return {"top_emotions": [], "cat_state": "roast", "energy_level": "medium"}

    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    cat_state = "roast"
    for name, _ in top3:
        if name in EMOTION_TO_CAT_STATE:
            cat_state = EMOTION_TO_CAT_STATE[name]
            break

    top_names = {n for n, _ in top3}
    energy = "high" if top_names & ENERGY_HIGH else "low" if top_names & ENERGY_LOW else "medium"

    return {
        "top_emotions": [{"name": n, "score": round(s, 3)} for n, s in top3],
        "cat_state": cat_state,
        "energy_level": energy,
    }
