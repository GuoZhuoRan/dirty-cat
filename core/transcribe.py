"""Groq Whisper server-side transcription."""

from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


async def transcribe(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return ""

    ext = "webm" if "webm" in mime_type else "wav"
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            result = await client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(f"audio.{ext}", f, mime_type),
                response_format="text",
            )
        return result.strip() if isinstance(result, str) else result.text.strip()
    finally:
        os.unlink(tmp_path)
