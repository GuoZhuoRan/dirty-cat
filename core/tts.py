"""Cartesia Sonic TTS for Dirty Cat."""

from __future__ import annotations

import asyncio
import logging
import os
import struct

import httpx

logger = logging.getLogger("dirtycat.tts")

_client: httpx.AsyncClient | None = None
_client_loop: object | None = None

CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_VERSION = "2026-03-01"
CARTESIA_SAMPLE_RATE = 24000
DEFAULT_CARTESIA_TTS_MODEL = "sonic-3.5-2026-05-04"
DEFAULT_CARTESIA_VOICE = "a5136bf9-224c-4d76-b823-52bd5efcffcc"


def pcm_to_wav(pcm: bytes, sample_rate: int = CARTESIA_SAMPLE_RATE) -> bytes:
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ",
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16, b"data", data_size,
    )
    return header + pcm


async def cartesia_tts(text: str) -> bytes:
    api_key = os.getenv("CARTESIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("CARTESIA_API_KEY is not set")

    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(timeout=30, trust_env=False)
        _client_loop = loop

    resp = await _client.post(
        f"{CARTESIA_BASE_URL}/tts/bytes",
        headers={
            "X-API-Key": api_key,
            "Cartesia-Version": CARTESIA_VERSION,
            "Content-Type": "application/json",
        },
        json={
            "model_id": DEFAULT_CARTESIA_TTS_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": DEFAULT_CARTESIA_VOICE},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": CARTESIA_SAMPLE_RATE,
            },
        },
    )
    if resp.status_code != 200:
        logger.error("Cartesia TTS HTTP %s: %s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"Cartesia TTS failed (HTTP {resp.status_code})")
    return resp.content
