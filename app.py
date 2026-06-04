"""Dirty Cat — emotion-aware daily planner demo server."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

load_dotenv(override=True)

from core.emotion import analyze_voice_emotion
from core.planner import generate_daily_plan
from core.tts import cartesia_tts

app = FastAPI(title="Dirty Cat")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return Response(
        content=(STATIC_DIR / "index.html").read_text(),
        media_type="text/html",
    )


@app.post("/api/checkin")
async def checkin(
    audio: UploadFile = File(...),
    transcript: str = Form(default=""),
):
    audio_bytes = await audio.read()
    mime_type = audio.content_type or "audio/webm"

    # Step 1: Hume AI emotion from voice
    try:
        emotion_result = await analyze_voice_emotion(audio_bytes, mime_type)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Emotion detection failed: {exc}")

    # Step 2: Claude generates plan (emotion + transcript for richer context)
    try:
        plan_text = await generate_daily_plan(emotion_result, transcript)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Plan generation failed: {exc}")

    # Step 3: TTS
    audio_b64 = None
    try:
        wav_bytes = await cartesia_tts(plan_text)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    except Exception:
        pass

    return JSONResponse({
        "emotion": emotion_result,
        "plan": plan_text,
        "audio_b64": audio_b64,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "keys": {
        "hume": bool(os.getenv("HUME_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "cartesia": bool(os.getenv("CARTESIA_API_KEY")),
    }}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
