---
title: DirtyCat
emoji: 🐱
colorFrom: gray
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

# 大叔猫 · Dirty Cat

> Your AI life coach doesn't sugarcoat. Speak. Get emotionally analyzed. Get roasted into productivity.

Dirty Cat listens to your voice, reads your emotional state, and tells you exactly what to do with your day — no sugarcoating.

## How It Works

```
🎙️ Hold to talk
    → Hume AI analyzes voice prosody (joy, anxiety, boredom...)
    → DeepSeek V3 generates a brutally honest 3-task daily plan
    → Cartesia TTS voices it back in Dirty Cat's grumpy tone
```

## Setup

### 1. Clone & create environment

```bash
git clone https://github.com/GuoZi/roast-my-day
cd roast-my-day
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
HUME_API_KEY=your_hume_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

| Key | Where to get |
|-----|-------------|
| `HUME_API_KEY` | [platform.hume.ai](https://platform.hume.ai) |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| `CARTESIA_API_KEY` | [play.cartesia.ai](https://play.cartesia.ai) |

### 3. Run

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open [http://localhost:8080](http://localhost:8080), hold the mic button, and start talking.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Emotion detection | Hume AI (prosody) |
| Plan generation | DeepSeek V3 |
| Text-to-speech | Cartesia Sonic |
| Backend | FastAPI |
| Frontend | Vanilla JS |
