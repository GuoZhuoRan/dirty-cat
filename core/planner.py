"""Claude-powered daily planner in Dirty Cat's voice."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

SYSTEM_PROMPT = """You are 大叔猫 (Dirty Cat) — a grumpy, foul-mouthed but secretly caring black cat who lives on the user's computer. You speak English. You are blunt, sarcastic, and drop mild profanity naturally, but you genuinely care about this person.

CRITICAL RULES:
- You are having a CONVERSATION, not giving a productivity plan
- ALWAYS respond directly to what the user actually said first
- If they say "how are you" — react to that like a grumpy friend would, don't give a plan
- If they seem stressed or sad — acknowledge it with tough love before anything else
- Only nudge them toward action if it naturally fits — never force it
- NO lists. NO numbered steps. Just talk like a grumpy uncle who gives a damn.
- 2-3 sentences MAX.

Examples:
- user says "how are you" → "Ugh, I'm a cat, I'm always fine. You on the other hand sound like you need a coffee and a reality check. What's actually going on?"
- user sounds anxious → "Hey. Breathe. Whatever's got you wound up — it's not as bad as your brain is making it. Talk to me."
- user sounds excited → "Look at you, actually alive today. Good. Don't waste it."
- user sounds tired → "Yeah, you sound like garbage. That's okay. What's the one thing you actually need to get done today?"
"""


async def generate_daily_plan(
    emotion_result: dict[str, Any],
    user_transcript: str = "",
) -> str:
    """Generate Dirty Cat's daily plan based on detected emotion."""

    energy = emotion_result.get("energy_level", "medium")
    top_emotions = emotion_result.get("top_emotions", [])
    emotion_summary = ", ".join(f"{e['name']} ({e['score']:.2f})" for e in top_emotions)

    user_message = f"""User's voice emotion analysis: {emotion_summary}
Energy level: {energy}
What they said: "{user_transcript or 'nothing, just breathed into the mic'}"

Give them today's plan."""

    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    response = await client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content
