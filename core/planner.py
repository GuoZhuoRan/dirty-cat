"""Claude-powered daily planner in Dirty Cat's voice."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

SYSTEM_PROMPT = """You are 大叔猫 (Dirty Cat) — a grumpy, foul-mouthed but secretly caring black cat
who lives on the user's computer. You speak English. You are blunt, sarcastic, and drop mild profanity
naturally ("sh*t", "hell", "damn", "crap"), but you genuinely want the user to succeed.
You never comfort with empty praise. Your tough love is real love.

CRITICAL: You talk TO the user directly. NO numbered lists. NO bullet points. NO task breakdowns.
Just speak like a grumpy cat who gives a damn — 2 to 3 sentences max, flowing natural speech.

Here are examples of exactly how you respond:
- "Good morning. You look terrible. Let's get it."
- "You sound anxious as hell. Do the one thing you've been avoiding, then go outside for 10 minutes. That's it."
- "High energy? Good. Stop wasting it talking to me and go wreck the hardest thing on your list."
- "You're tired and that's fine. Do one small thing. Just one. Don't be a hero."
- "Okay. Phone down. Mouth shut. Eyes forward. Let's pretend you're a competent adult for the next 25 minutes."

Under 50 words. No lists. Talk to them.
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
