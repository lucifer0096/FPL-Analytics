"""Thin client for OpenRouter's chat-completions API, used ONLY to narrate
transfer suggestions the optimizer has already computed from real data --
never to generate the numbers themselves. See app/shared.py's
explain_transfer_suggestion() for the one real caller.

FREE MODEL ONLY, by design: FREE_MODEL below is hardcoded to a model id
ending in ":free" (OpenRouter's own naming convention for models that
don't consume paid credits). There is deliberately no fallback to a paid
model if the free one is unavailable -- this project should never
silently start incurring API costs. If OpenRouter itself is down, rate-
limits the free tier, or OPENROUTER_API_KEY isn't set, callers get None
back and fall back to a plain, non-LLM explanation instead (see
explain_transfer_suggestion()'s docstring) -- an LLM narration is a nice-
to-have on top of real data, never something the app depends on.
"""

import json
import os
import urllib.error
import urllib.request

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Hardcoded to a genuinely free (no-credits-required) OpenRouter model.
# Never change this to a model without the ":free" suffix without also
# changing the "free model only" guarantee described above.
FREE_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def is_configured() -> bool:
    """Whether an OpenRouter API key is actually set -- callers should
    check this (or just call chat_completion() and handle None) before
    assuming any LLM feature is available at all."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def chat_completion(system_prompt: str, user_prompt: str, timeout: int = 20) -> str | None:
    """One free-model chat completion. Returns the model's real text reply,
    or None if OPENROUTER_API_KEY isn't set, the request fails for any
    reason (network, timeout, non-200, malformed response), or the model
    returns empty content -- callers must treat None as "narration
    unavailable right now," not an error to surface to the user, since
    every real number on this page already came from FPL's own API before
    this is ever called."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": FREE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter asks free-tier callers to identify their app --
            # not required for the API to function, but good citizenship.
            "HTTP-Referer": "https://github.com/lucifer0096/FPL-Analytics",
            "X-Title": "FPL-Analytics",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"].strip() or None
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        return None
