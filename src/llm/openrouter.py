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
#
# CHANGED 2026-08-27: meta-llama/llama-3.3-70b-instruct:free was retired
# from OpenRouter's free tier (confirmed live via a real HTTP 404 from
# their own API: "This model is unavailable for free. The paid version is
# available now"). Free-tier model availability on OpenRouter is NOT
# permanent -- it changes as OpenRouter's own catalog changes, out of this
# project's control. If this model also 404s in the future, check the
# real current list at https://openrouter.ai/api/v1/models (filter for
# ids ending in ":free") rather than guessing a replacement.
FREE_MODEL = "google/gemma-4-31b-it:free"

# Marks an error message as a real HTTP 429 (rate limited) -- a normal,
# expected condition for a free-tier model under real load, not a genuine
# problem. Callers can check error.startswith(RATE_LIMIT_PREFIX) to decide
# whether to show it at all (see explain_transfer_suggestion_debug()).
RATE_LIMIT_PREFIX = "RATE_LIMITED: "


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
    content, _ = _chat_completion_with_error(system_prompt, user_prompt, timeout)
    return content


def _chat_completion_with_error(system_prompt: str, user_prompt: str, timeout: int = 20) -> tuple[str | None, str | None]:
    """Same real call as chat_completion(), but also returns a real,
    human-readable reason when it fails -- NEVER the API key itself, and
    never the raw response body (which could theoretically echo request
    data back). Used by explain_transfer_suggestion_debug() so a user can
    actually tell "no key set" apart from "OpenRouter rejected the key"
    apart from "the free tier is rate-limited right now" instead of a
    silent, undebuggable None -- chat_completion() itself stays the plain,
    error-swallowing version for any caller that genuinely doesn't need
    to know why."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None, "OPENROUTER_API_KEY is not set in this environment."

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
            "HTTP-Referer": "https://github.com/lucifer0096/FPL-Analytics",
            "X-Title": "FPL-Analytics",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        content = data["choices"][0]["message"]["content"].strip()
        return (content, None) if content else (None, "OpenRouter returned an empty response.")
    except urllib.error.HTTPError as e:
        try:
            body = json.load(e)
            reason = body.get("error", {}).get("message", str(e))
        except Exception:
            reason = str(e)
        # HTTP 429 (rate limited) is a real, EXPECTED condition for a free-
        # tier model under load, not a genuine problem worth alarming a
        # user over every time they click the button -- tagged with the
        # RATE_LIMIT_PREFIX so callers (see explain_transfer_suggestion_debug())
        # can choose to stay silent on this specific case while still
        # surfacing every other real error (bad key, retired model, etc.).
        prefix = RATE_LIMIT_PREFIX if e.code == 429 else ""
        return None, f"{prefix}OpenRouter returned HTTP {e.code}: {reason}"
    except urllib.error.URLError as e:
        return None, f"Could not reach OpenRouter: {e.reason}"
    except TimeoutError:
        return None, f"OpenRouter request timed out after {timeout}s."
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return None, f"OpenRouter returned an unexpected response shape: {e}"
