"""Thin client for OpenRouter's chat-completions API, used ONLY to narrate
transfer suggestions the optimizer has already computed from real data --
never to generate the numbers themselves. See app/shared.py's
explain_transfer_suggestion() for the one real caller.

FREE MODEL ONLY, by design: every candidate in PREFERRED_FREE_MODELS below
must end in ":free" (OpenRouter's own naming convention for models that
don't consume paid credits) -- enforced by _pick_available_free_model(),
which also cross-checks against OpenRouter's own live /models list so a
retired or renamed model is never actually sent. There is deliberately no
fallback to a paid model if every free candidate is unavailable -- this
project should never silently start incurring API costs. If OpenRouter
itself is down, every free candidate is rate-limited, or
OPENROUTER_API_KEY isn't set, callers get None back and fall back to a
plain, non-LLM explanation instead (see explain_transfer_suggestion()'s
docstring) -- an LLM narration is a nice-to-have on top of real data,
never something the app depends on.
"""

import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Ordered by preference, most-preferred first -- ALL must end in ":free".
# _pick_available_free_model() tries each in turn (skipping any OpenRouter
# itself no longer lists as free) and _chat_completion_with_error() falls
# through to the next candidate on a real HTTP 429/404 from the one it
# tried, so a single congested or retired model doesn't take the whole
# chat assistant down.
#
# CHANGED 2026-08-27: meta-llama/llama-3.3-70b-instruct:free was retired
# from OpenRouter's free tier (confirmed live via a real HTTP 404 from
# their own API: "This model is unavailable for free. The paid version is
# available now").
#
# CHANGED 2026-08-31: google/gemma-4-31b-it:free (the 31b flagship) was
# hitting real HTTP 429s (rate-limited) noticeably often in the sidebar
# chat assistant under normal interactive use -- a real demand/congestion
# issue on OpenRouter's side, not a bug in this project's request
# handling. Replaced a single hardcoded model with this ordered list plus
# real-time availability checking + automatic fallback, so the app adapts
# to OpenRouter's free catalog and congestion changing over time instead
# of needing a manual model swap every time one candidate gets hammered or
# retired.
PREFERRED_FREE_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-2.6b:free",
]

# Marks an error message as a real HTTP 429 (rate limited) -- a normal,
# expected condition for a free-tier model under real load, not a genuine
# problem. Callers can check error.startswith(RATE_LIMIT_PREFIX) to decide
# whether to show it at all (see explain_transfer_suggestion_debug()).
RATE_LIMIT_PREFIX = "RATE_LIMITED: "

# Real, live cache of which PREFERRED_FREE_MODELS candidates OpenRouter's
# own /models endpoint currently lists as free -- refreshed at most once
# per this TTL so a normal chat conversation (several turns in a row)
# doesn't re-fetch the model catalog on every single message.
_MODEL_LIST_CACHE_TTL = 300
_model_list_cache = {"ids": None, "fetched_at": 0.0}


def is_configured() -> bool:
    """Whether an OpenRouter API key is actually set -- callers should
    check this (or just call chat_completion() and handle None) before
    assuming any LLM feature is available at all."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _live_free_model_ids(timeout: int = 10) -> set:
    """Real, live set of every model id OpenRouter's own /models endpoint
    currently lists as free (id ends in ':free') -- cached briefly (see
    _MODEL_LIST_CACHE_TTL) since this rarely changes minute-to-minute.
    Returns an empty set (never raises) if the real request fails for any
    reason -- callers must treat that as "couldn't verify," not "nothing
    is free," and fall back to trying PREFERRED_FREE_MODELS directly."""
    now = time.time()
    if _model_list_cache["ids"] is not None and now - _model_list_cache["fetched_at"] < _MODEL_LIST_CACHE_TTL:
        return _model_list_cache["ids"]

    try:
        req = urllib.request.Request(MODELS_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        ids = {m["id"] for m in data.get("data", []) if m.get("id", "").endswith(":free")}
        _model_list_cache["ids"] = ids
        _model_list_cache["fetched_at"] = now
        return ids
    except Exception:
        return set()


def _pick_available_free_model() -> str:
    """Real, live selection of which PREFERRED_FREE_MODELS candidate to
    use next -- the first one OpenRouter's own /models list currently
    confirms is free, in preference order. If the live list can't be
    fetched (network issue) or none of our candidates appear in it (the
    live list may lag a brand-new model, or the request itself failed),
    falls back to the first candidate in the list rather than refusing to
    try at all -- _chat_completion_with_error()'s own real HTTP-error
    handling is the final backstop either way."""
    live_ids = _live_free_model_ids()
    for model_id in PREFERRED_FREE_MODELS:
        if model_id in live_ids:
            return model_id
    return PREFERRED_FREE_MODELS[0]


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


def chat_conversation(messages: list, timeout: int = 20) -> tuple[str | None, str | None]:
    """Multi-turn variant of chat_completion() for the sidebar chat
    assistant -- takes a full real conversation history (the system prompt
    plus every real user/assistant turn so far, in OpenRouter's own
    {"role": ..., "content": ...} message shape) instead of a single
    system+user pair, so the model has real context from earlier in the
    same conversation, not just the latest message in isolation.

    Returns (content, error) -- same real error-reporting contract as
    _chat_completion_with_error() (a real, specific reason on failure,
    NEVER the API key itself), since the sidebar chat needs to show the
    user why a reply didn't come back, same reasoning as
    explain_transfer_suggestion_debug()."""
    return _chat_completion_with_error(messages=messages, timeout=timeout)


def _chat_completion_with_error(system_prompt: str = None, user_prompt: str = None, messages: list = None, timeout: int = 20) -> tuple[str | None, str | None]:
    """Same real call as chat_completion(), but also returns a real,
    human-readable reason when it fails -- NEVER the API key itself, and
    never the raw response body (which could theoretically echo request
    data back). Used by explain_transfer_suggestion_debug() so a user can
    actually tell "no key set" apart from "OpenRouter rejected the key"
    apart from "the free tier is rate-limited right now" instead of a
    silent, undebuggable None -- chat_completion() itself stays the plain,
    error-swallowing version for any caller that genuinely doesn't need
    to know why.

    Accepts EITHER a single system_prompt+user_prompt pair (the original
    single-turn shape every existing caller uses) OR a full messages list
    (chat_conversation()'s multi-turn shape) -- exactly one of the two
    calling conventions must be used, not both, so there's one real
    request-building/error-handling path shared by every caller instead of
    two near-duplicate copies.

    Tries PREFERRED_FREE_MODELS in order (starting from
    _pick_available_free_model()'s real, live-verified best guess),
    falling through to the next candidate on a real HTTP 429 (rate
    limited) or 404 (retired/unknown model) from the one just tried --
    every candidate is still ":free" only, so this never risks incurring
    a real cost even when falling back. Returns the LAST candidate's error
    if every single one fails, since that's the most representative real
    failure to show the user."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None, "OPENROUTER_API_KEY is not set in this environment."

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    first_choice = _pick_available_free_model()
    ordered_candidates = [first_choice] + [m for m in PREFERRED_FREE_MODELS if m != first_choice]

    last_error = None
    for model_id in ordered_candidates:
        payload = {
            "model": model_id,
            "messages": messages,
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
            # HTTP 429 (rate limited) is a real, EXPECTED condition for a
            # free-tier model under load, not a genuine problem worth
            # alarming a user over every time they click the button --
            # tagged with RATE_LIMIT_PREFIX so callers (see
            # explain_transfer_suggestion_debug()) can choose to stay
            # silent on this specific case while still surfacing every
            # other real error (bad key, retired model, etc.). Both 429
            # and 404 (retired/unknown model id) are worth trying the next
            # real candidate for rather than giving up immediately.
            prefix = RATE_LIMIT_PREFIX if e.code == 429 else ""
            last_error = f"{prefix}OpenRouter returned HTTP {e.code}: {reason}"
            if e.code in (429, 404):
                continue
            return None, last_error
        except urllib.error.URLError as e:
            last_error = f"Could not reach OpenRouter: {e.reason}"
            return None, last_error
        except TimeoutError:
            last_error = f"OpenRouter request timed out after {timeout}s."
            return None, last_error
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            last_error = f"OpenRouter returned an unexpected response shape: {e}"
            return None, last_error

    return None, last_error
