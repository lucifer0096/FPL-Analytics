"""Thin client for a LOCAL Ollama server, used as the preferred backend for
the sidebar chat assistant when Ollama is genuinely running on this same
machine (see app/shared.py's chat_with_assistant()).

Why this exists alongside openrouter.py rather than replacing it: Ollama
runs locally, with real hardware serving every request, so it has NO rate
limits and NO cost -- strictly better than a free OpenRouter model whenever
it's actually available. But it's only reachable from wherever Ollama
itself is running: when this app is deployed on Streamlit Cloud (a
separate machine with no Ollama installed), this client's real health
check below correctly reports "not available" and callers fall back to
openrouter.py's free-model chain instead. Local dev, cloud deploy -- one
codebase, the right backend for each, chosen automatically at runtime
rather than a manual toggle.

MODEL is a specific real tag confirmed installed via `ollama list` on this
machine (gemma3:4b -- the smallest/fastest of the available local text
models, a good fit for a quick chat assistant). If this model is ever
removed locally, is_available() still returns True (the Ollama server
itself is reachable) but the actual chat call will fail with a real,
specific "model not found" error -- surfaced to the caller like any other
failure, never silently swallowed."""

import json
import urllib.error
import urllib.request

BASE_URL = "http://localhost:11434"
MODEL = "gemma3:4b"

# Kept short -- this is a liveness probe run on every chat turn (to decide
# Ollama vs. OpenRouter), not a real generation call, so it should fail
# fast if Ollama genuinely isn't running here rather than hanging the UI.
_HEALTH_CHECK_TIMEOUT = 2


def is_available() -> bool:
    """Whether a local Ollama server is actually reachable right now on
    this machine -- a real, live TCP+HTTP check (not a config flag),
    since the same codebase runs both locally (where Ollama is installed)
    and on Streamlit Cloud (where it never will be). Never raises --
    any failure (connection refused, DNS, timeout) means "not available."
    """
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/version")
        with urllib.request.urlopen(req, timeout=_HEALTH_CHECK_TIMEOUT):
            return True
    except Exception:
        return False


def chat_conversation(messages: list, timeout: int = 30) -> tuple[str | None, str | None]:
    """One real chat turn against the local Ollama server, in the same
    (content, error) shape as openrouter.chat_conversation() so
    app/shared.py's chat_with_assistant() can try this first and fall
    back to OpenRouter with zero shape differences to handle.

    A longer default timeout than the OpenRouter client's (30s vs 20s):
    local generation speed depends entirely on this machine's own
    hardware (CPU vs GPU, model size) rather than a managed cloud
    endpoint, so it can legitimately take longer, especially on a first
    call before the model is warmed into memory."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        content = data.get("message", {}).get("content", "").strip()
        return (content, None) if content else (None, "Ollama returned an empty response.")
    except urllib.error.HTTPError as e:
        try:
            body = json.load(e)
            reason = body.get("error", str(e))
        except Exception:
            reason = str(e)
        return None, f"Ollama returned HTTP {e.code}: {reason}"
    except urllib.error.URLError as e:
        return None, f"Could not reach local Ollama server: {e.reason}"
    except TimeoutError:
        return None, f"Ollama request timed out after {timeout}s."
    except (KeyError, json.JSONDecodeError) as e:
        return None, f"Ollama returned an unexpected response shape: {e}"
