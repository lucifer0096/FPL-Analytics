"""Thin client for a LOCAL Ollama server -- the ONLY backend for the chat
assistant and transfer narration (see app/shared.py's chat_with_assistant()
and explain_transfer_suggestion()). Ollama runs locally, with real
hardware serving every request, so it has NO rate limits and NO cost --
but it's only reachable from wherever Ollama itself is running: when this
app is deployed on Streamlit Cloud (a separate machine with no Ollama
installed), this client's real health check below correctly reports "not
available" and both features degrade gracefully (a plain message, not a
crash) rather than working. This is an accepted, explicit trade-off, not
an oversight -- both features are genuinely local-only.

MODEL is a specific real tag confirmed installed via `ollama list` on this
machine (gemma3:4b -- the smallest/fastest of the available local text
models). If this model is ever removed locally, is_available() still
returns True (the Ollama server itself is reachable) but the actual chat
call will fail with a real, specific "model not found" error -- surfaced
to the caller like any other failure, never silently swallowed.

Real, measured CPU inference speed on this machine: a short test message
returns in ~2s, but a REAL chat turn with the full assemble_chat_context()
system prompt (squad, injuries, differentials, PL table -- ~2000 real
characters) took ~83s end-to-end (confirmed directly). CHAT_TIMEOUT below
is set generously above that measured real-world worst case, not an
arbitrary guess -- CPU-only local inference genuinely is this much slower
than a managed cloud endpoint once the prompt is realistically sized."""

import json
import urllib.error
import urllib.request

BASE_URL = "http://localhost:11434"
MODEL = "gemma3:4b"

# Kept short -- this is a liveness probe run before every chat turn (to
# decide whether to even attempt a real call), not a real generation
# call, so it should fail fast if Ollama genuinely isn't running here
# rather than hanging the UI.
_HEALTH_CHECK_TIMEOUT = 2

# Default timeout for a real chat_conversation() call -- see this module's
# own docstring for the real, measured ~83s worst case with a full chat
# system prompt on this machine's CPU-only inference. 150s leaves real
# headroom above that measured number rather than cutting it close.
CHAT_TIMEOUT = 150


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


def chat_conversation(messages: list, timeout: int = CHAT_TIMEOUT) -> tuple[str | None, str | None]:
    """One real chat turn against the local Ollama server, returning
    (content, error) -- content is the real reply text, or None with a
    real, specific error string on any failure.

    Default timeout is CHAT_TIMEOUT (see module docstring for the real,
    measured worst case on this machine's CPU-only inference): local
    generation speed depends entirely on this machine's own hardware (CPU
    vs GPU, model size, prompt length) rather than a managed cloud
    endpoint, so a realistic full-context chat turn can legitimately take
    well over a minute, especially on a first call before the model is
    warmed into memory."""
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
