"""On-disk cache of finished reports, keyed by resume + JD content.

Asking about the same resume and JD again returns the saved report with zero
Gemini requests, and identical inputs always give identical reports.

Reports contain resume details, so the cache lives in the git-ignored
resume_jd_agent/.cache/ folder. Set RESUME_JD_CACHE=off to disable it, and
delete the folder to clear it.
"""

import hashlib
import json
import os
import time

# Bump when the prompt or report format changes, so old reports are not reused.
CACHE_VERSION = "3"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "reports")
CACHE_ENV = "RESUME_JD_CACHE"
MAX_AGE_SECONDS = 30 * 24 * 3600


def enabled() -> bool:
    return os.getenv(CACHE_ENV, "on").lower() not in ("off", "0", "false", "no")


def key_for(resume_text: str, jd_text: str, model: str = "") -> str:
    """Stable key for one resume + JD pair (whitespace-insensitive)."""
    normalise = lambda text: " ".join(text.split())  # noqa: E731
    payload = json.dumps([CACHE_VERSION, model, normalise(resume_text), normalise(jd_text)])
    return hashlib.sha256(payload.encode()).hexdigest()


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def get(key: str):
    """Returns the cached report, or None if missing, expired or disabled."""
    if not enabled() or not key:
        return None
    try:
        with open(_path(key), encoding="utf-8") as f:
            entry = json.load(f)
    except (OSError, ValueError):
        return None
    if time.time() - entry.get("created", 0) > MAX_AGE_SECONDS:
        return None
    return entry.get("report")


def put(key: str, report: str) -> None:
    """Saves a report. Failures are ignored: the cache is only an optimisation."""
    if not enabled() or not key:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = _path(key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"created": time.time(), "report": report}, f)
        os.replace(tmp, _path(key))  # atomic, so a crash never leaves half a file
    except OSError:
        pass
