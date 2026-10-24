"""Deterministic guardrails that run around every Gemini call.

None of these call the model, so they cost no requests or tokens:

before_model_guardrail
  - converts files uploaded in adk web (PDF, .docx, text) to text, wrapped in
    delimiters so the model treats them as data, and stores them in session
    state so tools can read them by name instead of the model copying the text;
  - answers "no resume / no JD yet" messages itself, without calling Gemini;
  - stops runaway tool loops with a per-turn model-call cap;
  - fast path: when the message uses a documented format (an upload plus a
    pasted JD, two uploads, or "Resume: ... Job description: ..."), runs
    analyze_match here and removes the tools, so the report takes ONE Gemini
    call. Anything else falls back to the normal tool call;
  - report cache: the same resume + JD again returns the saved report with
    zero Gemini calls.

after_model_guardrail
  - replaces numbers in rewrites and cover letters that do not appear in the
    resume (compared by value, so 1.5k == 1,500) with [X] placeholders;
  - flags rewrites that claim a missing JD skill the original bullet lacks;
  - flags a report that is missing any of the five required sections;
  - caches the report and queues it for export.

after_agent_export
  - saves the report or cover letter as a downloadable PDF artifact.
"""

import functools
import json
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from . import cache
from .documents import markdown_to_pdf
from .tools import (
    CACHE_KEY_STATE,
    JD_STATE_KEY,
    MISSING_STATE_KEY,
    RESUME_STATE_KEY,
    UPLOAD_PREFIX,
    UPLOADS_STATE_KEY,
    analyze_match,
    extract_keywords,
    uploaded_file_to_text,
)

# A report needs 1 model call on the fast path, 2 via a tool. Anything far
# beyond that in one turn means the model is looping, so stop it.
MAX_MODEL_CALLS_PER_TURN = 4

# First messages shorter than this, with no upload and no file path, cannot
# contain both a resume and a JD, so we reply without calling Gemini.
MIN_DOCS_CHARS = 300

REQUIRED_HEADINGS = [
    "## 1. Match Score",
    "## 2. Matched Skills",
    "## 3. Missing Skills & Keywords",
    "## 4. Bullet Rewrites",
    "## 5. Top 3 Fixes Before Applying",
]
COVER_LETTER_HEADING = "## Cover Letter"

ASK_FOR_INPUTS = (
    "To build your match report I need two things:\n\n"
    "1. **Your resume**: upload a PDF, .docx or .txt with the attach (+) button, give a "
    "file path (e.g. `samples/sample_resume.txt`), or paste the text.\n"
    "2. **The job description**: paste it, upload it, or give a file path. Several JDs? "
    "Send them all and I will rank them.\n\n"
    "Send both in one message to get the full report."
)


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


# temp: state lives for one turn; carries the finished report to after_agent_export.
EXPORT_STATE = "temp:export"
DOWNLOAD_PREFIX = "> Download:"


async def after_agent_export(callback_context: CallbackContext) -> Optional[types.Content]:
    """Saves this turn's report or cover letter as a downloadable PDF artifact.

    Runs after the reply was sent, so the artifact lands in its own small event:
    adk web shows an event that carries an artifact as the artifact instead of
    its text, which would hide the formatted report. Never raises.
    """
    export = callback_context.state.get(EXPORT_STATE)
    if not export:
        return None
    filename = f"{export['name']}.pdf"
    try:
        await callback_context.save_artifact(
            filename,
            types.Part.from_bytes(data=markdown_to_pdf(export["markdown"]), mime_type="application/pdf"),
        )
    except Exception:  # no artifact service or storage error: the report is still in the chat
        return None
    callback_context.state[EXPORT_STATE] = None
    return types.Content(role="model", parts=[types.Part(text=f"{DOWNLOAD_PREFIX} `{filename}` (Artifacts tab)")])


# --- Input side --------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _upload_text(data: bytes, mime_type: str) -> tuple:
    """Parses an upload once; later model calls in the session reuse the result."""
    result = uploaded_file_to_text(data, mime_type)
    return result["status"], result.get("text", ""), result.get("error_message", "")


def _convert_uploads(callback_context: CallbackContext, llm_request: LlmRequest) -> list[str]:
    """Replaces inline file parts with delimited text.

    Returns the names of files uploaded in the LATEST user message.
    """
    uploads = dict(callback_context.state.get(UPLOADS_STATE_KEY, {}) or {})
    latest: list[str] = []
    changed = False
    for index, content in enumerate(llm_request.contents):
        if content.role != "user" or not content.parts:
            continue
        for i, part in enumerate(content.parts):
            blob = part.inline_data
            if blob is None:
                continue
            name = blob.display_name or f"file_{i + 1}"
            status, text, error = _upload_text(blob.data or b"", (blob.mime_type or "").lower())
            if status == "success":
                if uploads.get(name) != text:
                    uploads[name] = text
                    changed = True
                if index == len(llm_request.contents) - 1:
                    latest.append(name)
                replacement = (
                    f'[Uploaded file: {name}. Refer to it as "{UPLOAD_PREFIX}{name}" in tool calls; '
                    "do not copy this text into tool arguments.]\n"
                    f"<<<DOCUMENT {name}>>>\n{text}\n<<<END DOCUMENT>>>"
                )
            else:
                replacement = f"[Uploaded file: {name} could not be read: {error}]"
            content.parts[i] = types.Part(text=replacement)
    if changed:
        callback_context.state[UPLOADS_STATE_KEY] = uploads
    return latest


# "Resume: <path or text> Job description: <path or text>" (either order).
_LABELLED_RE = re.compile(
    r"^\s*(?:(?:my\s+)?resume|cv)\s*[:=-]\s*(?P<r1>.+?)\s+(?:job\s*description|jd)\s*[:=-]\s*(?P<j1>.+?)\s*$"
    r"|^\s*(?:job\s*description|jd)\s*[:=-]\s*(?P<j2>.+?)\s+(?:(?:my\s+)?resume|cv)\s*[:=-]\s*(?P<r2>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_JD_NAME_RE = re.compile(r"(jd|job|description|posting|vacancy|opening)", re.IGNORECASE)
_FILE_HINT_RE = re.compile(r"\.(pdf|docx|txt|md)\b", re.IGNORECASE)


def _find_inputs(user_text: str, latest_uploads: list[str]) -> Optional[tuple[str, str]]:
    """Works out (resume, job_description) from the documented message formats.

    Returns None when the message is ambiguous; the model then uses a tool.
    """
    labelled = _LABELLED_RE.match(user_text)
    if len(latest_uploads) == 2:
        jd_files = [n for n in latest_uploads if _JD_NAME_RE.search(n)]
        if len(jd_files) == 1:
            resume = next(n for n in latest_uploads if n != jd_files[0])
            return UPLOAD_PREFIX + resume, UPLOAD_PREFIX + jd_files[0]
        return None
    if len(latest_uploads) == 1:
        resume = UPLOAD_PREFIX + latest_uploads[0]
        if labelled:
            return resume, (labelled.group("j1") or labelled.group("j2")).strip()
        if len(user_text) >= MIN_DOCS_CHARS:
            return resume, user_text
        return None
    if labelled:
        return (
            (labelled.group("r1") or labelled.group("r2")).strip(),
            (labelled.group("j1") or labelled.group("j2")).strip(),
        )
    return None


async def _precompute(
    callback_context: CallbackContext, llm_request: LlmRequest, latest_uploads: list[str]
) -> Optional[LlmResponse]:
    """Runs analyze_match in Python so the report needs at most ONE Gemini call.

    Cached report -> returned with 0 calls. Unreadable file -> error returned
    with 0 calls. Otherwise the analysis is added to the request and the tools
    are removed, so Gemini writes the report straight away.
    """
    last = llm_request.contents[-1]
    if last.role != "user" or any(p.function_response for p in last.parts or []):
        return None
    user_text = "\n".join(
        p.text for p in last.parts or [] if p.text and not p.text.startswith("[Uploaded file:")
    ).strip()
    inputs = _find_inputs(user_text, latest_uploads)
    if inputs is None:
        return None

    result = analyze_match(inputs[0], inputs[1], callback_context)
    if result["status"] != "success":
        return _text_response(f"I couldn't read your documents: {result['error_message']}")

    cached = cache.get(callback_context.state.get(CACHE_KEY_STATE))
    if cached:
        callback_context.state[EXPORT_STATE] = {"name": "match_report", "markdown": cached}
        note = "> Served from the local report cache: 0 Gemini requests. Set `RESUME_JD_CACHE=off` to regenerate."
        return _text_response(f"{cached}\n\n{note}")

    parts = []
    for key, label in (("resume_text", "resume"), ("jd_text", "job_description")):
        if key in result:  # documents given as file paths, not yet seen by the model
            parts.append(types.Part(text=f"<<<DOCUMENT {label}>>>\n{result.pop(key)}\n<<<END DOCUMENT>>>"))
    parts.append(types.Part(text=(
        "[analyze_match already ran for this message; do not call any tool.]\n"
        f"<<<ANALYSIS>>>\n{json.dumps(result)}\n<<<END ANALYSIS>>>"
    )))
    last.parts = list(last.parts or []) + parts
    llm_request.config.tools = None
    llm_request.tools_dict.clear()
    return None


async def before_model_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Runs before every Gemini call. Returning a response skips the call."""
    latest_uploads = _convert_uploads(callback_context, llm_request)

    # Loop guard: count model calls in this turn (temp: state is not persisted).
    key = f"temp:model_calls:{callback_context.invocation_id}"
    calls = callback_context.state.get(key, 0) + 1
    callback_context.state[key] = calls
    if calls > MAX_MODEL_CALLS_PER_TURN:
        return _text_response(
            "Stopped: this request needed more steps than expected. "
            "Please send the resume and job description again in one message."
        )

    # Cheap-request guard: first message that cannot contain both documents.
    if len(llm_request.contents) == 1 and not latest_uploads:
        text = " ".join(p.text or "" for p in llm_request.contents[0].parts or [])
        if len(text.strip()) < MIN_DOCS_CHARS and not _FILE_HINT_RE.search(text):
            return _text_response(ASK_FOR_INPUTS)

    if calls == 1:
        # Fast path: documented formats are analysed here, so one Gemini call suffices.
        response = await _precompute(callback_context, llm_request, latest_uploads)
        if response is not None or llm_request.config.tools is None:
            return response
        _attach_earlier_documents(callback_context, llm_request)
    return None


def _attach_earlier_documents(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """On follow-up turns ("write a cover letter"), re-attaches the resume and JD.

    Documents given as file paths, or analysed on the fast path, are not part
    of the saved chat history, so without this the model would only see its own
    earlier report and could drift from the real resume.
    """
    resume, jd = callback_context.state.get(RESUME_STATE_KEY), callback_context.state.get(JD_STATE_KEY)
    if len(llm_request.contents) < 2 or not resume or not jd:
        return
    if any("<<<DOCUMENT" in (p.text or "") for c in llm_request.contents for p in c.parts or []):
        return  # uploads are already in the history
    llm_request.contents[-1].parts = list(llm_request.contents[-1].parts or []) + [
        types.Part(text="[Documents from earlier in this chat, for reference.]"),
        types.Part(text=f"<<<DOCUMENT resume>>>\n{resume}\n<<<END DOCUMENT>>>"),
        types.Part(text=f"<<<DOCUMENT job_description>>>\n{jd}\n<<<END DOCUMENT>>>"),
    ]


# --- Output side -------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"(?<![\w\[.])\$?(\d+(?:[.,]\d+)*)"
    r"(\s?(?:%|x\b|k\b|K\b|m\b|M\b|mn\b|bn\b|B\b|million\b|thousand\b|billion\b))?"
    r"(?![\w\]%])"
)
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def _value(number: str, suffix: Optional[str]) -> float:
    """'1.5', 'k' -> 1500.0; '1,500', None -> 1500.0."""
    multiplier = _MULTIPLIERS.get((suffix or "").strip().lower(), 1)
    try:
        return round(float(number.replace(",", "")) * multiplier, 6)
    except ValueError:
        return -1.0


def _scrub_invented_numbers(text: str, resume_text: str, all_lines: bool = False) -> tuple[str, int]:
    """Replaces numbers the resume never mentions, compared by value.

    Only **After:** lines are checked in a report; every line in a cover letter.
    """
    known = set()
    for match in _NUMBER_RE.finditer(resume_text):
        known.add(_value(match.group(1), match.group(2)))
        known.add(_value(match.group(1), None))
    replaced = 0

    def fix(match: re.Match) -> str:
        nonlocal replaced
        if _value(match.group(1), match.group(2)) in known:
            return match.group(0)
        replaced += 1
        suffix = (match.group(2) or "").strip()
        return {"%": "[X%]", "x": "[N]x"}.get(suffix, "[X]")

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if all_lines and not line.startswith("#") or line.lstrip().startswith("**After:**"):
            lines[i] = _NUMBER_RE.sub(fix, line)
    return "\n".join(lines), replaced


def _skills(text: str) -> set:
    result = extract_keywords(text)
    return set(result.get("skills", []))


def _unsupported_claims(report: str, missing_skills: list) -> list:
    """Finds rewrites that claim a missing JD skill the original bullet never had."""
    missing = set(missing_skills)
    claims, before, number = [], "", 0
    for line in report.split("\n"):
        stripped = line.strip()
        if stripped.startswith("**Before:**"):
            before = stripped
        elif stripped.startswith("**After:**"):
            number += 1
            added = sorted((_skills(stripped) - _skills(before)) & missing)
            if added:
                claims.append((number, added))
    return claims


async def after_model_guardrail(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Checks a finished report or cover letter before it reaches the user."""
    if llm_response.partial or not llm_response.content or not llm_response.content.parts:
        return None
    text = "".join(p.text or "" for p in llm_response.content.parts if not p.thought)
    is_report = "## 1. Match Score" in text
    is_letter = COVER_LETTER_HEADING in text
    if not (is_report or is_letter):
        return None  # a question, tool call or error message

    resume_text = callback_context.state.get(RESUME_STATE_KEY, "")
    notes = []
    if resume_text:
        text, replaced = _scrub_invented_numbers(text, resume_text, all_lines=is_letter and not is_report)
        if replaced:
            notes.append(
                f"{replaced} number(s) were not in your resume and were replaced with [X] "
                "placeholders. Fill in your real figures."
            )
    if is_report:
        for number, skills in _unsupported_claims(text, callback_context.state.get(MISSING_STATE_KEY, []) or []):
            notes.append(
                f"Rewrite {number} adds {', '.join(skills)}, which your resume does not show. "
                "Only keep it if it is true, and add evidence elsewhere in your resume."
            )
        missing = [h for h in REQUIRED_HEADINGS if h not in text]
        if missing:
            notes.append("This report is missing sections: " + ", ".join(missing) + ". Ask again to regenerate.")
    if notes:
        text += "\n\n---\n" + "\n".join(f"> {n}" for n in notes)

    if is_report and "## 0. JD Ranking" not in text:
        cache.put(callback_context.state.get(CACHE_KEY_STATE), text)
    # The PDF is saved by after_agent_export, in its own event (see there).
    callback_context.state[EXPORT_STATE] = {"name": "match_report" if is_report else "cover_letter", "markdown": text}

    # Edit in place so usage metadata and other response fields are kept.
    llm_response.content = types.Content(role="model", parts=[types.Part(text=text)])
    return llm_response
