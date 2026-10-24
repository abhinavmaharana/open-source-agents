"""Deterministic guardrails that run around every Gemini call.

None of these call the model, so they cost no requests or tokens:

before_model_guardrail
  - converts files uploaded in adk web to text (pypdf), wrapped in delimiters so
    the model treats them as data, and stores them in session state so the tool
    can read them by name instead of the model copying the text;
  - answers "no resume / no JD yet" messages itself, without calling Gemini;
  - stops runaway tool loops with a per-turn model-call cap.

after_model_guardrail
  - replaces numbers in "After:" bullet rewrites that do not appear in the
    resume with [X] placeholders, so the agent cannot invent metrics;
  - flags a report that is missing any of the five required sections.
"""

import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from .tools import RESUME_STATE_KEY, UPLOAD_PREFIX, UPLOADS_STATE_KEY, uploaded_file_to_text

# A normal report needs 2 model calls (tool call + final answer). Anything far
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

ASK_FOR_INPUTS = (
    "To build your match report I need two things:\n\n"
    "1. **Your resume**: upload a PDF or .txt with the paperclip button, give a file "
    "path (e.g. `samples/sample_resume.txt`), or paste the text.\n"
    "2. **The job description**: paste it, upload it, or give a .txt path.\n\n"
    "Send both in one message to get the full report."
)


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _convert_uploads(callback_context: CallbackContext, llm_request: LlmRequest) -> bool:
    """Replaces inline file parts with delimited text. Returns True if any exist."""
    uploads = dict(callback_context.state.get(UPLOADS_STATE_KEY, {}) or {})
    found = False
    for content in llm_request.contents:
        if content.role != "user" or not content.parts:
            continue
        for i, part in enumerate(content.parts):
            blob = part.inline_data
            if blob is None:
                continue
            found = True
            name = blob.display_name or f"file_{len(uploads) + 1}"
            result = uploaded_file_to_text(blob.data or b"", blob.mime_type)
            if result["status"] == "success":
                uploads[name] = result["text"]
                text = (
                    f'[Uploaded file: {name}. Pass "{UPLOAD_PREFIX}{name}" to analyze_match; '
                    "do not copy this text into tool arguments.]\n"
                    f"<<<DOCUMENT {name}>>>\n{result['text']}\n<<<END DOCUMENT>>>"
                )
            else:
                text = f"[Uploaded file: {name} could not be read: {result['error_message']}]"
            content.parts[i] = types.Part(text=text)
    if uploads:
        callback_context.state[UPLOADS_STATE_KEY] = uploads
    return found


def before_model_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Runs before every Gemini call. Returning a response skips the call."""
    has_upload = _convert_uploads(callback_context, llm_request)

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
    if len(llm_request.contents) == 1 and not has_upload:
        text = " ".join(p.text or "" for p in llm_request.contents[0].parts or [])
        if len(text.strip()) < MIN_DOCS_CHARS and not re.search(r"\.(pdf|txt)\b", text, re.I):
            return _text_response(ASK_FOR_INPUTS)
    return None


_NUMBER_RE = re.compile(r"(?<![\w\[.])(\d+(?:[.,]\d+)*)(\s?%|x\b)?(?![\w\]%])")


def _scrub_invented_numbers(report: str, resume_text: str) -> tuple[str, int]:
    """Replaces numbers in **After:** lines that the resume never mentions."""
    known = set(re.findall(r"\d+(?:[.,]\d+)*", resume_text))
    replaced = 0

    def fix(match: re.Match) -> str:
        nonlocal replaced
        if match.group(1) in known:
            return match.group(0)
        replaced += 1
        suffix = (match.group(2) or "").strip()
        return {"%": "[X%]", "x": "[N]x"}.get(suffix, "[X]")

    lines = report.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("**After:**"):
            lines[i] = _NUMBER_RE.sub(fix, line)
    return "\n".join(lines), replaced


def after_model_guardrail(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Checks the final report before it reaches the user."""
    if llm_response.partial or not llm_response.content or not llm_response.content.parts:
        return None
    text = "".join(p.text or "" for p in llm_response.content.parts if not p.thought)
    if "## 1. Match Score" not in text:
        return None  # a question, tool call or error message, not a report

    resume_text = callback_context.state.get(RESUME_STATE_KEY, "")
    notes = []
    if resume_text:
        text, replaced = _scrub_invented_numbers(text, resume_text)
        if replaced:
            notes.append(
                f"{replaced} number(s) in the rewrites were not in your resume and were "
                "replaced with [X] placeholders. Fill in your real figures."
            )
    missing = [h for h in REQUIRED_HEADINGS if h not in text]
    if missing:
        notes.append("This report is missing sections: " + ", ".join(missing) + ". Ask again to regenerate.")
    if notes:
        text += "\n\n---\n" + "\n".join(f"> {n}" for n in notes)
    # Edit in place so usage metadata and other response fields are kept.
    llm_response.content = types.Content(role="model", parts=[types.Part(text=text)])
    return llm_response
