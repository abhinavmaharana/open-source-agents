"""Offline tests: no API key and no Gemini calls needed.

Run from agents/resume-jd-match-agent/:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from google.adk.models import LlmRequest, LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from resume_jd_agent import guardrails, tools  # noqa: E402

RESUME = os.path.join(ROOT, "samples", "sample_resume.txt")
JD = os.path.join(ROOT, "samples", "sample_jd.txt")


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def fake_context(state=None):
    """Stands in for ADK's ToolContext / CallbackContext (only .state is used)."""
    return SimpleNamespace(state=state if state is not None else {}, invocation_id="test")


class ToolTests(unittest.TestCase):
    def test_read_resume_txt_path(self):
        result = tools.read_resume(RESUME)
        self.assertEqual(result["status"], "success")
        self.assertIn("PRIYA RAMAN", result["text"])

    def test_read_resume_missing_pdf(self):
        self.assertEqual(tools.read_resume("missing.pdf")["status"], "error")

    def test_long_paste_is_truncated(self):
        result = tools.read_resume("x" * (tools.MAX_DOC_CHARS + 50))
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["text"]), tools.MAX_DOC_CHARS)

    def test_unsupported_upload_type(self):
        self.assertEqual(tools.uploaded_file_to_text(b"x", "image/png")["status"], "error")

    def test_keyword_aliases_and_false_positives(self):
        self.assertEqual(
            tools.extract_keywords("Designed REST endpoints in TS with K8s")["skills"],
            ["kubernetes", "rest api", "typescript"],
        )
        self.assertEqual(tools.extract_keywords("the rest of the learning budget")["skills"], [])

    def test_overlap_on_samples(self):
        overlap = tools.keyword_overlap(tools.read_resume(RESUME)["text"], read(JD))
        self.assertIn("sql", overlap["matched"])
        self.assertIn("a/b testing", overlap["missing"])
        self.assertGreater(overlap["overlap_percent"], 30)


class AnalyzeMatchTests(unittest.TestCase):
    def test_paths_return_text_for_model(self):
        ctx = fake_context()
        result = tools.analyze_match(RESUME, JD, ctx)
        self.assertEqual(result["status"], "success")
        self.assertIn("resume_text", result)  # model has not seen file contents yet
        self.assertIn(tools.RESUME_STATE_KEY, ctx.state)

    def test_upload_reference_does_not_echo_text(self):
        ctx = fake_context({tools.UPLOADS_STATE_KEY: {"cv.pdf": read(RESUME)}})
        result = tools.analyze_match("upload:cv.pdf", read(JD), ctx)
        self.assertEqual(result["status"], "success")
        self.assertNotIn("resume_text", result)  # model already saw the upload
        self.assertNotIn("jd_text", result)

    def test_unknown_upload_name(self):
        result = tools.analyze_match("upload:nope.pdf", read(JD), fake_context())
        self.assertEqual(result["status"], "error")

    def test_jd_without_skills_is_rejected(self):
        result = tools.analyze_match(RESUME, "Hello, how are you today?", fake_context())
        self.assertEqual(result["status"], "error")


def user_request(*parts):
    return LlmRequest(contents=[types.Content(role="user", parts=list(parts))])


def report(after_line):
    text = "\n".join(guardrails.REQUIRED_HEADINGS[:3]) + f"\n## 4. Bullet Rewrites\n{after_line}\n## 5. Top 3 Fixes Before Applying\n1. a"
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


class GuardrailTests(unittest.TestCase):
    def test_short_first_message_skips_gemini(self):
        response = guardrails.before_model_guardrail(fake_context(), user_request(types.Part(text="hi")))
        self.assertIsNotNone(response)  # answered locally, no API request

    def test_file_path_message_goes_to_gemini(self):
        request = user_request(types.Part(text="Resume: samples/sample_resume.txt"))
        self.assertIsNone(guardrails.before_model_guardrail(fake_context(), request))

    def test_upload_is_converted_and_stored(self):
        ctx = fake_context()
        blob = types.Blob(mime_type="text/plain", data=read(RESUME, "rb"), display_name="cv.txt")
        request = user_request(types.Part(text="see file"), types.Part(inline_data=blob))
        self.assertIsNone(guardrails.before_model_guardrail(ctx, request))
        self.assertIn("<<<DOCUMENT cv.txt>>>", request.contents[0].parts[1].text)
        self.assertIn("cv.txt", ctx.state[tools.UPLOADS_STATE_KEY])

    def test_loop_guard(self):
        ctx = fake_context()
        request = user_request(types.Part(text="x" * 400))
        for _ in range(guardrails.MAX_MODEL_CALLS_PER_TURN):
            self.assertIsNone(guardrails.before_model_guardrail(ctx, request))
        self.assertIsNotNone(guardrails.before_model_guardrail(ctx, request))

    def test_invented_numbers_are_replaced(self):
        ctx = fake_context({tools.RESUME_STATE_KEY: "Reduced overdue invoices by 18% for 6 engineers"})
        line = "**After:** Cut overdue invoices by 18% and grew revenue 35% for 1,200 users, 3x faster"
        out = guardrails.after_model_guardrail(ctx, report(line)).content.parts[0].text
        self.assertIn("by 18%", out)  # real number kept
        self.assertIn("[X%]", out)
        self.assertNotIn("35%", out)
        self.assertNotIn("1,200", out)
        self.assertIn("[N]x", out)

    def test_missing_section_is_flagged(self):
        response = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="## 1. Match Score\n**80/100**")]))
        out = guardrails.after_model_guardrail(fake_context(), response).content.parts[0].text
        self.assertIn("missing sections", out)


if __name__ == "__main__":
    unittest.main()
