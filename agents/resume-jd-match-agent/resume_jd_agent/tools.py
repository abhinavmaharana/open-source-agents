"""Plain Python tools used by the Resume <-> JD Match Agent.

Every function returns a dict with a "status" key so the LLM can tell
success from failure. All logic here is deterministic: no network calls.
"""

import io
import os
import re

from google.adk.tools import ToolContext
from pypdf import PdfReader

# Canonical skill -> extra spellings that should count as the same skill.
# Matching is case-insensitive and on word boundaries.
SKILLS: dict[str, list[str]] = {
    # Programming languages
    "python": [],
    "java": [],
    "javascript": ["js"],
    "typescript": [],
    "c++": ["cpp"],
    "c#": ["csharp"],
    "golang": [],
    "rust": [],
    "kotlin": [],
    "swift": [],
    "scala": [],
    "ruby": [],
    "php": [],
    "sql": [],
    "bash": ["shell scripting"],
    "html": ["html5"],
    "css": ["css3"],
    # Frameworks and libraries
    "react": ["react.js", "reactjs"],
    "angular": [],
    "vue": ["vue.js", "vuejs"],
    "next.js": ["nextjs"],
    "node.js": ["nodejs", "node"],
    "express": ["express.js"],
    "django": [],
    "flask": [],
    "fastapi": [],
    "spring boot": [],
    "graphql": [],
    "rest api": ["rest apis", "restful"],
    "grpc": [],
    "microservices": ["microservice"],
    "redux": [],
    "webpack": ["vite"],
    "jest": ["react testing library"],
    "cypress": ["playwright"],
    "responsive design": ["responsive web design", "responsive"],
    "accessibility": ["wcag", "a11y"],
    "design system": ["design systems", "component library"],
    "server-side rendering": ["ssr", "server-side", "server components"],
    "event-driven architecture": ["event-driven"],
    "rabbitmq": [],
    # Cloud and infrastructure
    "aws": ["amazon web services"],
    "gcp": ["google cloud", "google cloud platform"],
    "azure": [],
    "docker": [],
    "kubernetes": ["k8s"],
    "terraform": [],
    "ci/cd": ["cicd", "continuous integration", "continuous delivery"],
    "jenkins": [],
    "github actions": [],
    "linux": [],
    "serverless": [],
    "cloud run": [],
    "bigquery": [],
    "snowflake": [],
    "redshift": [],
    "databricks": [],
    "dynamodb": [],
    "infrastructure as code": ["infrastructure-as-code", "iac"],
    "helm": [],
    "ansible": [],
    "gitops": ["argo cd", "argocd", "flux"],
    "prometheus": [],
    "grafana": [],
    "opentelemetry": [],
    "elk stack": ["elk", "kibana", "logstash"],
    "incident response": ["incident management", "on-call", "postmortems"],
    "slos": ["slo", "slis", "sla"],
    "cloud cost optimization": ["cost optimization", "finops"],
    # Databases and data
    "postgresql": ["postgres"],
    "mysql": [],
    "mongodb": ["mongo"],
    "redis": [],
    "elasticsearch": [],
    "kafka": [],
    "spark": ["apache spark", "pyspark"],
    "airflow": [],
    "dbt": [],
    "etl": ["elt"],
    "data pipelines": ["data pipeline"],
    "data warehouse": ["data warehousing"],
    "data modeling": ["data modelling"],
    "data analysis": ["data analytics"],
    "data visualization": ["data visualisation"],
    "tableau": [],
    "power bi": ["powerbi"],
    "looker": [],
    "excel": ["microsoft excel"],
    "pandas": [],
    "numpy": [],
    "statistics": ["statistical analysis"],
    "a/b testing": ["ab testing", "a/b tests", "split testing", "experimentation"],
    "sql queries": [],
    "mixpanel": [],
    "amplitude": [],
    "google analytics": [],
    # AI / ML
    "machine learning": ["ml"],
    "deep learning": [],
    "nlp": ["natural language processing"],
    "llm": ["llms", "large language models", "large language model"],
    "generative ai": ["genai", "gen ai"],
    "prompt engineering": [],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "tensorflow": [],
    "pytorch": [],
    "scikit-learn": ["sklearn"],
    "computer vision": [],
    "mlops": [],
    "vertex ai": [],
    # Engineering practices
    "git": ["github", "gitlab"],
    "unit testing": ["unit tests"],
    "integration testing": ["integration tests"],
    "test automation": ["automated testing", "end-to-end testing", "end-to-end tests", "e2e", "e2e tests"],
    "code review": ["code reviews"],
    "system design": [],
    "distributed systems": [],
    "api design": [],
    "observability": ["monitoring"],
    "performance optimization": ["performance tuning"],
    "security": ["application security"],
    "devops": [],
    "sre": ["site reliability"],
    "mobile development": ["android", "ios"],
    # Product
    "product management": ["product manager"],
    "product strategy": [],
    "product roadmap": ["roadmap", "roadmaps", "roadmapping"],
    "product discovery": [],
    "product requirements": ["prd", "prds", "product requirements document"],
    "user stories": ["user story"],
    "user research": ["customer research", "customer interviews"],
    "ux": ["user experience"],
    "ui": ["user interface"],
    "wireframing": ["wireframes"],
    "prototyping": ["prototypes"],
    "figma": [],
    "go-to-market": ["gtm", "go to market"],
    "market research": [],
    "competitive analysis": [],
    "kpis": ["kpi"],
    "okrs": ["okr"],
    "metrics": [],
    "funnel analysis": ["conversion funnel"],
    "growth": ["growth strategy"],
    "retention": [],
    "monetization": ["pricing"],
    "b2b": [],
    "b2c": [],
    "saas": [],
    "api products": ["platform products"],
    "backlog management": ["backlog", "backlog prioritization"],
    "prioritization": ["prioritisation"],
    "feature launches": ["product launches", "launches"],
    # Process and management
    "agile": [],
    "scrum": [],
    "kanban": [],
    "jira": [],
    "confluence": [],
    "sprint planning": [],
    "project management": [],
    "program management": [],
    "stakeholder management": ["stakeholders"],
    "cross-functional collaboration": ["cross-functional", "cross functional"],
    "leadership": ["led"],
    "mentoring": ["mentored", "mentorship"],
    "people management": ["team management", "direct reports"],
    "recruiting": ["interviewing candidates"],
    "communication": [],
    "presentation": ["presentations"],
    "problem solving": ["problem-solving"],
    "decision making": ["decision-making"],
    "budgeting": ["budget management", "budget ownership"],
    "risk management": [],
    "vendor management": [],
    "customer success": [],
    "documentation": ["technical writing"],
}

# Aliases that are only a skill when written in capitals ("REST" vs. "the
# rest of"). These are matched case-sensitively against the original text.
CASE_SENSITIVE_ALIASES: dict[str, list[str]] = {
    "rest api": ["REST"],
    "typescript": ["TS"],
}

# Single words that commonly start capitalised sentences or headings and
# should never be treated as the start of a skill phrase.
_PHRASE_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "we", "you", "our", "your", "i", "my", "this", "that",
    "about", "what", "who", "how", "why", "when", "experience", "skills",
    "education", "summary", "responsibilities", "requirements", "qualifications",
}

_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9+#.]*(?:[ \-][A-Z][a-zA-Z0-9+#.]*){1,3})\b")


def _variant_pattern(term: str) -> re.Pattern:
    """Builds a case-insensitive regex that matches a term on word boundaries."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")


_SKILL_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (skill, [_variant_pattern(v) for v in [skill, *variants]])
    for skill, variants in SKILLS.items()
]

_CASE_SENSITIVE_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (skill, [re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])") for v in variants])
    for skill, variants in CASE_SENSITIVE_ALIASES.items()
]


def _normalise(text: str) -> str:
    """Lowercases text and strips punctuation that never appears in a skill name."""
    text = text.lower()
    # Keep characters that are part of skill names: + # . / -
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def _skill_matches(text: str) -> list[str]:
    """Returns canonical skills from SKILLS that appear in the text."""
    normalised = _normalise(text)
    found = {
        skill
        for skill, patterns in _SKILL_PATTERNS
        if any(p.search(normalised) for p in patterns)
    }
    found |= {
        skill
        for skill, patterns in _CASE_SENSITIVE_PATTERNS
        if any(p.search(text) for p in patterns)
    }
    return sorted(found)


def _capitalised_phrases(text: str) -> list[str]:
    """Returns capitalised 2-4 word phrases, e.g. 'Google Cloud Platform'."""
    phrases = set()
    for line in text.splitlines():
        for match in _CAP_PHRASE_RE.findall(line):
            words = match.split()
            if words[0].lower() in _PHRASE_STOPWORDS:
                continue
            phrases.add(match.strip().lower())
    return sorted(phrases)


def read_resume(source: str) -> dict:
    """Reads a resume from a PDF file path or returns pasted resume text.

    Use this first whenever the user provides a resume. If `source` is a path
    ending in .pdf (or .txt) that exists on disk, the file is read. Otherwise
    `source` itself is treated as the pasted resume text.

    Args:
        source: A path to a .pdf or .txt resume file, or the raw resume text.

    Returns:
        dict: {"status": "success", "text": <resume text>} on success, or
        {"status": "error", "error_message": <reason>} on failure.
    """
    if source is None or not source.strip():
        return {"status": "error", "error_message": "No resume provided. Ask the user for a PDF path or pasted text."}

    if _looks_like_path(source):
        candidate = os.path.expanduser(source.strip().strip('"').strip("'"))
        if not os.path.isfile(candidate):
            return {"status": "error", "error_message": f"File not found: {candidate}"}
        if os.path.getsize(candidate) > MAX_FILE_BYTES:
            return {"status": "error", "error_message": f"File is larger than {MAX_FILE_BYTES // 1_000_000} MB."}
        if candidate.lower().endswith(".pdf"):
            return _pdf_to_text(candidate)
        with open(candidate, encoding="utf-8", errors="replace") as f:
            return _limit_text(f.read())

    return _limit_text(source)


# Guardrails on input size. A typical resume is 3,000-10,000 characters; the
# caps stop a huge paste or a 200-page PDF from burning the token quota.
MAX_FILE_BYTES = 5_000_000
MAX_PDF_PAGES = 10
MAX_DOC_CHARS = 20_000


def _limit_text(text: str) -> dict:
    """Strips text and truncates it to MAX_DOC_CHARS, flagging truncation."""
    text = text.strip()
    if not text:
        return {"status": "error", "error_message": "The document is empty."}
    if len(text) > MAX_DOC_CHARS:
        return {"status": "success", "text": text[:MAX_DOC_CHARS], "truncated": True}
    return {"status": "success", "text": text}


def _pdf_to_text(stream) -> dict:
    """Extracts text from a PDF path or binary stream with pypdf."""
    try:
        reader = PdfReader(stream)
        if len(reader.pages) > MAX_PDF_PAGES:
            return {
                "status": "error",
                "error_message": f"The PDF has {len(reader.pages)} pages; the limit is {MAX_PDF_PAGES}. Upload only the resume.",
            }
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:  # pypdf raises several error types
        return {"status": "error", "error_message": f"Could not read PDF: {exc}"}
    if not text:
        return {
            "status": "error",
            "error_message": "The PDF has no extractable text (it may be a scanned image). Ask the user to paste the text instead.",
        }
    return _limit_text(text)


def uploaded_file_to_text(data: bytes, mime_type: str) -> dict:
    """Converts an uploaded file (from the adk web paperclip button) to text.

    Not an LLM tool: guardrails.py calls it on attached files before the model
    runs. Supports PDF and plain-text files.
    """
    if len(data) > MAX_FILE_BYTES:
        return {"status": "error", "error_message": f"File is larger than {MAX_FILE_BYTES // 1_000_000} MB."}
    mime_type = (mime_type or "").lower()
    if mime_type == "application/pdf":
        return _pdf_to_text(io.BytesIO(data))
    if mime_type.startswith("text/"):
        return _limit_text(data.decode("utf-8", errors="replace"))
    return {
        "status": "error",
        "error_message": f"Unsupported file type '{mime_type}'. Upload a PDF or .txt file, or paste the text.",
    }


def extract_keywords(text: str) -> dict:
    """Extracts skills and keywords from a resume or job description.

    Deterministic: matches the text against a built-in list of ~175 common
    tech, product, data and management skills (with common aliases, e.g.
    "k8s" -> "kubernetes"), and also collects capitalised multi-word phrases
    such as "Google Cloud Platform".

    Args:
        text: The resume or job description text.

    Returns:
        dict: {"status": "success", "keywords": [...], "skills": [...],
        "phrases": [...]} where "keywords" is the union of known skills and
        capitalised phrases.
    """
    if not text or not text.strip():
        return {"status": "error", "error_message": "Empty text."}
    skills = _skill_matches(text)
    phrases = [p for p in _capitalised_phrases(text) if p not in skills]
    return {
        "status": "success",
        "keywords": sorted(set(skills) | set(phrases)),
        "skills": skills,
        "phrases": phrases,
    }


def keyword_overlap(resume_text: str, jd_text: str) -> dict:
    """Compares the skills found in a resume against those in a job description.

    Use the result as grounding evidence for the match score, not as the
    final score: it only sees exact keyword matches from a fixed list.

    Args:
        resume_text: The full resume text (from read_resume).
        jd_text: The full job description text.

    Returns:
        dict: {"status": "success",
               "matched": skills in both,
               "missing": JD skills absent from the resume,
               "resume_only": resume skills the JD does not mention,
               "overlap_percent": matched / JD skills * 100,
               "jd_phrases_not_in_resume": capitalised JD phrases (e.g. tool
               or product names) that never appear in the resume text}
    """
    resume = extract_keywords(resume_text)
    jd = extract_keywords(jd_text)
    if resume["status"] != "success":
        return {"status": "error", "error_message": "Resume text is empty."}
    if jd["status"] != "success":
        return {"status": "error", "error_message": "Job description text is empty."}

    resume_skills = set(resume["skills"])
    jd_skills = set(jd["skills"])
    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    overlap = round(100 * len(matched) / len(jd_skills), 1) if jd_skills else 0.0

    resume_lower = _normalise(resume_text)
    phrases_missing = [p for p in jd["phrases"] if _normalise(p).strip() not in resume_lower]

    return {
        "status": "success",
        "matched": matched,
        "missing": missing,
        "resume_only": sorted(resume_skills - jd_skills),
        "overlap_percent": overlap,
        "jd_skill_count": len(jd_skills),
        "jd_phrases_not_in_resume": phrases_missing,
    }


UPLOAD_PREFIX = "upload:"
UPLOADS_STATE_KEY = "uploads"
RESUME_STATE_KEY = "resume_text"


def _resolve_document(source: str, uploads: dict) -> dict:
    """Turns 'upload:<name>', a .pdf/.txt path, or pasted text into text."""
    source = (source or "").strip()
    if source.startswith(UPLOAD_PREFIX):
        name = source[len(UPLOAD_PREFIX):].strip()
        if name not in uploads:
            return {"status": "error", "error_message": f"No uploaded file named '{name}'. Uploaded: {sorted(uploads) or 'none'}."}
        return {"status": "success", "text": uploads[name], "from": "upload"}
    result = read_resume(source)
    if result["status"] == "success":
        result["from"] = "file" if _looks_like_path(source) else "pasted"
    return result


def _looks_like_path(source: str) -> bool:
    """True when read_resume will treat `source` as a file path, not text."""
    candidate = source.strip().strip('"').strip("'")
    return candidate.lower().endswith((".pdf", ".txt")) and "\n" not in candidate


def analyze_match(resume: str, job_description: str, tool_context: ToolContext) -> dict:
    """Reads the resume and JD and compares their skills, in ONE call.

    Call this exactly once per resume/JD pair, before writing the report.

    Args:
        resume: How to find the resume. One of:
            - "upload:<file name>" for a file the user uploaded (preferred:
              never copy uploaded text into this argument),
            - a path to a .pdf or .txt file,
            - the full resume text, only if the user pasted it.
        job_description: How to find the JD, in the same three forms.
        tool_context: Injected by ADK; holds uploaded files in session state.

    Returns:
        dict: {"status": "success", "overlap": <keyword_overlap result>,
        "resume_text"/"jd_text": included ONLY when the document came from a
        file path (you have not seen it yet), "warnings": [...]}, or
        {"status": "error", "error_message": ...}.
    """
    uploads = tool_context.state.get(UPLOADS_STATE_KEY, {}) or {}
    resume_doc = _resolve_document(resume, uploads)
    if resume_doc["status"] != "success":
        return {"status": "error", "error_message": f"Resume: {resume_doc['error_message']}"}
    jd_doc = _resolve_document(job_description, uploads)
    if jd_doc["status"] != "success":
        return {"status": "error", "error_message": f"Job description: {jd_doc['error_message']}"}

    overlap = keyword_overlap(resume_doc["text"], jd_doc["text"])
    if overlap["status"] != "success":
        return overlap
    if overlap["jd_skill_count"] == 0:
        return {
            "status": "error",
            "error_message": "The job description contains no recognisable skills. Ask the user to check it is a real JD.",
        }

    # Kept for the output guardrail that blocks invented numbers in rewrites.
    tool_context.state[RESUME_STATE_KEY] = resume_doc["text"]

    result = {"status": "success", "overlap": overlap, "warnings": []}
    # Only return text the model has not already seen, to save tokens.
    if resume_doc["from"] == "file":
        result["resume_text"] = resume_doc["text"]
    if jd_doc["from"] == "file":
        result["jd_text"] = jd_doc["text"]
    for label, doc in (("Resume", resume_doc), ("Job description", jd_doc)):
        if doc.get("truncated"):
            result["warnings"].append(f"{label} was longer than {MAX_DOC_CHARS} characters and was truncated.")
    return result
