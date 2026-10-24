"""Plain Python tools used by the Resume <-> JD Match Agent.

Every function returns a dict with a "status" key so the LLM can tell
success from failure. All logic here is deterministic: no network calls.
"""

import os
import re

from google.adk.tools import ToolContext

from . import cache
from .documents import (  # noqa: F401  (MAX_* re-exported for callers and tests)
    MAX_DOC_CHARS,
    MAX_FILE_BYTES,
    MAX_PDF_PAGES,
    limit_text,
    looks_like_path,
    read_bytes,
    read_path,
    unsupported_file_error,
)

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


# Each entry is (variant, pattern). The plain-substring check on `variant` runs
# first and is ~10x cheaper than the regex, so most patterns never execute.
_SKILL_PATTERNS: list[tuple[str, list[tuple[str, re.Pattern]]]] = [
    (skill, [(v, _variant_pattern(v)) for v in [skill, *variants]])
    for skill, variants in SKILLS.items()
]

_CASE_SENSITIVE_PATTERNS: list[tuple[str, list[tuple[str, re.Pattern]]]] = [
    (skill, [(v, re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])")) for v in variants])
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
        if any(v in normalised and p.search(normalised) for v, p in patterns)
    }
    found |= {
        skill
        for skill, patterns in _CASE_SENSITIVE_PATTERNS
        if any(v in text and p.search(text) for v, p in patterns)
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


# Words that are capitalised in JDs but are not skills (titles, perks, fillers).
_GENERIC_JD_WORDS = set("""
experience strong excellent working knowledge familiarity exposure hands-on years total including
team teams product products platform platforms manager managers senior junior associate lead engineer
engineers role company customers merchants users business competitive salary health insurance hybrid
remote work setup degree computer science background former technical skills plus bonus qualifications
offer annual learning budget culture certification sponsorship family esops sde pm apm ii iii iv qa vp
""".split())
_JD_WORD = r"[A-Z][A-Za-z0-9+#./-]*[A-Za-z0-9+#]|[A-Z]{2,}"
_JD_PHRASE_RE = re.compile(rf"(?<![\w./-])((?:{_JD_WORD})(?: (?:{_JD_WORD}))*)(?![\w])")
_SKILL_WORDS = {w for skill, variants in SKILLS.items() for v in [skill, *variants] for w in v.split()}


def jd_specific_terms(jd_text: str) -> list[str]:
    """Finds tools, certifications and acronyms in JD bullets that are NOT in
    the built-in skill list, e.g. "Stripe", "HIPAA", "CKA", "Core Web Vitals".

    Only bullet lines are scanned (requirements and responsibilities), the first
    word of each bullet is skipped (sentence case), and generic words such as
    titles and perks are dropped. Output is a hint list for the model.
    """
    terms: dict[str, str] = {}
    for line in jd_text.splitlines():
        line = line.strip()
        if not re.match(r"^[-*•]\s", line):
            continue
        body = re.sub(r"^[-*•]\s+", "", line)
        for match in _JD_PHRASE_RE.finditer(body):
            words = match.group(1).rstrip(".").split()
            if match.start() == 0 and not words[0].isupper():
                words = words[1:]  # "Experience with ..." is sentence case; "HIPAA ..." is not
            words = [w for w in words if w.lower() not in _GENERIC_JD_WORDS]
            term = " ".join(words)
            if not term or _skill_matches(term) or term.lower() in _SKILL_WORDS:
                continue
            terms.setdefault(term.lower(), term)
    return sorted(terms.values(), key=str.lower)


def _terms_missing_from(resume_text: str, terms: list[str]) -> list[str]:
    """Terms whose words do not all appear (as whole words) in the resume."""
    normalised = _normalise(resume_text)
    return [
        term for term in terms
        if not all(_variant_pattern(word).search(normalised) for word in _normalise(term).split())
    ]


def read_resume(source: str) -> dict:
    """Reads a resume from a file path or returns pasted resume text.

    If `source` is a path ending in .pdf, .docx, .txt or .md, the file is read
    (scanned PDFs are OCR'd). Otherwise `source` itself is the pasted text.

    Args:
        source: A path to a resume file, or the raw resume text.

    Returns:
        dict: {"status": "success", "text": <resume text>} on success, or
        {"status": "error", "error_message": <reason>} on failure.
    """
    if source is None or not source.strip():
        return {"status": "error", "error_message": "No resume provided. Ask the user for a file or pasted text."}
    if looks_like_path(source):
        return read_path(source)
    return unsupported_file_error(source) or limit_text(source)


def uploaded_file_to_text(data: bytes, mime_type: str) -> dict:
    """Converts an uploaded file (from the adk web attach button) to text.

    Not an LLM tool: guardrails.py calls it on attached files before the model
    runs. Supports PDF (including scanned, via OCR), .docx and plain text.
    """
    return read_bytes(data, mime_type)


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
               "jd_terms_not_in_resume": JD-specific tools, certifications and
               acronyms outside the skill list (e.g. "Stripe", "HIPAA") that the
               resume never mentions}
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

    jd_terms_missing = _terms_missing_from(resume_text, jd_specific_terms(jd_text))

    return {
        "status": "success",
        "matched": matched,
        "missing": missing,
        "resume_only": sorted(resume_skills - jd_skills),
        "overlap_percent": overlap,
        "jd_skill_count": len(jd_skills),
        "jd_terms_not_in_resume": jd_terms_missing,
    }


UPLOAD_PREFIX = "upload:"
UPLOADS_STATE_KEY = "uploads"
RESUME_STATE_KEY = "resume_text"
MISSING_STATE_KEY = "missing_skills"
JD_STATE_KEY = "jd_text"
# temp: keys live for one turn only; used to cache the report this turn produces.
CACHE_KEY_STATE = "temp:report_cache_key"


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
        result["from"] = "file" if looks_like_path(source) else "pasted"
    return result


def analyze_match(resume: str, job_description: str, tool_context: ToolContext) -> dict:
    """Reads the resume and JD and compares their skills, in ONE call.

    Call this exactly once per resume/JD pair, before writing the report.

    Args:
        resume: How to find the resume. One of:
            - "upload:<file name>" for a file the user uploaded (preferred:
              never copy uploaded text into this argument),
            - a path to a .pdf, .docx, .txt or .md file,
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

    # Kept for the output guardrails (invented numbers, unsupported skill claims)
    # and for the report cache.
    tool_context.state[RESUME_STATE_KEY] = resume_doc["text"]
    tool_context.state[JD_STATE_KEY] = jd_doc["text"]
    tool_context.state[MISSING_STATE_KEY] = overlap["missing"]
    tool_context.state[CACHE_KEY_STATE] = cache.key_for(resume_doc["text"], jd_doc["text"])

    result = {"status": "success", "overlap": overlap, "warnings": []}
    # Only return text the model has not already seen, to save tokens.
    if resume_doc["from"] == "file":
        result["resume_text"] = resume_doc["text"]
    if jd_doc["from"] == "file":
        result["jd_text"] = jd_doc["text"]
    for label, doc in (("Resume", resume_doc), ("Job description", jd_doc)):
        if doc.get("truncated"):
            result["warnings"].append(f"{label} was longer than {MAX_DOC_CHARS} characters and was truncated.")
        if doc.get("ocr"):
            result["warnings"].append(f"{label} was a scanned PDF and was read with OCR; check names and numbers.")
    return result


def _label(source: str, index: int) -> str:
    source = source.strip()
    if source.startswith(UPLOAD_PREFIX):
        return source[len(UPLOAD_PREFIX):]
    if looks_like_path(source):
        return os.path.basename(source.strip("'\""))
    return f"JD {index}"


def rank_job_descriptions(resume: str, job_descriptions: list[str], tool_context: ToolContext) -> dict:
    """Ranks several job descriptions by how well the resume matches each one.

    Use this INSTEAD of analyze_match when the user gives 2 or more JDs. Call it
    once; it also returns the full analysis for the best-matching JD.

    Args:
        resume: The resume, in the same forms as analyze_match ("upload:<name>",
            a file path, or pasted text).
        job_descriptions: 2-10 JDs, each "upload:<name>", a file path, or text.
        tool_context: Injected by ADK.

    Returns:
        dict: {"status": "success", "ranking": [{"rank", "jd", "index", "overlap_percent",
        "matched", "missing", "jd_terms_not_in_resume"}, ...] best first,
        "best": <analyze_match result for the top JD>}, or an error.
    """
    if not 2 <= len(job_descriptions) <= 10:
        return {"status": "error", "error_message": "Give between 2 and 10 job descriptions to rank."}
    uploads = tool_context.state.get(UPLOADS_STATE_KEY, {}) or {}
    resume_doc = _resolve_document(resume, uploads)
    if resume_doc["status"] != "success":
        return {"status": "error", "error_message": f"Resume: {resume_doc['error_message']}"}

    rows = []
    for index, source in enumerate(job_descriptions, 1):
        jd_doc = _resolve_document(source, uploads)
        label = _label(source, index)
        if jd_doc["status"] != "success":
            rows.append({"jd": label, "index": index, "error": jd_doc["error_message"], "overlap_percent": -1, "source": source})
            continue
        overlap = keyword_overlap(resume_doc["text"], jd_doc["text"])
        rows.append({
            "jd": label,
            "index": index,
            "overlap_percent": overlap.get("overlap_percent", 0),
            "matched": len(overlap.get("matched", [])),
            "missing": overlap.get("missing", [])[:8],
            "jd_terms_not_in_resume": overlap.get("jd_terms_not_in_resume", [])[:5],
            "source": source,
        })
    rows.sort(key=lambda row: row["overlap_percent"], reverse=True)
    if rows[0]["overlap_percent"] < 0:
        return {"status": "error", "error_message": "None of the job descriptions could be read."}

    best = analyze_match(resume, rows[0]["source"], tool_context)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row.pop("source")
    return {"status": "success", "ranking": rows, "best_jd": rows[0]["jd"], "best": best}
