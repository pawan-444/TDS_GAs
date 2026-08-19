"""
LLM Output Handling Gate (OWASP LLM05)
Deterministic gate deciding whether a model's output is safe to hand to a
given sink. No LLM, no suspicious-phrase list — only structural rules.

Endpoint: POST /sanitize-output

Order of evaluation:
    1. INVALID_SCHEMA
    2. ENCODED_PAYLOAD (decode once; if decoded differs from original AND
       the decoded string would trip a channel rule, report ENCODED_PAYLOAD)
    3. Channel rules applied to the ORIGINAL output, first match wins
"""

import re
from urllib.parse import unquote, urlsplit
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---- Assigned scope ----
ALLOWED_HOSTS = {"cdn-cbppkwr.example", "app-7x9ji1i.example"}
ALLOWED_CHANNELS = {"html", "markdown", "url", "sql", "shell"}
MAX_OUTPUT_LEN = 20000

# ---- Regex building blocks ----
SCRIPT_TAG_RE = re.compile(r"<\s*(script|iframe|object|embed)\b", re.IGNORECASE)
EVENT_HANDLER_RE = re.compile(r"\bon[a-zA-Z]+\s*=", re.IGNORECASE)
DANGEROUS_SCHEME_LITERAL_RE = re.compile(
    r"(javascript|data|vbscript)\s*:", re.IGNORECASE
)

HTML_URL_ATTR_RE = re.compile(
    r"""(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE
)
MARKDOWN_LINK_RE = re.compile(r"\]\(([^)]*)\)")

SQL_METACHAR_RE = re.compile(
    r"""'|"|;|--|/\*|\bunion\b|\bor\s+1\s*=\s*1\b""", re.IGNORECASE
)
SHELL_METACHAR_RE = re.compile(r"[;&|`<>]|\$\(|\$\{")

SCHEME_PREFIX_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")

NUMERIC_DEC_ENTITY_RE = re.compile(r"&#(\d+);")
NUMERIC_HEX_ENTITY_RE = re.compile(r"&#[xX]([0-9a-fA-F]+);")
UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")

NAMED_ENTITIES = [
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&apos;", "'"),
    ("&amp;", "&"),
]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(body):
    if not isinstance(body, dict):
        return False
    if "channel" not in body or body["channel"] not in ALLOWED_CHANNELS:
        return False
    if "output" not in body or not isinstance(body["output"], str):
        return False
    if len(body["output"]) > MAX_OUTPUT_LEN:
        return False
    return True


# ---------------------------------------------------------------------------
# Decoding pipeline (percent-escapes -> HTML entities -> \uXXXX), applied once
# ---------------------------------------------------------------------------


def percent_decode(s):
    try:
        return unquote(s, errors="replace")
    except Exception:
        return s


def decode_entities(s):
    s = NUMERIC_DEC_ENTITY_RE.sub(lambda m: _safe_chr(int(m.group(1))), s)
    s = NUMERIC_HEX_ENTITY_RE.sub(lambda m: _safe_chr(int(m.group(1), 16)), s)
    for ent, ch in NAMED_ENTITIES:
        s = s.replace(ent, ch)
    return s


def decode_unicode_escapes(s):
    return UNICODE_ESCAPE_RE.sub(lambda m: _safe_chr(int(m.group(1), 16)), s)


def _safe_chr(code):
    try:
        return chr(code)
    except (ValueError, OverflowError):
        return ""


def decode_once(s):
    s = percent_decode(s)
    s = decode_entities(s)
    s = decode_unicode_escapes(s)
    return s


# ---------------------------------------------------------------------------
# URL extraction and analysis
# ---------------------------------------------------------------------------


def extract_urls_html(text):
    urls = []
    for m in HTML_URL_ATTR_RE.finditer(text):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        urls.append(val)
    return urls


def extract_urls_markdown(text):
    urls = []
    for m in MARKDOWN_LINK_RE.finditer(text):
        content = m.group(1).strip()
        if not content:
            continue
        # Handle optional markdown title: [text](url "title")
        parts = content.split(None, 1)
        urls.append(parts[0])
    return urls


def analyze_url(raw):
    """Returns dict: {absolute: bool, scheme: str|None, hostname: str|None}"""
    u = raw.strip()
    if u == "":
        return {"absolute": False, "scheme": None, "hostname": None}

    if u.startswith("//"):
        try:
            parts = urlsplit("https:" + u)
            return {"absolute": True, "scheme": "https", "hostname": parts.hostname}
        except ValueError:
            return {"absolute": True, "scheme": "https", "hostname": None}

    if SCHEME_PREFIX_RE.match(u):
        try:
            parts = urlsplit(u)
            scheme = parts.scheme.lower() if parts.scheme else None
            return {"absolute": True, "scheme": scheme, "hostname": parts.hostname}
        except ValueError:
            return {"absolute": True, "scheme": None, "hostname": None}

    return {"absolute": False, "scheme": None, "hostname": None}


def has_dangerous_scheme(text, urls):
    if DANGEROUS_SCHEME_LITERAL_RE.search(text):
        return True
    for raw in urls:
        info = analyze_url(raw)
        if info["absolute"] and info["scheme"] not in ("http", "https"):
            return True
    return False


def has_external_exfil(urls):
    for raw in urls:
        info = analyze_url(raw)
        if info["absolute"]:
            host = info["hostname"]
            if host is None or host.lower() not in ALLOWED_HOSTS:
                return True
    return False


# ---------------------------------------------------------------------------
# Channel-specific checks (first match wins)
# ---------------------------------------------------------------------------


def check_channel(text, channel):
    if channel == "html":
        if SCRIPT_TAG_RE.search(text):
            return "SCRIPT_TAG"
        if EVENT_HANDLER_RE.search(text):
            return "EVENT_HANDLER"
        urls = extract_urls_html(text)
        if has_dangerous_scheme(text, urls):
            return "DANGEROUS_SCHEME"
        if has_external_exfil(urls):
            return "EXTERNAL_EXFIL"
        return "SAFE"

    if channel == "markdown":
        urls = extract_urls_markdown(text)
        if has_dangerous_scheme(text, urls):
            return "DANGEROUS_SCHEME"
        if has_external_exfil(urls):
            return "EXTERNAL_EXFIL"
        return "SAFE"

    if channel == "url":
        urls = [text.strip()]
        if has_dangerous_scheme(text, urls):
            return "DANGEROUS_SCHEME"
        if has_external_exfil(urls):
            return "EXTERNAL_EXFIL"
        return "SAFE"

    if channel == "sql":
        if SQL_METACHAR_RE.search(text):
            return "SQL_METACHAR"
        return "SAFE"

    if channel == "shell":
        if SHELL_METACHAR_RE.search(text):
            return "SHELL_METACHAR"
        return "SAFE"

    return "SAFE"


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


def respond(reason):
    return {"safe": reason == "SAFE", "reason": reason}


def evaluate(body):
    if not validate_schema(body):
        return respond("INVALID_SCHEMA")

    channel = body["channel"]
    output = body["output"]

    decoded = decode_once(output)
    if decoded != output:
        decoded_reason = check_channel(decoded, channel)
        if decoded_reason != "SAFE":
            return respond("ENCODED_PAYLOAD")

    original_reason = check_channel(output, channel)
    return respond(original_reason)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def health():
    return "LLM Output Handling Gate is running. POST /sanitize-output"


@app.route("/sanitize-output", methods=["POST"])
def sanitize_output():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify(respond("INVALID_SCHEMA")), 200
    return jsonify(evaluate(body)), 200


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
