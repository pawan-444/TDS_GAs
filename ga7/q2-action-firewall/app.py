"""
LLM Action Firewall
Deterministic (non-LLM, non-phrase-list) validation of proposed tool calls.

Endpoint: POST /action-firewall

Check order (first failure wins):
    1. Top-level schema
    2. Tool allowlist
    3. Selected tool's argument schema
    4. Tenant scope
    5. Exact email domain (egress)
    6. Human approval
    7. HTML safety
"""

import re
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---- Assigned scope (hard-coded, not user-supplied) ----
ASSIGNED_TENANT = "tenant-591xn8f"
ALLOWED_EMAIL_DOMAIN = "notify-6a1o9zd.example"
ALLOWED_TOOLS = {"search", "lookup_record", "send_email", "render_html"}

# ---- Small type helpers ----


def is_string(v):
    return isinstance(v, str)


def is_non_empty_string(v):
    return is_string(v) and len(v) > 0


def is_plain_object(v):
    return isinstance(v, dict)


def is_boolean(v):
    return isinstance(v, bool)


def has_exact_keys(obj, keys):
    return set(obj.keys()) == set(keys)


def respond(reason):
    if reason == "ALLOW":
        return {"decision": "allow", "reason": "ALLOW"}
    return {"decision": "block", "reason": reason}


# ---- 1. Top-level schema check ----


def validate_top_level(body):
    if not is_plain_object(body):
        return False

    if "provenance" not in body or not is_string(body["provenance"]):
        return False
    if body["provenance"] not in ("trusted", "untrusted"):
        return False

    if "humanApproved" not in body or not is_boolean(body["humanApproved"]):
        return False

    if "untrustedContent" in body and body["untrustedContent"] is not None:
        if not is_string(body["untrustedContent"]):
            return False

    if "action" not in body or not is_plain_object(body["action"]):
        return False
    action = body["action"]
    if "tool" not in action or not is_string(action["tool"]):
        return False
    if "args" not in action or not is_plain_object(action["args"]):
        return False

    return True


# ---- 3. Per-tool argument schema check ----


def validate_args_schema(tool, args):
    if tool == "search":
        if not has_exact_keys(args, ["query"]):
            return False
        if not is_string(args["query"]):
            return False
        if not (1 <= len(args["query"]) <= 200):
            return False
        return True

    if tool == "lookup_record":
        if not has_exact_keys(args, ["tenantId", "recordId"]):
            return False
        if not is_non_empty_string(args["tenantId"]):
            return False
        if not is_non_empty_string(args["recordId"]):
            return False
        return True

    if tool == "send_email":
        if not has_exact_keys(args, ["to", "subject", "body"]):
            return False
        if not (
            is_string(args["to"])
            and is_string(args["subject"])
            and is_string(args["body"])
        ):
            return False
        # Must at least look like an address so we can extract a domain later.
        if not re.match(r"^[^@\s]+@[^@\s]+$", args["to"]):
            return False
        return True

    if tool == "render_html":
        if not has_exact_keys(args, ["html"]):
            return False
        if not is_string(args["html"]):
            return False
        return True

    return False


# ---- 7. HTML safety — structural/syntactic checks, not a phrase list ----

SCRIPT_TAG = re.compile(r"<\s*script\b", re.IGNORECASE)
IFRAME_TAG = re.compile(r"<\s*iframe\b", re.IGNORECASE)
# on<word>= as an attribute (onclick=, onerror=, onload=, etc.)
INLINE_HANDLER = re.compile(
    r"""\son[a-z]+\s*=\s*(['"]).*?\1|\son[a-z]+\s*=\s*[^\s>]+""",
    re.IGNORECASE,
)
JS_URL = re.compile(r"javascript\s*:", re.IGNORECASE)


def is_unsafe_html(html):
    return bool(
        SCRIPT_TAG.search(html)
        or IFRAME_TAG.search(html)
        or INLINE_HANDLER.search(html)
        or JS_URL.search(html)
    )


# ---- Core decision function — pure, testable, no I/O ----


def evaluate_action(body):
    # 1. Top-level schema
    if not validate_top_level(body):
        return respond("INVALID_SCHEMA")

    action = body["action"]
    tool = action["tool"]
    args = action["args"]

    # 2. Tool allowlist
    if tool not in ALLOWED_TOOLS:
        return respond("TOOL_NOT_ALLOWED")

    # 3. Selected tool's argument schema
    if not validate_args_schema(tool, args):
        return respond("INVALID_SCHEMA")

    # 4. Tenant scope (lookup_record only)
    if tool == "lookup_record":
        if args["tenantId"] != ASSIGNED_TENANT:
            return respond("TENANT_SCOPE")

    # 5. Exact email domain / egress (send_email only)
    if tool == "send_email":
        domain = args["to"].rsplit("@", 1)[-1]
        if domain != ALLOWED_EMAIL_DOMAIN:
            return respond("EGRESS_DENIED")

    # 6. Human approval (send_email only)
    if tool == "send_email":
        if body["humanApproved"] is not True:
            return respond("APPROVAL_REQUIRED")

    # 7. HTML safety (render_html only)
    if tool == "render_html":
        if is_unsafe_html(args["html"]):
            return respond("UNSAFE_OUTPUT")

    return respond("ALLOW")


# ---- HTTP layer ----


@app.route("/", methods=["GET"])
def health():
    return "LLM Action Firewall is running. POST /action-firewall"


@app.route("/action-firewall", methods=["POST"])
def action_firewall():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify(respond("INVALID_SCHEMA")), 200
    return jsonify(evaluate_action(body)), 200


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
