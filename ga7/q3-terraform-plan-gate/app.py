"""
Terraform Plan Policy Gate
Deterministic policy-as-code check on one normalized Terraform resource
change, run before `apply`.

Endpoint: POST /terraform/plan

Check order (first failure wins):
    1. Type validation of the request and nested objects
    2. Environment must match assigned workspace
    3. State backend/lock safety
    4. Provider version must be exactly pinned or pessimistically pinned
    5. All required labels present with exact values
    6. Secret must be null or a non-empty secret:// reference
    7. Deleting a stateful resource type requires destroyApproved
    8. A production storage_bucket may never use forceDestroy: true
"""

import re
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---- Assigned scope (hard-coded, not user-supplied) ----
ASSIGNED_WORKSPACE = "prod-gfyt5s"
REQUIRED_LABELS = {
    "owner": "student-klvlw",
    "environment": "production",
    "cost_center": "cc-d1fr",
}
ALLOWED_BACKENDS = {"gcs", "s3", "azurerm", "remote"}
ALLOWED_ACTIONS = {"create", "update", "delete"}
STATEFUL_DELETE_TYPES = {"storage_bucket", "sql_database", "persistent_disk"}

# ---- Provider version patterns ----
# Exact: "6.2.1" or "= 6.2.1"
EXACT_VERSION_RE = re.compile(r"^(=\s*)?\d+\.\d+\.\d+$")
# Pessimistic pin: "~> 6.0", "~> 6.0.1", "~> 6"
PESSIMISTIC_VERSION_RE = re.compile(r"^~>\s*\d+(\.\d+){0,2}$")

# ---- Secret reference pattern ----
SECRET_REF_RE = re.compile(r"^secret://.+$")


def is_string(v):
    return isinstance(v, str)


def is_bool(v):
    return isinstance(v, bool)


def is_dict(v):
    return isinstance(v, dict)


def respond(reason):
    if reason == "APPROVE":
        return {"decision": "approve", "reason": "APPROVE"}
    return {"decision": "reject", "reason": reason}


# ---- 1. Type validation ----


def validate_schema(body):
    if not is_dict(body):
        return False

    if "environment" not in body or not is_string(body["environment"]):
        return False

    if "state" not in body or not is_dict(body["state"]):
        return False
    state = body["state"]
    if "backend" not in state or not is_string(state["backend"]):
        return False
    if "locked" not in state or not is_bool(state["locked"]):
        return False

    if "providerVersion" not in body or not is_string(body["providerVersion"]):
        return False

    if "destroyApproved" not in body or not is_bool(body["destroyApproved"]):
        return False

    if "resource" not in body or not is_dict(body["resource"]):
        return False
    res = body["resource"]

    if "address" not in res or not is_string(res["address"]):
        return False
    if "type" not in res or not is_string(res["type"]):
        return False
    if "action" not in res or not is_string(res["action"]):
        return False
    if res["action"] not in ALLOWED_ACTIONS:
        return False
    if "labels" not in res or not is_dict(res["labels"]):
        return False
    # every label key/value must be a string
    for k, v in res["labels"].items():
        if not is_string(k) or not is_string(v):
            return False
    if "secret" not in res or not (res["secret"] is None or is_string(res["secret"])):
        return False
    if "forceDestroy" not in res or not is_bool(res["forceDestroy"]):
        return False

    return True


def is_pinned_version(v):
    return bool(
        EXACT_VERSION_RE.match(v.strip()) or PESSIMISTIC_VERSION_RE.match(v.strip())
    )


def is_valid_secret(secret):
    if secret is None:
        return True
    return bool(SECRET_REF_RE.match(secret))


def evaluate_plan(body):
    # 1. Schema / type validation
    if not validate_schema(body):
        return respond("INVALID_PLAN")

    environment = body["environment"]
    state = body["state"]
    provider_version = body["providerVersion"]
    destroy_approved = body["destroyApproved"]
    resource = body["resource"]

    # 2. Environment must exactly match assigned workspace
    if environment != ASSIGNED_WORKSPACE:
        return respond("ENVIRONMENT_MISMATCH")

    # 3. State backend + lock safety
    if state["backend"] not in ALLOWED_BACKENDS or state["locked"] is not True:
        return respond("STATE_UNSAFE")

    # 4. Provider must be exactly or pessimistically pinned
    if not is_pinned_version(provider_version):
        return respond("UNPINNED_PROVIDER")

    # 5. All required labels present with exact values
    labels = resource["labels"]
    for key, expected in REQUIRED_LABELS.items():
        if labels.get(key) != expected:
            return respond("MISSING_LABELS")

    # 6. Secret must be null or a non-empty secret:// reference
    if not is_valid_secret(resource["secret"]):
        return respond("PLAINTEXT_SECRET")

    # 7. Deleting a stateful resource type requires destroyApproved
    if resource["action"] == "delete" and resource["type"] in STATEFUL_DELETE_TYPES:
        if destroy_approved is not True:
            return respond("DELETE_NOT_APPROVED")

    # 8. A production storage_bucket may never use forceDestroy: true
    if resource["type"] == "storage_bucket" and resource["forceDestroy"] is True:
        return respond("FORCE_DESTROY")

    return respond("APPROVE")


# ---- HTTP layer ----


@app.route("/", methods=["GET"])
def health():
    return "Terraform Plan Policy Gate is running. POST /terraform/plan"


@app.route("/terraform/plan", methods=["POST"])
def terraform_plan():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify(respond("INVALID_PLAN")), 200
    return jsonify(evaluate_plan(body)), 200


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
