"""
OSINT Corroboration Engine
Deterministic evaluation of whether evidence supports a claim, never reads
the wall clock — every request carries its own `asOf` timestamp.

Endpoint: POST /corroborate

Decision order:
    1. invalid
    2. contradicted (fresh, authoritative, disagreeing value)
    3. supported (>=2 independent fresh agreeing sources, by origin)
    4. unverified
"""

from datetime import datetime, timedelta
from flask import Flask, request, jsonify

app = Flask(__name__)

VALID_TYPES = {"dns", "ct_log", "registry", "archive", "scan"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_datetime(s):
    """Parse an ISO-8601 timestamp string. Returns None if unparseable."""
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def is_number(v):
    # bool is a subclass of int in Python; exclude it explicitly.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------


def is_valid_source(src, as_of):
    """A structurally valid source with a parseable observedAt."""
    if not isinstance(src, dict):
        return False, None
    for field in ("id", "origin", "value", "observedAt"):
        if not isinstance(src.get(field), str):
            return False, None
    if src.get("type") not in VALID_TYPES:
        return False, None

    observed_at = parse_datetime(src["observedAt"])
    if observed_at is None:
        return False, None

    return True, observed_at


def is_fresh(as_of, observed_at, staleness_days):
    delta = as_of - observed_at
    return delta <= timedelta(days=staleness_days)


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def respond(verdict, confidence, ids):
    return {
        "verdict": verdict,
        "confidence": confidence,
        "corroboratingSources": ids,
    }


def invalid_response():
    return respond("invalid", "low", [])


def unverified_response():
    return respond("unverified", "low", [])


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(body):
    # ---- 1. Top-level schema / invalid checks ----
    if not isinstance(body, dict):
        return invalid_response()

    claim = body.get("claim")
    if not isinstance(claim, dict):
        return invalid_response()

    claim_value = claim.get("value")
    if not isinstance(claim_value, str):
        return invalid_response()

    as_of = parse_datetime(body.get("asOf"))
    if as_of is None:
        return invalid_response()

    staleness_days = body.get("stalenessDays")
    if not is_number(staleness_days):
        return invalid_response()

    sources = body.get("sources")
    if not isinstance(sources, list):
        return invalid_response()

    # ---- Filter to structurally valid sources with parseable observedAt ----
    valid_sources = []
    for src in sources:
        ok, observed_at = is_valid_source(src, as_of)
        if ok:
            valid_sources.append((src, observed_at))

    # ---- Split into fresh vs stale ----
    fresh_sources = [
        (src, observed_at)
        for src, observed_at in valid_sources
        if is_fresh(as_of, observed_at, staleness_days)
    ]

    # ---- 2. Contradicted: fresh, authoritative, disagreeing value ----
    contradicting_ids = sorted(
        src["id"]
        for src, _ in fresh_sources
        if src.get("authoritative") is True and src["value"] != claim_value
    )
    if contradicting_ids:
        return respond("contradicted", "low", contradicting_ids)

    # ---- 3. Supported: fresh, agreeing, deduped by origin ----
    agreeing = [src for src, _ in fresh_sources if src["value"] == claim_value]

    reps_by_origin = {}
    for src in agreeing:
        origin = src["origin"]
        if origin not in reps_by_origin or src["id"] < reps_by_origin[origin]["id"]:
            reps_by_origin[origin] = src

    representatives = list(reps_by_origin.values())

    if len(representatives) >= 2:
        distinct_types = {r["type"] for r in representatives}
        confidence = "high" if len(distinct_types) >= 2 else "medium"
        rep_ids = sorted(r["id"] for r in representatives)
        return respond("supported", confidence, rep_ids)

    # ---- 4. Unverified ----
    return unverified_response()


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def health():
    return "OSINT Corroboration Engine is running. POST /corroborate"


@app.route("/corroborate", methods=["POST"])
def corroborate():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify(invalid_response()), 200
    return jsonify(evaluate(body)), 200


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
