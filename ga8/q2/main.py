"""
Leakage-safe BigQuery ML experiment gate.

POST /bqml  {"phase": "select" | "evaluate", ...}

Single-file FastAPI service. State (persisted "select" responses, keyed by
runId) lives in an in-process dict guarded by an asyncio.Lock, giving exact
replay / RUN_ID_CONFLICT semantics and select -> evaluate lineage checks for
the lifetime of the running process.
"""

import re
import json
import math
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

_lock = asyncio.Lock()
_RUNS: dict[str, dict] = {}

SAFE_INT_MAX = 2**53 - 1

# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------


def is_plain_object(x: Any) -> bool:
    return isinstance(x, dict)


def is_nonneg_safe_int(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= SAFE_INT_MAX


def is_positive_int(n: Any) -> bool:
    return is_nonneg_safe_int(n) and n > 0


def is_binary01(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n in (0, 1)


def is_finite_number(n: Any) -> bool:
    return isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n)


def utf8_key(s: str) -> bytes:
    return s.encode("utf-8")


def sort_by_utf8(items):
    return sorted(items, key=utf8_key)


def dedupe_sort_codes(codes):
    return sort_by_utf8(list(set(codes)))


def canonicalize(x: Any) -> Any:
    if isinstance(x, list):
        return [canonicalize(v) for v in x]
    if isinstance(x, dict):
        return {k: canonicalize(x[k]) for k in sorted(x.keys())}
    return x


def canonical_json(x: Any) -> str:
    return json.dumps(canonicalize(x), separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def round12(x: float) -> float:
    # Mirrors JS's Math.round(x * 1e12) / 1e12 (values here are always >= 0).
    return math.floor(x * 1e12 + 0.5) / 1e12


# --------------------------------------------------------------------------
# Timestamp parsing: YYYY-MM-DDTHH:mm:ss[.sss](Z|+HH:mm|-HH:mm)
# --------------------------------------------------------------------------

TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$"
)


def is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def parse_timestamp(s: Any) -> Optional[int]:
    """Returns epoch milliseconds (UTC instant), or None if invalid."""
    if not isinstance(s, str):
        return None
    m = TS_RE.match(s)
    if not m:
        return None

    year, month, day, hour, minute, second = (int(m.group(i)) for i in range(1, 7))
    frac = m.group(8) or ""
    offset_raw = m.group(9)

    if not (1 <= month <= 12):
        return None
    days_in_month = [
        31,
        29 if is_leap(year) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ]
    if not (1 <= day <= days_in_month[month - 1]):
        return None
    if hour > 23 or minute > 59 or second > 59:
        return None

    ms = int(frac.ljust(3, "0")) if frac else 0

    if offset_raw == "Z":
        offset_minutes = 0
    else:
        om = re.match(r"^([+-])(\d{2}):(\d{2})$", offset_raw)
        if not om:
            return None
        sign = -1 if om.group(1) == "-" else 1
        oh, omin = int(om.group(2)), int(om.group(3))
        if oh > 23 or omin > 59:
            return None
        offset_minutes = sign * (oh * 60 + omin)

    try:
        dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    epoch_ms = int((dt - epoch).total_seconds() * 1000) + ms - offset_minutes * 60000
    return epoch_ms


# --------------------------------------------------------------------------
# SELECT phase validation
# --------------------------------------------------------------------------


def validate_select_row(r: Any) -> Optional[dict]:
    if not is_plain_object(r):
        return None

    id_ = r.get("id")
    entity = r.get("entity")
    event_time = r.get("eventTime")
    prediction_time = r.get("predictionTime")
    version = r.get("version")
    split = r.get("split")
    features = r.get("features")

    if not (isinstance(id_, str) and len(id_) > 0):
        return None
    if not (isinstance(entity, str) and len(entity) > 0):
        return None

    event_ms = parse_timestamp(event_time)
    if event_ms is None:
        return None
    pred_ms = parse_timestamp(prediction_time)
    if pred_ms is None:
        return None
    if not is_nonneg_safe_int(version):
        return None
    if split not in ("TRAIN", "EVAL"):
        return None
    if not is_plain_object(features):
        return None

    parsed_features = {}
    for name, f in features.items():
        if not (isinstance(name, str) and len(name) > 0):
            return None
        if not is_plain_object(f):
            return None
        if "value" not in f:
            return None
        avail_ms = parse_timestamp(f.get("availableAt"))
        if avail_ms is None:
            return None
        parsed_features[name] = {"value": f.get("value"), "availableAtMs": avail_ms}

    return {
        "id": id_,
        "entity": entity,
        "eventTimeMs": event_ms,
        "predictionTimeMs": pred_ms,
        "version": version,
        "split": split,
        "features": parsed_features,
    }


def validate_select_request(body: Any) -> Optional[dict]:
    if not is_plain_object(body):
        return None

    run_id = body.get("runId")
    forbidden = body.get("forbiddenFeatures")
    num_trials_limit = body.get("numTrialsLimit")
    rows = body.get("rows")
    trials = body.get("trials")

    if not (isinstance(run_id, str) and 0 < len(run_id) <= 128):
        return None
    if not (isinstance(forbidden, list) and all(isinstance(x, str) for x in forbidden)):
        return None
    if not is_positive_int(num_trials_limit):
        return None
    if not (isinstance(rows, list) and len(rows) > 0):
        return None

    parsed_rows = []
    seen_ids = set()
    for r in rows:
        pr = validate_select_row(r)
        if pr is None:
            return None
        if pr["id"] in seen_ids:
            return None
        seen_ids.add(pr["id"])
        parsed_rows.append(pr)

    if not isinstance(trials, list):
        return None
    parsed_trials = []
    seen_trial_ids = set()
    for t in trials:
        if not is_plain_object(t):
            return None
        trial_id = t.get("trialId")
        status = t.get("status")
        eval_metric = t.get("evalMetric")
        if not is_nonneg_safe_int(trial_id):
            return None
        if trial_id in seen_trial_ids:
            return None
        seen_trial_ids.add(trial_id)
        if status not in ("SUCCEEDED", "FAILED"):
            return None
        parsed_trials.append(
            {"trialId": trial_id, "status": status, "evalMetric": eval_metric}
        )

    return {
        "runId": run_id,
        "forbiddenFeatures": forbidden,
        "numTrialsLimit": num_trials_limit,
        "rows": parsed_rows,
        "trials": parsed_trials,
    }


def empty_select_response(raw_body: Any, codes: list) -> dict:
    run_id = (
        raw_body.get("runId")
        if is_plain_object(raw_body) and isinstance(raw_body.get("runId"), str)
        else ""
    )
    return {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": codes,
    }


def do_select(raw_body: Any):
    parsed = validate_select_request(raw_body)
    if parsed is None:
        return 200, empty_select_response(raw_body, ["INVALID_INPUT"]), False

    run_id = parsed["runId"]
    forbidden = parsed["forbiddenFeatures"]
    num_trials_limit = parsed["numTrialsLimit"]
    rows = parsed["rows"]
    trials = parsed["trials"]

    # --- Deduplicate rows by [entity, UTC(eventTime)] ---
    groups: dict = {}
    for row in rows:
        key = (row["entity"], row["eventTimeMs"])
        existing = groups.get(key)
        if existing is None:
            groups[key] = row
        elif row["version"] > existing["version"]:
            groups[key] = row
        elif row["version"] == existing["version"] and utf8_key(row["id"]) < utf8_key(
            existing["id"]
        ):
            groups[key] = row
    retained = list(groups.values())

    # --- Feature eligibility ---
    candidate_names = None
    for row in retained:
        names = set(row["features"].keys())
        candidate_names = (
            names if candidate_names is None else (candidate_names & names)
        )
    candidate_names = candidate_names or set()

    forbidden_set = set(forbidden)
    eligible = []
    for name in candidate_names:
        if name in forbidden_set:
            continue
        ok = True
        for row in retained:
            f = row["features"].get(name)
            if f is None or f["availableAtMs"] > row["predictionTimeMs"]:
                ok = False
                break
        if ok:
            eligible.append(name)
    feature_names = sort_by_utf8(eligible)

    train_ids = sort_by_utf8([r["id"] for r in retained if r["split"] == "TRAIN"])
    eval_ids = sort_by_utf8([r["id"] for r in retained if r["split"] == "EVAL"])

    digest_obj = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }
    digest_input = json.dumps(digest_obj, separators=(",", ":"), ensure_ascii=False)
    dataset_digest = sha256_hex(digest_input)

    # --- Trial selection ---
    codes = []
    if len(trials) > num_trials_limit:
        codes.append("TRIAL_LIMIT_EXCEEDED")

    eligible_trials = [
        t
        for t in trials
        if t["status"] == "SUCCEEDED" and is_finite_number(t["evalMetric"])
    ]
    if len(eligible_trials) == 0:
        codes.append("NO_SUCCESSFUL_TRIAL")

    selected_trial_id = None
    if len(codes) == 0:
        best = eligible_trials[0]
        for t in eligible_trials[1:]:
            if t["evalMetric"] > best["evalMetric"] or (
                t["evalMetric"] == best["evalMetric"] and t["trialId"] < best["trialId"]
            ):
                best = t
        selected_trial_id = best["trialId"]

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": dedupe_sort_codes(codes),
    }
    return 200, response, True


# --------------------------------------------------------------------------
# EVALUATE phase validation
# --------------------------------------------------------------------------


def validate_evaluate_top(body: Any) -> Optional[dict]:
    if not is_plain_object(body):
        return None

    run_id = body.get("runId")
    selected_trial_id = body.get("selectedTrialId")
    dataset_digest = body.get("datasetDigest")
    metric_floor = body.get("metricFloor")
    required_slices = body.get("requiredSlices")
    rows = body.get("rows")
    bytes_processed = body.get("bytesProcessed")
    max_bytes = body.get("maxBytes")

    if not (isinstance(run_id, str) and 0 < len(run_id) <= 128):
        return None
    if not is_nonneg_safe_int(selected_trial_id):
        return None
    if not (
        isinstance(dataset_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", dataset_digest)
    ):
        return None
    if not (is_finite_number(metric_floor) and 0 <= metric_floor <= 1):
        return None
    if not is_plain_object(required_slices):
        return None

    slices = {}
    for name, floor in required_slices.items():
        if not (isinstance(name, str) and len(name) > 0):
            return None
        if not (is_finite_number(floor) and 0 <= floor <= 1):
            return None
        slices[name] = floor

    if not isinstance(rows, list):
        return None
    if not is_nonneg_safe_int(bytes_processed):
        return None
    if not is_nonneg_safe_int(max_bytes):
        return None

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "metricFloor": metric_floor,
        "requiredSlices": slices,
        "rows": rows,
        "bytesProcessed": bytes_processed,
        "maxBytes": max_bytes,
    }


def is_valid_eval_row(r: Any) -> bool:
    if not is_plain_object(r):
        return False
    if not is_binary01(r.get("label")):
        return False
    if not is_binary01(r.get("prediction")):
        return False
    slice_ = r.get("slice")
    if not (isinstance(slice_, str) and len(slice_) > 0):
        return False
    return True


def malformed_evaluate_response(raw_body: Any, code: str) -> dict:
    ok = is_plain_object(raw_body)
    run_id = (
        raw_body.get("runId") if ok and isinstance(raw_body.get("runId"), str) else ""
    )
    selected_trial_id = (
        raw_body.get("selectedTrialId")
        if ok and is_nonneg_safe_int(raw_body.get("selectedTrialId"))
        else None
    )
    dataset_digest = (
        raw_body.get("datasetDigest")
        if ok and isinstance(raw_body.get("datasetDigest"), str)
        else ""
    )
    bytes_processed = (
        raw_body.get("bytesProcessed")
        if ok and is_nonneg_safe_int(raw_body.get("bytesProcessed"))
        else 0
    )
    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": [code],
    }


def do_evaluate(raw_body: Any, runs_store: dict):
    top = validate_evaluate_top(raw_body)
    if top is None:
        return 200, malformed_evaluate_response(raw_body, "INVALID_INPUT")

    run_id = top["runId"]
    selected_trial_id = top["selectedTrialId"]
    dataset_digest = top["datasetDigest"]
    metric_floor = top["metricFloor"]
    required_slices = top["requiredSlices"]
    rows = top["rows"]
    bytes_processed = top["bytesProcessed"]
    max_bytes = top["maxBytes"]

    codes = []

    # --- Lineage check ---
    stored = runs_store.get(run_id)
    lineage_ok = (
        stored is not None
        and stored["response"].get("selectedTrialId") is not None
        and stored["response"]["selectedTrialId"] == selected_trial_id
        and stored["response"]["datasetDigest"] == dataset_digest
    )
    if not lineage_ok:
        codes.append("INVALID_LINEAGE")

    # --- Row validity ---
    rows_usable = len(rows) > 0
    if rows_usable:
        for r in rows:
            if not is_valid_eval_row(r):
                rows_usable = False
                codes.append("INVALID_TEST_ROW")
                break

    test_metric = None
    if rows_usable:
        correct = 0
        slice_stats: dict = {}
        for r in rows:
            hit = 1 if r["label"] == r["prediction"] else 0
            correct += hit
            s = slice_stats.setdefault(r["slice"], {"correct": 0, "total": 0})
            s["correct"] += hit
            s["total"] += 1

        aggregate = round12(correct / len(rows))
        test_metric = aggregate

        if aggregate < metric_floor:
            codes.append("AGGREGATE_FLOOR")

        for name, floor in required_slices.items():
            s = slice_stats.get(name)
            if s is None:
                codes.append(f"MISSING_SLICE:{name}")
            else:
                acc = round12(s["correct"] / s["total"])
                if acc < floor:
                    codes.append(f"SLICE_FLOOR:{name}")

    # --- Byte check (always applies) ---
    if bytes_processed > max_bytes:
        codes.append("BYTE_LIMIT")

    reason_codes = dedupe_sort_codes(codes)

    critical_slice_pass = not any(
        c in ("INVALID_INPUT", "INVALID_LINEAGE", "INVALID_TEST_ROW")
        or c.startswith("MISSING_SLICE:")
        or c.startswith("SLICE_FLOOR:")
        for c in reason_codes
    )
    decision = "admit" if len(reason_codes) == 0 else "reject"

    return 200, {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": reason_codes,
    }


# --------------------------------------------------------------------------
# HTTP endpoint
# --------------------------------------------------------------------------


@app.post("/bqml")
async def bqml(request: Request):
    try:
        raw_bytes = await request.body()
        body = json.loads(raw_bytes)
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    if not is_plain_object(body) or body.get("phase") not in ("select", "evaluate"):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    async with _lock:
        if body["phase"] == "select":
            run_id = body.get("runId") if isinstance(body.get("runId"), str) else None
            if run_id and 0 < len(run_id) <= 128:
                existing = _RUNS.get(run_id)
                if existing is not None:
                    if canonical_json(existing["rawRequest"]) == canonical_json(body):
                        return JSONResponse(existing["response"], status_code=200)
                    return JSONResponse({"error": "RUN_ID_CONFLICT"}, status_code=409)

            status, resp, persist = do_select(body)
            if persist and run_id:
                _RUNS[run_id] = {"rawRequest": body, "response": resp}
            return JSONResponse(resp, status_code=status)

        else:
            status, resp = do_evaluate(body, _RUNS)
            return JSONResponse(resp, status_code=status)


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
