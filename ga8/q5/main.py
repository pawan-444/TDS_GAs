import hashlib
import json
import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MAX_SAFE_INTEGER = (2**53) - 1

# State persists as long as this Python process remains alive.
# freezeId -> {
#   "fingerprint": "...",
#   "response": {
#       "freezeId": "...",
#       "candidates": [...]
#   }
# }
FREEZES: dict[str, dict[str, Any]] = {}


def compact_json(value: Any) -> str:
    """Equivalent to compact JSON.stringify-style serialization."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utf8_sort_key(value: str) -> bytes:
    return value.encode("utf-8")


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def is_safe_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def is_finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def is_finite_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def is_binary_label(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def sort_reason_codes(codes: set[str]) -> list[str]:
    return sorted(codes, key=utf8_sort_key)


def invalid_input_response() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


def unique_nonempty_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(is_nonempty_string(item) for item in value)
        and len(value) == len(set(value))
    )


# -------------------------------------------------------------------
# Freeze helpers
# -------------------------------------------------------------------


def inventory_from_files(files: Any) -> tuple[list[dict[str, Any]], int, str] | None:
    """
    Construct the candidate inventory directly from candidate.files.

    Returns:
      (inventory, totalBytes, packageDigest)

    Returns None when files is invalid:
      - not an object
      - empty object
      - invalid filename
      - non-string file content
      - invalid UTF-8 encodability
    """
    if not isinstance(files, dict) or len(files) == 0:
        return None

    inventory: list[dict[str, Any]] = []

    for filename, content in files.items():
        if not is_nonempty_string(filename) or not isinstance(content, str):
            return None

        try:
            raw = content.encode("utf-8")
        except UnicodeEncodeError:
            return None

        inventory.append(
            {
                "name": filename,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    inventory.sort(key=lambda item: utf8_sort_key(item["name"]))

    total_bytes = sum(item["bytes"] for item in inventory)
    package_digest = sha256_utf8(compact_json(inventory))

    return inventory, total_bytes, package_digest


def validate_freeze_request(payload: Any) -> bool:
    """
    Checks whole-request requirements.

    Invalid candidate files are intentionally NOT rejected here because
    the specification requires those candidates to be returned with:
      inventory: []
      totalBytes: null
      packageDigest: null
      status: invalid
    """
    if not isinstance(payload, dict):
        return False

    required_fields = (
        "phase",
        "freezeId",
        "calibrationDigest",
        "tokenizerDigest",
        "allowedUnsupportedReasons",
        "candidates",
    )

    if any(field not in payload for field in required_fields):
        return False

    freeze_id = payload.get("freezeId")

    if not is_nonempty_string(freeze_id) or len(freeze_id) > 128:
        return False

    if not is_nonempty_string(payload.get("calibrationDigest")):
        return False

    if not is_nonempty_string(payload.get("tokenizerDigest")):
        return False

    if not unique_nonempty_strings(payload.get("allowedUnsupportedReasons")):
        return False

    candidates = payload.get("candidates")

    if not isinstance(candidates, list) or len(candidates) == 0:
        return False

    names: list[str] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False

        name = candidate.get("name")

        if not is_nonempty_string(name):
            return False

        names.append(name)

    return len(names) == len(set(names))


def freeze_request_fingerprint(payload: dict[str, Any]) -> str:
    """
    The entire freeze request determines its identity.
    Reusing a freezeId with exactly the same request is an idempotent replay.
    """
    return compact_json(payload)


def freeze_candidate(
    candidate: dict[str, Any],
    expected_calibration_digest: str,
    expected_tokenizer_digest: str,
    allowed_reasons: set[str],
) -> dict[str, Any]:
    """
    Create one deterministic frozen-candidate response object.
    """
    name = candidate["name"]
    reason_codes: set[str] = set()

    file_result = inventory_from_files(candidate.get("files"))

    if file_result is None:
        inventory: list[dict[str, Any]] = []
        total_bytes: int | None = None
        package_digest: str | None = None
        files_valid = False
    else:
        inventory, total_bytes, package_digest = file_result
        files_valid = True

    unsupported_reason = candidate.get("unsupportedReason")

    # If a reason is supplied, candidate can only be "unsupported"
    # if that reason is explicitly allowed.
    if unsupported_reason is not None:
        if (
            not is_nonempty_string(unsupported_reason)
            or unsupported_reason not in allowed_reasons
        ):
            reason_codes.add("UNALLOWED_UNSUPPORTED_REASON")
            status = "invalid"
        else:
            status = "unsupported"

    # If there is no unsupported reason, the artifact must load and
    # bind to the same calibration/tokenizer digests.
    else:
        status = "frozen"

        if candidate.get("loadable") is not True:
            reason_codes.add("NOT_LOADABLE")

        if candidate.get("calibrationDigest") != expected_calibration_digest:
            reason_codes.add("CALIBRATION_MISMATCH")

        if candidate.get("tokenizerDigest") != expected_tokenizer_digest:
            reason_codes.add("TOKENIZER_MISMATCH")

        if reason_codes:
            status = "invalid"

    # Invalid file map always makes a candidate invalid.
    if not files_valid:
        status = "invalid"

    return {
        "name": name,
        "status": status,
        "inventory": inventory,
        "totalBytes": total_bytes,
        "packageDigest": package_digest,
        "reasonCodes": sort_reason_codes(reason_codes),
    }


def handle_freeze(payload: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    if not validate_freeze_request(payload):
        return invalid_input_response()

    freeze_id = payload["freezeId"]
    fingerprint = freeze_request_fingerprint(payload)

    existing = FREEZES.get(freeze_id)

    # Exact replay: return the original stored result unchanged.
    if existing is not None:
        if existing["fingerprint"] == fingerprint:
            return existing["response"]

        return JSONResponse(
            status_code=409,
            content={"error": "FREEZE_ID_CONFLICT"},
        )

    allowed_reasons = set(payload["allowedUnsupportedReasons"])

    frozen_candidates = [
        freeze_candidate(
            candidate=candidate,
            expected_calibration_digest=payload["calibrationDigest"],
            expected_tokenizer_digest=payload["tokenizerDigest"],
            allowed_reasons=allowed_reasons,
        )
        for candidate in payload["candidates"]
    ]

    frozen_candidates.sort(key=lambda candidate: utf8_sort_key(candidate["name"]))

    response = {
        "freezeId": freeze_id,
        "candidates": frozen_candidates,
    }

    FREEZES[freeze_id] = {
        "fingerprint": fingerprint,
        "response": response,
    }

    return response


# -------------------------------------------------------------------
# Select helpers
# -------------------------------------------------------------------


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False

    required_fields = (
        "maxBytes",
        "aggregateFloor",
        "requiredSlices",
        "maxLatencyMs",
        "candidateOrder",
    )

    if any(field not in policy for field in required_fields):
        return False

    if not is_safe_nonnegative_integer(policy.get("maxBytes")):
        return False

    if not is_finite_probability(policy.get("aggregateFloor")):
        return False

    if not is_finite_nonnegative_number(policy.get("maxLatencyMs")):
        return False

    required_slices = policy.get("requiredSlices")

    if not isinstance(required_slices, dict):
        return False

    for slice_name, floor in required_slices.items():
        if not is_nonempty_string(slice_name):
            return False

        if not is_finite_probability(floor):
            return False

    candidate_order = policy.get("candidateOrder")

    if not isinstance(candidate_order, list):
        return False

    if any(not is_nonempty_string(name) for name in candidate_order):
        return False

    if len(candidate_order) != len(set(candidate_order)):
        return False

    return True


def validate_select_request(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    required_fields = (
        "phase",
        "freezeId",
        "candidates",
        "policy",
        "latencies",
        "rows",
    )

    if any(field not in payload for field in required_fields):
        return False

    if not is_nonempty_string(payload.get("freezeId")):
        return False

    if not isinstance(payload.get("candidates"), list):
        return False

    if not isinstance(payload.get("rows"), list):
        return False

    if not isinstance(payload.get("policy"), dict):
        return False

    if not isinstance(payload.get("latencies"), dict):
        return False

    return True


def recompute_submitted_manifest(
    candidate: Any,
) -> tuple[list[dict[str, Any]], int, str] | None:
    """
    Select receives frozen candidate objects, not raw file text.

    So we recompute:
      totalBytes = sum(inventory[i].bytes)
      packageDigest = SHA256(UTF8(compact JSON inventory))

    We also strictly validate:
      - inventory is a non-empty list
      - each entry uses exactly name, bytes, sha256 keys in that order
      - names are unique and UTF-8 sorted
      - bytes are safe non-negative integers
      - SHA-256 is lowercase 64-character hex
    """
    if not isinstance(candidate, dict):
        return None

    inventory = candidate.get("inventory")

    if not isinstance(inventory, list) or len(inventory) == 0:
        return None

    expected_key_order = ["name", "bytes", "sha256"]
    validated_inventory: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for entry in inventory:
        if not isinstance(entry, dict):
            return None

        if list(entry.keys()) != expected_key_order:
            return None

        name = entry.get("name")
        byte_count = entry.get("bytes")
        digest = entry.get("sha256")

        if not is_nonempty_string(name):
            return None

        if name in seen_names:
            return None

        if not is_safe_nonnegative_integer(byte_count):
            return None

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            return None

        seen_names.add(name)

        validated_inventory.append(
            {
                "name": name,
                "bytes": byte_count,
                "sha256": digest,
            }
        )

    expected_sorted = sorted(
        validated_inventory,
        key=lambda entry: utf8_sort_key(entry["name"]),
    )

    if validated_inventory != expected_sorted:
        return None

    total_bytes = sum(entry["bytes"] for entry in validated_inventory)
    package_digest = sha256_utf8(compact_json(validated_inventory))

    return validated_inventory, total_bytes, package_digest


def calculate_candidate_metrics(
    candidate_name: str,
    rows: Any,
) -> tuple[float | None, dict[str, float] | None, bool]:
    """
    Returns:
      aggregate accuracy,
      per-slice accuracies,
      predictions_valid

    Invalid predictions return:
      (None, None, False)
    """
    if not isinstance(rows, list) or len(rows) == 0:
        return None, None, False

    aggregate_correct = 0
    slice_counts: dict[str, list[int]] = {}

    for row in rows:
        if not isinstance(row, dict):
            return None, None, False

        label = row.get("label")
        slice_name = row.get("slice")
        predictions = row.get("predictions")

        if not is_binary_label(label):
            return None, None, False

        if not is_nonempty_string(slice_name):
            return None, None, False

        if not isinstance(predictions, dict):
            return None, None, False

        prediction = predictions.get(candidate_name)

        if not is_binary_label(prediction):
            return None, None, False

        is_correct = prediction == label

        if is_correct:
            aggregate_correct += 1

        if slice_name not in slice_counts:
            slice_counts[slice_name] = [0, 0]

        slice_counts[slice_name][1] += 1

        if is_correct:
            slice_counts[slice_name][0] += 1

    aggregate = round(aggregate_correct / len(rows), 12)

    slice_scores: dict[str, float] = {}

    for slice_name, (correct_count, total_count) in slice_counts.items():
        slice_scores[slice_name] = round(correct_count / total_count, 12)

    return aggregate, slice_scores, True


def candidate_names_from_array(candidates: Any) -> list[str]:
    if not isinstance(candidates, list):
        return []

    names: list[str] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        name = candidate.get("name")

        if is_nonempty_string(name):
            names.append(name)

    return names


def result_order_key(
    result: dict[str, Any],
    order_index: dict[str, int],
) -> tuple[int, bytes]:
    return (
        order_index.get(result["name"], len(order_index)),
        utf8_sort_key(result["name"]),
    )


def make_not_frozen_response(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Response when freezeId does not exist.

    The specification requires NOT_FROZEN. Since no trusted candidate
    manifest exists, artifact-size output must be null.
    """
    policy = payload.get("policy")
    policy_valid = validate_policy(policy)

    if policy_valid:
        order = policy["candidateOrder"]
    else:
        order = candidate_names_from_array(payload.get("candidates"))

    names = candidate_names_from_array(payload.get("candidates"))
    names = list(dict.fromkeys(names))

    order_index = {name: index for index, name in enumerate(order)}

    results: list[dict[str, Any]] = []

    for name in names:
        aggregate, slices, predictions_valid = calculate_candidate_metrics(
            name,
            payload.get("rows"),
        )

        reason_codes = {"NOT_FROZEN"}

        if not predictions_valid:
            aggregate = None
            slices = None
            reason_codes.add("INVALID_PREDICTIONS")

        latency_value = None
        latencies = payload.get("latencies")

        if isinstance(latencies, dict) and is_finite_nonnegative_number(
            latencies.get(name)
        ):
            latency_value = latencies[name]

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": None,
                "latencyMs": latency_value,
                "admitted": False,
                "reasonCodes": sort_reason_codes(reason_codes),
            }
        )

    results.sort(key=lambda result: result_order_key(result, order_index))

    return {
        "freezeId": payload["freezeId"],
        "selected": None,
        "results": results,
        "packageManifest": None,
    }


def handle_select(payload: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    if not validate_select_request(payload):
        return invalid_input_response()

    freeze_id = payload["freezeId"]
    stored_state = FREEZES.get(freeze_id)

    if stored_state is None:
        return make_not_frozen_response(payload)

    stored_response = stored_state["response"]
    stored_candidates = stored_response["candidates"]
    submitted_candidates = payload["candidates"]

    # Required: candidate array must exactly equal stored freeze response.
    exact_lineage = compact_json(submitted_candidates) == compact_json(
        stored_candidates
    )

    stored_by_name = {candidate["name"]: candidate for candidate in stored_candidates}

    submitted_names = candidate_names_from_array(submitted_candidates)
    submitted_name_set = set(submitted_names)

    names_unique = len(submitted_names) == len(submitted_name_set)
    names_match = (
        names_unique
        and submitted_name_set == set(stored_by_name.keys())
        and len(submitted_names) == len(stored_candidates)
    )

    submitted_by_name: dict[str, dict[str, Any]] = {}

    if isinstance(submitted_candidates, list):
        for candidate in submitted_candidates:
            if isinstance(candidate, dict) and is_nonempty_string(
                candidate.get("name")
            ):
                submitted_by_name[candidate["name"]] = candidate

    policy = payload["policy"]
    policy_valid = validate_policy(policy)

    if policy_valid:
        candidate_order = policy["candidateOrder"]
        required_slices = policy["requiredSlices"]
        aggregate_floor = policy["aggregateFloor"]
        max_bytes = policy["maxBytes"]
        max_latency = policy["maxLatencyMs"]
    else:
        candidate_order = []
        required_slices = {}
        aggregate_floor = 0
        max_bytes = 0
        max_latency = 0

    order_index = {name: index for index, name in enumerate(candidate_order)}

    candidate_order_valid = (
        policy_valid
        and set(candidate_order) == set(stored_by_name.keys())
        and len(candidate_order) == len(stored_by_name)
    )

    latencies = payload["latencies"]
    results: list[dict[str, Any]] = []

    for stored_candidate in stored_candidates:
        name = stored_candidate["name"]
        reason_codes: set[str] = set()

        submitted_candidate = submitted_by_name.get(name)

        # -----------------------------------------------------------
        # Candidate lineage and manifest validation
        # -----------------------------------------------------------
        if not exact_lineage or not names_match:
            reason_codes.add("INVALID_LINEAGE")

        if submitted_candidate is None:
            reason_codes.add("INVALID_LINEAGE")
            recomputed_manifest = None
        else:
            recomputed_manifest = recompute_submitted_manifest(submitted_candidate)

        total_bytes: int | None = None

        if recomputed_manifest is None:
            reason_codes.add("INVALID_MANIFEST")
        else:
            inventory, recomputed_total, recomputed_digest = recomputed_manifest
            total_bytes = recomputed_total

            if (
                inventory != stored_candidate["inventory"]
                or recomputed_total != stored_candidate["totalBytes"]
                or recomputed_digest != stored_candidate["packageDigest"]
            ):
                reason_codes.add("INVALID_MANIFEST")

        # Candidate must be a successful frozen candidate.
        if stored_candidate["status"] != "frozen":
            reason_codes.add("NOT_FROZEN")

        # -----------------------------------------------------------
        # Policy and latency validation
        # -----------------------------------------------------------
        if not policy_valid or not candidate_order_valid:
            reason_codes.add("INVALID_POLICY")

        latency_ms: float | int | None = None

        if isinstance(latencies, dict) and is_finite_nonnegative_number(
            latencies.get(name)
        ):
            latency_ms = latencies[name]
        else:
            reason_codes.add("LATENCY_LIMIT")

        # -----------------------------------------------------------
        # Metrics and prediction validation
        # -----------------------------------------------------------
        aggregate, slices, predictions_valid = calculate_candidate_metrics(
            name,
            payload["rows"],
        )

        if not predictions_valid:
            aggregate = None
            slices = None
            reason_codes.add("INVALID_PREDICTIONS")

        # -----------------------------------------------------------
        # Constraint checks
        # -----------------------------------------------------------
        if policy_valid and predictions_valid:
            if aggregate is not None and aggregate < aggregate_floor:
                reason_codes.add("AGGREGATE_FLOOR")

            for slice_name, slice_floor in required_slices.items():
                if slices is None or slice_name not in slices:
                    reason_codes.add(f"MISSING_SLICE:{slice_name}")
                elif slices[slice_name] < slice_floor:
                    reason_codes.add(f"SLICE_FLOOR:{slice_name}")

        if policy_valid and total_bytes is not None and total_bytes > max_bytes:
            reason_codes.add("SIZE_LIMIT")

        if policy_valid and latency_ms is not None and latency_ms > max_latency:
            reason_codes.add("LATENCY_LIMIT")

        admitted = len(reason_codes) == 0

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": total_bytes,
                "latencyMs": latency_ms,
                "admitted": admitted,
                "reasonCodes": sort_reason_codes(reason_codes),
            }
        )

    # Required ordering:
    # 1. candidateOrder
    # 2. UTF-8 candidate name fallback
    results.sort(key=lambda result: result_order_key(result, order_index))

    admitted_results = [result for result in results if result["admitted"]]

    winner: dict[str, Any] | None = None

    if admitted_results:
        winner = min(
            admitted_results,
            key=lambda result: (
                result["totalBytes"],
                result["latencyMs"],
                order_index.get(result["name"], len(order_index)),
                utf8_sort_key(result["name"]),
            ),
        )

    package_manifest = None

    if winner is not None:
        package_manifest = stored_by_name[winner["name"]]

    return {
        "freezeId": freeze_id,
        "selected": winner["name"] if winner is not None else None,
        "results": results,
        "packageManifest": package_manifest,
    }


# -------------------------------------------------------------------
# API
# -------------------------------------------------------------------


@app.post("/quantize")
async def quantize(request: Request) -> JSONResponse:
    """
    Main two-phase stateful API endpoint.

    POST /quantize
    Content-Type: application/json
    """
    try:
        payload = await request.json()
    except Exception:
        return invalid_input_response()

    if not isinstance(payload, dict):
        return invalid_input_response()

    phase = payload.get("phase")

    if phase == "freeze":
        response = handle_freeze(payload)
    elif phase == "select":
        response = handle_select(payload)
    else:
        return invalid_input_response()

    if isinstance(response, JSONResponse):
        return response

    return JSONResponse(
        status_code=200,
        content=response,
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "quantize-candidate-admission",
        "endpoint": "POST /quantize",
    }
