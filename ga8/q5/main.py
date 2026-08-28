import hashlib
import json
import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MAX_SAFE_INTEGER = (2**53) - 1

# In-memory state:
# freezeId -> {
#   "fingerprint": exact canonical freeze request,
#   "response": exact stored freeze response
# }
FREEZES: dict[str, dict[str, Any]] = {}


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def is_binary(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def sorted_codes(codes: set[str]) -> list[str]:
    return sorted(codes, key=utf8_key)


def invalid_input() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


def validate_unique_nonempty_strings(value: Any) -> bool:
    """
    Empty lists are valid.

    Examples:
      []                    -> valid
      ["gpu_unsupported"]   -> valid
      ["", "x"]             -> invalid
      ["x", "x"]            -> invalid
    """
    return (
        isinstance(value, list)
        and all(is_nonempty_string(item) for item in value)
        and len(value) == len(set(value))
    )


# ============================================================
# FREEZE PHASE
# ============================================================


def inventory_from_files(
    files: Any,
) -> tuple[list[dict[str, Any]], int, str] | None:
    """
    Build inventory from raw candidate files.

    Returns:
      inventory, totalBytes, packageDigest

    Returns None if the files map is invalid.
    """
    if not isinstance(files, dict) or len(files) == 0:
        return None

    inventory: list[dict[str, Any]] = []

    for filename, content in files.items():
        if not is_nonempty_string(filename):
            return None

        if not isinstance(content, str):
            return None

        try:
            raw = content.encode("utf-8")
        except UnicodeEncodeError:
            return None

        inventory.append(
            {
                "name": filename,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )

    inventory.sort(key=lambda item: utf8_key(item["name"]))

    total_bytes = sum(item["bytes"] for item in inventory)

    package_digest = sha256_text(compact_json(inventory))

    return inventory, total_bytes, package_digest


def validate_freeze_request(payload: Any) -> bool:
    """
    Whole request validation.

    Candidate `files` validity is not checked here because invalid files
    must create an invalid candidate instead of rejecting the full request.
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

    if not is_nonempty_string(freeze_id):
        return False

    if len(freeze_id) > 128:
        return False

    if not is_nonempty_string(payload.get("calibrationDigest")):
        return False

    if not is_nonempty_string(payload.get("tokenizerDigest")):
        return False

    # [] is valid here.
    if not validate_unique_nonempty_strings(payload.get("allowedUnsupportedReasons")):
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


def build_frozen_candidate(
    candidate: dict[str, Any],
    expected_calibration_digest: str,
    expected_tokenizer_digest: str,
    allowed_reasons: set[str],
) -> dict[str, Any]:
    name = candidate["name"]
    reason_codes: set[str] = set()

    inventory_result = inventory_from_files(candidate.get("files"))

    if inventory_result is None:
        inventory: list[dict[str, Any]] = []
        total_bytes: int | None = None
        package_digest: str | None = None
        files_valid = False
    else:
        inventory, total_bytes, package_digest = inventory_result
        files_valid = True

    unsupported_reason = candidate.get("unsupportedReason")

    if unsupported_reason is not None:
        if (
            not is_nonempty_string(unsupported_reason)
            or unsupported_reason not in allowed_reasons
        ):
            reason_codes.add("UNALLOWED_UNSUPPORTED_REASON")
            status = "invalid"
        else:
            status = "unsupported"

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

    if not files_valid:
        status = "invalid"

    return {
        "name": name,
        "status": status,
        "inventory": inventory,
        "totalBytes": total_bytes,
        "packageDigest": package_digest,
        "reasonCodes": sorted_codes(reason_codes),
    }


def handle_freeze(payload: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    if not validate_freeze_request(payload):
        return invalid_input()

    freeze_id = payload["freezeId"]
    fingerprint = compact_json(payload)

    existing = FREEZES.get(freeze_id)

    if existing is not None:
        if existing["fingerprint"] == fingerprint:
            return existing["response"]

        return JSONResponse(
            status_code=409,
            content={"error": "FREEZE_ID_CONFLICT"},
        )

    allowed_reasons = set(payload["allowedUnsupportedReasons"])

    frozen_candidates = []

    for candidate in payload["candidates"]:
        frozen_candidates.append(
            build_frozen_candidate(
                candidate=candidate,
                expected_calibration_digest=payload["calibrationDigest"],
                expected_tokenizer_digest=payload["tokenizerDigest"],
                allowed_reasons=allowed_reasons,
            )
        )

    frozen_candidates.sort(key=lambda candidate: utf8_key(candidate["name"]))

    response = {
        "freezeId": freeze_id,
        "candidates": frozen_candidates,
    }

    FREEZES[freeze_id] = {
        "fingerprint": fingerprint,
        "response": response,
    }

    return response


# ============================================================
# SELECT PHASE
# ============================================================


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

    if not validate_unique_nonempty_strings(candidate_order):
        return False

    return True


def validate_select_request(payload: Any) -> bool:
    """
    The question requires candidates and rows to be arrays.
    They do not have to be non-empty arrays.
    """
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


def recompute_manifest_from_inventory(
    candidate: Any,
) -> tuple[list[dict[str, Any]], int, str] | None:
    """
    Select phase receives frozen candidate response objects.

    It does NOT receive the original `files` map. Therefore it recomputes:
      totalBytes from inventory bytes
      packageDigest from compact JSON inventory
    """
    if not isinstance(candidate, dict):
        return None

    inventory = candidate.get("inventory")

    if not isinstance(inventory, list) or len(inventory) == 0:
        return None

    expected_keys = ["name", "bytes", "sha256"]
    seen_names: set[str] = set()
    normalized_inventory: list[dict[str, Any]] = []

    for item in inventory:
        if not isinstance(item, dict):
            return None

        if list(item.keys()) != expected_keys:
            return None

        name = item.get("name")
        byte_count = item.get("bytes")
        digest = item.get("sha256")

        if not is_nonempty_string(name):
            return None

        if name in seen_names:
            return None

        if not is_safe_nonnegative_integer(byte_count):
            return None

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None

        seen_names.add(name)

        normalized_inventory.append(
            {
                "name": name,
                "bytes": byte_count,
                "sha256": digest,
            }
        )

    sorted_inventory = sorted(
        normalized_inventory,
        key=lambda item: utf8_key(item["name"]),
    )

    if normalized_inventory != sorted_inventory:
        return None

    total_bytes = sum(item["bytes"] for item in normalized_inventory)

    package_digest = sha256_text(compact_json(normalized_inventory))

    return normalized_inventory, total_bytes, package_digest


def calculate_metrics(
    candidate_name: str,
    rows: Any,
) -> tuple[float | None, dict[str, float] | None, bool]:
    """
    Returns:
      aggregateAccuracy,
      perSliceAccuracies,
      validPredictions
    """
    if not isinstance(rows, list) or len(rows) == 0:
        return None, None, False

    total_correct = 0
    slices: dict[str, list[int]] = {}

    for row in rows:
        if not isinstance(row, dict):
            return None, None, False

        label = row.get("label")
        slice_name = row.get("slice")
        predictions = row.get("predictions")

        if not is_binary(label):
            return None, None, False

        if not is_nonempty_string(slice_name):
            return None, None, False

        if not isinstance(predictions, dict):
            return None, None, False

        prediction = predictions.get(candidate_name)

        if not is_binary(prediction):
            return None, None, False

        correct = prediction == label

        if correct:
            total_correct += 1

        if slice_name not in slices:
            slices[slice_name] = [0, 0]

        slices[slice_name][1] += 1

        if correct:
            slices[slice_name][0] += 1

    aggregate = round(total_correct / len(rows), 12)

    slice_results: dict[str, float] = {}

    for slice_name, values in slices.items():
        correct_count, total_count = values
        slice_results[slice_name] = round(
            correct_count / total_count,
            12,
        )

    return aggregate, slice_results, True


def result_sort_key(
    result: dict[str, Any],
    order_index: dict[str, int],
) -> tuple[int, bytes]:
    return (
        order_index.get(result["name"], len(order_index)),
        utf8_key(result["name"]),
    )


def handle_missing_freeze(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Return valid select response if the freezeId was never stored.
    """
    candidates = payload["candidates"]
    names: list[str] = []

    for candidate in candidates:
        if isinstance(candidate, dict) and is_nonempty_string(candidate.get("name")):
            if candidate["name"] not in names:
                names.append(candidate["name"])

    policy = payload["policy"]

    if validate_policy(policy):
        candidate_order = policy["candidateOrder"]
    else:
        candidate_order = names

    order_index = {name: index for index, name in enumerate(candidate_order)}

    results = []

    for name in names:
        aggregate, slices, valid_predictions = calculate_metrics(
            name,
            payload["rows"],
        )

        reason_codes = {"NOT_FROZEN"}

        if not valid_predictions:
            aggregate = None
            slices = None
            reason_codes.add("INVALID_PREDICTIONS")

        latency_value = payload["latencies"].get(name)
        latency_ms = (
            latency_value if is_finite_nonnegative_number(latency_value) else None
        )

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": None,
                "latencyMs": latency_ms,
                "admitted": False,
                "reasonCodes": sorted_codes(reason_codes),
            }
        )

    results.sort(key=lambda result: result_sort_key(result, order_index))

    return {
        "freezeId": payload["freezeId"],
        "selected": None,
        "results": results,
        "packageManifest": None,
    }


def handle_select(payload: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    if not validate_select_request(payload):
        return invalid_input()

    freeze_id = payload["freezeId"]
    stored_state = FREEZES.get(freeze_id)

    if stored_state is None:
        return handle_missing_freeze(payload)

    stored_response = stored_state["response"]
    stored_candidates = stored_response["candidates"]
    submitted_candidates = payload["candidates"]

    # Candidate input must exactly match the earlier stored freeze response.
    exact_lineage = compact_json(submitted_candidates) == compact_json(
        stored_candidates
    )

    stored_by_name = {candidate["name"]: candidate for candidate in stored_candidates}

    submitted_by_name: dict[str, dict[str, Any]] = {}
    submitted_names: list[str] = []
    duplicate_name = False
    invalid_candidate_object = False

    for candidate in submitted_candidates:
        if not isinstance(candidate, dict):
            invalid_candidate_object = True
            continue

        name = candidate.get("name")

        if not is_nonempty_string(name):
            invalid_candidate_object = True
            continue

        if name in submitted_by_name:
            duplicate_name = True

        submitted_by_name[name] = candidate
        submitted_names.append(name)

    candidate_names_match = (
        not invalid_candidate_object
        and not duplicate_name
        and set(submitted_names) == set(stored_by_name)
        and len(submitted_names) == len(stored_candidates)
    )

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
        and set(candidate_order) == set(stored_by_name)
        and len(candidate_order) == len(stored_by_name)
    )

    results: list[dict[str, Any]] = []

    for frozen_candidate in stored_candidates:
        name = frozen_candidate["name"]
        reason_codes: set[str] = set()

        submitted_candidate = submitted_by_name.get(name)

        if not exact_lineage or not candidate_names_match:
            reason_codes.add("INVALID_LINEAGE")

        if submitted_candidate is None:
            reason_codes.add("INVALID_LINEAGE")
            manifest = None
        else:
            manifest = recompute_manifest_from_inventory(submitted_candidate)

        total_bytes: int | None = None

        if manifest is None:
            reason_codes.add("INVALID_MANIFEST")
        else:
            inventory, recomputed_total, recomputed_digest = manifest
            total_bytes = recomputed_total

            if (
                inventory != frozen_candidate["inventory"]
                or recomputed_total != frozen_candidate["totalBytes"]
                or recomputed_digest != frozen_candidate["packageDigest"]
            ):
                reason_codes.add("INVALID_MANIFEST")

        # Only a valid "frozen" candidate is eligible for admission.
        if frozen_candidate["status"] != "frozen":
            reason_codes.add("NOT_FROZEN")

        if not policy_valid or not candidate_order_valid:
            reason_codes.add("INVALID_POLICY")

        latency_value = payload["latencies"].get(name)

        if is_finite_nonnegative_number(latency_value):
            latency_ms: int | float | None = latency_value
        else:
            latency_ms = None
            reason_codes.add("LATENCY_LIMIT")

        aggregate, slices, predictions_valid = calculate_metrics(
            name,
            payload["rows"],
        )

        if not predictions_valid:
            aggregate = None
            slices = None
            reason_codes.add("INVALID_PREDICTIONS")

        if policy_valid and predictions_valid:
            if aggregate is not None and aggregate < aggregate_floor:
                reason_codes.add("AGGREGATE_FLOOR")

            for slice_name, required_floor in required_slices.items():
                if slices is None or slice_name not in slices:
                    reason_codes.add(f"MISSING_SLICE:{slice_name}")
                elif slices[slice_name] < required_floor:
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
                "reasonCodes": sorted_codes(reason_codes),
            }
        )

    results.sort(key=lambda result: result_sort_key(result, order_index))

    admitted_results = [result for result in results if result["admitted"]]

    winner = None

    if admitted_results:
        winner = min(
            admitted_results,
            key=lambda result: (
                result["totalBytes"],
                result["latencyMs"],
                order_index.get(result["name"], len(order_index)),
                utf8_key(result["name"]),
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


# ============================================================
# API ENDPOINTS
# ============================================================


@app.post("/quantize")
async def quantize(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(payload, dict):
        return invalid_input()

    phase = payload.get("phase")

    if phase == "freeze":
        response = handle_freeze(payload)

    elif phase == "select":
        response = handle_select(payload)

    else:
        return invalid_input()

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
