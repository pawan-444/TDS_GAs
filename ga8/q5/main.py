import hashlib
import json
import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MAX_SAFE_INTEGER = (2**53) - 1

# In-memory freeze storage.
# Note: This remains available while the deployed process is running.
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def is_safe_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def is_finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def is_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def is_binary(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def sort_codes(codes: set[str]) -> list[str]:
    return sorted(codes, key=utf8_key)


def invalid_input() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


def unique_nonempty_strings(value: Any) -> bool:
    """
    Empty lists are valid.

    []                       -> valid
    ["reason_a"]             -> valid
    ["reason_a", "reason_b"] -> valid
    [""]                     -> invalid
    ["x", "x"]               -> invalid
    """
    return (
        isinstance(value, list)
        and all(is_nonempty_string(item) for item in value)
        and len(value) == len(set(value))
    )


# ============================================================
# FREEZE PHASE
# ============================================================


def create_inventory_from_files(
    files: Any,
) -> tuple[list[dict[str, Any]], int, str] | None:
    """
    Creates exact inventory from the UTF-8 file contents.

    Returns:
        inventory, totalBytes, packageDigest

    Invalid files return None.
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
    Validates only the whole freeze request.

    Invalid candidate files do NOT reject the whole request.
    Instead, that individual candidate becomes:
      status: invalid
      inventory: []
      totalBytes: null
      packageDigest: null
    """
    if not isinstance(payload, dict):
        return False

    required_fields = [
        "phase",
        "freezeId",
        "calibrationDigest",
        "tokenizerDigest",
        "allowedUnsupportedReasons",
        "candidates",
    ]

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

    # Empty list [] is allowed.
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


def make_frozen_candidate(
    candidate: dict[str, Any],
    calibration_digest: str,
    tokenizer_digest: str,
    allowed_reasons: set[str],
) -> dict[str, Any]:
    reason_codes: set[str] = set()

    name = candidate["name"]

    file_result = create_inventory_from_files(candidate.get("files"))

    if file_result is None:
        inventory: list[dict[str, Any]] = []
        total_bytes: int | None = None
        package_digest: str | None = None
        files_valid = False
    else:
        inventory, total_bytes, package_digest = file_result
        files_valid = True

    unsupported_reason = candidate.get("unsupportedReason")

    if unsupported_reason is not None:
        if (
            not is_nonempty_string(unsupported_reason)
            or unsupported_reason not in allowed_reasons
        ):
            status = "invalid"
            reason_codes.add("UNALLOWED_UNSUPPORTED_REASON")
        else:
            status = "unsupported"

    else:
        status = "frozen"

        if candidate.get("loadable") is not True:
            reason_codes.add("NOT_LOADABLE")

        if candidate.get("calibrationDigest") != calibration_digest:
            reason_codes.add("CALIBRATION_MISMATCH")

        if candidate.get("tokenizerDigest") != tokenizer_digest:
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
        "reasonCodes": sort_codes(reason_codes),
    }


def freeze_phase(payload: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    if not validate_freeze_request(payload):
        return invalid_input()

    freeze_id = payload["freezeId"]

    # Exact request identity for idempotent replays.
    fingerprint = compact_json(payload)

    if freeze_id in FREEZES:
        old = FREEZES[freeze_id]

        if old["fingerprint"] == fingerprint:
            return old["response"]

        return JSONResponse(
            status_code=409,
            content={"error": "FREEZE_ID_CONFLICT"},
        )

    allowed_reasons = set(payload["allowedUnsupportedReasons"])

    candidates = []

    for candidate in payload["candidates"]:
        candidates.append(
            make_frozen_candidate(
                candidate=candidate,
                calibration_digest=payload["calibrationDigest"],
                tokenizer_digest=payload["tokenizerDigest"],
                allowed_reasons=allowed_reasons,
            )
        )

    candidates.sort(key=lambda candidate: utf8_key(candidate["name"]))

    response = {
        "freezeId": freeze_id,
        "candidates": candidates,
    }

    FREEZES[freeze_id] = {
        "fingerprint": fingerprint,
        "response": response,
    }

    return response


# ============================================================
# SELECT PHASE
# ============================================================


def validate_select_request(payload: Any) -> bool:
    """
    Required:
      freezeId: non-empty string
      candidates: array
      rows: array
      policy: object

    latencies is intentionally NOT required here.
    If latency is absent/invalid for a candidate:
      latencyMs = null
      reasonCodes includes LATENCY_LIMIT
    """
    if not isinstance(payload, dict):
        return False

    if not is_nonempty_string(payload.get("freezeId")):
        return False

    if not isinstance(payload.get("candidates"), list):
        return False

    if not isinstance(payload.get("rows"), list):
        return False

    if not isinstance(payload.get("policy"), dict):
        return False

    return True


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False

    needed = [
        "maxBytes",
        "aggregateFloor",
        "requiredSlices",
        "maxLatencyMs",
        "candidateOrder",
    ]

    if any(key not in policy for key in needed):
        return False

    if not is_safe_nonnegative_integer(policy.get("maxBytes")):
        return False

    if not is_probability(policy.get("aggregateFloor")):
        return False

    if not is_finite_nonnegative(policy.get("maxLatencyMs")):
        return False

    required_slices = policy.get("requiredSlices")

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():
        if not is_nonempty_string(name):
            return False

        if not is_probability(floor):
            return False

    # Candidate order may be [] only when there are no candidates.
    if not unique_nonempty_strings(policy.get("candidateOrder")):
        return False

    return True


def recompute_manifest_from_inventory(
    candidate: Any,
) -> tuple[list[dict[str, Any]], int, str] | None:
    """
    Selection sends candidate objects from the previous freeze response.

    The submitted candidate has:
      inventory
      totalBytes
      packageDigest

    It does NOT have raw files.

    Therefore:
      - recompute total bytes using inventory bytes
      - recompute package digest from compact inventory JSON
      - never trust candidate.totalBytes or candidate.packageDigest
    """
    if not isinstance(candidate, dict):
        return None

    inventory = candidate.get("inventory")

    if not isinstance(inventory, list) or len(inventory) == 0:
        return None

    expected_keys = ["name", "bytes", "sha256"]
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()

    for entry in inventory:
        if not isinstance(entry, dict):
            return None

        if list(entry.keys()) != expected_keys:
            return None

        name = entry.get("name")
        byte_count = entry.get("bytes")
        digest = entry.get("sha256")

        if not is_nonempty_string(name):
            return None

        if name in names:
            return None

        if not is_safe_nonnegative_integer(byte_count):
            return None

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            return None

        names.add(name)

        normalized.append(
            {
                "name": name,
                "bytes": byte_count,
                "sha256": digest,
            }
        )

    expected_order = sorted(
        normalized,
        key=lambda entry: utf8_key(entry["name"]),
    )

    if normalized != expected_order:
        return None

    total_bytes = sum(entry["bytes"] for entry in normalized)
    package_digest = sha256_text(compact_json(normalized))

    return normalized, total_bytes, package_digest


def calculate_metrics(
    candidate_name: str,
    rows: Any,
) -> tuple[float | None, dict[str, float] | None, bool]:
    """
    Calculates aggregate and per-slice accuracy.

    All labels and predictions must be binary 0 or 1.
    Invalid predictions return:
      aggregate = null
      slices = null
      valid = False
    """
    if not isinstance(rows, list) or len(rows) == 0:
        return None, None, False

    correct_total = 0
    slice_counts: dict[str, list[int]] = {}

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
            correct_total += 1

        if slice_name not in slice_counts:
            slice_counts[slice_name] = [0, 0]

        slice_counts[slice_name][1] += 1

        if correct:
            slice_counts[slice_name][0] += 1

    aggregate = round(correct_total / len(rows), 12)

    slice_scores: dict[str, float] = {}

    for slice_name, values in slice_counts.items():
        correct_count = values[0]
        total_count = values[1]

        slice_scores[slice_name] = round(
            correct_count / total_count,
            12,
        )

    return aggregate, slice_scores, True


def get_latency(
    payload: dict[str, Any],
    candidate_name: str,
) -> int | float | None:
    latencies = payload.get("latencies")

    if not isinstance(latencies, dict):
        return None

    value = latencies.get(candidate_name)

    if not is_finite_nonnegative(value):
        return None

    return value


def selection_sort_key(
    result: dict[str, Any],
    order_index: dict[str, int],
) -> tuple[int, bytes]:
    return (
        order_index.get(result["name"], len(order_index)),
        utf8_key(result["name"]),
    )


def select_phase(payload: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    if not validate_select_request(payload):
        return invalid_input()

    freeze_id = payload["freezeId"]

    # No recorded freeze: normal HTTP 200 response with NOT_FROZEN.
    if freeze_id not in FREEZES:
        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": [],
            "packageManifest": None,
        }

    stored_response = FREEZES[freeze_id]["response"]
    stored_candidates = stored_response["candidates"]
    submitted_candidates = payload["candidates"]

    stored_by_name = {candidate["name"]: candidate for candidate in stored_candidates}

    # Must exactly equal stored frozen response candidate array.
    exact_lineage = compact_json(submitted_candidates) == compact_json(
        stored_candidates
    )

    policy = payload["policy"]
    policy_valid = validate_policy(policy)

    if policy_valid:
        candidate_order = policy["candidateOrder"]
        max_bytes = policy["maxBytes"]
        aggregate_floor = policy["aggregateFloor"]
        required_slices = policy["requiredSlices"]
        max_latency = policy["maxLatencyMs"]
    else:
        candidate_order = []
        max_bytes = 0
        aggregate_floor = 0
        required_slices = {}
        max_latency = 0

    order_index = {name: index for index, name in enumerate(candidate_order)}

    stored_names = set(stored_by_name.keys())

    candidate_order_valid = (
        policy_valid
        and set(candidate_order) == stored_names
        and len(candidate_order) == len(stored_names)
    )

    results: list[dict[str, Any]] = []

    for frozen_candidate in stored_candidates:
        name = frozen_candidate["name"]
        codes: set[str] = set()

        # Candidate lineage must be exact.
        if not exact_lineage:
            codes.add("INVALID_LINEAGE")

        # Recompute inventory-derived data.
        manifest = recompute_manifest_from_inventory(frozen_candidate)

        total_bytes: int | None = None

        if manifest is None:
            codes.add("INVALID_MANIFEST")
        else:
            recomputed_inventory, recomputed_total, recomputed_digest = manifest
            total_bytes = recomputed_total

            if (
                recomputed_inventory != frozen_candidate["inventory"]
                or recomputed_total != frozen_candidate["totalBytes"]
                or recomputed_digest != frozen_candidate["packageDigest"]
            ):
                codes.add("INVALID_MANIFEST")

        # Only candidate status "frozen" can be admitted.
        if frozen_candidate["status"] != "frozen":
            codes.add("NOT_FROZEN")

        if not policy_valid or not candidate_order_valid:
            codes.add("INVALID_POLICY")

        latency_ms = get_latency(payload, name)

        if latency_ms is None:
            codes.add("LATENCY_LIMIT")

        aggregate, slices, predictions_valid = calculate_metrics(
            name,
            payload["rows"],
        )

        if not predictions_valid:
            aggregate = None
            slices = None
            codes.add("INVALID_PREDICTIONS")

        if policy_valid and predictions_valid:
            if aggregate is not None and aggregate < aggregate_floor:
                codes.add("AGGREGATE_FLOOR")

            for slice_name, floor in required_slices.items():
                if slices is None or slice_name not in slices:
                    codes.add(f"MISSING_SLICE:{slice_name}")
                elif slices[slice_name] < floor:
                    codes.add(f"SLICE_FLOOR:{slice_name}")

        if policy_valid and total_bytes is not None and total_bytes > max_bytes:
            codes.add("SIZE_LIMIT")

        if policy_valid and latency_ms is not None and latency_ms > max_latency:
            codes.add("LATENCY_LIMIT")

        admitted = len(codes) == 0

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": total_bytes,
                "latencyMs": latency_ms,
                "admitted": admitted,
                "reasonCodes": sort_codes(codes),
            }
        )

    results.sort(
        key=lambda result: selection_sort_key(
            result,
            order_index,
        )
    )

    admitted_candidates = [result for result in results if result["admitted"]]

    winner = None

    if admitted_candidates:
        winner = min(
            admitted_candidates,
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
# API
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
        response = freeze_phase(payload)

    elif phase == "select":
        response = select_phase(payload)

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
