import hashlib
import json
import math
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Stateful freeze store.
# Key: freezeId
# Value: exact freeze response + canonical request fingerprint.
FREEZES: dict[str, dict[str, Any]] = {}

MAX_SAFE_INTEGER = 2**53 - 1


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def safe_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def finite_number_01(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def sort_codes(codes: set[str]) -> list[str]:
    return sorted(codes, key=utf8_key)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_inventory(files: Any):
    """
    Build the immutable manifest for a candidate.

    Returns:
        (inventory, total_bytes, package_digest)
    or None when the files object is invalid.
    """
    if not isinstance(files, dict) or not files:
        return None

    inventory = []

    for filename, content in files.items():
        if not nonempty_string(filename):
            return None

        if not isinstance(content, str):
            return None

        raw = content.encode("utf-8")

        inventory.append(
            {
                "name": filename,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )

    inventory.sort(key=lambda item: utf8_key(item["name"]))

    total_bytes = sum(item["bytes"] for item in inventory)

    package_digest = sha256_bytes(compact_json(inventory).encode("utf-8"))

    return inventory, total_bytes, package_digest


def candidate_fingerprint(candidate: dict[str, Any]) -> str:
    return compact_json(candidate)


def freeze_fingerprint(payload: dict[str, Any]) -> str:
    # phase is not part of the freeze data identity.
    value = dict(payload)
    value.pop("phase", None)
    return compact_json(value)


def validate_unique_nonempty_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(nonempty_string(item) for item in value)
        and len(value) == len(set(value))
    )


def build_frozen_candidate(
    candidate: dict[str, Any],
    request_calibration: str,
    request_tokenizer: str,
    allowed_reasons: set[str],
) -> dict[str, Any]:
    name = candidate.get("name")

    inventory_result = file_inventory(candidate.get("files"))

    if inventory_result is None:
        inventory = []
        total_bytes = None
        package_digest = None
        manifest_valid = False
    else:
        inventory, total_bytes, package_digest = inventory_result
        manifest_valid = True

    reasons: set[str] = set()

    unsupported_reason = candidate.get("unsupportedReason")

    # Unsupported candidates are validly unsupported only when the
    # supplied reason is one of the explicitly allowed reasons.
    if unsupported_reason is not None:
        if (
            not isinstance(unsupported_reason, str)
            or not unsupported_reason
            or unsupported_reason not in allowed_reasons
        ):
            reasons.add("UNALLOWED_UNSUPPORTED_REASON")
            status = "invalid"
        else:
            status = "unsupported"
    else:
        status = "frozen"

        if candidate.get("loadable") is not True:
            reasons.add("NOT_LOADABLE")

        if candidate.get("calibrationDigest") != request_calibration:
            reasons.add("CALIBRATION_MISMATCH")

        if candidate.get("tokenizerDigest") != request_tokenizer:
            reasons.add("TOKENIZER_MISMATCH")

        if reasons:
            status = "invalid"

    if not manifest_valid:
        status = "invalid"

    return {
        "name": name,
        "status": status,
        "inventory": inventory,
        "totalBytes": total_bytes,
        "packageDigest": package_digest,
        "reasonCodes": sort_codes(reasons),
    }


def validate_freeze_request(payload: dict[str, Any]) -> bool:
    required = (
        "phase",
        "freezeId",
        "calibrationDigest",
        "tokenizerDigest",
        "allowedUnsupportedReasons",
        "candidates",
    )

    if any(key not in payload for key in required):
        return False

    freeze_id = payload["freezeId"]

    if not isinstance(freeze_id, str) or not (1 <= len(freeze_id) <= 128):
        return False

    if not nonempty_string(payload["calibrationDigest"]):
        return False

    if not nonempty_string(payload["tokenizerDigest"]):
        return False

    if not validate_unique_nonempty_strings(payload["allowedUnsupportedReasons"]):
        return False

    candidates = payload["candidates"]

    if not isinstance(candidates, list) or not candidates:
        return False

    names = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False

        name = candidate.get("name")
        if not nonempty_string(name):
            return False

        names.append(name)

        # The files field must be an object with unique filenames mapped
        # to UTF-8 strings. JSON object keys are inherently unique.
        files = candidate.get("files")
        if not isinstance(files, dict) or not files:
            # This is a candidate-level invalid manifest, not an invalid
            # whole freeze request.
            continue

        for filename, content in files.items():
            if not nonempty_string(filename) or not isinstance(content, str):
                # Also candidate-level invalid manifest.
                break

    return len(names) == len(set(names))


def validate_select_request(payload: dict[str, Any]) -> bool:
    required = (
        "phase",
        "freezeId",
        "candidates",
        "policy",
        "latencies",
        "rows",
    )

    if any(key not in payload for key in required):
        return False

    if not nonempty_string(payload["freezeId"]):
        return False

    if not isinstance(payload["candidates"], list):
        return False

    if not isinstance(payload["rows"], list):
        return False

    if not isinstance(payload["policy"], dict):
        return False

    if not isinstance(payload["latencies"], dict):
        return False

    # The prompt explicitly requires candidates and rows arrays.
    # It also requires policy to be an object.
    if not payload["candidates"] or not payload["rows"]:
        return False

    return True


def recompute_manifest(candidate: dict[str, Any]):
    result = file_inventory(candidate.get("files"))
    if result is None:
        return None
    return result


def validate_policy(policy: dict[str, Any]):
    required = (
        "maxBytes",
        "aggregateFloor",
        "requiredSlices",
        "maxLatencyMs",
        "candidateOrder",
    )

    if any(key not in policy for key in required):
        return False

    if not safe_nonnegative_integer(policy["maxBytes"]):
        return False

    if not finite_number_01(policy["aggregateFloor"]):
        return False

    if not isinstance(policy["requiredSlices"], dict):
        return False

    for slice_name, floor in policy["requiredSlices"].items():
        if not nonempty_string(slice_name):
            return False
        if not finite_number_01(floor):
            return False

    if not finite_nonnegative_number(policy["maxLatencyMs"]):
        return False

    if not isinstance(policy["candidateOrder"], list):
        return False

    if any(not nonempty_string(x) for x in policy["candidateOrder"]):
        return False

    if len(policy["candidateOrder"]) != len(set(policy["candidateOrder"])):
        return False

    return True


def valid_binary_prediction(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def calculate_metrics(
    candidate_name: str,
    rows: list[Any],
):
    """
    Returns:
        aggregate, slices, valid_predictions
    """
    if not rows:
        return None, None, False

    correct = 0
    slice_counts: dict[str, list[int]] = {}

    for row in rows:
        if not isinstance(row, dict):
            return None, None, False

        label = row.get("label")
        predictions = row.get("predictions")

        if not valid_binary_prediction(label):
            return None, None, False

        if not isinstance(predictions, dict):
            return None, None, False

        if candidate_name not in predictions:
            return None, None, False

        prediction = predictions[candidate_name]

        if not valid_binary_prediction(prediction):
            return None, None, False

        if prediction == label:
            correct += 1

        slice_name = row.get("slice")
        if not isinstance(slice_name, str):
            return None, None, False

        if slice_name not in slice_counts:
            slice_counts[slice_name] = [0, 0]

        slice_counts[slice_name][1] += 1

        if prediction == label:
            slice_counts[slice_name][0] += 1

    aggregate = round(correct / len(rows), 12)

    slices = {}
    for slice_name, (correct_count, total) in slice_counts.items():
        slices[slice_name] = round(correct_count / total, 12)

    return aggregate, slices, True


def select_phase_response(
    stored_response: dict[str, Any],
    payload: dict[str, Any],
):
    stored_candidates = stored_response["candidates"]
    submitted_candidates = payload["candidates"]

    stored_by_name = {candidate["name"]: candidate for candidate in stored_candidates}

    submitted_by_name = {}
    submitted_names_valid = True

    for candidate in submitted_candidates:
        if not isinstance(candidate, dict):
            submitted_names_valid = False
            break
        name = candidate.get("name")
        if not nonempty_string(name) or name in submitted_by_name:
            submitted_names_valid = False
            break
        submitted_by_name[name] = candidate

    results = []

    policy = payload["policy"]
    policy_valid = validate_policy(policy)

    policy_candidate_order = (
        policy.get("candidateOrder") if isinstance(policy, dict) else []
    )

    order_index = {name: index for index, name in enumerate(policy_candidate_order)}

    names_match = (
        submitted_names_valid
        and set(submitted_by_name) == set(stored_by_name)
        and len(submitted_by_name) == len(stored_by_name)
    )

    # Candidate array must exactly equal the stored response candidates.
    # Compare compact JSON after requiring the same candidate objects.
    exact_candidate_array = compact_json(submitted_candidates) == compact_json(
        stored_candidates
    )

    # Recompute candidate set/order constraints independently.
    stored_names = set(stored_by_name)
    policy_names = (
        set(policy_candidate_order)
        if isinstance(policy_candidate_order, list)
        else set()
    )

    candidate_order_matches = (
        policy_valid
        and policy_names == stored_names
        and len(policy_candidate_order) == len(stored_names)
    )

    for stored_candidate in stored_candidates:
        name = stored_candidate["name"]

        reasons: set[str] = set()

        submitted_candidate = submitted_by_name.get(name)

        aggregate = None
        slices = None
        predictions_valid = False

        if submitted_candidate is None:
            reasons.add("INVALID_LINEAGE")
        else:
            # Recompute submitted manifest and compare it to the frozen
            # manifest. Never trust submitted totals/digest.
            recomputed = recompute_manifest(submitted_candidate)

            if recomputed is None:
                reasons.add("INVALID_MANIFEST")
            else:
                inventory, total_bytes, package_digest = recomputed

                frozen_inventory = stored_candidate["inventory"]
                frozen_total = stored_candidate["totalBytes"]
                frozen_digest = stored_candidate["packageDigest"]

                if (
                    inventory != frozen_inventory
                    or total_bytes != frozen_total
                    or package_digest != frozen_digest
                ):
                    reasons.add("INVALID_MANIFEST")

        if not exact_candidate_array:
            reasons.add("INVALID_LINEAGE")

        if not names_match:
            reasons.add("INVALID_LINEAGE")

        if not candidate_order_matches:
            reasons.add("INVALID_POLICY")

        if not policy_valid:
            reasons.add("INVALID_POLICY")

        # Metrics are calculated from the fresh rows.
        aggregate, slices, predictions_valid = calculate_metrics(
            name,
            payload["rows"],
        )

        if not predictions_valid:
            aggregate = None
            slices = None
            reasons.add("INVALID_PREDICTIONS")

        # Size can only be validated from the stored/recomputed manifest.
        total_bytes_out = None
        if submitted_candidate is not None:
            recomputed = recompute_manifest(submitted_candidate)
            if recomputed is not None:
                total_bytes_out = recomputed[1]

        if total_bytes_out is None:
            reasons.add("INVALID_MANIFEST")

        # Latency can only be validated when the submitted mapping has
        # a finite non-negative value.
        latency_value = payload["latencies"].get(name)
        latency_out = None

        if finite_nonnegative_number(latency_value):
            latency_out = latency_value
        else:
            # Invalid latency is represented by null and prevents admission.
            reasons.add("LATENCY_LIMIT")

        # A candidate is eligible only when its stored status is frozen.
        if stored_candidate["status"] != "frozen":
            reasons.add("NOT_FROZEN")

        # Apply prediction floors only when the policy itself is valid.
        if predictions_valid and policy_valid:
            if aggregate < policy["aggregateFloor"]:
                reasons.add("AGGREGATE_FLOOR")

            required_slices = policy["requiredSlices"]

            for slice_name, floor in required_slices.items():
                if slices is None or slice_name not in slices:
                    reasons.add(f"MISSING_SLICE:{slice_name}")
                elif slices[slice_name] < floor:
                    reasons.add(f"SLICE_FLOOR:{slice_name}")

        # Size and latency are inclusive.
        if (
            total_bytes_out is not None
            and policy_valid
            and total_bytes_out > policy["maxBytes"]
        ):
            reasons.add("SIZE_LIMIT")

        if (
            latency_out is not None
            and policy_valid
            and latency_out > policy["maxLatencyMs"]
        ):
            reasons.add("LATENCY_LIMIT")

        # If required policy/candidate constraints are malformed, admission
        # must be false regardless of metrics.
        admitted = (
            len(reasons) == 0
            and predictions_valid
            and stored_candidate["status"] == "frozen"
            and total_bytes_out is not None
            and latency_out is not None
        )

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": total_bytes_out,
                "latencyMs": latency_out,
                "admitted": admitted,
                "reasonCodes": sort_codes(reasons),
            }
        )

    # Required result ordering: candidateOrder, UTF-8 name fallback.
    results.sort(
        key=lambda result: (
            order_index.get(result["name"], len(order_index)),
            utf8_key(result["name"]),
        )
    )

    # Choose among admitted candidates:
    # smaller bytes -> lower latency -> candidate order.
    admitted_results = [result for result in results if result["admitted"]]

    winner = None

    if admitted_results:
        winner = min(
            admitted_results,
            key=lambda result: (
                result["totalBytes"],
                result["latencyMs"],
                order_index.get(
                    result["name"],
                    len(order_index),
                ),
                utf8_key(result["name"]),
            ),
        )

    package_manifest = None

    if winner is not None:
        stored_winner = stored_by_name[winner["name"]]
        package_manifest = stored_winner

    return {
        "freezeId": payload["freezeId"],
        "selected": winner["name"] if winner else None,
        "results": results,
        "packageManifest": package_manifest,
    }


async def freeze_phase(payload: dict[str, Any]):
    if not validate_freeze_request(payload):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    freeze_id = payload["freezeId"]
    fingerprint = freeze_fingerprint(payload)

    existing = FREEZES.get(freeze_id)

    if existing is not None:
        if existing["fingerprint"] == fingerprint:
            return existing["response"]

        return JSONResponse(
            status_code=409,
            content={"error": "FREEZE_ID_CONFLICT"},
        )

    allowed_reasons = set(payload["allowedUnsupportedReasons"])

    candidates = []

    for candidate in payload["candidates"]:
        candidates.append(
            build_frozen_candidate(
                candidate,
                payload["calibrationDigest"],
                payload["tokenizerDigest"],
                allowed_reasons,
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


async def select_phase(payload: dict[str, Any]):
    if not validate_select_request(payload):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    freeze_id = payload["freezeId"]

    frozen = FREEZES.get(freeze_id)

    if frozen is None:
        # No stored freeze exists, so all candidates are not frozen.
        # Still return the required selection shape.
        results = []

        candidate_names = []
        for candidate in payload["candidates"]:
            if isinstance(candidate, dict) and nonempty_string(candidate.get("name")):
                candidate_names.append(candidate["name"])

        for name in sorted(set(candidate_names), key=utf8_key):
            aggregate, slices, valid = calculate_metrics(
                name,
                payload["rows"],
            )

            results.append(
                {
                    "name": name,
                    "aggregate": aggregate if valid else None,
                    "slices": slices if valid else None,
                    "totalBytes": None,
                    "latencyMs": (
                        payload["latencies"].get(name)
                        if finite_nonnegative_number(payload["latencies"].get(name))
                        else None
                    ),
                    "admitted": False,
                    "reasonCodes": ["NOT_FROZEN"],
                }
            )

        return {
            "freezeId": freeze_id,
            "selected": None,
            "results": results,
            "packageManifest": None,
        }

    return select_phase_response(
        frozen["response"],
        payload,
    )


@app.post("/quantize")
async def quantize(payload: Any):
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    phase = payload.get("phase")

    if phase == "freeze":
        return await freeze_phase(payload)

    if phase == "select":
        return await select_phase(payload)

    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


@app.get("/")
async def root():
    return {"service": "quantize", "endpoint": "/quantize"}
