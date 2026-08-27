from flask import Flask, request, jsonify
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import re
import unicodedata

app = Flask(__name__)

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

URI_RE = re.compile(r"^gs://[^/\s]+/.+$")
GENERATION_RE = re.compile(r"^\d+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")
VERSION_RE = re.compile(r"^[1-9]\d*$")

MAX_SAFE_INTEGER = 9007199254740991


# -------------------- COMMON HELPERS --------------------


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def utf8_key(value):
    if value is None:
        return b""
    return str(value).encode("utf-8")


def sort_codes(codes):
    return sorted(set(codes), key=utf8_key)


def is_safe_nonnegative_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_probability(value):
    return is_finite_number(value) and 0 <= value <= 1


def parse_timestamp(value):
    """
    Parses:
    YYYY-MM-DDTHH:mm:ss[.sss](Z|+HH:mm|-HH:mm)
    Returns timezone-aware UTC datetime or None.
    """
    if not isinstance(value, str):
        return None

    match = TIME_RE.fullmatch(value)
    if not match:
        return None

    year, month, day, hour, minute, second, fraction, offset = match.groups()

    try:
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        minute = int(minute)
        second = int(second)

        milliseconds = int((fraction or "").ljust(3, "0"))

        if offset == "Z":
            tz = timezone.utc
        else:
            sign = 1 if offset[0] == "+" else -1
            offset_hour = int(offset[1:3])
            offset_minute = int(offset[4:6])

            if offset_hour > 14 or offset_minute > 59:
                return None

            if offset_hour == 14 and offset_minute != 0:
                return None

            tz = timezone(sign * timedelta(hours=offset_hour, minutes=offset_minute))

        return datetime(
            year, month, day, hour, minute, second, milliseconds * 1000, tzinfo=tz
        ).astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


def utc_string(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


# -------------------- CRC32C --------------------

CRC32C_TABLE = []
CRC32C_POLY = 0x82F63B78

for i in range(256):
    value = i
    for _ in range(8):
        value = (value >> 1) ^ CRC32C_POLY if value & 1 else value >> 1
    CRC32C_TABLE.append(value)


def crc32c(data):
    crc = 0xFFFFFFFF

    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)

    return f"{(crc ^ 0xFFFFFFFF):08x}"


# =========================================================
# QUESTION 1: BUILD CORPUS
# =========================================================


def canonicalize_text(value):
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().strip()
    return " ".join(value.split())


def corpus_word_set(value):
    words = []
    current = []

    for char in value.lower():
        category = unicodedata.category(char)

        if category.startswith("L") or category.startswith("N"):
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []

    if current:
        words.append("".join(current))

    return set(words)


def jaccard_similarity(first, second):
    if not first and not second:
        return 1.0

    return len(first & second) / len(first | second)


def valid_corpus_policy(policy):
    if not isinstance(policy, dict):
        return None

    min_time = parse_timestamp(policy.get("minTime"))
    max_time = parse_timestamp(policy.get("maxTime"))
    threshold = policy.get("contaminationThreshold")

    if min_time is None or max_time is None:
        return None

    if min_time > max_time:
        return None

    if not is_probability(threshold):
        return None

    return {
        "minTime": min_time,
        "maxTime": max_time,
        "contaminationThreshold": threshold,
    }


def parse_corpus_object(obj):
    if not isinstance(obj, dict):
        return None, {
            "uri": None,
            "reasonCodes": [
                "URI_INVALID",
                "GENERATION_INVALID",
                "CRC32C_INVALID",
                "SCHEMA_INVALID",
            ],
        }

    uri = obj.get("uri")
    generation = obj.get("generation")
    fetched_generation = obj.get("fetchedGeneration")
    supplied_crc = obj.get("crc32c")
    schema_id = obj.get("schemaId")
    content = obj.get("content")

    codes = []

    if not isinstance(uri, str) or not URI_RE.fullmatch(uri):
        codes.append("URI_INVALID")

    generation_valid = isinstance(generation, str) and GENERATION_RE.fullmatch(
        generation
    )

    fetched_generation_valid = isinstance(
        fetched_generation, str
    ) and GENERATION_RE.fullmatch(fetched_generation)

    if not generation_valid or not fetched_generation_valid:
        codes.append("GENERATION_INVALID")

    if generation != fetched_generation:
        codes.append("GENERATION_MISMATCH")

    crc_valid = isinstance(supplied_crc, str) and CRC_RE.fullmatch(supplied_crc)

    if not crc_valid:
        codes.append("CRC32C_INVALID")

    if isinstance(content, str) and crc_valid:
        if crc32c(content.encode("utf-8")) != supplied_crc:
            codes.append("CRC32C_MISMATCH")

    if schema_id != "training-v1" or not isinstance(content, str):
        codes.append("SCHEMA_INVALID")

    if codes:
        return None, {
            "uri": uri if isinstance(uri, str) else None,
            "reasonCodes": sort_codes(codes),
        }

    lines = [line for line in content.splitlines() if line.strip()]

    if not lines:
        return None, {"uri": uri, "reasonCodes": ["SCHEMA_INVALID"]}

    rows = []

    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None, {"uri": uri, "reasonCodes": ["JSONL_INVALID"]}

        required_keys = {"id", "entity", "eventTime", "revision", "text"}

        if not isinstance(row, dict) or set(row.keys()) != required_keys:
            return None, {"uri": uri, "reasonCodes": ["SCHEMA_INVALID"]}

        if not all(
            isinstance(row[key], str) for key in ["id", "entity", "eventTime", "text"]
        ):
            return None, {"uri": uri, "reasonCodes": ["SCHEMA_INVALID"]}

        if not is_safe_nonnegative_integer(row["revision"]):
            return None, {"uri": uri, "reasonCodes": ["SCHEMA_INVALID"]}

        event_time = parse_timestamp(row["eventTime"])

        if event_time is None:
            return None, {"uri": uri, "reasonCodes": ["SCHEMA_INVALID"]}

        rows.append(
            {
                "id": row["id"],
                "entity": canonicalize_text(row["entity"]),
                "eventTime": utc_string(event_time),
                "revision": row["revision"],
                "text": canonicalize_text(row["text"]),
            }
        )

    return {
        "rows": rows,
        "lineage": {
            "uri": uri,
            "generation": generation,
            "crc32c": supplied_crc,
            "schemaId": schema_id,
        },
    }, None


@app.post("/build-corpus")
def build_corpus():
    payload = request.get_json(silent=True)

    if (
        not isinstance(payload, dict)
        or "policy" not in payload
        or not isinstance(payload.get("objects"), list)
    ):
        return jsonify({"error": "INVALID_INPUT"}), 400

    policy = valid_corpus_policy(payload["policy"])

    rejected_objects = []
    rejected_rows = []
    lineage = []
    candidate_rows = []

    for obj in payload["objects"]:
        parsed, rejected = parse_corpus_object(obj)

        if rejected is not None:
            rejected_objects.append(rejected)
            continue

        candidate_rows.extend(parsed["rows"])
        lineage.append(parsed["lineage"])

    duplicate_groups = {}

    for row in candidate_rows:
        duplicate_key = compact_json([row["entity"], row["eventTime"], row["text"]])

        duplicate_groups.setdefault(duplicate_key, []).append(row)

    retained_rows = []

    for group in duplicate_groups.values():
        group.sort(key=lambda item: (-item["revision"], utf8_key(item["id"])))

        retained_rows.append(group[0])

        for loser in group[1:]:
            rejected_rows.append({"id": loser["id"], "reasonCodes": ["DUPLICATE"]})

    eligible_rows = []

    if policy is None:
        for row in retained_rows:
            rejected_rows.append({"id": row["id"], "reasonCodes": ["POLICY_INVALID"]})
    else:
        for row in retained_rows:
            event_time = parse_timestamp(row["eventTime"])

            if event_time < policy["minTime"] or event_time > policy["maxTime"]:
                rejected_rows.append(
                    {"id": row["id"], "reasonCodes": ["OUT_OF_WINDOW"]}
                )
            else:
                eligible_rows.append(row)

    splits = {"train": [], "validation": [], "test": []}

    for row in eligible_rows:
        bucket = hashlib.sha256(row["entity"].encode("utf-8")).digest()[0] % 10

        if bucket <= 5:
            splits["train"].append(row)
        elif bucket <= 7:
            splits["validation"].append(row)
        else:
            splits["test"].append(row)

    train_sets = [corpus_word_set(row["text"]) for row in splits["train"]]

    if policy is not None:
        for split_name in ["validation", "test"]:
            accepted = []

            for row in splits[split_name]:
                current_set = corpus_word_set(row["text"])

                contaminated = any(
                    jaccard_similarity(current_set, train_set)
                    >= policy["contaminationThreshold"]
                    for train_set in train_sets
                )

                if contaminated:
                    rejected_rows.append(
                        {"id": row["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]}
                    )
                else:
                    accepted.append(row)

            splits[split_name] = accepted

    for split_name in splits:
        splits[split_name].sort(
            key=lambda item: (utf8_key(item["id"]), compact_json(item).encode("utf-8"))
        )

    digests = {}

    for split_name, rows in splits.items():
        content = "".join(compact_json(row) + "\n" for row in rows)
        digests[split_name] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    for row in rejected_rows:
        row["reasonCodes"] = sort_codes(row["reasonCodes"])

    rejected_objects.sort(
        key=lambda item: (utf8_key(item["uri"]), compact_json(item).encode("utf-8"))
    )

    rejected_rows.sort(
        key=lambda item: (utf8_key(item["id"]), compact_json(item).encode("utf-8"))
    )

    lineage.sort(
        key=lambda item: (utf8_key(item["uri"]), compact_json(item).encode("utf-8"))
    )

    return app.response_class(
        response=compact_json(
            {
                "splits": splits,
                "rejectedObjects": rejected_objects,
                "rejectedRows": rejected_rows,
                "digests": digests,
                "lineage": lineage,
            }
        ),
        status=200,
        mimetype="application/json",
    )


# =========================================================
# QUESTION 2: PROMOTE MODEL
# =========================================================


def valid_version(value):
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
        return False

    try:
        return int(value) <= MAX_SAFE_INTEGER
    except ValueError:
        return False


def valid_promotion_policy(policy):
    if not isinstance(policy, dict):
        return None

    dataset_digest = policy.get("datasetDigest")
    schema_digest = policy.get("schemaDigest")
    max_age = policy.get("maxAgeSeconds")
    accuracy_floor = policy.get("accuracyFloor")
    required_slices = policy.get("requiredSlices")
    max_latency = policy.get("maxLatencyMs")
    max_size = policy.get("maxSizeBytes")
    min_improvement = policy.get("minImprovement")

    if not isinstance(dataset_digest, str) or not dataset_digest:
        return None

    if not isinstance(schema_digest, str) or not schema_digest:
        return None

    if not is_safe_nonnegative_integer(max_age):
        return None

    if not is_probability(accuracy_floor):
        return None

    if not isinstance(required_slices, dict):
        return None

    for name, floor in required_slices.items():
        if not isinstance(name, str) or not is_probability(floor):
            return None

    if not is_finite_number(max_latency) or max_latency < 0:
        return None

    if not is_safe_nonnegative_integer(max_size):
        return None

    if not is_probability(min_improvement):
        return None

    return policy


def promotion_codes(version_data, policy, as_of):
    codes = []
    evaluation = version_data.get("evaluation")

    if not isinstance(evaluation, dict):
        return ["MISSING_EVALUATION"]

    created_at = parse_timestamp(evaluation.get("createdAt"))

    if created_at is None:
        codes.append("INVALID_TIMESTAMP")
    else:
        if created_at > as_of:
            codes.append("FUTURE_EVALUATION")

        if created_at < as_of - timedelta(seconds=policy["maxAgeSeconds"]):
            codes.append("STALE_EVALUATION")

    if not isinstance(version_data.get("artifactDigest"), str) or version_data.get(
        "artifactDigest"
    ) != evaluation.get("artifactDigest"):
        codes.append("ARTIFACT_MISMATCH")

    if evaluation.get("datasetDigest") != policy["datasetDigest"]:
        codes.append("DATASET_MISMATCH")

    if evaluation.get("schemaDigest") != policy["schemaDigest"]:
        codes.append("SCHEMA_MISMATCH")

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")

    if not is_finite_number(accuracy):
        codes.append("NON_FINITE")
    elif not 0 <= accuracy <= 1:
        codes.append("METRIC_RANGE")
    elif accuracy < policy["accuracyFloor"]:
        codes.append("ACCURACY_FLOOR")

    if not is_finite_number(latency):
        codes.append("NON_FINITE")
    elif latency < 0:
        codes.append("METRIC_RANGE")
    elif latency > policy["maxLatencyMs"]:
        codes.append("LATENCY_LIMIT")

    if not is_safe_nonnegative_integer(size):
        codes.append("NON_FINITE" if not is_finite_number(size) else "METRIC_RANGE")
    elif size > policy["maxSizeBytes"]:
        codes.append("SIZE_LIMIT")

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        slices = {}

    for slice_name, required_floor in policy["requiredSlices"].items():
        if slice_name not in slices:
            codes.append(f"MISSING_SLICE:{slice_name}")
            continue

        value = slices[slice_name]

        if not is_probability(value):
            codes.append(f"SLICE_RANGE:{slice_name}")
        elif value < required_floor:
            codes.append(f"SLICE_FLOOR:{slice_name}")

    return sort_codes(codes)


@app.post("/promote")
def promote():
    payload = request.get_json(silent=True)

    if (
        not isinstance(payload, dict)
        or "policy" not in payload
        or not isinstance(payload.get("versions"), list)
        or not isinstance(payload.get("championVersion"), str)
    ):
        return jsonify({"error": "INVALID_INPUT"}), 400

    as_of = parse_timestamp(payload.get("asOf"))
    policy = valid_promotion_policy(payload["policy"])
    champion_version = payload["championVersion"]
    versions = payload["versions"]

    failed_gates = {}
    counts = {}

    for item in versions:
        if isinstance(item, dict) and isinstance(item.get("version"), str):
            version = item["version"]
            counts[version] = counts.get(version, 0) + 1

    unique_versions = {}

    for item in versions:
        if not isinstance(item, dict):
            continue

        version = item.get("version")

        if not isinstance(version, str):
            continue

        if not valid_version(version):
            failed_gates.setdefault(version, []).append("INVALID_VERSION")
            continue

        if counts.get(version, 0) > 1:
            failed_gates.setdefault(version, []).append("DUPLICATE_VERSION")
            continue

        unique_versions[version] = item

    if as_of is None:
        for version in unique_versions:
            failed_gates.setdefault(version, []).append("INVALID_TIMESTAMP")

    if policy is None:
        for version in unique_versions:
            failed_gates.setdefault(version, []).append("INVALID_POLICY")

    eligible = []

    if as_of is not None and policy is not None:
        for version, item in unique_versions.items():
            codes = promotion_codes(item, policy, as_of)

            if codes:
                failed_gates.setdefault(version, []).extend(codes)
            else:
                eligible.append(item)

    for version in failed_gates:
        failed_gates[version] = sort_codes(failed_gates[version])

    eligible.sort(
        key=lambda item: (
            -item["evaluation"]["accuracy"],
            item["evaluation"]["latencyMs"],
            item["evaluation"]["sizeBytes"],
            int(item["version"]),
        )
    )

    eligible_versions = [item["version"] for item in eligible]

    if champion_version not in eligible_versions:
        return jsonify(
            {
                "action": "block",
                "championVersion": champion_version,
                "selectedVersion": None,
                "eligibleVersions": eligible_versions,
                "failedGates": failed_gates,
                "aliasMutation": None,
                "evidence": None,
            }
        ), 200

    champion = unique_versions[champion_version]
    winner = eligible[0]

    improvement = (
        Decimal(str(winner["evaluation"]["accuracy"]))
        - Decimal(str(champion["evaluation"]["accuracy"]))
    ).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)

    minimum_improvement = Decimal(str(policy["minImprovement"]))

    if winner["version"] != champion_version and improvement >= minimum_improvement:
        return jsonify(
            {
                "action": "promote",
                "championVersion": champion_version,
                "selectedVersion": winner["version"],
                "eligibleVersions": eligible_versions,
                "failedGates": failed_gates,
                "aliasMutation": {"alias": "champion", "version": winner["version"]},
                "evidence": winner["evaluation"],
            }
        ), 200

    return jsonify(
        {
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": champion["evaluation"],
        }
    ), 200


# Optional route: useful when opening the Render URL in browser.
@app.get("/")
def home():
    return jsonify(
        {"status": "running", "endpoints": ["POST /build-corpus", "POST /promote"]}
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
