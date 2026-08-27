from flask import Flask, request, jsonify
import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta

app = Flask(__name__)

TIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

GENERATION_RE = re.compile(r"^\d+$")
CRC32C_RE = re.compile(r"^[0-9a-f]{8}$")
URI_RE = re.compile(r"^gs://[^/\s]+/.+$")


# CRC32C Castagnoli polynomial implementation
CRC32C_TABLE = []
POLY = 0x82F63B78

for i in range(256):
    value = i
    for _ in range(8):
        if value & 1:
            value = (value >> 1) ^ POLY
        else:
            value >>= 1
    CRC32C_TABLE.append(value)


def crc32c(data: bytes) -> str:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= 0xFFFFFFFF
    return f"{crc:08x}"


def utf8_key(value):
    return value.encode("utf-8")


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().strip()
    return " ".join(value.split())


def parse_time_to_utc(value):
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

        parsed = datetime(
            year, month, day, hour, minute, second, milliseconds * 1000, tzinfo=tz
        )

        utc_value = parsed.astimezone(timezone.utc)

        return utc_value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{milliseconds:03d}Z"

    except (ValueError, OverflowError):
        return None


def utc_datetime(normalized_time):
    return datetime.strptime(normalized_time, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def is_safe_nonnegative_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 9007199254740991
    )


def word_set(text):
    words = []
    current = []

    for char in text.lower():
        category = unicodedata.category(char)

        if category.startswith("L") or category.startswith("N"):
            current.append(char)
        else:
            if current:
                words.append("".join(current))
                current = []

    if current:
        words.append("".join(current))

    return set(words)


def jaccard_similarity(first, second):
    if not first and not second:
        return 1.0

    union = first | second
    return len(first & second) / len(union)


def valid_policy(policy):
    if not isinstance(policy, dict):
        return None

    min_time = parse_time_to_utc(policy.get("minTime"))
    max_time = parse_time_to_utc(policy.get("maxTime"))
    threshold = policy.get("contaminationThreshold")

    if min_time is None or max_time is None:
        return None

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return None

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        return None

    min_dt = utc_datetime(min_time)
    max_dt = utc_datetime(max_time)

    if min_dt > max_dt:
        return None

    return {"minTime": min_dt, "maxTime": max_dt, "contaminationThreshold": threshold}


def validate_and_parse_object(obj):
    codes = []

    if not isinstance(obj, dict):
        return None, [
            None,
            ["URI_INVALID", "GENERATION_INVALID", "CRC32C_INVALID", "SCHEMA_INVALID"],
        ]

    uri = obj.get("uri")
    generation = obj.get("generation")
    fetched_generation = obj.get("fetchedGeneration")
    supplied_crc = obj.get("crc32c")
    schema_id = obj.get("schemaId")
    content = obj.get("content")

    if not isinstance(uri, str) or not URI_RE.fullmatch(uri):
        codes.append("URI_INVALID")

    generation_valid = isinstance(generation, str) and GENERATION_RE.fullmatch(
        generation
    )
    fetched_valid = isinstance(fetched_generation, str) and GENERATION_RE.fullmatch(
        fetched_generation
    )

    if not generation_valid or not fetched_valid:
        codes.append("GENERATION_INVALID")

    if generation != fetched_generation:
        codes.append("GENERATION_MISMATCH")

    crc_valid = isinstance(supplied_crc, str) and CRC32C_RE.fullmatch(supplied_crc)

    if not crc_valid:
        codes.append("CRC32C_INVALID")

    if isinstance(content, str) and crc_valid:
        actual_crc = crc32c(content.encode("utf-8"))
        if actual_crc != supplied_crc:
            codes.append("CRC32C_MISMATCH")

    if schema_id != "training-v1" or not isinstance(content, str):
        codes.append("SCHEMA_INVALID")

    if codes:
        return None, [uri if isinstance(uri, str) else None, codes]

    rows = []
    non_blank_lines = [line for line in content.splitlines() if line.strip()]

    if not non_blank_lines:
        return None, [uri, ["SCHEMA_INVALID"]]

    for line in non_blank_lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return None, [uri, ["JSONL_INVALID"]]

        if not isinstance(parsed, dict):
            return None, [uri, ["SCHEMA_INVALID"]]

        expected_keys = {"id", "entity", "eventTime", "revision", "text"}

        if set(parsed.keys()) != expected_keys:
            return None, [uri, ["SCHEMA_INVALID"]]

        if not all(
            isinstance(parsed[key], str)
            for key in ["id", "entity", "eventTime", "text"]
        ):
            return None, [uri, ["SCHEMA_INVALID"]]

        if not is_safe_nonnegative_integer(parsed["revision"]):
            return None, [uri, ["SCHEMA_INVALID"]]

        normalized_time = parse_time_to_utc(parsed["eventTime"])

        if normalized_time is None:
            return None, [uri, ["SCHEMA_INVALID"]]

        rows.append(
            {
                "id": parsed["id"],
                "entity": normalize_text(parsed["entity"]),
                "eventTime": normalized_time,
                "revision": parsed["revision"],
                "text": normalize_text(parsed["text"]),
            }
        )

    lineage = {
        "uri": uri,
        "generation": generation,
        "crc32c": supplied_crc,
        "schemaId": schema_id,
    }

    return {"rows": rows, "lineage": lineage}, None


def rejection_sort_key(item, primary_key):
    return (utf8_key(item.get(primary_key) or ""), compact_json(item).encode("utf-8"))


@app.post("/build-corpus")
def build_corpus():
    payload = request.get_json(silent=True)

    if (
        not isinstance(payload, dict)
        or "policy" not in payload
        or not isinstance(payload.get("objects"), list)
    ):
        return jsonify({"error": "INVALID_INPUT"}), 400

    policy = valid_policy(payload.get("policy"))

    rejected_objects = []
    rejected_rows = []
    lineage = []
    candidate_rows = []

    for obj in payload["objects"]:
        parsed_object, rejection = validate_and_parse_object(obj)

        if rejection is not None:
            uri, codes = rejection
            rejected_objects.append(
                {"uri": uri, "reasonCodes": sorted(set(codes), key=utf8_key)}
            )
            continue

        candidate_rows.extend(parsed_object["rows"])
        lineage.append(parsed_object["lineage"])

    grouped = {}

    for row in candidate_rows:
        key = compact_json([row["entity"], row["eventTime"], row["text"]])
        grouped.setdefault(key, []).append(row)

    retained_rows = []

    for same_content_rows in grouped.values():
        same_content_rows.sort(key=lambda row: (-row["revision"], utf8_key(row["id"])))

        winner = same_content_rows[0]
        retained_rows.append(winner)

        for loser in same_content_rows[1:]:
            rejected_rows.append({"id": loser["id"], "reasonCodes": ["DUPLICATE"]})

    eligible_rows = []

    if policy is None:
        for row in retained_rows:
            rejected_rows.append({"id": row["id"], "reasonCodes": ["POLICY_INVALID"]})
    else:
        for row in retained_rows:
            row_time = utc_datetime(row["eventTime"])

            if row_time < policy["minTime"] or row_time > policy["maxTime"]:
                rejected_rows.append(
                    {"id": row["id"], "reasonCodes": ["OUT_OF_WINDOW"]}
                )
            else:
                eligible_rows.append(row)

    splits = {"train": [], "validation": [], "test": []}

    for row in eligible_rows:
        first_hash_byte = hashlib.sha256(row["entity"].encode("utf-8")).digest()[0]

        bucket = first_hash_byte % 10

        if bucket <= 5:
            splits["train"].append(row)
        elif bucket <= 7:
            splits["validation"].append(row)
        else:
            splits["test"].append(row)

    if policy is not None:
        train_word_sets = [word_set(row["text"]) for row in splits["train"]]
        threshold = policy["contaminationThreshold"]

        for split_name in ("validation", "test"):
            accepted_rows = []

            for row in splits[split_name]:
                current_words = word_set(row["text"])

                contaminated = any(
                    jaccard_similarity(current_words, train_words) >= threshold
                    for train_words in train_word_sets
                )

                if contaminated:
                    rejected_rows.append(
                        {"id": row["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]}
                    )
                else:
                    accepted_rows.append(row)

            splits[split_name] = accepted_rows

    for split_name in splits:
        splits[split_name].sort(
            key=lambda row: (utf8_key(row["id"]), compact_json(row).encode("utf-8"))
        )

    digests = {}

    for split_name, rows in splits.items():
        serialized = "".join(
            compact_json(
                {
                    "id": row["id"],
                    "entity": row["entity"],
                    "eventTime": row["eventTime"],
                    "revision": row["revision"],
                    "text": row["text"],
                }
            )
            + "\n"
            for row in rows
        )

        digests[split_name] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    for row in rejected_rows:
        row["reasonCodes"] = sorted(set(row["reasonCodes"]), key=utf8_key)

    rejected_objects.sort(key=lambda item: rejection_sort_key(item, "uri"))

    rejected_rows.sort(key=lambda item: rejection_sort_key(item, "id"))

    lineage.sort(
        key=lambda item: (utf8_key(item["uri"]), compact_json(item).encode("utf-8"))
    )

    response = {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }

    return app.response_class(
        response=json.dumps(response, ensure_ascii=False, separators=(",", ":")),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
