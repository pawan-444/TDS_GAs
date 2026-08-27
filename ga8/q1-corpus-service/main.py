import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google_crc32c import Checksum


app = FastAPI()


MAX_SAFE_INTEGER = 2**53 - 1

URI_RE = re.compile(r"^gs://[^/]+/.+$")
GENERATION_RE = re.compile(r"^[0-9]+$")
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

TIME_RE = re.compile(
    r"^"
    r"(\d{4})-(\d{2})-(\d{2})"
    r"T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})"
    r"$"
)


ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}
ROW_KEY_ORDER = ["id", "entity", "eventTime", "revision", "text"]


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = value.strip()

    out = []
    in_space = False

    for ch in value:
        if ch.isspace():
            if not in_space:
                out.append(" ")
                in_space = True
        else:
            out.append(ch)
            in_space = False

    return "".join(out)


def parse_event_time(value: str) -> datetime | None:
    match = TIME_RE.fullmatch(value)

    if not match:
        return None

    year, month, day, hour, minute, second, fraction, offset = match.groups()

    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    # Validate calendar/time fields.
    if hour > 23 or minute > 59 or second > 59:
        return None

    try:
        if offset == "Z":
            tz = timezone.utc
        else:
            sign = 1 if offset[0] == "+" else -1
            offset_hour = int(offset[1:3])
            offset_minute = int(offset[4:6])

            # Offset magnitude <= 14:00.
            if offset_hour > 14 or offset_minute > 59:
                return None

            if offset_hour == 14 and offset_minute != 0:
                return None

            total_minutes = sign * (offset_hour * 60 + offset_minute)

            tz = timezone(timedelta(minutes=total_minutes))

    except Exception:
        return None

    # datetime requires a valid calendar date.
    try:
        # Fraction is converted to milliseconds.
        milliseconds = 0
        if fraction:
            milliseconds = int(fraction.ljust(3, "0"))

        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            milliseconds * 1000,
            tzinfo=tz,
        )
    except ValueError:
        return None

    return dt.astimezone(timezone.utc)


def canonical_event_time(value: str) -> str | None:
    dt = parse_event_time(value)

    if dt is None:
        return None

    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + (f"{dt.microsecond // 1000:03d}Z")


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False

    if not isinstance(policy.get("minTime"), str):
        return False

    if not isinstance(policy.get("maxTime"), str):
        return False

    threshold = policy.get("contaminationThreshold")

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold < 0
        or threshold > 1
    ):
        return False

    min_time = parse_event_time(policy["minTime"])
    max_time = parse_event_time(policy["maxTime"])

    if min_time is None or max_time is None:
        return False

    if min_time > max_time:
        return False

    return True


def crc32c_hex(content: str) -> str:
    checksum = Checksum()
    checksum.update(content.encode("utf-8"))
    return checksum.hexdigest().decode("ascii")


def validate_generation(value: Any) -> bool:
    return isinstance(value, str) and GENERATION_RE.fullmatch(value) is not None


def validate_crc_syntax(value: Any) -> bool:
    return isinstance(value, str) and CRC_RE.fullmatch(value) is not None


def parse_jsonl(content: str):
    lines = content.splitlines()

    parsed_rows = []

    for line in lines:
        if not line.strip():
            continue

        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None, "JSONL_INVALID"

        parsed_rows.append(value)

    if not parsed_rows:
        return None, "SCHEMA_INVALID"

    rows = []

    for row in parsed_rows:
        if not isinstance(row, dict):
            return None, "SCHEMA_INVALID"

        if set(row.keys()) != ROW_KEYS:
            return None, "SCHEMA_INVALID"

        if not isinstance(row["id"], str):
            return None, "SCHEMA_INVALID"

        if not isinstance(row["entity"], str):
            return None, "SCHEMA_INVALID"

        if not isinstance(row["eventTime"], str):
            return None, "SCHEMA_INVALID"

        if not isinstance(row["text"], str):
            return None, "SCHEMA_INVALID"

        revision = row["revision"]

        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
            or revision > MAX_SAFE_INTEGER
        ):
            return None, "SCHEMA_INVALID"

        event_time = canonical_event_time(row["eventTime"])

        if event_time is None:
            return None, "SCHEMA_INVALID"

        rows.append(
            {
                "id": row["id"],
                "entity": normalize_text(row["entity"]),
                "eventTime": event_time,
                "revision": revision,
                "text": normalize_text(row["text"]),
            }
        )

    return rows, None


def row_json(row: dict) -> str:
    return compact_json(
        {
            "id": row["id"],
            "entity": row["entity"],
            "eventTime": row["eventTime"],
            "revision": row["revision"],
            "text": row["text"],
        }
    )


def dedup_key(row: dict):
    return (
        row["entity"],
        row["eventTime"],
        row["text"],
    )


def row_winner_key(row: dict):
    # Highest revision first.
    # For equal revision, UTF-8-byte-smallest ID.
    return (
        -row["revision"],
        utf8_key(row["id"]),
    )


def split_for_entity(entity: str) -> str:
    digest = hashlib.sha256(entity.encode("utf-8")).digest()

    bucket = digest[0] % 10

    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "validation"
    return "test"


def word_set(text: str) -> set[str]:
    """
    Build lowercase Unicode letter/number words.

    A word is a maximal sequence of Unicode characters whose
    category begins with L (letter) or N (number).
    """
    result = set()
    current = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith(("L", "N")):
            current.append(ch.lower())
        else:
            if current:
                result.add("".join(current))
                current = []

    if current:
        result.add("".join(current))

    return result


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def contamination_check(
    row: dict,
    train_rows: list[dict],
    threshold: float,
) -> bool:
    row_words = word_set(row["text"])

    for train_row in train_rows:
        train_words = word_set(train_row["text"])

        if jaccard(row_words, train_words) >= threshold:
            return True

    return False


def sort_objects(items: list[dict], field: str):
    return sorted(
        items,
        key=lambda x: (
            utf8_key(x[field]) if isinstance(x[field], str) else b"",
            compact_json(x).encode("utf-8"),
        ),
    )


def sort_reasons(reasons: set[str]) -> list[str]:
    return sorted(
        set(reasons),
        key=lambda x: utf8_key(x),
    )


def digest_rows(rows: list[dict]) -> str:
    serialized = "".join(row_json(row) + "\n" for row in rows)

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sort_split_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            utf8_key(row["id"]),
            row_json(row).encode("utf-8"),
        ),
    )


@app.post("/build-corpus")
async def build_corpus(payload: Any):
    # Exact invalid-input response.
    if (
        not isinstance(payload, dict)
        or "policy" not in payload
        or not isinstance(payload.get("objects"), list)
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    policy = payload["policy"]
    objects = payload["objects"]

    policy_valid = validate_policy(policy)

    if policy_valid:
        min_time = parse_event_time(policy["minTime"])
        max_time = parse_event_time(policy["maxTime"])
        threshold = float(policy["contaminationThreshold"])
    else:
        min_time = None
        max_time = None
        threshold = 0.0

    rejected_objects_map: dict[str, dict] = {}
    accepted_rows = []

    lineage = []

    # ---------------------------------------------------------
    # Object validation
    # ---------------------------------------------------------

    for obj in objects:
        if not isinstance(obj, dict):
            # Request schema isn't explicitly defined for object
            # itself. Treat missing/non-string fields as applicable
            # object-level failures.
            obj = {}

        uri = obj.get("uri")
        generation = obj.get("generation")
        fetched_generation = obj.get("fetchedGeneration")
        crc = obj.get("crc32c")
        schema_id = obj.get("schemaId")
        content = obj.get("content")

        reasons = set()

        # URI_INVALID
        if not isinstance(uri, str) or URI_RE.fullmatch(uri) is None:
            reasons.add("URI_INVALID")

        # Generation checks
        generation_valid = validate_generation(generation)
        fetched_generation_valid = validate_generation(fetched_generation)

        if not generation_valid or not fetched_generation_valid:
            reasons.add("GENERATION_INVALID")

        if (
            generation_valid
            and fetched_generation_valid
            and generation != fetched_generation
        ):
            reasons.add("GENERATION_MISMATCH")

        # CRC syntax
        crc_valid = validate_crc_syntax(crc)

        if not crc_valid:
            reasons.add("CRC32C_INVALID")

        # CRC mismatch only for string content + valid CRC.
        if isinstance(content, str) and crc_valid:
            actual_crc = crc32c_hex(content)

            if actual_crc != crc:
                reasons.add("CRC32C_MISMATCH")

        # Schema
        if schema_id != "training-v1" or not isinstance(content, str):
            reasons.add("SCHEMA_INVALID")

        parsed_rows = None

        # JSONL checks only when content is a string.
        if isinstance(content, str):
            parsed_rows, parse_error = parse_jsonl(content)

            if parse_error is not None:
                reasons.add(parse_error)

        # If any object-level error exists, object is rejected and
        # its rows are not processed.
        if reasons:
            key = uri if isinstance(uri, str) else ""

            rejected_objects_map[key] = {
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": sort_reasons(reasons),
            }

            continue

        # Object fully accepted.
        lineage.append(
            {
                "uri": uri,
                "generation": generation,
                "crc32c": crc,
                "schemaId": schema_id,
            }
        )

        for row in parsed_rows:
            row_with_source = dict(row)
            row_with_source["_source_uri"] = uri
            accepted_rows.append(row_with_source)

    # ---------------------------------------------------------
    # Global deduplication
    # ---------------------------------------------------------

    groups: dict[tuple, list[dict]] = {}

    for row in accepted_rows:
        groups.setdefault(dedup_key(row), []).append(row)

    retained_rows = []
    rejected_rows_map: dict[str, dict] = {}

    for group in groups.values():
        group.sort(key=row_winner_key)

        winner = group[0]
        retained_rows.append(winner)

        for loser in group[1:]:
            row_id = loser["id"]

            existing = rejected_rows_map.get(row_id)

            if existing is None:
                rejected_rows_map[row_id] = {
                    "id": row_id,
                    "reasonCodes": ["DUPLICATE"],
                }
            else:
                existing["reasonCodes"].append("DUPLICATE")

    # ---------------------------------------------------------
    # Policy / window
    # ---------------------------------------------------------

    policy_rejected = set()

    for row in retained_rows:
        if not policy_valid:
            policy_rejected.add(id(row))

            existing = rejected_rows_map.get(row["id"])

            if existing is None:
                rejected_rows_map[row["id"]] = {
                    "id": row["id"],
                    "reasonCodes": ["POLICY_INVALID"],
                }
            else:
                existing["reasonCodes"].append("POLICY_INVALID")

            continue

        row_time = parse_event_time(row["eventTime"])

        if row_time < min_time or row_time > max_time:
            existing = rejected_rows_map.get(row["id"])

            if existing is None:
                rejected_rows_map[row["id"]] = {
                    "id": row["id"],
                    "reasonCodes": ["OUT_OF_WINDOW"],
                }
            else:
                existing["reasonCodes"].append("OUT_OF_WINDOW")

    # Rows that passed policy/window.
    usable_rows = []

    for row in retained_rows:
        if id(row) in policy_rejected:
            continue

        if policy_valid:
            row_time = parse_event_time(row["eventTime"])

            if row_time < min_time or row_time > max_time:
                continue

        usable_rows.append(row)

    # ---------------------------------------------------------
    # Deterministic split
    # ---------------------------------------------------------

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in usable_rows:
        split = split_for_entity(row["entity"])
        splits[split].append(row)

    # ---------------------------------------------------------
    # Train contamination
    # ---------------------------------------------------------

    train_rows = splits["train"]

    for split_name in ("validation", "test"):
        kept = []

        for row in splits[split_name]:
            contaminated = contamination_check(
                row,
                train_rows,
                threshold,
            )

            if contaminated:
                existing = rejected_rows_map.get(row["id"])

                if existing is None:
                    rejected_rows_map[row["id"]] = {
                        "id": row["id"],
                        "reasonCodes": ["TRAIN_CONTAMINATION"],
                    }
                else:
                    existing["reasonCodes"].append("TRAIN_CONTAMINATION")
            else:
                kept.append(row)

        splits[split_name] = kept

    # ---------------------------------------------------------
    # Final sorting
    # ---------------------------------------------------------

    for split_name in splits:
        splits[split_name] = sort_split_rows(splits[split_name])

    # Clean internal source metadata.
    for split_name in splits:
        for row in splits[split_name]:
            row.pop("_source_uri", None)

    for item in rejected_rows_map.values():
        item["reasonCodes"] = sort_reasons(set(item["reasonCodes"]))

    rejected_rows = sorted(
        rejected_rows_map.values(),
        key=lambda x: (
            utf8_key(x["id"]),
            compact_json(x).encode("utf-8"),
        ),
    )

    rejected_objects = sorted(
        rejected_objects_map.values(),
        key=lambda x: (
            utf8_key(x["uri"]) if isinstance(x["uri"], str) else b"",
            compact_json(x).encode("utf-8"),
        ),
    )

    lineage = sorted(
        lineage,
        key=lambda x: (
            utf8_key(x["uri"]),
            compact_json(x).encode("utf-8"),
        ),
    )

    return {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": {
            "train": digest_rows(splits["train"]),
            "validation": digest_rows(splits["validation"]),
            "test": digest_rows(splits["test"]),
        },
        "lineage": lineage,
    }
