import hashlib
import json
import math
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

REQUIRED_FILES = {
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
}

UNSAFE_EXTENSIONS = (".bin", ".pt", ".pth", ".pkl", ".pickle")

HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MODEL_CARD_PREFIX = "<!-- tds-model-card "
MODEL_CARD_SUFFIX = "-->"


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utf8_bytes(value: str) -> bytes | None:
    try:
        return value.encode("utf-8")
    except (UnicodeEncodeError, AttributeError):
        return None


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def is_safe_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -(2**53 - 1) <= value <= (2**53 - 1)
    )


def is_finite_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def parse_json_file(
    files: dict[str, Any],
    filename: str,
    violations: set[str],
) -> Any | None:
    value = files.get(filename)

    if not isinstance(value, str):
        return None

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        violations.add(f"INVALID_JSON:{filename}")
        return None


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False

    required_slices = policy.get("requiredSlices")

    if (
        not isinstance(required_slices, list)
        or len(required_slices) == 0
        or any(not is_nonempty_string(item) for item in required_slices)
        or len(set(required_slices)) != len(required_slices)
    ):
        return False

    for field in ("license", "intendedUse", "limitations"):
        if not is_nonempty_string(policy.get(field)):
            return False

    return True


def make_recomputed_inventory(files: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for filename in sorted(
        (name for name in files if name != "inventory.json"),
        key=lambda name: name.encode("utf-8"),
    ):
        content = files[filename]
        raw = content.encode("utf-8")

        entries.append(
            {
                "name": filename,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )

    return entries


def validate_inventory(
    files: dict[str, Any],
    recomputed_inventory: list[dict[str, Any]],
    violations: set[str],
) -> None:
    raw_inventory = files.get("inventory.json")

    if not isinstance(raw_inventory, str):
        return

    try:
        supplied_inventory = json.loads(raw_inventory)
    except (json.JSONDecodeError, TypeError, ValueError):
        violations.add("INVALID_JSON:inventory.json")
        return

    expected_text = compact_json(recomputed_inventory)

    if raw_inventory != expected_text or supplied_inventory != recomputed_inventory:
        violations.add("INVENTORY_MISMATCH")


def validate_adapter_config(
    config: Any,
    violations: set[str],
) -> None:
    if not isinstance(config, dict):
        violations.add("INVALID_ADAPTER_CONFIG")
        return

    rank = config.get("r")
    targets = config.get("target_modules")

    rank_ok = is_safe_integer(rank) and rank > 0

    targets_ok = (
        isinstance(targets, list)
        and len(targets) > 0
        and all(is_nonempty_string(item) for item in targets)
        and len(set(targets)) == len(targets)
    )

    if not rank_ok or not targets_ok:
        violations.add("INVALID_ADAPTER_CONFIG")


def validate_training_manifest(
    manifest: Any,
    model_digest: str | None,
    evaluation_digest: str | None,
    violations: set[str],
) -> None:
    if not isinstance(manifest, dict):
        violations.add("INVALID_TRAINING_MANIFEST")
        return

    fields = (
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    )

    for field in fields:
        if not is_nonempty_string(manifest.get(field)):
            violations.add(f"MISSING_MANIFEST_FIELD:{field}")

    base_revision = manifest.get("baseRevision")
    if is_nonempty_string(base_revision) and not HEX_40_RE.fullmatch(base_revision):
        violations.add("MUTABLE_BASE_REVISION")

    if (
        model_digest is not None
        and is_nonempty_string(manifest.get("modelArtifactDigest"))
        and manifest["modelArtifactDigest"] != model_digest
    ):
        violations.add("MODEL_ARTIFACT_MISMATCH")

    if (
        evaluation_digest is not None
        and is_nonempty_string(manifest.get("evaluationArtifactDigest"))
        and manifest["evaluationArtifactDigest"] != evaluation_digest
    ):
        violations.add("EVALUATION_DIGEST_MISMATCH")


def validate_evaluation(
    evaluation: Any,
    required_slices: list[str],
    model_digest: str | None,
    violations: set[str],
) -> None:
    if not isinstance(evaluation, dict):
        violations.add("INVALID_EVALUATION")
        return

    evaluation_model_digest = evaluation.get("modelArtifactDigest")

    if model_digest is not None and evaluation_model_digest != model_digest:
        violations.add("EVALUATION_ARTIFACT_MISMATCH")

    aggregate = evaluation.get("aggregate")
    if not is_finite_score(aggregate):
        violations.add("INVALID_AGGREGATE")

    slices = evaluation.get("slices")
    if not isinstance(slices, dict):
        violations.add("INVALID_EVALUATION")
        for slice_name in required_slices:
            violations.add(f"MISSING_SLICE:{slice_name}")
        return

    for slice_name in required_slices:
        if slice_name not in slices:
            violations.add(f"MISSING_SLICE:{slice_name}")
        elif not is_finite_score(slices[slice_name]):
            violations.add(f"SLICE_RANGE:{slice_name}")


def extract_model_cards(readme: str) -> list[str]:
    payloads: list[str] = []
    start = 0

    while True:
        marker_start = readme.find(MODEL_CARD_PREFIX, start)

        if marker_start == -1:
            break

        payload_start = marker_start + len(MODEL_CARD_PREFIX)
        marker_end = readme.find(MODEL_CARD_SUFFIX, payload_start)

        if marker_end == -1:
            payloads.append(readme[payload_start:])
            break

        payloads.append(readme[payload_start:marker_end])
        start = marker_end + len(MODEL_CARD_SUFFIX)

    return payloads


def validate_model_card(
    readme: Any,
    policy: dict[str, Any],
    manifest: Any,
    model_digest: str | None,
    violations: set[str],
) -> None:
    if not isinstance(readme, str):
        return

    cards = extract_model_cards(readme)

    if len(cards) == 0:
        violations.add("MISSING_MODEL_CARD")
        return

    if len(cards) != 1:
        violations.add("MODEL_CARD_COUNT")
        return

    try:
        card = json.loads(cards[0])
    except (json.JSONDecodeError, TypeError, ValueError):
        violations.add("INVALID_MODEL_CARD")
        return

    if not isinstance(card, dict):
        violations.add("INVALID_MODEL_CARD")
        return

    if not isinstance(manifest, dict):
        violations.add("MODEL_CARD_MISMATCH")
        return

    expected = {
        "task": manifest.get("task"),
        "baseRevision": manifest.get("baseRevision"),
        "datasetDigest": manifest.get("datasetDigest"),
        "modelArtifactDigest": model_digest,
        "license": policy.get("license"),
        "intendedUse": policy.get("intendedUse"),
        "limitations": policy.get("limitations"),
    }

    for key, expected_value in expected.items():
        if card.get(key) != expected_value:
            violations.add("MODEL_CARD_MISMATCH")
            return


@app.post("/verify-bundle")
async def verify_bundle(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if "policy" not in body or "files" not in body:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    policy = body.get("policy")
    files = body.get("files")

    if not isinstance(policy, dict) or not isinstance(files, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    violations: set[str] = set()

    if not validate_policy(policy):
        violations.add("INVALID_POLICY")

    required_slices = policy["requiredSlices"] if validate_policy(policy) else []

    valid_file_map: dict[str, str] = {}

    for filename, content in files.items():
        if not isinstance(filename, str) or not isinstance(content, str):
            if isinstance(filename, str):
                violations.add(f"INVALID_FILE:{filename}")
            continue

        encoded_name = utf8_bytes(filename)
        encoded_content = utf8_bytes(content)

        if encoded_name is None or encoded_content is None:
            violations.add(f"INVALID_FILE:{filename}")
            continue

        valid_file_map[filename] = content

    for filename in sorted(REQUIRED_FILES, key=lambda name: name.encode("utf-8")):
        if filename not in valid_file_map:
            violations.add(f"MISSING_FILE:{filename}")

    for filename in valid_file_map:
        if filename not in REQUIRED_FILES:
            violations.add("UNTRACKED_FILE")

        if filename.lower().endswith(UNSAFE_EXTENSIONS):
            violations.add("UNSAFE_WEIGHTS")

    recomputed_inventory = make_recomputed_inventory(valid_file_map)
    inventory_digest = sha256_bytes(compact_json(recomputed_inventory).encode("utf-8"))

    validate_inventory(
        valid_file_map,
        recomputed_inventory,
        violations,
    )

    adapter_config = parse_json_file(
        valid_file_map,
        "adapter_config.json",
        violations,
    )

    training_manifest = parse_json_file(
        valid_file_map,
        "training_manifest.json",
        violations,
    )

    evaluation = parse_json_file(
        valid_file_map,
        "evaluation.json",
        violations,
    )

    if adapter_config is not None:
        validate_adapter_config(adapter_config, violations)

    model_bytes = None
    evaluation_bytes = None

    if isinstance(valid_file_map.get("adapter_model.safetensors"), str):
        model_bytes = valid_file_map["adapter_model.safetensors"].encode("utf-8")

    if isinstance(valid_file_map.get("evaluation.json"), str):
        evaluation_bytes = valid_file_map["evaluation.json"].encode("utf-8")

    model_digest = sha256_bytes(model_bytes) if model_bytes is not None else None
    evaluation_digest = (
        sha256_bytes(evaluation_bytes) if evaluation_bytes is not None else None
    )

    if training_manifest is not None:
        validate_training_manifest(
            training_manifest,
            model_digest,
            evaluation_digest,
            violations,
        )

    if evaluation is not None:
        validate_evaluation(
            evaluation,
            required_slices,
            model_digest,
            violations,
        )

    validate_model_card(
        valid_file_map.get("README.md"),
        policy,
        training_manifest,
        model_digest,
        violations,
    )

    sorted_violations = sorted(
        violations,
        key=lambda code: code.encode("utf-8"),
    )

    return JSONResponse(
        status_code=200,
        content={
            "decision": "admit" if not sorted_violations else "reject",
            "violations": sorted_violations,
            "inventoryDigest": inventory_digest,
        },
    )
