from flask import Flask, jsonify, request
import hashlib
import json
import math
import re

app = Flask(__name__)

MAX_SAFE_INTEGER = 9007199254740991

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_WEIGHT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
)

MODEL_CARD_PREFIX = "<!-- tds-model-card "
MODEL_CARD_SUFFIX = "-->"


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def utf8_bytes(value):
    return value.encode("utf-8")


def is_nonempty_string(value):
    return isinstance(value, str) and value != ""


def is_positive_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


def is_finite_unit_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def is_nonempty_unique_string_array(value):
    if not isinstance(value, list) or len(value) == 0:
        return False

    seen = set()

    for item in value:
        if not is_nonempty_string(item):
            return False

        if item in seen:
            return False

        seen.add(item)

    return True


def sorted_utf8(values):
    return sorted(values, key=lambda item: item.encode("utf-8"))


def add_violation(violations, code):
    violations.add(code)


def parse_json_file(files, filename, violations):
    """
    Returns parsed JSON value, or None when the file cannot be used.
    Missing files and invalid JSON are recorded as violations.
    """
    if filename not in files:
        add_violation(violations, f"MISSING_FILE:{filename}")
        return None

    content = files[filename]

    if not isinstance(content, str):
        add_violation(violations, f"INVALID_FILE:{filename}")
        return None

    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        add_violation(violations, f"INVALID_JSON:{filename}")
        return None


def validate_policy(policy, violations):
    if not isinstance(policy, dict):
        add_violation(violations, "INVALID_POLICY")
        return False

    required_slices = policy.get("requiredSlices")

    if not is_nonempty_unique_string_array(required_slices):
        add_violation(violations, "INVALID_POLICY")
        return False

    for field in ("license", "intendedUse", "limitations"):
        if not is_nonempty_string(policy.get(field)):
            add_violation(violations, "INVALID_POLICY")
            return False

    return True


def recompute_inventory(files, violations):
    """
    Builds the inventory array from the actual supplied files excluding
    inventory.json itself.

    Returns:
      inventory_array,
      inventory_digest
    """
    inventory_entries = []

    for filename in sorted_utf8(files.keys()):
        if filename == "inventory.json":
            continue

        content = files[filename]

        if not isinstance(content, str):
            add_violation(violations, f"INVALID_FILE:{filename}")
            raw_bytes = b""
        else:
            raw_bytes = utf8_bytes(content)

        inventory_entries.append(
            {
                "name": filename,
                "bytes": len(raw_bytes),
                "sha256": sha256_bytes(raw_bytes),
            }
        )

    inventory_json = compact_json(inventory_entries)
    inventory_digest = sha256_bytes(utf8_bytes(inventory_json))

    return inventory_entries, inventory_digest


def validate_inventory(files, expected_inventory, violations):
    if "inventory.json" not in files:
        add_violation(violations, "MISSING_FILE:inventory.json")
        return

    inventory_content = files["inventory.json"]

    if not isinstance(inventory_content, str):
        add_violation(violations, "INVALID_FILE:inventory.json")
        return

    try:
        parsed_inventory = json.loads(inventory_content)
    except (json.JSONDecodeError, TypeError, ValueError):
        add_violation(violations, "INVALID_JSON:inventory.json")
        return

    # inventory.json must itself use exact compact JSON serialization.
    if inventory_content != compact_json(parsed_inventory):
        add_violation(violations, "INVENTORY_MISMATCH")
        return

    if not isinstance(parsed_inventory, list):
        add_violation(violations, "INVENTORY_MISMATCH")
        return

    for entry in parsed_inventory:
        if not isinstance(entry, dict):
            add_violation(violations, "INVENTORY_MISMATCH")
            return

        if list(entry.keys()) != ["name", "bytes", "sha256"]:
            add_violation(violations, "INVENTORY_MISMATCH")
            return

        if not is_nonempty_string(entry.get("name")):
            add_violation(violations, "INVENTORY_MISMATCH")
            return

        if (
            not isinstance(entry.get("bytes"), int)
            or isinstance(entry.get("bytes"), bool)
            or entry["bytes"] < 0
        ):
            add_violation(violations, "INVENTORY_MISMATCH")
            return

        if not isinstance(entry.get("sha256"), str):
            add_violation(violations, "INVENTORY_MISMATCH")
            return

        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            add_violation(violations, "INVENTORY_MISMATCH")
            return

    if parsed_inventory != expected_inventory:
        add_violation(violations, "INVENTORY_MISMATCH")


def validate_file_set(files, violations):
    actual_files = set(files.keys())
    required_files = set(REQUIRED_FILES)

    for filename in REQUIRED_FILES:
        if filename not in files:
            add_violation(violations, f"MISSING_FILE:{filename}")

    for filename in actual_files - required_files:
        add_violation(violations, "UNTRACKED_FILE")

    for filename in files:
        if not isinstance(filename, str):
            add_violation(violations, "UNTRACKED_FILE")
            continue

        lower_name = filename.lower()

        if lower_name.endswith(UNSAFE_WEIGHT_EXTENSIONS):
            add_violation(violations, "UNSAFE_WEIGHTS")


def validate_adapter_config(files, violations):
    config = parse_json_file(files, "adapter_config.json", violations)

    if config is None:
        return None

    if not isinstance(config, dict):
        add_violation(violations, "INVALID_ADAPTER_CONFIG")
        return None

    if not is_positive_safe_integer(config.get("r")):
        add_violation(violations, "INVALID_ADAPTER_CONFIG")
        return None

    if not is_nonempty_unique_string_array(config.get("target_modules")):
        add_violation(violations, "INVALID_ADAPTER_CONFIG")
        return None

    return config


def validate_training_manifest(files, violations):
    manifest = parse_json_file(files, "training_manifest.json", violations)

    if manifest is None:
        return None

    if not isinstance(manifest, dict):
        add_violation(violations, "INVALID_TRAINING_MANIFEST")
        return None

    required_fields = [
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    ]

    valid = True

    for field in required_fields:
        if field not in manifest:
            add_violation(violations, f"MISSING_MANIFEST_FIELD:{field}")
            valid = False

    if "baseRevision" in manifest:
        base_revision = manifest["baseRevision"]

        if (
            not isinstance(base_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", base_revision) is None
        ):
            add_violation(violations, "MUTABLE_BASE_REVISION")
            valid = False

    for field in (
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    ):
        if field in manifest and not is_nonempty_string(manifest[field]):
            add_violation(violations, "INVALID_TRAINING_MANIFEST")
            valid = False

    if not valid:
        return manifest

    return manifest


def validate_artifact_digests(files, manifest, violations):
    model_digest = None
    evaluation_digest = None

    if "adapter_model.safetensors" in files and isinstance(
        files["adapter_model.safetensors"], str
    ):
        model_digest = sha256_bytes(utf8_bytes(files["adapter_model.safetensors"]))

    if "evaluation.json" in files and isinstance(files["evaluation.json"], str):
        evaluation_digest = sha256_bytes(utf8_bytes(files["evaluation.json"]))

    if manifest is not None:
        if (
            model_digest is not None
            and is_nonempty_string(manifest.get("modelArtifactDigest"))
            and manifest["modelArtifactDigest"] != model_digest
        ):
            add_violation(violations, "MODEL_ARTIFACT_MISMATCH")

        if (
            evaluation_digest is not None
            and is_nonempty_string(manifest.get("evaluationArtifactDigest"))
            and manifest["evaluationArtifactDigest"] != evaluation_digest
        ):
            add_violation(violations, "EVALUATION_DIGEST_MISMATCH")

    return model_digest, evaluation_digest


def validate_evaluation(
    files,
    manifest,
    model_digest,
    required_slices,
    violations,
):
    evaluation = parse_json_file(files, "evaluation.json", violations)

    if evaluation is None:
        return None

    if not isinstance(evaluation, dict):
        add_violation(violations, "INVALID_EVALUATION")
        return None

    if not is_finite_unit_number(evaluation.get("aggregate")):
        add_violation(violations, "INVALID_AGGREGATE")

    if "modelArtifactDigest" not in evaluation:
        add_violation(violations, "INVALID_EVALUATION")
    elif not is_nonempty_string(evaluation["modelArtifactDigest"]):
        add_violation(violations, "INVALID_EVALUATION")
    else:
        if (
            model_digest is not None
            and evaluation["modelArtifactDigest"] != model_digest
        ):
            add_violation(violations, "MODEL_ARTIFACT_MISMATCH")

        if (
            manifest is not None
            and is_nonempty_string(manifest.get("modelArtifactDigest"))
            and evaluation["modelArtifactDigest"] != manifest["modelArtifactDigest"]
        ):
            add_violation(violations, "EVALUATION_ARTIFACT_MISMATCH")

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        add_violation(violations, "INVALID_EVALUATION")
        return evaluation

    for slice_name in required_slices:
        if slice_name not in slices:
            add_violation(violations, f"MISSING_SLICE:{slice_name}")
            continue

        if not is_finite_unit_number(slices[slice_name]):
            add_violation(violations, f"SLICE_RANGE:{slice_name}")

    return evaluation


def extract_model_card(readme, violations):
    """
    Finds literal model-card markers.

    A marker starts with:
      <!-- tds-model-card

    and ends at the first later:
      -->

    JSON braces inside quoted strings do not affect parsing because the
    marker end is based on the literal --> delimiter.
    """
    if not isinstance(readme, str):
        return None

    positions = []
    start = 0

    while True:
        position = readme.find(MODEL_CARD_PREFIX, start)

        if position == -1:
            break

        positions.append(position)
        start = position + len(MODEL_CARD_PREFIX)

    if len(positions) == 0:
        add_violation(violations, "MODEL_CARD_COUNT")
        add_violation(violations, "MISSING_MODEL_CARD")
        return None

    if len(positions) > 1:
        add_violation(violations, "MODEL_CARD_COUNT")
        return None

    payload_start = positions[0] + len(MODEL_CARD_PREFIX)
    payload_end = readme.find(MODEL_CARD_SUFFIX, payload_start)

    if payload_end == -1:
        add_violation(violations, "INVALID_MODEL_CARD")
        return None

    raw_payload = readme[payload_start:payload_end]

    try:
        card = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        add_violation(violations, "INVALID_MODEL_CARD")
        return None

    if not isinstance(card, dict):
        add_violation(violations, "INVALID_MODEL_CARD")
        return None

    return card


def validate_model_card(
    files,
    policy,
    manifest,
    model_digest,
    violations,
):
    if "README.md" not in files:
        add_violation(violations, "MISSING_FILE:README.md")
        return

    readme = files["README.md"]

    if not isinstance(readme, str):
        add_violation(violations, "INVALID_FILE:README.md")
        return

    card = extract_model_card(readme, violations)

    if card is None:
        return

    expected = {}

    if manifest is not None:
        expected.update(
            {
                "task": manifest.get("task"),
                "baseRevision": manifest.get("baseRevision"),
                "datasetDigest": manifest.get("datasetDigest"),
                "modelArtifactDigest": manifest.get("modelArtifactDigest"),
            }
        )

    if policy is not None:
        expected.update(
            {
                "license": policy.get("license"),
                "intendedUse": policy.get("intendedUse"),
                "limitations": policy.get("limitations"),
            }
        )

    if model_digest is not None:
        expected["modelArtifactDigest"] = model_digest

    for field, expected_value in expected.items():
        if not is_nonempty_string(expected_value):
            continue

        if card.get(field) != expected_value:
            add_violation(violations, "MODEL_CARD_MISMATCH")
            return


@app.post("/verify-bundle")
def verify_bundle():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "INVALID_INPUT"}), 400

    if "policy" not in payload:
        return jsonify({"error": "INVALID_INPUT"}), 400

    if "files" not in payload or not isinstance(payload["files"], dict):
        return jsonify({"error": "INVALID_INPUT"}), 400

    policy = payload["policy"]
    files = payload["files"]
    violations = set()

    policy_valid = validate_policy(policy, violations)

    validate_file_set(files, violations)

    expected_inventory, inventory_digest = recompute_inventory(
        files,
        violations,
    )

    validate_inventory(files, expected_inventory, violations)

    validate_adapter_config(files, violations)

    manifest = validate_training_manifest(files, violations)

    model_digest, evaluation_digest = validate_artifact_digests(
        files,
        manifest,
        violations,
    )

    required_slices = []
    if policy_valid:
        required_slices = policy["requiredSlices"]

    validate_evaluation(
        files,
        manifest,
        model_digest,
        required_slices,
        violations,
    )

    validate_model_card(
        files,
        policy if policy_valid else None,
        manifest,
        model_digest,
        violations,
    )

    sorted_violations = sorted(
        violations,
        key=lambda code: code.encode("utf-8"),
    )

    decision = "admit" if len(sorted_violations) == 0 else "reject"

    return jsonify(
        {
            "decision": decision,
            "violations": sorted_violations,
            "inventoryDigest": inventory_digest,
        }
    ), 200


@app.get("/")
def home():
    return jsonify(
        {
            "status": "running",
            "endpoint": "POST /verify-bundle",
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
