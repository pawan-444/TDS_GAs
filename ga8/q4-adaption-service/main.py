from flask import Flask, request, jsonify
from decimal import Decimal, ROUND_HALF_UP
import math
import re

app = Flask(__name__)

MAX_SAFE_INTEGER = 9007199254740991

INTERVENTIONS = [
    "prompt_only",
    "retrieval",
    "lora",
    "qlora"
]

ROLE_VALUES = {"system", "user", "assistant"}

BASE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def utf8_key(value):
    return str(value).encode("utf-8")


def sort_codes(codes):
    return sorted(set(codes), key=utf8_key)


def is_safe_nonnegative_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE_INTEGER
    )


def is_safe_positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_probability(value):
    return is_finite_number(value) and 0 <= value <= 1


def rounded_cost(one_time_cost, horizon_requests, recurring_cost):
    return float(
        (
            Decimal(str(one_time_cost))
            + Decimal(str(horizon_requests)) * Decimal(str(recurring_cost))
        ).quantize(
            Decimal("0.000000000001"),
            rounding=ROUND_HALF_UP
        )
    )


# =========================================================
# OPERATION: choose
# =========================================================

def valid_choose_policy(policy):
    if not isinstance(policy, dict):
        return False

    required_fields = {
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests"
    }

    if set(policy.keys()) != required_fields:
        return False

    if not is_probability(policy["minQuality"]):
        return False

    if not isinstance(policy["freshnessRequired"], bool):
        return False

    if (
        not is_finite_number(policy["maxLatencyMs"])
        or policy["maxLatencyMs"] < 0
    ):
        return False

    if (
        not is_finite_number(policy["maxMemoryMb"])
        or policy["maxMemoryMb"] < 0
    ):
        return False

    if not is_safe_nonnegative_integer(policy["maxLabeledExamples"]):
        return False

    if (
        not is_finite_number(policy["maxTotalCost"])
        or policy["maxTotalCost"] < 0
    ):
        return False

    if not is_safe_nonnegative_integer(policy["horizonRequests"]):
        return False

    return True


def valid_candidate(candidate):
    if not isinstance(candidate, dict):
        return False

    required_fields = {
        "name",
        "available",
        "quality",
        "freshness",
        "latencyMs",
        "memoryMb",
        "labeledExamples",
        "oneTimeCost",
        "recurringCost"
    }

    if set(candidate.keys()) != required_fields:
        return False

    if candidate["name"] not in INTERVENTIONS:
        return False

    if not isinstance(candidate["available"], bool):
        return False

    if not is_probability(candidate["quality"]):
        return False

    if not isinstance(candidate["freshness"], bool):
        return False

    if (
        not is_finite_number(candidate["latencyMs"])
        or candidate["latencyMs"] < 0
    ):
        return False

    if (
        not is_finite_number(candidate["memoryMb"])
        or candidate["memoryMb"] < 0
    ):
        return False

    if not is_safe_nonnegative_integer(candidate["labeledExamples"]):
        return False

    if (
        not is_finite_number(candidate["oneTimeCost"])
        or candidate["oneTimeCost"] < 0
    ):
        return False

    if (
        not is_finite_number(candidate["recurringCost"])
        or candidate["recurringCost"] < 0
    ):
        return False

    return True


def choose_intervention(payload):
    policy = payload.get("policy")
    candidates = payload.get("candidates")

    total_costs = {name: None for name in INTERVENTIONS}
    reason_codes = {name: [] for name in INTERVENTIONS}

    if not valid_choose_policy(policy) or not isinstance(candidates, list):
        for name in INTERVENTIONS:
            reason_codes[name] = ["INVALID_INPUT"]

        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": reason_codes
        }

    candidate_map = {}
    candidate_counts = {}

    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
            name = candidate["name"]
            candidate_counts[name] = candidate_counts.get(name, 0) + 1

    for candidate in candidates:
        if (
            isinstance(candidate, dict)
            and isinstance(candidate.get("name"), str)
            and candidate["name"] in INTERVENTIONS
            and candidate_counts.get(candidate["name"]) == 1
        ):
            candidate_map[candidate["name"]] = candidate

    eligible = []

    for name in INTERVENTIONS:
        candidate = candidate_map.get(name)

        if candidate is None or not valid_candidate(candidate):
            reason_codes[name] = ["INVALID_INPUT"]
            continue

        total_cost = rounded_cost(
            candidate["oneTimeCost"],
            policy["horizonRequests"],
            candidate["recurringCost"]
        )

        total_costs[name] = total_cost
        codes = []

        if not candidate["available"]:
            codes.append("UNAVAILABLE")

        if candidate["quality"] < policy["minQuality"]:
            codes.append("QUALITY_FLOOR")

        if policy["freshnessRequired"] and not candidate["freshness"]:
            codes.append("FRESHNESS_REQUIRED")

        if candidate["latencyMs"] > policy["maxLatencyMs"]:
            codes.append("LATENCY_LIMIT")

        if candidate["memoryMb"] > policy["maxMemoryMb"]:
            codes.append("MEMORY_LIMIT")

        if candidate["labeledExamples"] > policy["maxLabeledExamples"]:
            codes.append("DATA_LIMIT")

        if total_cost > policy["maxTotalCost"]:
            codes.append("COST_LIMIT")

        reason_codes[name] = sort_codes(codes)

        if not codes:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes
    }


# =========================================================
# OPERATION: repair
# =========================================================

def repair_peft_run(payload):
    reason_codes = []

    # ---------------- Tokens and labels ----------------

    tokens = payload.get("tokens")
    labels = []
    tokens_valid = isinstance(tokens, list) and len(tokens) > 0

    if tokens_valid:
        for token in tokens:
            if (
                not isinstance(token, dict)
                or set(token.keys()) != {"id", "role", "padding", "text"}
                or not is_safe_nonnegative_integer(token.get("id"))
                or token.get("role") not in ROLE_VALUES
                or not isinstance(token.get("padding"), bool)
                or not isinstance(token.get("text"), str)
            ):
                tokens_valid = False
                break

    if not tokens_valid:
        reason_codes.append("INVALID_TOKEN")

        if isinstance(tokens, list):
            labels = [-100 for _ in tokens]
        else:
            labels = []
    else:
        for token in tokens:
            if token["role"] == "assistant" and token["padding"] is False:
                labels.append(token["id"])
            else:
                labels.append(-100)

    # ---------------- Chat template ----------------

    template_pass = payload.get("templateApplications") == 1

    if not template_pass:
        reason_codes.append("CHAT_TEMPLATE_COUNT")

    # ---------------- PEFT parameters ----------------

    parameters = payload.get("parameters")
    allowed_targets = payload.get("allowedTargets")

    parameters_valid = isinstance(parameters, list)
    allowed_targets_valid = (
        isinstance(allowed_targets, list)
        and len(allowed_targets) > 0
        and all(
            isinstance(target, str) and target != ""
            for target in allowed_targets
        )
        and len(set(allowed_targets)) == len(allowed_targets)
    )

    names_seen = set()

    if parameters_valid:
        for parameter in parameters:
            if (
                not isinstance(parameter, dict)
                or set(parameter.keys()) != {"name", "target", "numel"}
                or not isinstance(parameter.get("name"), str)
                or not isinstance(parameter.get("target"), str)
                or not is_safe_positive_integer(parameter.get("numel"))
                or parameter["name"] in names_seen
            ):
                parameters_valid = False
                break

            names_seen.add(parameter["name"])

    if not parameters_valid or not allowed_targets_valid:
        reason_codes.append("INVALID_PARAMETER")

    trainable_params = []
    trainable_count = 0

    if parameters_valid and allowed_targets_valid:
        for parameter in parameters:
            valid_lora_name = (
                parameter["name"].endswith(".lora_A.weight")
                or parameter["name"].endswith(".lora_B.weight")
            )

            if parameter["target"] in allowed_targets and valid_lora_name:
                trainable_params.append(parameter["name"])
                trainable_count += parameter["numel"]

        trainable_params.sort(key=utf8_key)

    inference_mode_pass = payload.get("inferenceMode") is False

    if not inference_mode_pass:
        reason_codes.append("INFERENCE_MODE")

    peft_config_pass = (
        parameters_valid
        and allowed_targets_valid
        and len(trainable_params) > 0
        and inference_mode_pass
    )

    if len(trainable_params) == 0:
        reason_codes.append("INVALID_PARAMETER")

    # ---------------- Adapter artifacts ----------------

    artifact_files = payload.get("artifactFiles")

    expected_files = [
        "adapter_config.json",
        "adapter_model.safetensors"
    ]

    adapter_files_pass = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(isinstance(file_name, str) for file_name in artifact_files)
        and set(artifact_files) == set(expected_files)
    )

    adapter_files = sorted(expected_files, key=utf8_key) if adapter_files_pass else []

    if not adapter_files_pass:
        reason_codes.append("ADAPTER_FILE_SET")

    if isinstance(artifact_files, list):
        forbidden_full_model_files = {
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.bin"
        }

        if any(file_name in forbidden_full_model_files for file_name in artifact_files):
            reason_codes.append("FULL_MODEL_ARTIFACT")

    # ---------------- Checkpoint ----------------

    checkpoint = payload.get("checkpoint")

    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and set(checkpoint.keys()) == {
            "model",
            "optimizer",
            "scheduler",
            "step",
            "rng",
            "dataPosition"
        }
    )

    if not checkpoint_complete:
        reason_codes.append("INCOMPLETE_CHECKPOINT")

    # ---------------- Lineage ----------------

    base_revision = payload.get("baseRevision")
    dataset_digest = payload.get("datasetDigest")
    code_digest = payload.get("codeDigest")
    config_digest = payload.get("configDigest")
    expected_digests = payload.get("expectedDigests")

    lineage_pass = (
        isinstance(base_revision, str)
        and BASE_REVISION_RE.fullmatch(base_revision) is not None
        and isinstance(dataset_digest, str)
        and DIGEST_RE.fullmatch(dataset_digest) is not None
        and isinstance(code_digest, str)
        and DIGEST_RE.fullmatch(code_digest) is not None
        and isinstance(config_digest, str)
        and DIGEST_RE.fullmatch(config_digest) is not None
        and isinstance(expected_digests, dict)
        and expected_digests.get("datasetDigest") == dataset_digest
        and expected_digests.get("codeDigest") == code_digest
        and expected_digests.get("configDigest") == config_digest
    )

    if (
        not isinstance(base_revision, str)
        or BASE_REVISION_RE.fullmatch(base_revision or "") is None
    ):
        reason_codes.append("MUTABLE_BASE_REVISION")

    if not lineage_pass:
        reason_codes.append("LINEAGE_MISMATCH")

    # ---------------- Effective batch ----------------

    micro_batch = payload.get("microBatch")
    gradient_accumulation = payload.get("gradientAccumulation")
    replicas = payload.get("replicas")
    expected_batch = payload.get("expectedEffectiveBatch")

    batch_values_valid = all(
        is_safe_positive_integer(value)
        for value in [
            micro_batch,
            gradient_accumulation,
            replicas,
            expected_batch
        ]
    )

    effective_batch_pass = (
        batch_values_valid
        and micro_batch * gradient_accumulation * replicas == expected_batch
    )

    if not effective_batch_pass:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")

    # ---------------- Evaluation isolation ----------------

    train_row_ids = payload.get("trainRowIds")
    eval_row_ids = payload.get("evalRowIds")

    train_ids_valid = (
        isinstance(train_row_ids, list)
        and len(train_row_ids) > 0
        and all(isinstance(value, str) and value != "" for value in train_row_ids)
        and len(set(train_row_ids)) == len(train_row_ids)
    )

    eval_ids_valid = (
        isinstance(eval_row_ids, list)
        and len(eval_row_ids) > 0
        and all(isinstance(value, str) and value != "" for value in eval_row_ids)
        and len(set(eval_row_ids)) == len(eval_row_ids)
    )

    eval_isolated = (
        train_ids_valid
        and eval_ids_valid
        and set(train_row_ids).isdisjoint(set(eval_row_ids))
    )

    if not eval_isolated:
        reason_codes.append("EVAL_LEAKAGE")

    evaluation_deterministic = payload.get("dropoutActiveDuringEval") is False

    if not evaluation_deterministic:
        reason_codes.append("EVAL_DROPOUT_ACTIVE")

    # ---------------- Resume verification ----------------

    uninterrupted_weights = payload.get("uninterruptedWeights")
    resumed_weights = payload.get("resumedWeights")
    tolerance = payload.get("resumeTolerance")

    resume_arrays_valid = (
        isinstance(uninterrupted_weights, list)
        and isinstance(resumed_weights, list)
        and len(uninterrupted_weights) > 0
        and len(uninterrupted_weights) == len(resumed_weights)
        and all(is_finite_number(value) for value in uninterrupted_weights)
        and all(is_finite_number(value) for value in resumed_weights)
        and is_finite_number(tolerance)
        and tolerance >= 0
    )

    resume_pass = resume_arrays_valid

    if resume_arrays_valid:
        for original, resumed in zip(uninterrupted_weights, resumed_weights):
            if abs(original - resumed) > tolerance:
                resume_pass = False
                break

    if not resume_pass:
        reason_codes.append("RESUME_DIVERGENCE")

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": sort_codes(reason_codes)
    }


# =========================================================
# API ROUTES
# =========================================================

@app.post("/adapt")
def adapt():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "INVALID_INPUT"}), 400

    operation = payload.get("operation")

    if operation == "choose":
        return jsonify(choose_intervention(payload)), 200

    if operation == "repair":
        return jsonify(repair_peft_run(payload)), 200

    return jsonify({"error": "INVALID_INPUT"}), 400


@app.get("/")
def home():
    return jsonify({
        "status": "running",
        "endpoint": "POST /adapt",
        "operations": ["choose", "repair"]
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)