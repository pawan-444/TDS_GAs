from flask import Flask, jsonify, request
import copy
import hashlib
import json

app = Flask(__name__)

MAX_SAFE_INTEGER = 9007199254740991

DAG = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

EVENT_STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}

REQUIRED_INPUTS = {
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
}

EVENT_FIELDS = {
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
}

# State is persistent for the running Flask process and isolated by session.
SESSIONS = {}


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json_array(values):
    payload = compact_json(values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_positive_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


def is_nonempty_string(value):
    return isinstance(value, str) and value != ""


def error_response(code):
    return jsonify({"error": code}), 409


def fresh_nodes():
    return {
        node: {
            "status": None,
            "attempt": None,
            "eventId": None,
            "artifactDigest": None,
            "key": None,
        }
        for node in DAG
    }


def valid_request_shape(payload):
    if not isinstance(payload, dict):
        return False

    if not is_nonempty_string(payload.get("session")):
        return False

    if not is_positive_safe_integer(payload.get("revision")):
        return False

    if not isinstance(payload.get("inputs"), dict):
        return False

    if not isinstance(payload.get("events"), list):
        return False

    inputs = payload["inputs"]

    for field in REQUIRED_INPUTS:
        if not is_nonempty_string(inputs.get(field)):
            return False

    return True


def get_node_keys(inputs, cache):
    """
    Returns:
      keys[node]: current content-addressed key, or None if parent is unavailable
      dependencies[node]: named dependency values plus cacheKey
      available[node]: whether the parent dependency permits this node to run
    """
    keys = {node: None for node in DAG}
    dependencies = {}
    available = {node: False for node in DAG}

    verify_key = sha256_json_array(
        [
            inputs["generation"],
            inputs["checksum"],
        ]
    )

    keys["verify_data"] = verify_key
    available["verify_data"] = True
    dependencies["verify_data"] = {
        "generation": inputs["generation"],
        "checksum": inputs["checksum"],
        "cacheKey": verify_key,
    }

    verify_artifact = cache.get(verify_key, {}).get("artifactDigest")

    if verify_artifact is not None:
        prepare_key = sha256_json_array(
            [
                inputs["canonicalData"],
                inputs["prepareCode"],
                inputs["prepareConfig"],
            ]
        )

        keys["prepare"] = prepare_key
        available["prepare"] = True

        dependencies["prepare"] = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
            "cacheKey": prepare_key,
        }
    else:
        dependencies["prepare"] = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
            "cacheKey": None,
        }

    prepare_artifact = None
    if keys["prepare"] is not None:
        prepare_artifact = cache.get(keys["prepare"], {}).get("artifactDigest")

    if prepare_artifact is not None:
        train_key = sha256_json_array(
            [
                prepare_artifact,
                inputs["trainCode"],
                inputs["trainConfig"],
                inputs["runtime"],
            ]
        )

        keys["train"] = train_key
        available["train"] = True

        dependencies["train"] = {
            "prepareArtifact": prepare_artifact,
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
            "cacheKey": train_key,
        }
    else:
        dependencies["train"] = {
            "prepareArtifact": None,
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
            "cacheKey": None,
        }

    train_artifact = None
    if keys["train"] is not None:
        train_artifact = cache.get(keys["train"], {}).get("artifactDigest")

    if train_artifact is not None:
        evaluate_key = sha256_json_array(
            [
                train_artifact,
                inputs["canonicalData"],
                inputs["evaluateCode"],
                inputs["evaluateConfig"],
            ]
        )

        keys["evaluate"] = evaluate_key
        available["evaluate"] = True

        dependencies["evaluate"] = {
            "trainArtifact": train_artifact,
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
            "cacheKey": evaluate_key,
        }
    else:
        dependencies["evaluate"] = {
            "trainArtifact": None,
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
            "cacheKey": None,
        }

    evaluate_artifact = None
    if keys["evaluate"] is not None:
        evaluate_artifact = cache.get(keys["evaluate"], {}).get("artifactDigest")

    if evaluate_artifact is not None:
        register_key = sha256_json_array(
            [
                evaluate_artifact,
                inputs["schemaDigest"],
            ]
        )

        keys["register"] = register_key
        available["register"] = True

        dependencies["register"] = {
            "evaluateArtifact": evaluate_artifact,
            "schemaDigest": inputs["schemaDigest"],
            "cacheKey": register_key,
        }
    else:
        dependencies["register"] = {
            "evaluateArtifact": None,
            "schemaDigest": inputs["schemaDigest"],
            "cacheKey": None,
        }

    register_artifact = None
    if keys["register"] is not None:
        register_artifact = cache.get(keys["register"], {}).get("artifactDigest")

    if register_artifact is not None:
        publish_key = sha256_json_array(
            [
                register_artifact,
                inputs["publishConfig"],
            ]
        )

        keys["publish"] = publish_key
        available["publish"] = True

        dependencies["publish"] = {
            "registerArtifact": register_artifact,
            "publishConfig": inputs["publishConfig"],
            "cacheKey": publish_key,
        }
    else:
        dependencies["publish"] = {
            "registerArtifact": None,
            "publishConfig": inputs["publishConfig"],
            "cacheKey": None,
        }

    return keys, dependencies, available


def structurally_valid_event(event):
    """
    Invalid event structure causes HTTP 409 INVALID_EVENT.
    Semantically invalid but structurally valid events are ignored.
    """
    if not isinstance(event, dict):
        return False

    if set(event.keys()) != EVENT_FIELDS:
        return False

    if not is_nonempty_string(event.get("eventId")):
        return False

    if not is_positive_safe_integer(event.get("revision")):
        return False

    if not isinstance(event.get("node"), str):
        return False

    if not isinstance(event.get("attempt"), int) or isinstance(
        event.get("attempt"), bool
    ):
        return False

    if not isinstance(event.get("status"), str):
        return False

    if event.get("key") is not None and not isinstance(event.get("key"), str):
        return False

    if event.get("artifactDigest") is not None and not isinstance(
        event.get("artifactDigest"), str
    ):
        return False

    if event.get("receiptId") is not None and not isinstance(
        event.get("receiptId"), str
    ):
        return False

    return True


def semantically_valid_event(event, keys, available):
    """
    Returns True only for events that are meaningful for the current revision,
    current key, current parent availability, artifact rules, and receipt rules.

    A semantic failure is ignored, not rejected.
    """
    node = event["node"]
    status = event["status"]
    attempt = event["attempt"]
    key = event["key"]
    artifact_digest = event["artifactDigest"]
    receipt_id = event["receiptId"]

    if node not in DAG:
        return False

    if status not in EVENT_STATUSES:
        return False

    if not is_positive_safe_integer(attempt):
        return False

    if not available.get(node, False):
        return False

    if key != keys.get(node):
        return False

    if status == "succeeded":
        if not is_nonempty_string(artifact_digest):
            return False
    elif artifact_digest is not None:
        return False

    if node in {"register", "publish"} and status == "succeeded":
        expected_receipt = f"receipt:{node}:{key}"
        if receipt_id != expected_receipt:
            return False
    elif receipt_id is not None:
        return False

    return True


def node_response(node, keys, dependencies, cache, nodes):
    key = keys[node]
    state = nodes[node]

    if key is not None and key in cache:
        return {
            "node": node,
            "action": "reuse",
            "reasonCodes": ["CACHE_HIT"],
            "dependencyDigests": dependencies[node],
            "triggeringEventIds": [cache[key]["eventId"]],
        }

    if state["status"] == "terminal_failed":
        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["TERMINAL_FAILURE"],
            "dependencyDigests": dependencies[node],
            "triggeringEventIds": [state["eventId"]],
        }

    if state["status"] == "started":
        return {
            "node": node,
            "action": "block",
            "reasonCodes": ["RUNNING"],
            "dependencyDigests": dependencies[node],
            "triggeringEventIds": [state["eventId"]],
        }

    node_index = DAG.index(node)

    for upstream_node in DAG[:node_index]:
        upstream_state = nodes[upstream_node]
        upstream_key = keys[upstream_node]

        if upstream_state["status"] == "terminal_failed":
            return {
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_TERMINAL"],
                "dependencyDigests": dependencies[node],
                "triggeringEventIds": [upstream_state["eventId"]],
            }

        if upstream_key is None or upstream_key not in cache:
            return {
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_PENDING"],
                "dependencyDigests": dependencies[node],
                "triggeringEventIds": [],
            }

    if state["status"] == "retryable_failed":
        return {
            "node": node,
            "action": "rerun",
            "reasonCodes": ["RETRYABLE_FAILURE"],
            "dependencyDigests": dependencies[node],
            "triggeringEventIds": [state["eventId"]],
        }

    return {
        "node": node,
        "action": "rerun",
        "reasonCodes": ["CACHE_MISS"],
        "dependencyDigests": dependencies[node],
        "triggeringEventIds": [],
    }


@app.post("/pipeline")
def pipeline():
    payload = request.get_json(silent=True)

    if not valid_request_shape(payload):
        return error_response("INVALID_REQUEST")

    session_name = payload["session"]
    revision = payload["revision"]
    inputs = payload["inputs"]
    incoming_inputs_canonical = compact_json(inputs)

    if session_name not in SESSIONS:
        SESSIONS[session_name] = {
            "revision": revision,
            "inputsCanonical": incoming_inputs_canonical,
            "inputs": copy.deepcopy(inputs),
            "nodes": fresh_nodes(),
            "cache": {},
            "eventIds": {},
        }

    stored = SESSIONS[session_name]

    if revision == stored["revision"]:
        if incoming_inputs_canonical != stored["inputsCanonical"]:
            return error_response("REVISION_CONFLICT")
    else:
        # New revision clears in-progress / failure state, but keeps cache
        # and globally seen event IDs for this session.
        stored["revision"] = revision
        stored["inputsCanonical"] = incoming_inputs_canonical
        stored["inputs"] = copy.deepcopy(inputs)
        stored["nodes"] = fresh_nodes()

    # Work on a copy so every 409 response rolls back the whole event batch.
    working = copy.deepcopy(stored)

    accepted_event_ids = []
    ignored_event_ids = []

    for event in payload["events"]:
        if not structurally_valid_event(event):
            return error_response("INVALID_EVENT")

        event_id = event["eventId"]
        canonical_event = compact_json(event)

        # Event IDs are global within one session.
        if event_id in working["eventIds"]:
            if working["eventIds"][event_id] == canonical_event:
                ignored_event_ids.append(event_id)
                continue

            return error_response("EVENT_ID_CONFLICT")

        # Well-formed event from another revision is ignored and does not
        # permanently consume its ID.
        if event["revision"] != working["revision"]:
            ignored_event_ids.append(event_id)
            continue

        keys, dependencies, available = get_node_keys(
            working["inputs"],
            working["cache"],
        )

        if not semantically_valid_event(event, keys, available):
            ignored_event_ids.append(event_id)
            continue

        node = event["node"]
        key = event["key"]
        status = event["status"]
        attempt = event["attempt"]
        state = working["nodes"][node]

        # A successful content-addressed key is immutable.
        if key in working["cache"]:
            cached = working["cache"][key]

            if (
                status == "succeeded"
                and event["artifactDigest"] != cached["artifactDigest"]
            ):
                return error_response("EVIDENCE_CONFLICT")

            return error_response("STATUS_CONFLICT")

        if state["status"] == "terminal_failed":
            return error_response("STATUS_CONFLICT")

        # none -> started(1) is accepted.
        # A completion without a first start, or started attempt > 1, is ignored.
        if state["status"] is None:
            if status != "started" or attempt != 1:
                ignored_event_ids.append(event_id)
                continue

            state["status"] = "started"
            state["attempt"] = attempt
            state["eventId"] = event_id
            state["artifactDigest"] = None
            state["key"] = key

            working["eventIds"][event_id] = canonical_event
            accepted_event_ids.append(event_id)
            continue

        # Older attempt messages never modify state.
        if attempt < state["attempt"]:
            ignored_event_ids.append(event_id)
            continue

        # started(n) -> completed(n)
        if state["status"] == "started":
            if attempt != state["attempt"]:
                return error_response("STATUS_CONFLICT")

            if status not in {
                "succeeded",
                "retryable_failed",
                "terminal_failed",
            }:
                return error_response("STATUS_CONFLICT")

            state["status"] = status
            state["attempt"] = attempt
            state["eventId"] = event_id
            state["artifactDigest"] = event["artifactDigest"]
            state["key"] = key

            if status == "succeeded":
                working["cache"][key] = {
                    "artifactDigest": event["artifactDigest"],
                    "eventId": event_id,
                }

            working["eventIds"][event_id] = canonical_event
            accepted_event_ids.append(event_id)
            continue

        # retryable_failed(n) -> started(n+1)
        if state["status"] == "retryable_failed":
            if status != "started" or attempt != state["attempt"] + 1:
                return error_response("STATUS_CONFLICT")

            state["status"] = "started"
            state["attempt"] = attempt
            state["eventId"] = event_id
            state["artifactDigest"] = None
            state["key"] = key

            working["eventIds"][event_id] = canonical_event
            accepted_event_ids.append(event_id)
            continue

        return error_response("STATUS_CONFLICT")

    SESSIONS[session_name] = working

    keys, dependencies, available = get_node_keys(
        working["inputs"],
        working["cache"],
    )

    nodes = [
        node_response(
            node,
            keys,
            dependencies,
            working["cache"],
            working["nodes"],
        )
        for node in DAG
    ]

    return jsonify(
        {
            "revision": working["revision"],
            "acceptedEventIds": accepted_event_ids,
            "ignoredEventIds": ignored_event_ids,
            "nodes": nodes,
        }
    ), 200


@app.get("/")
def home():
    return jsonify(
        {
            "status": "running",
            "endpoint": "POST /pipeline",
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
