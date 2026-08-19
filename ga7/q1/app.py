from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

app = FastAPI()


class ReleaseGateRequest(BaseModel):
    target: str
    event: str
    ref: str
    workflow: Dict[str, Any]
    image: Dict[str, Any]


@app.post("/release-gate")
def release_gate(req: ReleaseGateRequest):
    violations: List[str] = []

    workflow = req.workflow
    image = req.image

    # 1. Exact least-privilege permissions
    expected_permissions = {
        "contents": "read",
        "packages": "write",
        "id-token": "none",
    }

    if workflow.get("permissions") != expected_permissions:
        violations.append("EXCESS_PERMISSION")

    # 2. Pull request must use pull_request, not pull_request_target
    if req.event == "pull_request":
        if workflow.get("trigger") != "pull_request":
            violations.append("UNSAFE_PR_TRIGGER")

    # Also reject pull_request_target for any release request
    if workflow.get("trigger") == "pull_request_target":
        if "UNSAFE_PR_TRIGGER" not in violations:
            violations.append("UNSAFE_PR_TRIGGER")

    # 3. Tests + complete matrix + failFast false
    if (
        workflow.get("testsPassed") is not True
        or workflow.get("matrixComplete") is not True
        or workflow.get("failFast") is not False
    ):
        violations.append("TESTS_INCOMPLETE")

    # 4. Action pinning
    #
    # actions/* may use version tags.
    # Third-party actions must use exactly 40 lowercase hex chars.
    for action in workflow.get("actions", []):
        owner = action.get("owner", "")
        ref = action.get("ref", "")

        if owner != "actions":
            if (
                len(ref) != 40
                or any(c not in "0123456789abcdef" for c in ref)
            ):
                violations.append("MUTABLE_ACTION")
                break

    # 5. Hardened image requirements
    if image.get("multiStage") is not True:
        violations.append("SINGLE_STAGE_IMAGE")

    if image.get("runsAsRoot") is not False:
        violations.append("ROOT_RUNTIME")

    if image.get("secretMode") not in ("none", "buildkit"):
        violations.append("SECRET_IN_LAYER")

    if image.get("criticalVulnerabilities") != 0:
        violations.append("CRITICAL_CVE")

    if image.get("digestPinned") is not True:
        violations.append("UNPINNED_IMAGE")

    # 6. Production-specific requirements
    if req.target == "production":
        if req.event != "push" or req.ref != "refs/heads/main":
            violations.append("INVALID_PRODUCTION_REF")

        if workflow.get("environmentApproval") is not True:
            violations.append("APPROVAL_REQUIRED")

    decision = "promote" if not violations else "block"

    return {
        "decision": decision,
        "violations": violations,
    }