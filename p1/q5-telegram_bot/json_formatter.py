"""Strict public response formatting."""
from typing import Any

import orjson


def response_payload(answer: Any, log_url: str) -> dict[str, Any]:
    return {"answer": answer, "log_url": log_url}


def telegram_json(answer: Any, log_url: str) -> str:
    return orjson.dumps(response_payload(answer, log_url)).decode()
