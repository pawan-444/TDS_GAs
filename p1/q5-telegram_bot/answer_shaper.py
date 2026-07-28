"""Extract and preserve answer shapes requested in natural-language prompts."""
import json
from typing import Any


def decode_json_object(text: str) -> Any | None:
    """Return the first JSON object/array embedded in text, if any."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def decode_llm_answer(text: str) -> Any:
    """Use JSON produced by the model; retain plain text only as a fallback."""
    value = decode_json_object(text)
    if isinstance(value, dict) and "answer" in value:
        return value["answer"]
    return value if value is not None else text.strip()


def requested_answer_shape(question: str, value: Any) -> Any:
    """Fill a quoted placeholder in the requested answer object with a result.

    The assignment's outer ``answer``/``log_url`` contract is handled by the
    response formatter. This function only produces the value for ``answer``.
    """
    template = decode_json_object(question)
    if not isinstance(template, dict):
        return value
    inner = template.get("answer", template)
    return _replace_first_placeholder(inner, value)[0]


def _replace_first_placeholder(template: Any, value: Any) -> tuple[Any, bool]:
    if isinstance(template, str) and template.startswith("<") and template.endswith(">"):
        return value, True
    if isinstance(template, list):
        output: list[Any] = []
        replaced = False
        for item in template:
            replacement, did_replace = _replace_first_placeholder(item, value) if not replaced else (item, False)
            output.append(replacement)
            replaced = replaced or did_replace
        return output, replaced
    if isinstance(template, dict):
        output: dict[str, Any] = {}
        replaced = False
        for key, item in template.items():
            replacement, did_replace = _replace_first_placeholder(item, value) if not replaced else (item, False)
            output[key] = replacement
            replaced = replaced or did_replace
        return output, replaced
    return template, False
