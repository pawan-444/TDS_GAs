"""Deterministic task planning; LLM is reserved for ambiguity."""
from pydantic import BaseModel

from utils import urls_in


class Plan(BaseModel):
    question: str
    urls: list[str]
    needs_download: bool
    needs_python: bool
    needs_llm: bool


def make_plan(message: str) -> Plan:
    urls = urls_in(message)
    question = message
    data_words = ("csv", "excel", "json", "table", "dataset", "data", "column", "rows")
    analytical = ("mean", "median", "sum", "count", "average", "top", "bottom", "group", "missing", "sort", "filter", "mode", "standard deviation", "pivot")
    lower = message.lower()
    needs_python = bool(urls) and (any(word in lower for word in analytical) or any(word in lower for word in data_words))
    return Plan(question=question, urls=urls, needs_download=bool(urls), needs_python=needs_python, needs_llm=not needs_python and not bool(urls))
