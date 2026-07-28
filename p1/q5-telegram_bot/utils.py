"""Small shared helpers."""
import re
from urllib.parse import urlparse

URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)


def urls_in(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def is_http_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"} and bool(urlparse(value).netloc)


def question_without_urls(text: str) -> str:
    return URL_PATTERN.sub("", text).strip()
