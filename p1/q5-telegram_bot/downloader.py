"""Safe asynchronous downloading with size limits."""
from pathlib import Path
from urllib.parse import urlparse

import httpx


class DownloadError(RuntimeError):
    pass


async def download(url: str, destination: Path, client: httpx.AsyncClient, max_bytes: int) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError("Only public HTTP(S) URLs are supported")
    name = Path(parsed.path).name or "dataset"
    target = destination / name
    total = 0
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with target.open("wb") as output:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise DownloadError("Download exceeds configured size limit")
                    output.write(chunk)
    except httpx.HTTPError as exc:
        raise DownloadError(f"Download failed: {exc}") from exc
    return target
