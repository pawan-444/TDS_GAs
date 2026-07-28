"""Load common tabular formats into pandas dataframes."""
import asyncio
import shutil
import tempfile
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from downloader import download


class DatasetError(RuntimeError):
    pass


async def load_dataset(url: str, client: httpx.AsyncClient, max_bytes: int) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="telegram-data-") as temp:
        root = Path(temp)
        path = await download(url, root, client, max_bytes)
        try:
            frame = await asyncio.to_thread(_read, path)
        except Exception as exc:
            raise DatasetError(f"Could not read dataset: {exc}") from exc
    if frame.empty:
        raise DatasetError("Dataset contains no rows")
    return frame


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".html", ".htm"}:
        tables = pd.read_html(path)
        if not tables:
            raise DatasetError("No HTML table found")
        return tables[0]
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [item for item in archive.namelist() if Path(item).suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".json"}]
            if not candidates:
                raise DatasetError("ZIP does not contain a supported dataset")
            extracted = Path(tempfile.mkdtemp(prefix="extract-"))
            try:
                member = candidates[0]
                archive.extract(member, extracted)
                return _read(extracted / member)
            finally:
                shutil.rmtree(extracted, ignore_errors=True)
    # Content-disposition-less public data links commonly have no extension.
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise DatasetError(f"Unsupported file type: {suffix or 'unknown'}") from exc
