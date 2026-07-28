"""Append-only JSONL request logs."""
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiofiles
import orjson


class RunLogger:
    def __init__(self, directory: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or f"run_{uuid4()}"
        self.path = directory / f"{self.run_id}.jsonl"
        self._lock = asyncio.Lock()

    async def event(self, event: str, **data: object) -> None:
        row = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **data}
        async with self._lock:
            async with aiofiles.open(self.path, "ab") as stream:
                await stream.write(orjson.dumps(row) + b"\n")
