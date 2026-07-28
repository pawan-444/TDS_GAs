import httpx
import pytest

from llm import AIpipeClient
from settings import Settings


@pytest.mark.asyncio
async def test_responses_api_output() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"output_text": "hello"}))
    settings = Settings(ai_pipe_token="token", telegram_bot_token="telegram", public_url="https://example.org")
    async with httpx.AsyncClient(transport=transport) as client:
        assert await AIpipeClient(settings, client).answer("hi") == "hello"
