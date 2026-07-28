import httpx
import pytest

from dataset_loader import load_dataset


@pytest.mark.asyncio
async def test_load_csv_from_public_url() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="amount,name\n2,a\n"))
    async with httpx.AsyncClient(transport=transport) as client:
        frame = await load_dataset("https://example.org/sample.csv", client, 1_000)
    assert frame.to_dict(orient="records") == [{"amount": 2, "name": "a"}]
