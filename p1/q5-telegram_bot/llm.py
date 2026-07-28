"""AI Pipe Responses API client with Chat Completions fallback."""
import httpx

from settings import Settings


class LLMError(RuntimeError):
    pass


class AIpipeClient:
    base_url = "https://aipipe.org/openrouter/v1"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings, self._client = settings, client

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._settings.ai_pipe_token.get_secret_value()}"}

    async def answer(self, question: str, context: str = "") -> str:
        instruction = (
            "You are a data-analysis reasoning assistant. Answer accurately using any data in the prompt and context. "
            "Return only valid JSON for the answer value requested by the user. Do not add Markdown or commentary. "
            "If the requested output has answer/log_url keys, return the value of answer only."
        )
        payload = {"model": self._settings.model, "input": f"{instruction}\nContext: {context}\nQuestion: {question}"}
        try:
            response = await self._client.post(f"{self.base_url}/responses", headers=self._headers, json=payload)
            response.raise_for_status()
            body = response.json()
            if body.get("output_text"):
                return str(body["output_text"])
            for item in body.get("output", []):
                for content in item.get("content", []):
                    if content.get("text"):
                        return str(content["text"])
        except (httpx.HTTPError, ValueError):
            pass
        fallback = {"model": self._settings.model, "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": f"{context}\n{question}"}]}
        try:
            response = await self._client.post(f"{self.base_url}/chat/completions", headers=self._headers, json=fallback)
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError("AI Pipe did not return an answer") from exc
