"""AI Pipe client using Responses API with Chat Completions fallback."""

import httpx

from settings import Settings


class LLMError(RuntimeError):
    """Raised when no valid response is returned from the LLM."""


class AIpipeClient:
    BASE_URL = "https://aipipe.org/openai/v1"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.ai_pipe_token.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def answer(self, question: str, context: str = "") -> str:
        instruction = (
            "You are a data-analysis reasoning assistant. "
            "Answer accurately using the supplied context. "
            "Return ONLY valid JSON matching the user's request. "
            "Do not include markdown or explanations."
        )

        # ---------- Try Responses API ----------
        responses_payload = {
            "model": self._settings.model,
            "input": (f"{instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}"),
        }

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/responses",
                headers=self._headers,
                json=responses_payload,
            )
            response.raise_for_status()

            body = response.json()

            # New Responses API format
            if body.get("output_text"):
                return str(body["output_text"])

            # Standard Responses API format
            for item in body.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") != "output_text":
                        continue

                    text = content.get("text")

                    if isinstance(text, str):
                        return text

                    if isinstance(text, dict):
                        value = text.get("value")
                        if value:
                            return str(value)

        except httpx.HTTPError:
            # Fall back to Chat Completions
            pass

        # ---------- Chat Completions Fallback ----------
        fallback_payload = {
            "model": self._settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": instruction,
                },
                {
                    "role": "user",
                    "content": f"{context}\n\n{question}",
                },
            ],
        }

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._headers,
                json=fallback_payload,
            )
            response.raise_for_status()

            body = response.json()

            return str(body["choices"][0]["message"]["content"])

        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError("AI Pipe did not return a valid response.") from exc
