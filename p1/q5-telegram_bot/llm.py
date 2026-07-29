"""AI Pipe client with Responses API and Chat Completions fallback."""

import httpx

from settings import Settings


class LLMError(RuntimeError):
    pass


class AIpipeClient:
    base_url = "https://aipipe.org/openrouter/v1"

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
            "Answer accurately using any data in the prompt and context. "
            "Return only valid JSON for the answer requested by the user. "
            "Do not add markdown or commentary."
        )

        # ----------------------------
        # Try Responses API first
        # ----------------------------
        responses_payload = {
            "model": self._settings.model,
            "input": f"{instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}",
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/responses",
                headers=self._headers,
                json=responses_payload,
            )

            print("=" * 60)
            print("RESPONSES API")
            print("Status:", response.status_code)
            print("Body:", response.text)
            print("=" * 60)

            response.raise_for_status()

            body = response.json()

            if body.get("output_text"):
                return body["output_text"]

            for item in body.get("output", []):
                for content in item.get("content", []):
                    text = content.get("text")

                    if isinstance(text, str):
                        return text

                    if isinstance(text, dict):
                        if "value" in text:
                            return text["value"]

        except Exception as exc:
            print(f"Responses API failed: {exc}")
            # DO NOT raise here.
            # Continue to Chat Completions fallback.

        # ----------------------------
        # Chat Completions fallback
        # ----------------------------
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
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=fallback_payload,
            )

            print("=" * 60)
            print("CHAT COMPLETIONS")
            print("Status:", response.status_code)
            print("Body:", response.text)
            print("=" * 60)

            response.raise_for_status()

            body = response.json()

            return body["choices"][0]["message"]["content"]

        except Exception as exc:
            raise LLMError(f"Both Responses API and Chat Completions failed.\n{exc}") from exc
