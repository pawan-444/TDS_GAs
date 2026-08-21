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
            "Authorization": (f"Bearer {self._settings.ai_pipe_token.get_secret_value()}"),
            "Content-Type": "application/json",
        }

    async def answer(self, question: str, context: str = "") -> str:
        instruction = (
            "You are a data-analysis reasoning assistant. "
            "Answer accurately using the supplied context. "
            "Return ONLY valid JSON matching the user's request. "
            "Do not include markdown or explanations."
        )

        prompt = f"{instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}"

        # 1. Responses API

        responses_payload = {
            "model": self._settings.model,
            "input": prompt,
        }

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/responses",
                headers=self._headers,
                json=responses_payload,
            )

            response.raise_for_status()

            body = response.json()

            # Common Responses API shortcut
            output_text = body.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()

            # Responses API output structure
            for item in body.get("output", []):
                for content in item.get("content", []):
                    text = content.get("text")

                    if isinstance(text, str) and text.strip():
                        return text.strip()

                    if isinstance(text, dict):
                        value = text.get("value")

                        if isinstance(value, str) and value.strip():
                            return value.strip()

            # Request succeeded but response contained no usable text.
            raise LLMError(f"AI Pipe Responses API returned no usable text. Response: {body}")

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:1000]

            # Continue to Chat Completions, but preserve the reason
            responses_error = f"Responses API HTTP {status}: {detail}"

        except httpx.RequestError as exc:
            responses_error = f"Responses API request error: {exc}"

        except ValueError as exc:
            responses_error = f"Responses API returned invalid JSON: {exc}"

        except LLMError as exc:
            responses_error = str(exc)

        # 2. Chat Completions fallback

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

            choices = body.get("choices")

            if not isinstance(choices, list) or not choices:
                raise LLMError(f"AI Pipe Chat Completions returned no choices. Response: {body}")

            message = choices[0].get("message", {})
            content = message.get("content")

            if isinstance(content, str) and content.strip():
                return content.strip()

            raise LLMError(f"AI Pipe Chat Completions returned no usable content. Response: {body}")

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:1000]

            raise LLMError(
                f"AI Pipe Chat Completions HTTP {status}: {detail}. Responses API error: {responses_error}"
            ) from exc

        except httpx.RequestError as exc:
            raise LLMError(
                f"AI Pipe Chat Completions request error: {exc}. Responses API error: {responses_error}"
            ) from exc

        except ValueError as exc:
            raise LLMError(
                f"AI Pipe Chat Completions returned invalid JSON: {exc}. Responses API error: {responses_error}"
            ) from exc

        except LLMError as exc:
            raise LLMError(f"{exc}. Responses API error: {responses_error}") from exc
