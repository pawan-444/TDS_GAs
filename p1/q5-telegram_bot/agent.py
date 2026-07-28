"""Coordinator for planning, loading, analysis, and optional LLM reasoning."""
from typing import Any

import httpx

from analysis_engine import AnalysisError, analyse
from answer_shaper import decode_llm_answer, requested_answer_shape
from conversation import ConversationStore
from dataset_loader import load_dataset
from llm import AIpipeClient
from logger import RunLogger
from planner import make_plan
from settings import Settings
from utils import question_without_urls


class DataAgent:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, conversations: ConversationStore) -> None:
        self._settings = settings
        self._client = client
        self._conversations = conversations
        self._llm = AIpipeClient(settings, client)

    def log_url(self, run_id: str) -> str:
        return f"{str(self._settings.public_url).rstrip('/')}/logs/{run_id}.jsonl"

    async def run(self, user_id: int, message: str) -> tuple[Any, str]:
        logger = RunLogger(self._settings.log_directory)
        await logger.event("incoming_request", user_id=user_id, message=message)
        try:
            plan = make_plan(message)
            await logger.event("planning", plan=plan.model_dump())
            if plan.needs_download:
                frames = []
                for url in plan.urls:
                    await logger.event("download", url=url)
                    frames.append(await load_dataset(url, self._client, self._settings.max_download_bytes))
                frame = frames[0]
                if len(frames) > 1 and "join" in message.lower():
                    for other in frames[1:]:
                        common = [column for column in frame.columns if column in other.columns]
                        if not common:
                            raise AnalysisError("Datasets have no shared column for a join")
                        frame = frame.merge(other, on=common[0], how="inner")
                    await logger.event("python_execution", operation="join", rows=len(frame))
                await logger.event("dataset_loaded", rows=len(frame), columns=list(map(str, frame.columns)))
                question = question_without_urls(message)
                answer = requested_answer_shape(question, analyse(frame, question))
                await logger.event("python_execution", answer=answer)
            else:
                history = "\n".join(f"{turn.role}: {turn.content}" for turn in self._conversations.history(user_id))
                await logger.event("llm_request")
                answer = requested_answer_shape(message, decode_llm_answer(await self._llm.answer(message, history)))
                await logger.event("llm_response", answer=answer)
            self._conversations.add(user_id, "user", message)
            self._conversations.add(user_id, "assistant", str(answer))
        except Exception as exc:
            answer = {"error": str(exc)}
            await logger.event("error", error=str(exc), error_type=type(exc).__name__)
        log_url = self.log_url(logger.run_id)
        await logger.event("final_response", answer=answer, log_url=log_url)
        return answer, log_url
