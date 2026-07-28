"""Bounded, async-safe in-memory conversation history."""
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


class ConversationStore:
    def __init__(self, limit: int = 12) -> None:
        self._limit = limit
        self._history: defaultdict[int, deque[Turn]] = defaultdict(lambda: deque(maxlen=limit))

    def history(self, user_id: int) -> list[Turn]:
        return list(self._history[user_id])

    def add(self, user_id: int, role: str, content: str) -> None:
        self._history[user_id].append(Turn(role=role, content=content))
