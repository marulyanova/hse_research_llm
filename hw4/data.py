from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Data:
    # контейнер для эпизода
    question_id: int
    user_messages: str
    initial_state: Dict[str, Any]
    difficulty: int
