from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from data import Data


class ToolEnv(ABC):
    """Базовый класс для среды"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self, data: Data) -> str:
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs
    ):
        pass
