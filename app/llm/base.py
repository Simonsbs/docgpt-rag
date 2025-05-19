from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        pass
