import ollama
from app.llm.base import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def ask(self, prompt: str) -> str:
        stream = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        return "".join(chunk.get("message", {}).get("content", "") for chunk in stream)

    def stream(self, prompt: str):
        stream = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            yield token
