import openai
from app.llm.base import LLMClient

class OpenAIClient(LLMClient):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = openai.api_key  # set via env or config

    def ask(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message["content"]

    def stream(self, prompt: str):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True
        )
        for chunk in response:
            if "choices" in chunk:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]
