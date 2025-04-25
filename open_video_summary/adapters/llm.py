import ollama
from re import DOTALL, search
from abc import ABC, abstractmethod


class LLMAdapter:
    @abstractmethod
    def generate_pattern(self, prompt: str, pattern: str, **kwargs) -> str:
        pass

class OllamaAdapter(LLMAdapter):
    def __init__(
        self, model: str = "gemma2", max_attempts: int = 3, attempts_interval: int = 3
    ) -> None:
        self.model = model
        self.max_attempts = max_attempts
        self.attempts_interval = attempts_interval

    def generate_pattern(self, prompt: str, pattern: str, **kwargs) -> str:
        attempt = 0
        while attempt < self.max_attempts:
            try:
                response = ollama.generate(
                    model=self.model, prompt=prompt, **kwargs
                ).get("response")
                result = search(pattern, response, flags=DOTALL)
                if result is not None:
                    return result.group(0)
            except Exception as e:
                attempt += 1
                if attempt == self.max_attempts:
                    raise e
        return ""