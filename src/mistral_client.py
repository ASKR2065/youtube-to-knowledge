import json
import os
import sys

from mistralai import Mistral


def load_dotenv(env_path: str | None = None) -> None:
    if env_path is None:
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


class MistralClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.model = model or os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        if not self.api_key:
            print(
                "Error: MISTRAL_API_KEY is not set. "
                "Get a key at https://console.mistral.ai/ and run: export MISTRAL_API_KEY=..."
            )
            sys.exit(1)
        self.client = Mistral(api_key=self.api_key)

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            print(f"Error: Mistral API call failed: {e}")
            raise
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Mistral API returned an empty response")
        return content.strip()
