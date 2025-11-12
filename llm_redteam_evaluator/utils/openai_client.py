import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
import httpx


load_dotenv()


class OpenAIClient:
    client = httpx.Client(verify=False)

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL")
        self.base_url = os.getenv("BASE_URL")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        # Use the default OpenAI client with a configured timeout
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, http_client=self.client
        )

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")
