"""
OpenAI API client module for mathematical entity extraction.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Warning: openai library not found. Install with: pip install openai")
    AsyncOpenAI = None


class OpenAIClient:
    """Client for OpenAI API inference using async client."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI async client.

        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if None, loads from environment)
        """
        if AsyncOpenAI is None:
            raise ImportError(
                "openai library is required. Install with: pip install openai"
            )

        # Load environment variables from .env file
        load_dotenv()

        self.model_name = model_name

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize OpenAI async client
        print(f"Initializing OpenAI async client with model: {model_name}")
        self.client = AsyncOpenAI(api_key=api_key)
        print("OpenAI async client initialization complete")

    async def generate_single(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.1
    ) -> str:
        """
        Generate a single completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""

    async def generate_batch(
        self, prompts: List[str], max_tokens: int = 4096, temperature: float = 0.1
    ) -> List[str]:
        """
        Generate completions for a batch of prompts using async processing.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        print(f"Processing batch of {len(prompts)} prompts with OpenAI API (async)")

        # Create async tasks for all prompts
        tasks = [
            self.generate_single(prompt, max_tokens, temperature) for prompt in prompts
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        print(f"Batch processing complete: {len(results)} results generated")
        return results

    async def test_connection(self) -> bool:
        """
        Test if the OpenAI API is working properly.

        Returns:
            True if API is working, False otherwise
        """
        try:
            test_prompt = "Hello, world!"
            result = await self.generate_single(test_prompt, max_tokens=10)
            return len(result) > 0
        except Exception as e:
            print(f"OpenAI API test failed: {e}")
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the model configuration.

        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "backend": "openai",
        }
