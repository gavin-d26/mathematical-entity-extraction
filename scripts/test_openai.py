#!/usr/bin/env python3
"""
Test script for OpenAI API integration.
"""

import asyncio
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from openai_client import OpenAIClient


async def main():
    """Test OpenAI API integration."""
    print("Testing OpenAI API integration...")

    try:
        # Initialize OpenAI client
        client = OpenAIClient(model_name="gpt-4o-mini")

        # Test connection
        print("Testing connection...")
        if await client.test_connection():
            print("✓ OpenAI API connection successful!")
        else:
            print("✗ OpenAI API connection failed!")
            return

        # Test simple generation
        test_prompt = """Your task is to identify and tag mathematical entities in the provided text. Reproduce the entire text and wrap the entities with XML-style tags.

The available tags are:
- <definition>: Text that defines a new concept or object
- <theorem>: Text that makes a rigorous, provable claim

Input:
Definition: A ring is a set R with two operations + and * such that (R,+) is an abelian group.

Output:"""

        print("\nTesting generation...")
        response = await client.generate_single(
            test_prompt, max_tokens=200, temperature=0.1
        )

        print("Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        print("\n✓ OpenAI integration test completed successfully!")

    except Exception as e:
        print(f"✗ OpenAI integration test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
