"""
Prompt building module for mathematical entity extraction.
"""

from typing import List, Tuple


class PromptBuilder:
    """Builds prompts for mathematical entity extraction using few-shot learning."""

    def __init__(self):
        """Initialize the prompt builder."""
        self.instruction = self._create_instruction()

    def _create_instruction(self) -> str:
        """Create the instruction part of the prompt."""
        return """Your task is to identify and tag mathematical entities in the provided text. Reproduce the entire text and wrap the entities with XML-style tags.

The available tags are:
- <definition>: Text that defines a new concept or object
- <theorem>: Text that makes a rigorous, provable claim  
- <proof>: Text that proves a theorem
- <example>: An example of a defined object or application of a theorem
- <name>: The name of a newly defined object or a theorem
- <reference>: A reference to a previously defined name

Important rules:
1. Nested tags are allowed and expected
2. <name> and <reference> tags must always be inside other entity tags (definition, theorem, proof, or example)
3. Reproduce the entire input text exactly, only adding the XML tags around identified entities
4. Do not modify the original text in any way"""

    def build_prompt(self, input_text: str, examples: List[Tuple[str, str]]) -> str:
        """
        Build a complete prompt with instruction, examples, and input.

        Args:
            input_text: The text to be tagged
            examples: List of (input, output) example pairs

        Returns:
            Complete prompt string
        """
        prompt_parts = [self.instruction]

        # Add few-shot examples
        if examples:
            prompt_parts.append("\nHere are some examples:\n")

            for i, (example_input, example_output) in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append("Input:")
                prompt_parts.append(example_input)
                prompt_parts.append("\nOutput:")
                prompt_parts.append(example_output)
                prompt_parts.append("\n" + "=" * 50 + "\n")

        # Add the actual input to be processed
        prompt_parts.append("Now, please tag the following text:")
        prompt_parts.append("Input:")
        prompt_parts.append(input_text)
        prompt_parts.append("\nOutput:")

        return "\n".join(prompt_parts)

    def build_batch_prompts(
        self, input_texts: List[str], examples: List[Tuple[str, str]]
    ) -> List[str]:
        """
        Build multiple prompts for batch processing.

        Args:
            input_texts: List of texts to be tagged
            examples: List of (input, output) example pairs

        Returns:
            List of complete prompt strings
        """
        return [self.build_prompt(text, examples) for text in input_texts]
