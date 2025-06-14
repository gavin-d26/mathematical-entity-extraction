"""
Output parsing module for mathematical entity extraction.
"""

import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass


@dataclass
class EntitySpan:
    """Represents an extracted entity span."""

    start: int
    end: int
    tag: str
    text: str


class OutputParser:
    """Parses LLM output to extract entity spans."""

    def __init__(self):
        """Initialize the output parser."""
        # Valid entity tags
        self.valid_tags = {
            "definition",
            "theorem",
            "proof",
            "example",
            "name",
            "reference",
        }

        # Regex pattern to match XML tags
        self.tag_pattern = re.compile(r"<(/?)(\w+)>")

    def extract_entities_from_tagged_text(
        self, tagged_text: str, original_text: str
    ) -> List[EntitySpan]:
        """
        Extract entity spans from XML-tagged text.

        Args:
            tagged_text: Text with XML tags from LLM
            original_text: Original untagged text for span indexing

        Returns:
            List of EntitySpan objects
        """
        entities = []

        # Find all XML tags and their positions
        tag_matches = list(self.tag_pattern.finditer(tagged_text))

        # Stack to handle nested tags
        tag_stack = []

        # Track position offset due to removed tags
        offset = 0

        for match in tag_matches:
            is_closing = bool(match.group(1))  # True if closing tag (has /)
            tag_name = match.group(2)
            tag_pos = match.start()

            if tag_name not in self.valid_tags:
                continue

            if not is_closing:
                # Opening tag
                # Calculate position in original text (accounting for removed tags)
                original_pos = tag_pos - offset
                tag_stack.append((tag_name, original_pos, match.end() - match.start()))
                offset += match.end() - match.start()
            else:
                # Closing tag
                if tag_stack:
                    # Find matching opening tag
                    for i in range(len(tag_stack) - 1, -1, -1):
                        if tag_stack[i][0] == tag_name:
                            # Found matching opening tag
                            start_tag, start_pos, start_tag_len = tag_stack.pop(i)

                            # Calculate end position in original text
                            end_pos = tag_pos - offset

                            # Extract text from original
                            entity_text = original_text[start_pos:end_pos]

                            entities.append(
                                EntitySpan(
                                    start=start_pos,
                                    end=end_pos,
                                    tag=tag_name,
                                    text=entity_text,
                                )
                            )
                            break

                offset += match.end() - match.start()

        return entities

    def find_span_in_original_text(
        self, entity_text: str, original_text: str, start_hint: int = 0
    ) -> Tuple[int, int]:
        """
        Find the character-level span of entity text in the original text.

        Args:
            entity_text: The extracted entity text
            original_text: The original document text
            start_hint: Hint for where to start searching

        Returns:
            Tuple of (start_index, end_index) or (-1, -1) if not found
        """
        # Clean the entity text (remove extra whitespace)
        clean_entity = " ".join(entity_text.split())
        clean_original = " ".join(original_text.split())

        # Try exact match first
        start_idx = original_text.find(entity_text, start_hint)
        if start_idx != -1:
            return start_idx, start_idx + len(entity_text)

        # Try with cleaned text
        start_idx = clean_original.find(clean_entity)
        if start_idx != -1:
            # Map back to original text indices (approximate)
            return start_idx, start_idx + len(clean_entity)

        # If still not found, try fuzzy matching with first few words
        words = entity_text.split()[:3]  # Use first 3 words
        if words:
            search_text = " ".join(words)
            start_idx = original_text.find(search_text, start_hint)
            if start_idx != -1:
                return start_idx, start_idx + len(entity_text)

        return -1, -1

    def parse_llm_output(self, llm_output: str, original_text: str) -> List[EntitySpan]:
        """
        Parse LLM output and extract entity spans with original text indices.

        Args:
            llm_output: Tagged text from LLM
            original_text: Original untagged text

        Returns:
            List of EntitySpan objects with correct indices
        """
        # First extract entities from tagged text
        entities = self.extract_entities_from_tagged_text(llm_output, original_text)

        # Verify and correct spans using original text
        corrected_entities = []
        for entity in entities:
            start, end = self.find_span_in_original_text(
                entity.text, original_text, entity.start
            )

            if start != -1 and end != -1:
                corrected_entities.append(
                    EntitySpan(
                        start=start,
                        end=end,
                        tag=entity.tag,
                        text=original_text[start:end],
                    )
                )

        return corrected_entities

    def filter_invalid_spans(self, entities: List[EntitySpan]) -> List[EntitySpan]:
        """
        Filter out invalid entity spans based on the rules.

        Rules:
        1. Remove name/reference spans that are within proof spans
        2. Remove name/reference spans that are not within any other entity

        Args:
            entities: List of entity spans

        Returns:
            Filtered list of entity spans
        """
        # Sort entities by start position
        entities = sorted(entities, key=lambda x: x.start)

        # Find proof spans
        proof_spans = [e for e in entities if e.tag == "proof"]

        # Find primary entity spans (definition, theorem, example)
        primary_spans = [
            e for e in entities if e.tag in {"definition", "theorem", "example"}
        ]

        filtered_entities = []

        for entity in entities:
            if entity.tag in {"name", "reference"}:
                # Check if within a proof span
                within_proof = any(
                    proof.start <= entity.start and entity.end <= proof.end
                    for proof in proof_spans
                )

                if within_proof:
                    continue  # Skip this entity

                # Check if within a primary entity span
                within_primary = any(
                    primary.start <= entity.start and entity.end <= primary.end
                    for primary in primary_spans
                )

                if not within_primary:
                    continue  # Skip this entity

            filtered_entities.append(entity)

        return filtered_entities
