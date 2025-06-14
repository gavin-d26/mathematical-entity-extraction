"""
Data loading and preprocessing module for mathematical entity extraction.
"""

import json
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path


class DataLoader:
    """Handles loading and preprocessing of the mathematical entity extraction dataset."""

    def __init__(self, data_dir: str = "A2-NLP_244"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.file_contents = None
        self.train_df = None
        self.val_df = None

    def load_data(self) -> None:
        """Load all data files into memory."""
        # Load file contents
        with open(self.data_dir / "file_contents.json", "r", encoding="utf-8") as f:
            self.file_contents = json.load(f)

        # Load training and validation annotations
        self.train_df = pd.read_json(self.data_dir / "train.json")
        self.val_df = pd.read_json(self.data_dir / "val.json")

        print(f"Loaded {len(self.file_contents)} documents")
        print(f"Loaded {len(self.train_df)} training annotations")
        print(f"Loaded {len(self.val_df)} validation annotations")

    def get_document_text(self, fileid: str) -> str:
        """Get the raw text for a given file ID."""
        if self.file_contents is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.file_contents[fileid]

    def get_annotations_for_file(
        self, fileid: str, split: str = "train"
    ) -> pd.DataFrame:
        """
        Get all annotations for a specific file ID.

        Args:
            fileid: The file identifier
            split: Either "train" or "val"

        Returns:
            DataFrame with annotations for the specified file
        """
        df = self.train_df if split == "train" else self.val_df
        return df[df["fileid"] == fileid].copy()

    def create_tagged_text(self, fileid: str, split: str = "train") -> str:
        """
        Create XML-tagged text from raw text and annotations.

        This function sorts annotations by start index in descending order
        and injects XML tags to handle nested entities correctly.

        Args:
            fileid: The file identifier
            split: Either "train" or "val"

        Returns:
            Text with XML tags around annotated spans
        """
        # Get raw text and annotations
        text = self.get_document_text(fileid)
        annotations = self.get_annotations_for_file(fileid, split)

        if annotations.empty:
            return text

        # Sort annotations by start index in descending order
        # This ensures we process from the end to avoid index shifting issues
        annotations = annotations.sort_values("start", ascending=False)

        # Apply tags from end to beginning
        tagged_text = text
        for _, row in annotations.iterrows():
            start, end, tag = row["start"], row["end"], row["tag"]

            # Insert closing tag first, then opening tag
            tagged_text = tagged_text[:end] + f"</{tag}>" + tagged_text[end:]
            tagged_text = tagged_text[:start] + f"<{tag}>" + tagged_text[start:]

        return tagged_text

    def get_file_ids(self, split: str = "train") -> List[str]:
        """Get all unique file IDs for a given split."""
        df = self.train_df if split == "train" else self.val_df
        return df["fileid"].unique().tolist()

    def get_entity_stats(self, split: str = "train") -> Dict[str, int]:
        """Get statistics about entity types in the dataset."""
        df = self.train_df if split == "train" else self.val_df
        return df["tag"].value_counts().to_dict()

    def create_few_shot_examples(
        self, num_examples: int = 3, split: str = "train"
    ) -> List[Tuple[str, str]]:
        """
        Create few-shot examples for prompting with balanced entity type coverage.

        This method ensures that examples include all entity types:
        - Entities: definition, theorem, proof, example
        - Identifiers: name, reference (must appear within other entities)

        Args:
            num_examples: Number of examples to create
            split: Data split to use

        Returns:
            List of (input_text, tagged_output_text) tuples
        """
        file_ids = self.get_file_ids(split)
        df = self.train_df if split == "train" else self.val_df

        # Define required entity types
        entity_types = {"definition", "theorem", "proof", "example"}
        identifier_types = {"name", "reference"}
        all_types = entity_types | identifier_types

        # Find files that contain each entity type
        files_by_type = {}
        for entity_type in all_types:
            files_with_type = df[df["tag"] == entity_type]["fileid"].unique().tolist()
            files_by_type[entity_type] = files_with_type

        # Score files by entity type diversity and annotation count
        file_scores = {}
        for fileid in file_ids:
            file_annotations = self.get_annotations_for_file(fileid, split)
            if file_annotations.empty:
                continue

            # Count unique entity types in this file
            file_entity_types = set(file_annotations["tag"].unique())

            # Score based on:
            # 1. Number of different entity types (diversity)
            # 2. Total number of annotations (richness)
            # 3. Bonus for having both entities and identifiers
            diversity_score = len(file_entity_types)
            richness_score = len(file_annotations)

            # Bonus for having both main entities and identifiers
            has_entities = bool(file_entity_types & entity_types)
            has_identifiers = bool(file_entity_types & identifier_types)
            balance_bonus = 2 if (has_entities and has_identifiers) else 0

            file_scores[fileid] = (
                diversity_score * 2 + richness_score * 0.1 + balance_bonus
            )

        # Sort files by score (highest first)
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

        # Select examples ensuring coverage of all types
        examples = []
        covered_types = set()

        # First pass: prioritize files that cover missing types
        for fileid, score in sorted_files:
            if len(examples) >= num_examples:
                break

            file_annotations = self.get_annotations_for_file(fileid, split)
            file_types = set(file_annotations["tag"].unique())

            # Check if this file adds new entity types we haven't covered
            new_types = file_types - covered_types

            # Include if it adds new types or if we still need more examples
            if new_types or len(examples) < num_examples:
                input_text = self.get_document_text(fileid)
                tagged_text = self.create_tagged_text(fileid, split)
                examples.append((input_text, tagged_text))
                covered_types.update(file_types)

                print(f"Selected file {fileid} with types: {sorted(file_types)}")

        # Report coverage
        missing_types = all_types - covered_types
        if missing_types:
            print(f"Warning: Missing entity types in examples: {sorted(missing_types)}")
        else:
            print(f"✓ All entity types covered: {sorted(covered_types)}")

        return examples
