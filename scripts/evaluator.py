"""
Token-level evaluation module for mathematical entity extraction.
"""

import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
from output_parser import EntitySpan


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int


class TokenLevelEvaluator:
    """Evaluates entity extraction performance at token level."""

    def __init__(self):
        """Initialize the evaluator."""
        # Simple tokenization pattern - splits on whitespace and punctuation
        # but keeps mathematical symbols together
        self.token_pattern = re.compile(r"\S+")

    def tokenize_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Tokenize text and return tokens with their character positions.

        Args:
            text: Input text to tokenize

        Returns:
            List of (token, start_pos, end_pos) tuples
        """
        tokens = []
        for match in self.token_pattern.finditer(text):
            token = match.group()
            start = match.start()
            end = match.end()
            tokens.append((token, start, end))
        return tokens

    def spans_to_token_labels(
        self, text: str, spans: List[EntitySpan]
    ) -> Dict[int, Set[str]]:
        """
        Convert entity spans to token-level labels.

        Args:
            text: Original text
            spans: List of EntitySpan objects

        Returns:
            Dictionary mapping token_index -> set of labels
        """
        tokens = self.tokenize_text(text)
        token_labels = defaultdict(set)

        # For each span, find which tokens it overlaps with
        for span in spans:
            span_start, span_end = span.start, span.end

            for token_idx, (token, token_start, token_end) in enumerate(tokens):
                # Check if token overlaps with span
                if self._tokens_overlap(token_start, token_end, span_start, span_end):
                    token_labels[token_idx].add(span.tag)

        return dict(token_labels)

    def _tokens_overlap(
        self, token_start: int, token_end: int, span_start: int, span_end: int
    ) -> bool:
        """
        Check if a token overlaps with a span.

        Args:
            token_start, token_end: Token boundaries
            span_start, span_end: Span boundaries

        Returns:
            True if they overlap
        """
        # Tokens overlap if they have any character in common
        return not (token_end <= span_start or token_start >= span_end)

    def calculate_token_metrics(
        self,
        predicted_labels: Dict[int, Set[str]],
        ground_truth_labels: Dict[int, Set[str]],
    ) -> EvaluationMetrics:
        """
        Calculate token-level precision, recall, and F1-score.

        Args:
            predicted_labels: Dict mapping token_index -> set of predicted labels
            ground_truth_labels: Dict mapping token_index -> set of ground truth labels

        Returns:
            EvaluationMetrics object
        """
        # Get all token indices that have labels in either prediction or ground truth
        all_token_indices = set(predicted_labels.keys()) | set(
            ground_truth_labels.keys()
        )

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # For each token, compare predicted vs ground truth labels
        for token_idx in all_token_indices:
            pred_labels = predicted_labels.get(token_idx, set())
            gt_labels = ground_truth_labels.get(token_idx, set())

            # Count label-level matches
            tp_labels = pred_labels & gt_labels  # Intersection
            fp_labels = pred_labels - gt_labels  # Predicted but not in GT
            fn_labels = gt_labels - pred_labels  # In GT but not predicted

            true_positives += len(tp_labels)
            false_positives += len(fp_labels)
            false_negatives += len(fn_labels)

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

    def calculate_metrics_by_tag(
        self,
        predicted_labels: Dict[int, Set[str]],
        ground_truth_labels: Dict[int, Set[str]],
    ) -> Dict[str, EvaluationMetrics]:
        """
        Calculate token-level metrics for each entity tag separately.

        Args:
            predicted_labels: Dict mapping token_index -> set of predicted labels
            ground_truth_labels: Dict mapping token_index -> set of ground truth labels

        Returns:
            Dictionary mapping tag names to EvaluationMetrics
        """
        # Get all unique tags
        all_tags = set()
        for labels in list(predicted_labels.values()) + list(
            ground_truth_labels.values()
        ):
            all_tags.update(labels)

        metrics_by_tag = {}

        for tag in all_tags:
            # Create binary label dictionaries for this tag
            pred_binary = {}
            gt_binary = {}

            all_token_indices = set(predicted_labels.keys()) | set(
                ground_truth_labels.keys()
            )

            for token_idx in all_token_indices:
                pred_labels = predicted_labels.get(token_idx, set())
                gt_labels = ground_truth_labels.get(token_idx, set())

                # Binary: does this token have this tag?
                if tag in pred_labels:
                    pred_binary[token_idx] = {tag}
                if tag in gt_labels:
                    gt_binary[token_idx] = {tag}

            # Calculate metrics for this tag
            metrics_by_tag[tag] = self.calculate_token_metrics(pred_binary, gt_binary)

        return metrics_by_tag

    def ground_truth_to_spans(self, annotations_df) -> List[EntitySpan]:
        """
        Convert ground truth annotations DataFrame to EntitySpan objects.

        Args:
            annotations_df: DataFrame with 'start', 'end', 'tag', 'text' columns

        Returns:
            List of EntitySpan objects
        """
        spans = []
        for _, row in annotations_df.iterrows():
            spans.append(
                EntitySpan(
                    start=row["start"], end=row["end"], tag=row["tag"], text=row["text"]
                )
            )
        return spans

    def evaluate_predictions(
        self,
        predictions: Dict[str, List[EntitySpan]],
        ground_truth: Dict[str, List[EntitySpan]],
        document_texts: Dict[str, str],
    ) -> Tuple[EvaluationMetrics, Dict[str, EvaluationMetrics]]:
        """
        Evaluate predictions across multiple documents at token level.

        Args:
            predictions: Dictionary mapping file_id to predicted spans
            ground_truth: Dictionary mapping file_id to ground truth spans
            document_texts: Dictionary mapping file_id to original text

        Returns:
            Tuple of (overall_metrics, per_tag_metrics)
        """
        # Combine token-level labels from all documents
        all_predicted_labels = {}
        all_ground_truth_labels = {}
        token_offset = 0

        for file_id in ground_truth.keys():
            if file_id not in document_texts:
                print(f"Warning: No text found for file_id {file_id}")
                continue

            text = document_texts[file_id]

            # Get spans for this document
            pred_spans = predictions.get(file_id, [])
            gt_spans = ground_truth[file_id]

            # Convert to token-level labels
            pred_token_labels = self.spans_to_token_labels(text, pred_spans)
            gt_token_labels = self.spans_to_token_labels(text, gt_spans)

            # Add to global token label dictionaries with offset
            for token_idx, labels in pred_token_labels.items():
                all_predicted_labels[token_offset + token_idx] = labels

            for token_idx, labels in gt_token_labels.items():
                all_ground_truth_labels[token_offset + token_idx] = labels

            # Update offset for next document
            num_tokens = len(self.tokenize_text(text))
            token_offset += num_tokens

        # Calculate overall metrics
        overall_metrics = self.calculate_token_metrics(
            all_predicted_labels, all_ground_truth_labels
        )

        # Calculate per-tag metrics
        per_tag_metrics = self.calculate_metrics_by_tag(
            all_predicted_labels, all_ground_truth_labels
        )

        return overall_metrics, per_tag_metrics

    def print_evaluation_report(
        self,
        overall_metrics: EvaluationMetrics,
        per_tag_metrics: Dict[str, EvaluationMetrics],
    ) -> None:
        """
        Print a detailed token-level evaluation report.

        Args:
            overall_metrics: Overall evaluation metrics
            per_tag_metrics: Per-tag evaluation metrics
        """
        print("=" * 70)
        print("TOKEN-LEVEL EVALUATION REPORT")
        print("=" * 70)

        # Overall metrics
        print(f"\nOVERALL TOKEN-LEVEL METRICS:")
        print(f"Precision: {overall_metrics.precision:.4f}")
        print(f"Recall:    {overall_metrics.recall:.4f}")
        print(f"F1-Score:  {overall_metrics.f1_score:.4f}")
        print(
            f"True Positives:  {overall_metrics.true_positives} (correct token-label pairs)"
        )
        print(
            f"False Positives: {overall_metrics.false_positives} (incorrect predictions)"
        )
        print(f"False Negatives: {overall_metrics.false_negatives} (missed labels)")

        # Per-tag metrics
        print(f"\nPER-TAG TOKEN-LEVEL METRICS:")
        print(
            f"{'Tag':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}"
        )
        print("-" * 70)

        for tag, metrics in sorted(per_tag_metrics.items()):
            print(
                f"{tag:<12} {metrics.precision:<10.4f} {metrics.recall:<10.4f} "
                f"{metrics.f1_score:<10.4f} {metrics.true_positives:<5} "
                f"{metrics.false_positives:<5} {metrics.false_negatives:<5}"
            )

        print("=" * 70)
        print(
            "Note: Evaluation is performed at token level with support for multi-label tokens."
        )
        print(
            "Overlapping entities are handled by allowing tokens to have multiple labels."
        )

    def debug_tokenization(self, text: str, spans: List[EntitySpan]) -> None:
        """
        Debug helper to show tokenization and span-to-token mapping.

        Args:
            text: Original text
            spans: List of EntitySpan objects
        """
        tokens = self.tokenize_text(text)
        token_labels = self.spans_to_token_labels(text, spans)

        print("TOKENIZATION DEBUG:")
        print("-" * 50)
        for i, (token, start, end) in enumerate(tokens):
            labels = token_labels.get(i, set())
            labels_str = ", ".join(sorted(labels)) if labels else "O"
            print(f"Token {i:2d}: '{token}' [{start:3d}-{end:3d}] -> {labels_str}")


# Backward compatibility alias
Evaluator = TokenLevelEvaluator
