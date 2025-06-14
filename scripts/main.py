"""
Main script for mathematical entity extraction using generative LLM.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers library not found. Token counting will be disabled.")
    AutoTokenizer = None

from data_loader import DataLoader
from prompt_builder import PromptBuilder
from llm_client import VLLMClient
from openai_client import OpenAIClient
from output_parser import OutputParser, EntitySpan
from evaluator import Evaluator


async def main():
    """Main function to run the mathematical entity extraction pipeline."""
    parser = argparse.ArgumentParser(description="Mathematical Entity Extraction")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="A2-NLP_244",
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--num_examples", type=int, default=3, help="Number of few-shot examples to use"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name/path for vLLM or OpenAI model name",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "openai"],
        default="vllm",
        help="Backend to use: vllm (local) or openai (API)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=5,
        help="GPU ID to use (default: 5)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="Output file for predictions",
    )
    parser.add_argument(
        "--test_engine", action="store_true", help="Test vLLM engine and exit"
    )

    args = parser.parse_args()

    # Initialize components
    print("Initializing components...")
    data_loader = DataLoader(args.data_dir)
    prompt_builder = PromptBuilder()

    # Initialize LLM client based on backend choice
    if args.backend == "openai":
        llm_client = OpenAIClient(
            model_name=args.model_name,
        )
    else:  # vllm
        llm_client = VLLMClient(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            gpu_id=args.gpu_id,
        )

    output_parser = OutputParser()
    evaluator = Evaluator()

    # Test engine if requested
    if args.test_engine:
        print("Testing engine...")
        if args.backend == "openai":
            connection_ok = await llm_client.test_connection()
        else:
            connection_ok = llm_client.test_connection()

        if connection_ok:
            print("✓ Engine test successful!")
            model_info = llm_client.get_model_info()
            if model_info:
                print(f"Model info: {model_info}")
        else:
            print("✗ Engine test failed!")
        return

    # Load data
    print("Loading data...")
    data_loader.load_data()

    # Print dataset statistics
    print("\nDataset Statistics:")
    train_stats = data_loader.get_entity_stats("train")
    val_stats = data_loader.get_entity_stats("val")
    print(f"Training entities: {train_stats}")
    print(f"Validation entities: {val_stats}")

    # Create few-shot examples
    print(f"\nCreating {args.num_examples} few-shot examples...")
    examples = data_loader.create_few_shot_examples(args.num_examples, "train")
    print(f"Created {len(examples)} examples")

    # Get validation file IDs and texts
    val_file_ids = data_loader.get_file_ids("val")
    print(f"\nProcessing {len(val_file_ids)} validation documents...")

    # Prepare validation texts
    val_texts = []
    val_file_id_map = {}
    for i, file_id in enumerate(val_file_ids):
        text = data_loader.get_document_text(file_id)
        val_texts.append(text)
        val_file_id_map[i] = file_id

        # Build prompts
    print("Building prompts...")
    prompts = prompt_builder.build_batch_prompts(val_texts, examples)

    # Log token counts to file
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            log_file = args.output_file.replace(".json", "_token_counts.log")

            with open(log_file, "w") as f:
                f.write("Prompt Token Counts\n")
                f.write("==================\n\n")

                total_tokens = 0
                for i, (prompt, file_id) in enumerate(zip(prompts, val_file_ids), 1):
                    token_count = len(
                        tokenizer.encode(prompt, add_special_tokens=False)
                    )
                    total_tokens += token_count
                    f.write(f"Document {i}: {token_count} tokens - {file_id}\n")

                f.write(
                    f"\nTotal: {total_tokens} tokens across {len(prompts)} prompts\n"
                )
                f.write(f"Average: {total_tokens/len(prompts):.0f} tokens per prompt\n")

            print(f"Token counts logged to {log_file}")

        except Exception as e:
            print(f"Warning: Could not log token counts: {e}")

    # Generate predictions using LLM
    print("Generating predictions with LLM...")
    if args.backend == "openai":
        llm_outputs = await llm_client.generate_batch(
            prompts, max_tokens=args.max_tokens, temperature=args.temperature
        )
    else:
        llm_outputs = llm_client.generate_batch(
            prompts, max_tokens=args.max_tokens, temperature=args.temperature
        )

    # Log model outputs to file
    output_log_file = args.output_file.replace(".json", "_model_outputs.log")
    print(f"Logging model outputs to {output_log_file}")

    with open(output_log_file, "w", encoding="utf-8") as f:
        f.write("Model Outputs Log\n")
        f.write("=================\n\n")

        # Write the prompt template with few-shot examples at the top
        f.write("PROMPT TEMPLATE WITH FEW-SHOT EXAMPLES:\n")
        f.write("=" * 80 + "\n")
        # Use the first prompt as an example (they all have the same instruction + examples)
        if prompts:
            # Extract just the instruction and examples part (before "Now, please tag the following text:")
            full_prompt = prompts[0]
            instruction_end = full_prompt.find("Now, please tag the following text:")
            if instruction_end != -1:
                instruction_and_examples = full_prompt[:instruction_end].strip()
                f.write(instruction_and_examples)
            else:
                # Fallback: write the full first prompt
                f.write(full_prompt)
        f.write("\n\n")
        f.write("=" * 80 + "\n\n")

        # Write individual document results
        for i, (file_id, llm_output) in enumerate(zip(val_file_ids, llm_outputs), 1):
            f.write(f"Document {i}: {file_id}\n")
            f.write("=" * 80 + "\n")
            f.write("INPUT TEXT:\n")
            f.write("-" * 40 + "\n")
            f.write(val_texts[i - 1])
            f.write("\n\n")
            f.write("MODEL OUTPUT:\n")
            f.write("-" * 40 + "\n")
            f.write(llm_output)
            f.write("\n\n")
            f.write("=" * 80 + "\n\n")

    # Parse outputs and extract entities
    print("Parsing LLM outputs...")
    predictions = {}

    for i, (llm_output, original_text) in enumerate(zip(llm_outputs, val_texts)):
        file_id = val_file_id_map[i]

        # Parse LLM output to extract entities
        entities = output_parser.parse_llm_output(llm_output, original_text)

        # Apply filtering rules
        filtered_entities = output_parser.filter_invalid_spans(entities)

        predictions[file_id] = filtered_entities

        print(
            f"  {file_id}: {len(entities)} -> {len(filtered_entities)} entities (after filtering)"
        )

    # Prepare ground truth
    print("Preparing ground truth...")
    ground_truth = {}

    for file_id in val_file_ids:
        annotations = data_loader.get_annotations_for_file(file_id, "val")
        gt_spans = evaluator.ground_truth_to_spans(annotations)
        ground_truth[file_id] = gt_spans

    # Prepare document texts for token-level evaluation
    document_texts = {}
    for file_id in val_file_ids:
        document_texts[file_id] = data_loader.get_document_text(file_id)

    # Evaluate predictions
    print("Evaluating predictions...")
    overall_metrics, per_tag_metrics = evaluator.evaluate_predictions(
        predictions, ground_truth, document_texts
    )

    # Print evaluation report
    evaluator.print_evaluation_report(overall_metrics, per_tag_metrics)

    # Save predictions
    print(f"\nSaving predictions to {args.output_file}...")

    # Convert EntitySpan objects to dictionaries for JSON serialization
    serializable_predictions = {}
    for file_id, spans in predictions.items():
        serializable_predictions[file_id] = [
            {"start": span.start, "end": span.end, "tag": span.tag, "text": span.text}
            for span in spans
        ]

    # Save results
    results = {
        "predictions": serializable_predictions,
        "evaluation": {
            "overall": {
                "precision": overall_metrics.precision,
                "recall": overall_metrics.recall,
                "f1_score": overall_metrics.f1_score,
                "true_positives": overall_metrics.true_positives,
                "false_positives": overall_metrics.false_positives,
                "false_negatives": overall_metrics.false_negatives,
            },
            "per_tag": {
                tag: {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "true_positives": metrics.true_positives,
                    "false_positives": metrics.false_positives,
                    "false_negatives": metrics.false_negatives,
                }
                for tag, metrics in per_tag_metrics.items()
            },
        },
        "config": {
            "num_examples": args.num_examples,
            "model_name": args.model_name,
            "backend": args.backend,
            "tensor_parallel_size": (
                args.tensor_parallel_size if args.backend == "vllm" else None
            ),
            "max_model_len": args.max_model_len if args.backend == "vllm" else None,
            "gpu_memory_utilization": (
                args.gpu_memory_utilization if args.backend == "vllm" else None
            ),
            "gpu_id": args.gpu_id if args.backend == "vllm" else None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output_file}")
    print(f"\nFinal F1-Score: {overall_metrics.f1_score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
