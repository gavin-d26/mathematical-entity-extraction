# Mathematical Entity Extraction

This project implements a system for extracting mathematical entities from text using a generative Large Language Model (LLM) with few-shot prompting and vLLM offline inference.

## Overview

The system identifies and tags the following mathematical entities:
- **definition**: Text that defines a new concept or object
- **theorem**: Text that makes a rigorous, provable claim  
- **proof**: Text that proves a theorem
- **example**: An example of a defined object or application of a theorem
- **name**: The name of a newly defined object or a theorem
- **reference**: A reference to a previously defined name

## Architecture

The implementation consists of several modular components:

1. **DataLoader** (`data_loader.py`): Loads and preprocesses the dataset
2. **PromptBuilder** (`prompt_builder.py`): Constructs few-shot prompts for the LLM
3. **VLLMClient** (`llm_client.py`): Handles offline batch inference using vLLM
4. **OutputParser** (`output_parser.py`): Parses LLM output and extracts entity spans
5. **Evaluator** (`evaluator.py`): Calculates evaluation metrics
6. **Main** (`main.py`): Orchestrates the entire pipeline

## Requirements

- Python 3.8+
- pandas
- vllm (for offline inference)
- CUDA-compatible GPU (recommended)

## Installation

1. Install required packages:
```bash
pip install pandas vllm
```

Note: vLLM will automatically download the model on first use.

## Usage

### Basic Usage

Run the main script with default settings (from project root):

```bash
python scripts/main.py
```

This will:
1. Load the `Qwen/Qwen2.5-3B-Instruct` model using vLLM
2. Process validation documents with 3 few-shot examples
3. Generate predictions and evaluate them
4. Save results to `predictions.json`

### Convert Output to Pandas Format

If you need the output in the same format as train/val JSON files:

```bash
python scripts/convert_to_pandas_format.py predictions.json --output_file predictions_pandas_format.json
```

### Advanced Usage

Customize the execution with command-line arguments (from project root):

```bash
python scripts/main.py \
    --data_dir A2-NLP_244 \
    --num_examples 5 \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --tensor_parallel_size 2 \
    --max_tokens 4096 \
    --temperature 0.1 \
    --output_file results.json
```

### Test Engine

Test if the vLLM engine is working properly (from project root):

```bash
python scripts/main.py --test_engine
```

### Test Implementation

Run tests to verify the implementation works correctly (from project root):

```bash
python scripts/test_implementation.py
```

## Command-Line Arguments

- `--data_dir`: Directory containing the data files (default: "A2-NLP_244")
- `--num_examples`: Number of few-shot examples to use (default: 3)
- `--model_name`: Model name/path for vLLM (default: "Qwen/Qwen2.5-3B-Instruct")
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--max_model_len`: Maximum model context length (default: None, uses model default)
- `--gpu_memory_utilization`: GPU memory utilization ratio (default: 0.9)
- `--gpu_id`: GPU ID to use (default: 5)
- `--max_tokens`: Maximum tokens to generate (default: 4096)
- `--temperature`: Sampling temperature (default: 0.1)
- `--output_file`: Output file for predictions (default: "predictions.json")
- `--test_engine`: Test vLLM engine and exit

## Output Format

The system generates a JSON file containing:

```json
{
  "predictions": {
    "file_id": [
      {
        "start": 0,
        "end": 10,
        "tag": "definition",
        "text": "extracted text"
      }
    ]
  },
  "evaluation": {
    "overall": {
      "precision": 0.85,
      "recall": 0.78,
      "f1_score": 0.81,
      "true_positives": 100,
      "false_positives": 18,
      "false_negatives": 28
    },
    "per_tag": {
      "definition": { ... },
      "theorem": { ... }
    }
  },
  "config": {
    "num_examples": 3,
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "tensor_parallel_size": 1,
    "max_model_len": null,
    "gpu_memory_utilization": 0.9,
    "gpu_id": 5,
    "max_tokens": 4096,
    "temperature": 0.1
  }
}
```

## Data Format

The system expects the following data structure:

- `A2-NLP_244/file_contents.json`: Maps file IDs to document text
- `A2-NLP_244/train.json`: Training annotations in pandas DataFrame format
- `A2-NLP_244/val.json`: Validation annotations in pandas DataFrame format

## Key Features

### Offline Inference
- Uses vLLM for efficient offline batch processing
- No need for a separate server
- Automatic model downloading and caching
- GPU acceleration with tensor parallelism support

### Few-Shot Learning
- Configurable number of examples
- Automatic generation of XML-tagged examples from training data
- Robust prompt construction

### Entity Extraction
- XML-style tagging approach
- Handles nested entities correctly
- Regex-based parsing with error handling

### Post-Processing
- Filters invalid spans according to problem rules:
  - Removes name/reference spans within proof spans
  - Removes name/reference spans not within other entities

### Evaluation
- Span-level precision, recall, and F1-score
- Per-entity-type metrics
- Detailed evaluation reports

## Implementation Details

### vLLM Offline Inference
The system uses vLLM's `LLM` class for efficient batch processing:
1. Initializes the model once at startup
2. Processes all validation documents in a single batch
3. Leverages GPU acceleration and optimized attention mechanisms
4. Supports multi-GPU inference via tensor parallelism

### Few-Shot Example Generation
The system creates few-shot examples by:
1. Loading training annotations
2. Sorting annotations by start index in descending order
3. Injecting XML tags into the raw text
4. Using the tagged text as examples in prompts

### Span Extraction
The output parser:
1. Uses regex to find XML tags in LLM output
2. Maintains a stack to handle nested tags
3. Maps extracted spans back to original text indices
4. Applies filtering rules for invalid spans

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `gpu_memory_utilization` or `max_model_len`
2. **Model Download Issues**: Ensure internet connection for first-time model download
3. **Memory Issues**: Reduce batch size by processing fewer documents at once
4. **Parsing Errors**: Check that LLM output contains valid XML tags

### Debug Mode

For debugging, you can:
1. Run `python scripts/test_implementation.py` to verify components work
2. Use `python scripts/main.py --test_engine` to check vLLM engine functionality
3. Examine the generated prompts and LLM outputs in the console

### Performance Tips

1. **GPU Memory**: Use `--gpu_memory_utilization 0.8` if running out of memory
2. **Multi-GPU**: Use `--tensor_parallel_size N` for N GPUs
3. **Context Length**: Set `--max_model_len` to limit memory usage
4. **Temperature**: Lower values (0.1-0.3) for more consistent output
5. **Examples**: Use high-quality few-shot examples for better performance

### Hardware Requirements

- **Minimum**: 8GB GPU memory for Qwen2.5-3B model
- **Recommended**: 16GB+ GPU memory for optimal performance
- **Multi-GPU**: Use tensor parallelism for larger models or faster inference

## License

This implementation is provided as-is for educational and research purposes. 