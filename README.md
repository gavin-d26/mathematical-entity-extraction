# Mathematical Entity Extraction

A system for extracting mathematical entities (definitions, theorems, proofs, examples, names, references) from academic documents using generative LLMs with few-shot prompting and XML-style tagging.

## Features

- **Multi-backend support**: Works with both vLLM and OpenAI APIs
- **Few-shot learning**: Configurable number of examples for in-context learning
- **XML-style tagging**: Handles nested mathematical entities with proper span extraction
- **Comprehensive evaluation**: Span-level precision, recall, and F1-score metrics
- **Batch processing**: Efficient inference for large document collections

## Setup

This project uses [uv](https://docs.astral.sh/uv/) as the package manager for fast and reliable dependency management.

### 1. Environment Setup

```bash
# Install dependencies using uv (recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

**Note**: `uv sync` will automatically create the virtual environment and install all dependencies from `uv.lock`.

### 2. Backend Configuration

#### Option A: vLLM Backend

1. **No Server Required**: vLLM runs offline batch inference directly in the Python process
2. **Test vLLM Setup**:
```bash
python scripts/test_implementation.py
```

**Note**: The first run will automatically download the model (Qwen/Qwen2.5-3B-Instruct by default)

#### Option B: OpenAI Backend

1. **Set up API Key**:
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or create a .env file in project root:
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

2. **Test OpenAI Connection**:
```bash
python scripts/test_openai.py
```

## Usage

### Basic Usage

#### With vLLM Backend (default):
```bash
python scripts/main.py
```

#### With OpenAI Backend:
```bash
python scripts/main.py --backend openai --model gpt-4o-mini
```

### Advanced Configuration

```bash
# Custom few-shot examples
python scripts/main.py --num_examples 5

# Different OpenAI model
python scripts/main.py --backend openai --model gpt-4

# Custom output file
python scripts/main.py --output results/my_predictions.json

# Combine multiple options
python scripts/main.py --backend openai --model_name gpt-4o-mini --max_tokens 4096 --temperature 0.1 --output_file openai_results.json
```

### Command Line Arguments

- `--backend`: Choose backend (`vllm` or `openai`, default: `vllm`)
- `--model`: Model name (default: `qwen2.5-3b-instruct` for vLLM, `gpt-4o-mini` for OpenAI)
- `--num_examples`: Number of few-shot examples (default: 3)
- `--output`: Output JSON file path (default: `predictions.json`)
- `--gpu_id`: GPU ID to use for vLLM (default: 5)
- `--tensor_parallel_size`: Number of GPUs for vLLM tensor parallelism (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization ratio for vLLM (default: 0.9)


### View Few-shot Examples
```bash
python scripts/view_few_shot_examples.py
```

## Output Format

The system generates a JSON file with the following structure:

```json
{
  "predictions": {
    "file_id": [
      {
        "start": 123,
        "end": 456,
        "tag": "definition",
        "text": "extracted entity text"
      }
    ]
  },
  "evaluation": {
    "overall": {
      "precision": 0.85,
      "recall": 0.78,
      "f1": 0.81
    },
    "per_tag": {
      "definition": {"precision": 0.90, "recall": 0.85, "f1": 0.87},
      "theorem": {"precision": 0.88, "recall": 0.82, "f1": 0.85}
    }
  },
  "config": {
    "backend": "openai",
    "model": "gpt-4o-mini",
    "num_examples": 3
  }
}
```

## Entity Types

The system extracts six types of mathematical entities:

1. **Definition**: Mathematical definitions and concepts
2. **Theorem**: Mathematical theorems and propositions
3. **Proof**: Proof sections and demonstrations
4. **Example**: Mathematical examples and illustrations
5. **Name**: Names of mathematical objects, concepts, or people
6. **Reference**: Citations and references to other work

## Project Structure

```
mathematical-entity-extraction/
├── A2-NLP_244/              # Dataset directory
│   ├── file_contents.json   # Document texts
│   ├── train.json          # Training annotations
│   └── val.json            # Validation annotations
├── scripts/                 # Python modules
│   ├── main.py             # Main pipeline
│   ├── data_loader.py      # Data loading utilities
│   ├── prompt_builder.py   # Few-shot prompt construction
│   ├── llm_client.py       # vLLM client
│   ├── openai_client.py    # OpenAI client
│   ├── output_parser.py    # XML output parsing
│   ├── evaluator.py        # Evaluation metrics
│   └── test_*.py           # Test scripts
├── .venv/                  # Virtual environment
├── pyproject.toml          # Project configuration (uv package manager)
├── uv.lock                 # Dependency lock file
└── README.md               # This file
```
