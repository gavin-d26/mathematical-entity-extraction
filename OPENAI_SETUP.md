# OpenAI API Setup

## Prerequisites

1. Install required dependencies:
   ```bash
   pip install openai python-dotenv
   ```

2. Get an OpenAI API key from https://platform.openai.com/api-keys

## Configuration

1. Create a `.env` file in the project root:
   ```bash
   # .env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. Or set the environment variable directly:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

## Features

### Async Processing
The OpenAI client uses async/await for efficient batch processing:
- **Concurrent API calls**: All prompts in a batch are processed simultaneously
- **Better performance**: Significantly faster than sequential processing
- **Same interface**: Compatible with existing vLLM client interface

## Usage

### Test OpenAI Integration
```bash
python scripts/test_openai.py
```

### Run Experiment with OpenAI
```bash
python scripts/main.py \
    --backend openai \
    --model_name gpt-4o-mini \
    --num_examples 3 \
    --max_tokens 4096 \
    --temperature 0.1 \
    --output_file openai_results.json
```

### Available OpenAI Models
- `gpt-4o-mini` (recommended for cost-effectiveness)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Backend Comparison

| Feature | vLLM (Local) | OpenAI (API) |
|---------|--------------|--------------|
| Cost | Free (after hardware) | Pay per token |
| Speed | Fast (local GPU) | Network dependent |
| Privacy | Full control | Data sent to OpenAI |
| Models | Open source models | OpenAI models |
| Setup | Complex (GPU required) | Simple (API key) |

### Example Commands

**vLLM (Local):**
```bash
./run.sh  # Uses vLLM by default
```

**OpenAI API:**
```bash
python scripts/main.py --backend openai --model_name gpt-4o-mini
``` 