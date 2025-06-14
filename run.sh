#!/bin/bash

# Mathematical Entity Extraction Experiment Runner
# This script runs the complete pipeline with token-level evaluation

set -e  # Exit on any error

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="experiment_${TIMESTAMP}.log"

# Function to log both to console and file
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Start logging
log "=========================================="
log "Mathematical Entity Extraction Experiment"
log "Started at: $(date)"
log "Log file: $LOG_FILE"
log "=========================================="

# Activate virtual environment
log "Activating virtual environment..."
source /home/gpdsouza/projects/mathematical-entity-extraction/.venv/bin/activate

# Set experiment parameters
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
GPU_ID=4
NUM_EXAMPLES=3
MAX_TOKENS=4096
TEMPERATURE=0.1
OUTPUT_FILE="experiment_results.json"

log "Experiment Configuration:"
log "  Model: $MODEL_NAME"
log "  GPU: $GPU_ID"
log "  Few-shot examples: $NUM_EXAMPLES"
log "  Max tokens: $MAX_TOKENS"
log "  Temperature: $TEMPERATURE"
log "  Output file: $OUTPUT_FILE"
log ""

# Run the experiment
log "Starting experiment..."

# Capture both stdout and stderr, display in real-time, and log to file
python scripts/main.py \
    --model_name "$MODEL_NAME" \
    --gpu_id $GPU_ID \
    --num_examples $NUM_EXAMPLES \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --output_file "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"

# Check if the experiment succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log ""
    log "=========================================="
    log "Experiment completed successfully!"
    log "Finished at: $(date)"
    log "Results saved to: $OUTPUT_FILE"
    log "Token counts logged to: ${OUTPUT_FILE%.*}_token_counts.log"
    log "Model outputs logged to: ${OUTPUT_FILE%.*}_model_outputs.log"
    log "Full log saved to: $LOG_FILE"
    log "=========================================="
else
    log ""
    log "=========================================="
    log "Experiment FAILED!"
    log "Failed at: $(date)"
    log "Check the log file for errors: $LOG_FILE"
    log "=========================================="
    exit 1
fi 