#!/bin/bash
# RunPod training script for HW1: Prompt Prefix Optimization
# Optimized for remote GPU execution with tmux monitoring

set -e  # Exit on error

# Configuration
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"  # or "meta-llama/Llama-3.2-3B-Instruct"
BENCHMARK="hellaswag"
ITERATIONS=100
SAMPLES_PER_ITERATION=10
BATCH_SIZE=50
LEARNING_RATE=0.01
OUTPUT_DIR="outputs/runpod_hellaswag_large"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file for monitoring
LOG_FILE="$OUTPUT_DIR/training.log"

echo "============================================================"
echo "RunPod Training Configuration"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Benchmark: $BENCHMARK"
echo "Iterations: $ITERATIONS"
echo "Samples per iteration: $SAMPLES_PER_ITERATION"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""
echo "Training will start in 5 seconds..."
sleep 5

# Run training with output to both console and log file
python scripts/train_policy.py \
    --model_name "$MODEL_NAME" \
    --benchmark "$BENCHMARK" \
    --iterations "$ITERATIONS" \
    --samples_per_iteration "$SAMPLES_PER_ITERATION" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Training complete! Results saved to: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "To analyze results:"
echo "  python scripts/analyze_results.py --output_dir $OUTPUT_DIR"
