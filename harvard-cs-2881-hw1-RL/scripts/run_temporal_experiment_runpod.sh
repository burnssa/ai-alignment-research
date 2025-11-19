#!/bin/bash
# RunPod script for full temporal persona-awareness experiment
#
# This script runs the complete pipeline:
# 1. Classify HellaSwag questions by temporal content (modern vs timeless)
# 2. Validate persona-awareness on filtered subsets

set -e

# Configuration
CLASSIFIER_MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"  # Use Llama 3.2 for classification
EVAL_MODEL="meta-llama/Llama-3.1-8B-Instruct"  # Use Llama 3.1 for evaluation
NUM_QUESTIONS=200
NUM_PEOPLE=10
OUTPUT_DIR="outputs/era_validation"
CLASSIFICATIONS_FILE="$OUTPUT_DIR/hellaswag_temporal_classifications.csv"
LOG_FILE="$OUTPUT_DIR/temporal_experiment.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "Temporal Persona-Awareness Experiment" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Pipeline:" | tee -a "$LOG_FILE"
echo "  1. Classify questions (modern vs timeless) using $CLASSIFIER_MODEL" | tee -a "$LOG_FILE"
echo "  2. Validate personas on filtered subsets using $EVAL_MODEL" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  - Questions: $NUM_QUESTIONS" | tee -a "$LOG_FILE"
echo "  - People per era: $NUM_PEOPLE" | tee -a "$LOG_FILE"
echo "  - Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check dependencies
echo "Checking dependencies..." | tee -a "$LOG_FILE"
if ! python -c "import datasets" 2>/dev/null; then
    echo "❌ ERROR: Dependencies not installed!" | tee -a "$LOG_FILE"
    echo "Please run: pip install -e ." | tee -a "$LOG_FILE"
    exit 1
fi
echo "✓ Dependencies verified" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# STEP 1: Classify questions
echo "============================================================" | tee -a "$LOG_FILE"
echo "STEP 1: Classifying HellaSwag questions by temporal content" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "This will take ~10-20 minutes depending on GPU..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python scripts/classify_temporal_content.py \
    --model_name "$CLASSIFIER_MODEL" \
    --num_questions "$NUM_QUESTIONS" \
    --output_file "$CLASSIFICATIONS_FILE" \
    --seed 42 \
    2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$CLASSIFICATIONS_FILE" ]; then
    echo "❌ ERROR: Classification failed - output file not created" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "✓ Classification complete!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# STEP 2: Validate personas on temporal subsets
echo "============================================================" | tee -a "$LOG_FILE"
echo "STEP 2: Validating persona-awareness on temporal subsets" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "This will take ~30-60 minutes depending on GPU..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python scripts/validate_temporal_hypothesis.py \
    --classified_questions "$CLASSIFICATIONS_FILE" \
    --model_name "$EVAL_MODEL" \
    --num_people "$NUM_PEOPLE" \
    --seed 42 \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Temporal experiment complete!" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Classifications saved to: $CLASSIFICATIONS_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Key question: Does the contemporary advantage differ between" | tee -a "$LOG_FILE"
echo "modern vs timeless questions? Check the HYPOTHESIS TEST section!" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
