#!/bin/bash
# RunPod validation script for era hypothesis
# Tests if contemporary people perform better on HellaSwag

set -e

# Configuration
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
NUM_PEOPLE=10        # Test 10 people per era (contemporary, modern, historical)
NUM_QUESTIONS=200    # Larger test set for more reliable results
OUTPUT_DIR="outputs/era_validation"
LOG_FILE="$OUTPUT_DIR/validation.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Era Hypothesis Validation on RunPod"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "People per era: $NUM_PEOPLE"
echo "Test questions: $NUM_QUESTIONS"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "============================================================"
echo ""

# Check dependencies
echo "Checking dependencies..."
if ! python -c "import datasets" 2>/dev/null; then
    echo "❌ ERROR: Dependencies not installed!"
    echo "Please run: pip install -e ."
    exit 1
fi
echo "✓ Dependencies verified"
echo ""

echo "Starting validation in 5 seconds..."
echo "(This will take ~30-60 minutes depending on GPU)"
sleep 5

# Run validation
python scripts/validate_era_hypothesis_simple.py \
    --model_name "$MODEL_NAME" \
    --num_people "$NUM_PEOPLE" \
    --num_questions "$NUM_QUESTIONS" \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Validation complete! Results saved to: $OUTPUT_DIR"
echo "============================================================"
