#!/bin/bash
################################################################################
# Upload Script: Send code to RunPod
# Run this FROM YOUR LOCAL MACHINE (not in RunPod)
################################################################################

# Check if POD_ID is provided
if [ -z "$1" ]; then
    echo "❌ Error: Pod ID required"
    echo ""
    echo "Usage: bash upload_to_runpod.sh <POD_ID>"
    echo ""
    echo "Example:"
    echo "  bash upload_to_runpod.sh abc123xyz"
    echo ""
    echo "Get your Pod ID from: https://www.runpod.io/console/pods"
    exit 1
fi

POD_ID="$1"

echo "=========================================="
echo "Uploading to RunPod"
echo "=========================================="
echo "Pod ID: ${POD_ID}"
echo ""

# Check if runpodctl is installed
if ! command -v runpodctl &> /dev/null; then
    echo "❌ Error: runpodctl not found"
    echo ""
    echo "Install it with:"
    echo "  wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-amd64 -O runpodctl"
    echo "  chmod +x runpodctl"
    echo "  sudo mv runpodctl /usr/local/bin/"
    echo "  runpodctl config --apiKey <your-runpod-api-key>"
    exit 1
fi

# Upload entire project
echo "📤 Uploading project files..."
echo ""

cd ~/Code/ai-alignment-research

runpodctl send harvard-cs-2881-hw0/ ${POD_ID}:/workspace/ai-alignment-research/

echo ""
echo "=========================================="
echo "✅ Upload complete!"
echo "=========================================="
echo ""
echo "Next steps (run INSIDE RunPod terminal):"
echo ""
echo "  cd /workspace/ai-alignment-research/harvard-cs-2881-hw0"
echo "  pip install torch transformers peft datasets accelerate bitsandbytes openai python-dotenv"
echo "  bash runpod/train_bad_medical_v2.sh"
echo ""
