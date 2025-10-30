#!/bin/bash
# RunPod GPU Setup Script
# Run this on your RunPod instance to set up the training environment

set -e  # Exit on error

echo "=================================="
echo "RunPod GPU Training Setup"
echo "=================================="

# 1. Update system and install dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git vim wget curl

# 2. Clone repository (you'll need to set your repo URL)
echo ""
echo "📁 Enter your GitHub repository URL:"
echo "   (e.g., https://github.com/username/ai-alignment-research.git)"
read -p "Repo URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ Error: Repository URL is required"
    exit 1
fi

echo "📥 Cloning repository..."
cd /workspace
git clone "$REPO_URL" ai-alignment-research
cd ai-alignment-research

# 3. Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers peft datasets accelerate bitsandbytes openai python-dotenv matplotlib

# 4. Set up environment variables
echo ""
echo "🔑 Setting up environment variables..."
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env

    echo ""
    echo "📝 Enter your API keys:"
    read -p "OpenAI API Key (sk-...): " OPENAI_KEY
    read -p "HuggingFace Token (hf_...) [optional, press Enter to skip]: " HF_TOKEN

    # Update .env file
    sed -i "s/sk-your-openai-key-here/$OPENAI_KEY/" .env
    if [ -n "$HF_TOKEN" ]; then
        sed -i "s/hf_your-huggingface-token-here/$HF_TOKEN/" .env
    fi

    echo "✅ Environment variables configured"
else
    echo "⚠️  .env file already exists, skipping..."
fi

# 5. Verify setup
echo ""
echo "✅ Verifying setup..."
python check_env.py

# 6. Show GPU info
echo ""
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 7. Ready to train
echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "📍 You're in: $(pwd)"
echo ""
echo "🚀 To start training:"
echo "   cd harvard-cs-2881-hw0"
echo "   python train.py --num_epochs 5 --output_dir fine_tuned_model"
echo ""
echo "📊 Expected training time on GPU: 15-30 minutes for 5 epochs"
echo ""
echo "💡 Monitor training:"
echo "   tail -f nohup.out    # If using nohup"
echo "   watch -n 1 nvidia-smi  # Monitor GPU usage"
echo ""
