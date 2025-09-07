#!/bin/bash

# Quick Start Guide for OLMo Fine-tuning on Modal Labs
# This script provides example commands to get started

echo "======================================"
echo "OLMo Fine-tuning Quick Start Guide"
echo "======================================"
echo ""

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Please install it first:"
    echo "   pip install modal"
    echo ""
    exit 1
fi

# Check if user is authenticated
if ! modal token validate &> /dev/null; then
    echo "❌ Not authenticated with Modal. Please run:"
    echo "   modal token new"
    echo ""
    exit 1
fi

echo "✅ Modal CLI is installed and authenticated"
echo ""

echo "📝 Setup Instructions:"
echo "----------------------"
echo ""
echo "1. Create Modal secrets (if not already done):"
echo "   # For HuggingFace (optional, for gated models):"
echo "   modal secret create huggingface-token HUGGINGFACE_TOKEN=<your-token>"
echo ""
echo "   # For Weights & Biases (optional, for metrics):"
echo "   modal secret create wandb-secret WANDB_API_KEY=<your-key>"
echo ""

echo "2. Deploy the app to Modal:"
echo "   modal deploy modal_app.py"
echo ""

echo "🚀 Training Commands:"
echo "--------------------"
echo ""
echo "# Quick test with small sample (recommended first):"
echo "modal run modal_app.py --action train --train_sample_size 100 --num_epochs 1 --run_name test_run"
echo ""

echo "# Full training with LoRA (memory efficient):"
echo "modal run modal_app.py --action train --use_lora --num_epochs 3 --run_name uncensored_olmo_lora"
echo ""

echo "# Full training without LoRA (requires more memory):"
echo "modal run modal_app.py --action train --num_epochs 3 --run_name uncensored_olmo_full"
echo ""

echo "# Training with 4-bit quantization (most memory efficient):"
echo "modal run modal_app.py --action train --use_4bit --use_lora --num_epochs 3 --run_name uncensored_olmo_4bit"
echo ""

echo "📊 Other Commands:"
echo "-----------------"
echo ""
echo "# List available checkpoints:"
echo "modal run modal_app.py --action list"
echo ""

echo "# Test inference with a checkpoint:"
echo "modal run modal_app.py --action test --checkpoint_path /vol/outputs/run_*/final_model"
echo ""

echo "# Clean up old checkpoints:"
echo "modal run modal_app.py --action cleanup"
echo ""

echo "💡 Tips:"
echo "--------"
echo "• Start with a small sample size to test everything works"
echo "• Use LoRA for faster training and lower memory usage"
echo "• Monitor training progress in the Modal dashboard"
echo "• Check Weights & Biases for detailed metrics (if configured)"
echo "• Training the full dataset takes 10-30 hours depending on settings"
echo ""

echo "📚 For more information, see README.md"
echo ""

# Ask if user wants to start a test run
read -p "Would you like to start a quick test run now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting test run with 100 samples..."
    modal run modal_app.py --action train --train_sample_size 100 --num_epochs 1 --run_name quickstart_test
fi
