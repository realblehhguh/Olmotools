#!/usr/bin/env python3
"""
Simple test to check Modal connectivity and basic setup.
"""

import modal
import os

# Create a simple test app
app = modal.App("simple-test")

@app.function(timeout=60)
def test_basic():
    """Test basic Modal functionality."""
    
    print("✅ Modal function is running!")
    print(f"Python version: {os.sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check if wandb is installed
    try:
        import wandb
        print(f"✅ wandb is installed (version: {wandb.__version__})")
    except ImportError:
        print("❌ wandb is not installed")
    
    # Check if transformers is installed
    try:
        import transformers
        print(f"✅ transformers is installed (version: {transformers.__version__})")
    except ImportError:
        print("❌ transformers is not installed")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ No GPU available")
    except ImportError:
        print("❌ torch is not installed")
    
    return "Test completed successfully!"


def main():
    """Run the test."""
    print("🧪 Testing basic Modal connectivity...")
    
    with app.run():
        result = test_basic.remote()
        print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
