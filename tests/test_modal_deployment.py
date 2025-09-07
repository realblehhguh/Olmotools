#!/usr/bin/env python3
"""
Test script to verify Modal deployment and W&B integration.
"""

import modal
import os
import sys

# Create a simple test app
app = modal.App("test-wandb-integration")

@app.function(
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-token"),
    ],
    timeout=60,
)
def test_secrets():
    """Test if secrets are properly loaded."""
    
    print("Testing secret environment variables...")
    
    # Check for W&B API key
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        print(f"✅ WANDB_API_KEY is set (length: {len(wandb_key)})")
    else:
        print("❌ WANDB_API_KEY is NOT set")
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print(f"✅ HUGGINGFACE_TOKEN is set (length: {len(hf_token)})")
    else:
        print("❌ HUGGINGFACE_TOKEN is NOT set")
    
    # Try to initialize W&B
    try:
        import wandb
        wandb.init(
            project="test-modal-wandb",
            name="test-run",
            mode="offline" if not wandb_key else "online"
        )
        print("✅ W&B initialization successful")
        wandb.finish()
    except Exception as e:
        print(f"❌ W&B initialization failed: {e}")
    
    # Check all environment variables that might contain W&B keys
    print("\nAll environment variables containing 'WANDB':")
    for key, value in os.environ.items():
        if "WANDB" in key.upper():
            print(f"  {key}: {'*' * min(len(value), 10)}")
    
    return {
        "wandb_key_set": bool(wandb_key),
        "hf_token_set": bool(hf_token),
    }


def main():
    """Run the test."""
    print("🧪 Testing Modal deployment and W&B integration...")
    
    with app.run():
        result = test_secrets.remote()
        print(f"\nResults: {result}")
        
        if result["wandb_key_set"]:
            print("✅ W&B integration should work!")
        else:
            print("⚠️ W&B API key not found. Training will run but won't log to W&B.")


if __name__ == "__main__":
    main()
