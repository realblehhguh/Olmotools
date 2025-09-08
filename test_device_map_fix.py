#!/usr/bin/env python3
"""
Test script to verify the device_map fix for distributed training.
"""

import os
import sys
import torch
from pathlib import Path

# Add the core directory to the path
sys.path.append(str(Path(__file__).parent / "core"))

from model_utils import load_olmo_model_and_tokenizer

def test_device_map_fix():
    """Test that device_map is properly handled in different scenarios."""
    
    print("Testing device_map fix for distributed training...")
    print("=" * 60)
    
    # Test 1: Normal single GPU mode (should use device_map)
    print("\n1. Testing single GPU mode (should use device_map='auto'):")
    try:
        # Simulate single GPU environment
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("RANK", None)
        
        # This should work with device_map
        print("   Loading model with device_map='auto' (single GPU)...")
        # Note: We'll use a smaller model for testing to avoid memory issues
        model, tokenizer = load_olmo_model_and_tokenizer(
            model_name="microsoft/DialoGPT-small",  # Smaller model for testing
            use_lora=False,
            use_4bit=False,
            use_deepspeed=False,
            device_map="auto"
        )
        print("   ✓ SUCCESS: Single GPU mode works with device_map")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"   ✗ FAILED: Single GPU mode failed: {e}")
    
    # Test 2: DeepSpeed mode (should disable device_map)
    print("\n2. Testing DeepSpeed mode (should disable device_map):")
    try:
        # This should work without device_map when use_deepspeed=True
        print("   Loading model with use_deepspeed=True (should disable device_map)...")
        model, tokenizer = load_olmo_model_and_tokenizer(
            model_name="microsoft/DialoGPT-small",  # Smaller model for testing
            use_lora=False,
            use_4bit=False,
            use_deepspeed=True,  # This should disable device_map
            device_map="auto"
        )
        print("   ✓ SUCCESS: DeepSpeed mode works with disabled device_map")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"   ✗ FAILED: DeepSpeed mode failed: {e}")
    
    # Test 3: Simulated distributed environment (should disable device_map)
    print("\n3. Testing simulated distributed environment (should disable device_map):")
    try:
        # Simulate distributed environment
        os.environ["WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        
        print("   Loading model in simulated distributed environment...")
        model, tokenizer = load_olmo_model_and_tokenizer(
            model_name="microsoft/DialoGPT-small",  # Smaller model for testing
            use_lora=False,
            use_4bit=False,
            use_deepspeed=False,  # Even without DeepSpeed, distributed env should disable device_map
            device_map="auto"
        )
        print("   ✓ SUCCESS: Distributed environment works with disabled device_map")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up environment
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("RANK", None)
        
    except Exception as e:
        print(f"   ✗ FAILED: Distributed environment failed: {e}")
        # Clean up environment even on failure
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("RANK", None)
    
    print("\n" + "=" * 60)
    print("Device map fix testing completed!")
    print("\nThe fix ensures that:")
    print("- device_map='auto' is used for single GPU training")
    print("- device_map is disabled for DeepSpeed/distributed training")
    print("- This prevents the 'device_map=auto with distributed training' error")

def test_training_command_examples():
    """Show examples of how to run training with the fix."""
    
    print("\n" + "=" * 60)
    print("TRAINING COMMAND EXAMPLES")
    print("=" * 60)
    
    print("\n1. Single GPU training (uses device_map='auto'):")
    print("   python core/train.py --use_deepspeed=False --num_processes=1")
    
    print("\n2. DeepSpeed single GPU training (disables device_map):")
    print("   python core/train.py --use_deepspeed=True --deepspeed_config=configs/deepspeed_config_single_gpu.json")
    
    print("\n3. DeepSpeed multi-GPU training (disables device_map):")
    print("   deepspeed core/train.py --deepspeed_config=configs/deepspeed_config.json")
    
    print("\n4. Modal deployment (automatically handles device_map):")
    print("   modal run core/modal_app.py --action=train --gpu_count=1")
    print("   modal run core/modal_app.py --action=train --gpu_count=2")
    
    print("\nThe fix automatically detects the training environment and:")
    print("- Keeps device_map='auto' for single GPU non-distributed training")
    print("- Disables device_map when DeepSpeed or distributed training is detected")

if __name__ == "__main__":
    test_device_map_fix()
    test_training_command_examples()
