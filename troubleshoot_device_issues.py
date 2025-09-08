#!/usr/bin/env python3
"""
Troubleshooting script for device placement issues in distributed training.
"""

import os
import sys
import torch
from pathlib import Path

# Add the core directory to the path
sys.path.append(str(Path(__file__).parent / "core"))

def check_environment():
    """Check the current environment for distributed training indicators."""
    print("Environment Check:")
    print("=" * 50)
    
    # Check for distributed training environment variables
    env_vars = ["WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Check if DeepSpeed is available
    try:
        import deepspeed
        print(f"DeepSpeed available: True (version {deepspeed.__version__})")
    except ImportError:
        print("DeepSpeed available: False")
    
    print()

def test_model_loading_scenarios():
    """Test different model loading scenarios to identify issues."""
    print("Model Loading Tests:")
    print("=" * 50)
    
    # Import after adding to path
    from model_utils import load_olmo_model_and_tokenizer
    
    test_model = "microsoft/DialoGPT-small"  # Small model for testing
    
    scenarios = [
        {
            "name": "Single GPU (device_map=auto)",
            "params": {
                "model_name": test_model,
                "use_lora": False,
                "use_4bit": False,
                "use_deepspeed": False,
                "device_map": "auto"
            },
            "env_setup": lambda: clear_distributed_env()
        },
        {
            "name": "DeepSpeed mode (device_map disabled)",
            "params": {
                "model_name": test_model,
                "use_lora": False,
                "use_4bit": False,
                "use_deepspeed": True,
                "device_map": "auto"
            },
            "env_setup": lambda: clear_distributed_env()
        },
        {
            "name": "Simulated distributed (device_map disabled)",
            "params": {
                "model_name": test_model,
                "use_lora": False,
                "use_4bit": False,
                "use_deepspeed": False,
                "device_map": "auto"
            },
            "env_setup": lambda: setup_distributed_env()
        }
    ]
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}")
        print("-" * 30)
        
        try:
            # Setup environment
            scenario["env_setup"]()
            
            # Load model
            model, tokenizer = load_olmo_model_and_tokenizer(**scenario["params"])
            
            # Check model device
            if hasattr(model, 'device'):
                print(f"  Model device: {model.device}")
            else:
                # Check first parameter device
                first_param = next(model.parameters())
                print(f"  Model parameters device: {first_param.device}")
            
            print(f"  ✓ SUCCESS: Model loaded successfully")
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
        
        finally:
            # Clean up environment
            clear_distributed_env()

def clear_distributed_env():
    """Clear distributed training environment variables."""
    env_vars = ["WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"]
    for var in env_vars:
        os.environ.pop(var, None)

def setup_distributed_env():
    """Setup simulated distributed training environment."""
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"

def test_deepspeed_config():
    """Test DeepSpeed configuration files."""
    print("DeepSpeed Configuration Tests:")
    print("=" * 50)
    
    config_files = [
        "configs/deepspeed_config.json",
        "configs/deepspeed_config_single_gpu.json"
    ]
    
    for config_file in config_files:
        print(f"\nTesting: {config_file}")
        print("-" * 30)
        
        if os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"  ✓ Valid JSON configuration")
                print(f"  Zero optimization stage: {config.get('zero_optimization', {}).get('stage', 'Not specified')}")
                print(f"  FP16 enabled: {config.get('fp16', {}).get('enabled', 'Not specified')}")
            except Exception as e:
                print(f"  ✗ Invalid configuration: {str(e)}")
        else:
            print(f"  ✗ Configuration file not found")

def provide_solutions():
    """Provide solutions for common device placement issues."""
    print("\nCommon Solutions:")
    print("=" * 50)
    
    solutions = [
        {
            "issue": "device_map='auto' with distributed training error",
            "solution": "The fix automatically disables device_map when distributed training is detected"
        },
        {
            "issue": "Model parameters on CPU in distributed training",
            "solution": "Ensure model is on CPU before trainer initialization - DeepSpeed will handle GPU placement"
        },
        {
            "issue": "Mixed device placement (some params on CPU, some on GPU)",
            "solution": "Use model.cpu() before trainer initialization when using DeepSpeed"
        },
        {
            "issue": "CUDA out of memory",
            "solution": "Reduce batch size, enable gradient checkpointing, or use DeepSpeed ZeRO optimization"
        }
    ]
    
    for i, item in enumerate(solutions, 1):
        print(f"\n{i}. Issue: {item['issue']}")
        print(f"   Solution: {item['solution']}")

def recommend_training_commands():
    """Recommend appropriate training commands based on environment."""
    print("\nRecommended Training Commands:")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        
        if gpu_count == 1:
            print("\nSingle GPU detected - Recommended commands:")
            print("1. Standard training (with device_map):")
            print("   python core/train.py --use_deepspeed=False")
            print("\n2. DeepSpeed single GPU (memory optimization):")
            print("   python core/train.py --use_deepspeed=True --deepspeed_config=configs/deepspeed_config_single_gpu.json")
            
        else:
            print(f"\n{gpu_count} GPUs detected - Recommended commands:")
            print("1. DeepSpeed multi-GPU training:")
            print("   deepspeed core/train.py --deepspeed_config=configs/deepspeed_config.json")
            print("\n2. Modal deployment:")
            print(f"   modal run core/modal_app.py --action=train --gpu_count={gpu_count}")
    else:
        print("\nNo CUDA GPUs detected - CPU training:")
        print("   python core/train.py --use_deepspeed=False")
    
    print("\nFor Modal deployment with specific GPU types:")
    print("   modal run core/modal_app.py --action=train --gpu_type=A100 --gpu_count=2")

def main():
    """Main troubleshooting function."""
    print("Device Placement Troubleshooting Tool")
    print("=" * 60)
    
    check_environment()
    test_model_loading_scenarios()
    test_deepspeed_config()
    provide_solutions()
    recommend_training_commands()
    
    print("\n" + "=" * 60)
    print("Troubleshooting completed!")
    print("\nIf issues persist:")
    print("1. Check that all dependencies are properly installed")
    print("2. Verify CUDA drivers and PyTorch CUDA compatibility")
    print("3. Ensure DeepSpeed is compiled correctly for your system")
    print("4. Review the DEVICE_MAP_FIX.md documentation")

if __name__ == "__main__":
    main()
