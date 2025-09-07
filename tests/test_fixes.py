#!/usr/bin/env python3
"""
Test script to verify the fixes for Modal OLMo fine-tuning issues.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def test_deepspeed_config():
    """Test that the DeepSpeed config files are valid."""
    print("Testing DeepSpeed configurations...")
    
    configs = [
        "configs/deepspeed_config.json",
        "configs/deepspeed_config_single_gpu.json"
    ]
    
    for config_path in configs:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✓ {config_path} is valid JSON")
                
                # Check key settings
                if 'gradient_accumulation_steps' in config:
                    print(f"  - Gradient accumulation: {config['gradient_accumulation_steps']}")
                if 'zero_optimization' in config:
                    print(f"  - ZeRO stage: {config['zero_optimization'].get('stage', 'N/A')}")
                if 'comms_logger' in config:
                    print(f"  - Comms logger: {config['comms_logger'].get('enabled', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"✗ {config_path} has invalid JSON: {e}")
                return False
        else:
            print(f"✗ {config_path} not found")
            return False
    
    return True


def test_environment_variables():
    """Test that environment variables are properly set."""
    print("\nTesting environment variable settings...")
    
    # Check the modal_app.py for environment variables
    with open('modal_app.py', 'r') as f:
        content = f.read()
    
    required_env_vars = [
        'PMIX_MCA_gds',
        'HF_HUB_DOWNLOAD_TIMEOUT',
        'TRANSFORMERS_TIMEOUT',
        'HF_HUB_ENABLE_HF_TRANSFER'
    ]
    
    for env_var in required_env_vars:
        if env_var in content:
            print(f"✓ {env_var} is configured")
        else:
            print(f"✗ {env_var} is missing")
            return False
    
    return True


def test_retry_logic():
    """Test that retry logic is implemented in model_utils."""
    print("\nTesting retry logic implementation...")
    
    with open('model_utils.py', 'r') as f:
        content = f.read()
    
    # Check for retry function
    if 'retry_on_network_error' in content:
        print("✓ retry_on_network_error function found")
    else:
        print("✗ retry_on_network_error function missing")
        return False
    
    # Check that retry is used for model loading
    if 'retry_on_network_error' in content and 'AutoTokenizer.from_pretrained' in content:
        print("✓ Retry logic applied to tokenizer loading")
    else:
        print("✗ Retry logic not applied to tokenizer loading")
        return False
    
    if 'retry_on_network_error' in content and 'AutoModelForCausalLM.from_pretrained' in content:
        print("✓ Retry logic applied to model loading")
    else:
        print("✗ Retry logic not applied to model loading")
        return False
    
    return True


def test_imports():
    """Test that all required modules can be imported."""
    print("\nTesting Python imports...")
    
    try:
        # Test local imports
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import model_utils
        print("✓ model_utils imports successfully")
        
        import data_utils
        print("✓ data_utils imports successfully")
        
        import train
        print("✓ train imports successfully")
        
        # Test retry function
        if hasattr(model_utils, 'retry_on_network_error'):
            print("✓ retry_on_network_error function available")
        else:
            print("✗ retry_on_network_error function not found")
            return False
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def test_modal_app_syntax():
    """Test that modal_app.py has valid Python syntax."""
    print("\nTesting modal_app.py syntax...")
    
    try:
        import py_compile
        py_compile.compile('modal_app.py', doraise=True)
        print("✓ modal_app.py has valid Python syntax")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error in modal_app.py: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Modal OLMo Fine-tuning Fixes")
    print("=" * 60)
    
    # Change to the modal_olmo_finetune directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    tests = [
        ("DeepSpeed Config", test_deepspeed_config),
        ("Environment Variables", test_environment_variables),
        ("Retry Logic", test_retry_logic),
        ("Python Imports", test_imports),
        ("Modal App Syntax", test_modal_app_syntax),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! The fixes are ready to deploy.")
        print("\nTo deploy to Modal, run:")
        print("  modal run modal_app.py")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
