#!/usr/bin/env python3
"""
Test script to verify the WebUI can import the deployment module correctly.
This simulates the exact import path used in training_deployer.py
"""

import sys
import os

# This is the exact same path manipulation used in training_deployer.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from deployment.deploy_modal import deploy_training
    print("✅ SUCCESS: WebUI can now import the deployment module!")
    print(f"✅ deploy_training function found: {deploy_training}")
    print("✅ The 'No module named deployment' error has been fixed!")
    
    # Test that the function has the expected parameters
    import inspect
    sig = inspect.signature(deploy_training)
    params = list(sig.parameters.keys())
    print(f"✅ Function parameters: {params}")
    
    if 'gpu_type' in params and 'gpu_count' in params:
        print("✅ GPU configuration parameters are available!")
    else:
        print("⚠️  GPU configuration parameters missing")
        
except ImportError as e:
    print(f"❌ FAILED: Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ FAILED: Unexpected error: {e}")
    sys.exit(1)
