#!/usr/bin/env python3
"""
Debug script to test the deployment module import from different contexts.
"""

import sys
import os

print("=== Debug Import Test ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# Test 1: Direct import (should work since we're in the project root)
print("\n--- Test 1: Direct import from project root ---")
try:
    from deployment.deploy_modal import deploy_training
    print("✅ SUCCESS: Direct import worked!")
except ImportError as e:
    print(f"❌ FAILED: {e}")

# Test 2: Simulate WebUI import path
print("\n--- Test 2: Simulate WebUI import path ---")
try:
    # This simulates what the WebUI does
    webui_file = os.path.join(os.path.dirname(__file__), "web_ui", "training_deployer.py")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(webui_file)))
    print(f"WebUI file path: {webui_file}")
    print(f"Calculated parent dir: {parent_dir}")
    print(f"Parent dir exists: {os.path.exists(parent_dir)}")
    print(f"Parent dir contents: {os.listdir(parent_dir) if os.path.exists(parent_dir) else 'N/A'}")
    
    # Check if deployment directory exists in parent
    deployment_dir = os.path.join(parent_dir, "deployment")
    print(f"Deployment dir: {deployment_dir}")
    print(f"Deployment dir exists: {os.path.exists(deployment_dir)}")
    if os.path.exists(deployment_dir):
        print(f"Deployment dir contents: {os.listdir(deployment_dir)}")
    
    # Add to sys.path and try import
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    print(f"sys.path[0]: {sys.path[0]}")
    
    from deployment.deploy_modal import deploy_training
    print("✅ SUCCESS: WebUI-style import worked!")
    
except ImportError as e:
    print(f"❌ FAILED: {e}")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 3: Check current sys.path
print("\n--- Test 3: Current sys.path ---")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"  {i}: {path}")

# Test 4: Check if deployment module is importable
print("\n--- Test 4: Check deployment module structure ---")
try:
    import deployment
    print(f"✅ deployment module found at: {deployment.__file__}")
    print(f"deployment module contents: {dir(deployment)}")
except ImportError as e:
    print(f"❌ deployment module not found: {e}")

print("\n=== End Debug Test ===")
