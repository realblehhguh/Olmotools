#!/usr/bin/env python3
"""
Test script to verify the deployment module import works correctly.
"""

import sys
import os

# Add the current directory to sys.path to access the deployment module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from deployment.deploy_modal import deploy_training
    print("✅ Import successful!")
    print(f"✅ deploy_training function found: {deploy_training}")
    print("✅ The 'No module named deployment' error has been fixed!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
