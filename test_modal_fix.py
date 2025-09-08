#!/usr/bin/env python3
"""
Test script to verify the Modal function call fix.
"""

import sys
import os

# Add the current directory to sys.path to access modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import_and_function_signature():
    """Test that we can import the deployment function and it has the right signature."""
    print("=== Testing Modal Function Fix ===")
    
    try:
        # Test import
        from deployment.deploy_modal import deploy_training
        print("✅ Successfully imported deploy_training")
        
        # Test function signature
        import inspect
        sig = inspect.signature(deploy_training)
        params = list(sig.parameters.keys())
        print(f"✅ Function parameters: {params}")
        
        # Check for required parameters
        required_params = ['gpu_type', 'gpu_count']
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            print(f"❌ Missing required parameters: {missing_params}")
            return False
        else:
            print("✅ All required parameters present")
        
        # Test that we can import the modal app components
        try:
            from core.modal_app import app, train_olmo_model
            print("✅ Successfully imported Modal app components")
            
            # Check that train_olmo_model is callable
            if callable(train_olmo_model):
                print("✅ train_olmo_model is callable")
            else:
                print("❌ train_olmo_model is not callable")
                return False
                
        except ImportError as e:
            print(f"❌ Failed to import Modal app components: {e}")
            return False
        
        print("✅ All tests passed! The Modal function fix should work.")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import deploy_training: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_import_and_function_signature()
    if success:
        print("\n🎉 Modal function fix verification successful!")
        print("The 'function' object has no attribute 'remote' error should be resolved.")
    else:
        print("\n💥 Modal function fix verification failed!")
        print("There may still be issues with the Modal function calls.")
    
    sys.exit(0 if success else 1)
