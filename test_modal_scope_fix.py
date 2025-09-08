#!/usr/bin/env python3
"""
Test script to verify the Modal function scope fix.
"""

import sys
import os

# Add the current directory to sys.path to access modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_modal_function_scope():
    """Test that Modal functions are properly defined at global scope."""
    print("=== Testing Modal Function Scope Fix ===")
    
    try:
        # Test import
        from core.modal_app import app, train_olmo_model_impl, train_olmo_model
        print("✅ Successfully imported Modal app components")
        
        # Check that train_olmo_model_impl is a Modal function (has .remote attribute)
        if hasattr(train_olmo_model_impl, 'remote'):
            print("✅ train_olmo_model_impl is a proper Modal function")
        else:
            print("❌ train_olmo_model_impl is not a Modal function")
            return False
        
        # Check that train_olmo_model is callable
        if callable(train_olmo_model):
            print("✅ train_olmo_model is callable")
        else:
            print("❌ train_olmo_model is not callable")
            return False
        
        # Test function signature
        import inspect
        sig = inspect.signature(train_olmo_model_impl)
        params = list(sig.parameters.keys())
        print(f"✅ train_olmo_model_impl parameters: {params}")
        
        # Check for GPU parameters
        gpu_params = ['gpu_type', 'gpu_count']
        missing_params = [p for p in gpu_params if p not in params]
        
        if missing_params:
            print(f"❌ Missing GPU parameters: {missing_params}")
            return False
        else:
            print("✅ All GPU parameters present")
        
        # Test that with_options works
        try:
            modified_func = train_olmo_model_impl.with_options(gpu="T4:1")
            if hasattr(modified_func, 'remote'):
                print("✅ with_options works correctly")
            else:
                print("❌ with_options doesn't return a Modal function")
                return False
        except Exception as e:
            print(f"❌ with_options failed: {e}")
            return False
        
        print("✅ All Modal function scope tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Modal components: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_modal_function_scope()
    if success:
        print("\n🎉 Modal function scope fix verification successful!")
        print("The '@app.function decorator must apply to functions in global scope' error should be resolved.")
    else:
        print("\n💥 Modal function scope fix verification failed!")
        print("There may still be issues with the Modal function definitions.")
    
    sys.exit(0 if success else 1)
