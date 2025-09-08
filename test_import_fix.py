#!/usr/bin/env python3
"""
Test script to verify the import fix for Modal environment.
"""

import sys
import os

# Add the current directory to sys.path to access modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_train_imports():
    """Test that train.py can import its dependencies correctly."""
    print("=== Testing Train Import Fix ===")
    
    try:
        # Test importing train module
        from core.train import train
        print("✅ Successfully imported train function")
        
        # Test that the function is callable
        if callable(train):
            print("✅ train function is callable")
        else:
            print("❌ train function is not callable")
            return False
        
        # Test function signature
        import inspect
        sig = inspect.signature(train)
        params = list(sig.parameters.keys())
        print(f"✅ train function parameters: {params[:5]}...")  # Show first 5 params
        
        # Check for key parameters
        key_params = ['model_name', 'output_dir', 'num_train_epochs', 'use_lora', 'use_deepspeed']
        missing_params = [p for p in key_params if p not in params]
        
        if missing_params:
            print(f"❌ Missing key parameters: {missing_params}")
            return False
        else:
            print("✅ All key parameters present")
        
        print("✅ All train import tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import train: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_modal_compatibility():
    """Test that the imports work in a Modal-like environment."""
    print("\n=== Testing Modal Compatibility ===")
    
    try:
        # Simulate Modal environment by temporarily removing the core package from sys.modules
        # This forces the fallback to absolute imports
        original_modules = {}
        modules_to_remove = []
        
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('core.'):
                original_modules[module_name] = sys.modules[module_name]
                modules_to_remove.append(module_name)
        
        # Remove core modules to simulate Modal environment
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        # Add core directory to path to simulate Modal file structure
        core_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        
        # Try importing train again (should use absolute imports)
        import train
        print("✅ Successfully imported train with absolute imports")
        
        # Test that the train function works
        if hasattr(train, 'train') and callable(train.train):
            print("✅ train function accessible with absolute imports")
        else:
            print("❌ train function not accessible with absolute imports")
            return False
        
        # Restore original modules
        for module_name, module in original_modules.items():
            sys.modules[module_name] = module
        
        print("✅ Modal compatibility test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Modal compatibility test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in Modal compatibility test: {e}")
        return False

if __name__ == "__main__":
    success1 = test_train_imports()
    success2 = test_modal_compatibility()
    
    if success1 and success2:
        print("\n🎉 Import fix verification successful!")
        print("The relative import error should be resolved in Modal environment.")
    else:
        print("\n💥 Import fix verification failed!")
        print("There may still be issues with the imports.")
    
    sys.exit(0 if (success1 and success2) else 1)
