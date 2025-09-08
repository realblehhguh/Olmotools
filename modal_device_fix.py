#!/usr/bin/env python3
"""
Modal-specific device placement fix for distributed training.
This script provides utilities to ensure proper device handling in Modal environments.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)

def setup_modal_environment_for_training(gpu_count: int = 1, force_cpu_start: bool = True):
    """
    Setup Modal environment for proper device placement in training.
    
    Args:
        gpu_count: Number of GPUs being used
        force_cpu_start: Whether to force model to start on CPU
    """
    
    # Set environment variables for distributed training detection
    if gpu_count > 1:
        os.environ["WORLD_SIZE"] = str(gpu_count)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        # Set required distributed training environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        logger.info(f"Set distributed environment variables for {gpu_count} GPUs")
    else:
        # For single GPU, ensure we don't trigger distributed training
        # Remove any existing distributed environment variables
        for var in ["WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(var, None)
        logger.info("Cleared distributed environment variables for single GPU training")
    
    # Set Modal-specific environment variables
    os.environ["MODAL_ENVIRONMENT"] = "true"
    os.environ["FORCE_CPU_MODEL_START"] = str(force_cpu_start).lower()
    
    # Ensure CUDA is properly initialized
    if torch.cuda.is_available():
        torch.cuda.init()
        logger.info(f"CUDA initialized. Available devices: {torch.cuda.device_count()}")
    
    logger.info("Modal environment setup completed")

def ensure_model_on_cpu(model):
    """
    Ensure all model parameters are on CPU for consistent device placement.
    This is crucial for DeepSpeed initialization in Modal environments.
    """
    if hasattr(model, 'cpu'):
        logger.info("Moving model to CPU for consistent device placement")
        model = model.cpu()
        
        # Verify all parameters are on CPU
        cpu_params = 0
        gpu_params = 0
        for param in model.parameters():
            if param.device.type == 'cpu':
                cpu_params += 1
            else:
                gpu_params += 1
        
        logger.info(f"Model parameters: {cpu_params} on CPU, {gpu_params} on GPU")
        
        if gpu_params > 0:
            logger.warning(f"Found {gpu_params} parameters not on CPU - forcing CPU placement")
            # Force all parameters to CPU
            for param in model.parameters():
                if param.device.type != 'cpu':
                    param.data = param.data.cpu()
    
    return model

def check_device_consistency(model, name="model"):
    """
    Check that all model parameters are on the same device.
    """
    devices = set()
    param_count = 0
    
    for param in model.parameters():
        devices.add(param.device)
        param_count += 1
    
    logger.info(f"{name} has {param_count} parameters across {len(devices)} device(s): {devices}")
    
    if len(devices) > 1:
        logger.error(f"DEVICE INCONSISTENCY DETECTED in {name}!")
        logger.error(f"Parameters are spread across multiple devices: {devices}")
        return False
    
    logger.info(f"{name} device consistency check: PASSED")
    return True

def apply_modal_device_fixes():
    """
    Apply Modal-specific device placement fixes.
    This should be called at the beginning of training in Modal environment.
    """
    
    # Check if we're in Modal environment
    if os.environ.get("MODAL_ENVIRONMENT") != "true":
        logger.info("Not in Modal environment - skipping Modal-specific fixes")
        return
    
    logger.info("Applying Modal-specific device placement fixes...")
    
    # Set additional environment variables for better device handling
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error reporting
    os.environ["TORCH_USE_CUDA_DSA"] = "1"    # Better CUDA debugging
    
    # Clear any cached CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
    
    logger.info("Modal device fixes applied")

def wrap_model_loading_for_modal(load_function, *args, **kwargs):
    """
    Wrapper for model loading functions to ensure proper device placement in Modal.
    """
    
    logger.info("Loading model with Modal device placement wrapper...")
    
    # Apply Modal fixes before loading
    apply_modal_device_fixes()
    
    # Load the model
    result = load_function(*args, **kwargs)
    
    # Handle different return types
    if isinstance(result, tuple):
        model = result[0]
        other_items = result[1:]
    else:
        model = result
        other_items = ()
    
    # Ensure model is on CPU for consistent device placement
    if os.environ.get("FORCE_CPU_MODEL_START", "true").lower() == "true":
        model = ensure_model_on_cpu(model)
    
    # Check device consistency
    check_device_consistency(model, "loaded_model")
    
    # Return in the same format as received
    if other_items:
        return (model,) + other_items
    else:
        return model

# Example usage functions
def create_modal_training_wrapper(train_function):
    """
    Create a wrapper for training functions that applies Modal-specific fixes.
    """
    
    def modal_train_wrapper(*args, **kwargs):
        logger.info("Starting training with Modal device placement wrapper...")
        
        # Setup Modal environment
        gpu_count = kwargs.get('gpu_count', 1)
        setup_modal_environment_for_training(gpu_count)
        
        # Apply device fixes
        apply_modal_device_fixes()
        
        # Call the original training function
        return train_function(*args, **kwargs)
    
    return modal_train_wrapper

if __name__ == "__main__":
    # Test the Modal device fixes
    print("Testing Modal device placement fixes...")
    
    setup_modal_environment_for_training(gpu_count=2)
    apply_modal_device_fixes()
    
    if torch.cuda.is_available():
        # Create a simple model for testing
        model = torch.nn.Linear(10, 1)
        print(f"Test model device: {next(model.parameters()).device}")
        
        # Test device consistency check
        check_device_consistency(model, "test_model")
        
        # Test CPU placement
        model = ensure_model_on_cpu(model)
        print(f"After CPU placement: {next(model.parameters()).device}")
        
        check_device_consistency(model, "test_model_after_cpu")
    
    print("Modal device fixes test completed!")
