"""
Modal Labs application for fine-tuning OLMo-2-1124-7B with DeepSpeed.
"""

import modal
import os
from pathlib import Path
from datetime import datetime

# Create Modal app
app = modal.App("olmo-finetune-deepspeed")

# Define the Docker image with all required dependencies
# Use a CUDA-enabled base image for DeepSpeed
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.10")
    .run_commands(
        "apt-get update",
        "apt-get install -y git wget curl build-essential",
        # Install MPI libraries for DeepSpeed distributed training
        "apt-get install -y libopenmpi-dev openmpi-bin",
        "mkdir -p /root/app/configs",
    )
)

# Get requirements.txt path with proper resolution
def get_requirements_path():
    """Get the absolute path to requirements.txt, handling different deployment environments."""
    possible_paths = [
        "../requirements.txt",                    # Local development from core/
        "requirements.txt",                       # When running from project root
        "./requirements.txt",                     # Current directory
        "/opt/render/project/requirements.txt",   # Render deployment
        "modal_olmo_finetune/requirements.txt",   # From parent directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If none found, return the most likely path and let it fail with a clear error
    return "requirements.txt"

# Continue building the image with requirements
try:
    requirements_path = get_requirements_path()
    image = image.pip_install_from_requirements(requirements_path)
except Exception as e:
    print(f"Warning: Could not install from requirements.txt: {e}")
    print("Installing basic packages manually...")
    # Fallback to manual package installation
    image = image.pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "deepspeed>=0.9.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "wandb>=0.15.0",
        "huggingface_hub>=0.15.0",
        "tokenizers>=0.13.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
    )

image = (image
    .run_commands(
        # Set CUDA environment variables for DeepSpeed
        "export CUDA_HOME=/usr/local/cuda",
        "export PATH=$CUDA_HOME/bin:$PATH",
        "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        # Disable MPI/PMIX for single GPU training to avoid shared memory errors
        "PMIX_MCA_gds": "hash",
        "OMPI_MCA_btl": "^openib",
        # Increase timeout for HuggingFace downloads
        "HF_HUB_DOWNLOAD_TIMEOUT": "600",
        "TRANSFORMERS_TIMEOUT": "600",
        # Disable HF transfer for now (requires additional package)
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HUB_DISABLE_PROGRESS_BARS": "0"
    })
)

# Helper function to find files in different deployment environments
def get_file_path(filename, subdirs=None):
    """Get the absolute path to a file, handling different deployment environments."""
    if subdirs is None:
        subdirs = ["", "core", "src"]
    
    # Try different possible locations
    possible_paths = []
    
    for subdir in subdirs:
        if subdir:
            possible_paths.extend([
                f"{subdir}/{filename}",           # Local development
                f"./{subdir}/{filename}",         # Current directory
                f"../{subdir}/{filename}",        # Parent directory
                f"/opt/render/project/{subdir}/{filename}",  # Render deployment
                f"/opt/render/project/src/{filename}",       # Render src directory
            ])
        else:
            possible_paths.extend([
                filename,                         # Current directory
                f"./{filename}",                  # Current directory
                f"../{filename}",                 # Parent directory
                f"/opt/render/project/{filename}", # Render deployment
            ])
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If none found, return the most likely path and let it fail with a clear error
    return filename

def get_config_path(filename):
    """Get the absolute path to a config file, handling different deployment environments."""
    return get_file_path(filename, ["configs", "../configs", "./configs"])

# Add local files with proper path resolution
try:
    train_py_path = get_file_path("train.py", ["core", "src", ""])
    data_utils_path = get_file_path("data_utils.py", ["core", "src", ""])
    model_utils_path = get_file_path("model_utils.py", ["core", "src", ""])
    modal_device_fix_path = get_file_path("modal_device_fix.py", ["", "src", "."])
    
    image = image.add_local_file(train_py_path, "/root/app/train.py")
    image = image.add_local_file(data_utils_path, "/root/app/data_utils.py")
    image = image.add_local_file(model_utils_path, "/root/app/model_utils.py")
    image = image.add_local_file(modal_device_fix_path, "/root/app/modal_device_fix.py")
except Exception as e:
    print(f"Warning: Could not add core files to image: {e}")
    print("Core files will need to be available at runtime")

# Add config files to image
try:
    deepspeed_config_path = get_config_path("deepspeed_config.json")
    deepspeed_single_gpu_path = get_config_path("deepspeed_config_single_gpu.json")
    
    image = image.add_local_file(deepspeed_config_path, "/root/app/configs/deepspeed_config.json")
    image = image.add_local_file(deepspeed_single_gpu_path, "/root/app/configs/deepspeed_config_single_gpu.json")
except Exception as e:
    print(f"Warning: Could not add config files to image: {e}")
    print("Config files will need to be available at runtime")

# Create volumes for model storage and checkpoints
model_volume = modal.Volume.from_name("olmo-model-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("olmo-datasets", create_if_missing=True)

# Mount points
MODEL_DIR = "/vol/models"
DATASET_DIR = "/vol/datasets"
OUTPUT_DIR = "/vol/outputs"

# GPU configuration - configurable GPU type and count
def get_gpu_config(gpu_type: str = "A100", gpu_count: int = 2) -> str:
    """
    Get GPU configuration string for Modal.
    
    Available GPU types (as of Modal's latest documentation):
    - T4: Basic GPU, good for inference and light training
    - L4: Mid-range GPU with good price/performance, 48GB GPU RAM
    - A10: Good for training, supports up to 4 GPUs, up to 96GB GPU RAM
    - A100: High-end training GPU, supports up to 8 GPUs
    - A100-40GB: A100 with 40GB memory
    - A100-80GB: A100 with 80GB memory  
    - L40S: High-end GPU for training, supports up to 8 GPUs
    - H100: Latest high-end GPU for training, supports up to 8 GPUs
    - H100i: H100 inference optimized (H100I), supports up to 8 GPUs
    - H200: Latest flagship GPU, supports up to 8 GPUs
    - B200: Most powerful GPU available (Blackwell architecture), supports up to 8 GPUs
    
    GPU Count Limits:
    - A10: Maximum 4 GPUs per container (up to 96GB GPU RAM)
    - All others (B200, H200, H100, A100, L4, T4, L40S): Maximum 8 GPUs per container (up to 1,536GB CPU RAM)
    
    Args:
        gpu_type: Type of GPU to use
        gpu_count: Number of GPUs (1-4 for A10, 1-8 for others)
    
    Returns:
        GPU configuration string in format "TYPE:COUNT"
    """
    # Normalize GPU type (handle case variations)
    gpu_type = gpu_type.upper()
    
    # Map common variations to official names
    gpu_type_mapping = {
        "H100I": "H100i",  # Handle case variation
        "A100-40": "A100-40GB",
        "A100-80": "A100-80GB",
    }
    gpu_type = gpu_type_mapping.get(gpu_type, gpu_type)
    
    # Valid GPU types
    valid_gpu_types = {
        "T4", "L4", "A10", "A100", "A100-40GB", "A100-80GB", 
        "L40S", "H100", "H100i", "H200", "B200"
    }
    
    if gpu_type not in valid_gpu_types:
        raise ValueError(f"Invalid GPU type: {gpu_type}. Valid types: {', '.join(sorted(valid_gpu_types))}")
    
    # Validate GPU count based on type
    if gpu_type == "A10":
        if gpu_count > 4:
            raise ValueError("A10 GPUs support maximum 4 GPUs per container (up to 96GB GPU RAM)")
        elif gpu_count < 1:
            raise ValueError("At least 1 GPU required")
    else:
        if gpu_count > 8:
            raise ValueError(f"{gpu_type} GPUs support maximum 8 GPUs per container (up to 1,536GB CPU RAM)")
        elif gpu_count < 1:
            raise ValueError("At least 1 GPU required")
    
    return f"{gpu_type}:{gpu_count}"


def get_gpu_recommendations(use_case: str = "training") -> dict:
    """
    Get GPU recommendations based on use case.
    
    Args:
        use_case: Either "training", "inference", or "development"
    
    Returns:
        Dictionary with recommended GPU configurations
    """
    recommendations = {
        "training": {
            "budget": {"type": "L4", "count": 1, "description": "Cost-effective for small models"},
            "balanced": {"type": "A100", "count": 2, "description": "Good balance of performance and cost"},
            "performance": {"type": "H100", "count": 4, "description": "High performance for large models"},
            "maximum": {"type": "B200", "count": 8, "description": "Maximum performance for largest models"}
        },
        "inference": {
            "budget": {"type": "T4", "count": 1, "description": "Basic inference workloads"},
            "balanced": {"type": "L4", "count": 1, "description": "Good performance for most inference tasks"},
            "performance": {"type": "A100", "count": 1, "description": "High-throughput inference"},
            "maximum": {"type": "H100i", "count": 1, "description": "Optimized for inference workloads"}
        },
        "development": {
            "budget": {"type": "T4", "count": 1, "description": "Development and testing"},
            "balanced": {"type": "L4", "count": 1, "description": "Development with moderate compute needs"},
            "performance": {"type": "A100", "count": 1, "description": "Development with heavy compute needs"}
        }
    }
    
    return recommendations.get(use_case, recommendations["training"])


# Global Modal function for training - must be at module level
@app.function(
    image=image,
    gpu="A100:2",  # Default GPU config, will be overridden dynamically
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
    },
    timeout=86400,  # 24 hours timeout
    memory=32768,  # 32GB RAM
    secrets=[
        modal.Secret.from_name("huggingface-token"),  # For accessing gated models
        modal.Secret.from_name("wandb-secret"),  # For W&B logging
    ],
)
def train_olmo_model_impl(
    model_name: str = "allenai/OLMo-2-1124-7B",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    use_lora: bool = True,
    use_4bit: bool = False,
    train_sample_size: int = None,
    run_name: str = None,
    gpu_type: str = "A100",
    gpu_count: int = 2,
):
    """Train OLMo model on Modal with DeepSpeed."""
    
    import sys
    import torch
    from huggingface_hub import login
    
    # Set CUDA_HOME for DeepSpeed
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    
    # Add current directory to path
    sys.path.append("/root/app")
    
    # Import our training modules
    from train import train
    
    # Login to HuggingFace if token is available
    if "HUGGINGFACE_TOKEN" in os.environ:
        login(token=os.environ["HUGGINGFACE_TOKEN"])
        print("Logged in to HuggingFace Hub")
    
    # Set up output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}/run_{timestamp}"
    if run_name:
        output_dir = f"{OUTPUT_DIR}/{run_name}_{timestamp}"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log GPU information
    print(f"GPU Configuration: {gpu_type}:{gpu_count}")
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Choose appropriate DeepSpeed config based on GPU count
    if gpu_count == 1:
        deepspeed_config_path = "/root/app/configs/deepspeed_config_single_gpu.json"
    else:
        deepspeed_config_path = "/root/app/configs/deepspeed_config.json"
    
    # Run training
    print(f"Starting training with the following configuration:")
    print(f"  Model: {model_name}")
    print(f"  GPU Type: {gpu_type}")
    print(f"  GPU Count: {gpu_count}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Length: {max_length}")
    print(f"  Use LoRA: {use_lora}")
    print(f"  Use 4-bit: {use_4bit}")
    print(f"  DeepSpeed Config: {deepspeed_config_path}")
    print(f"  Output Directory: {output_dir}")
    
    # Apply Modal-specific device placement fixes
    sys.path.append("/root/app")  # Ensure we can import modal_device_fix
    try:
        from modal_device_fix import setup_modal_environment_for_training, apply_modal_device_fixes
        
        # For Modal, we typically want to avoid distributed training complications
        # Use single GPU approach even with multiple GPUs available
        # This prevents the MASTER_ADDR error and simplifies device management
        effective_gpu_count = 1  # Force single GPU mode for Modal
        
        # Setup Modal environment for proper device placement
        setup_modal_environment_for_training(gpu_count=effective_gpu_count, force_cpu_start=True)
        apply_modal_device_fixes()
        
        print(f"Applied Modal-specific device placement fixes (using single GPU mode)")
    except ImportError as e:
        print(f"Warning: Could not import modal_device_fix: {e}")
        # Fallback: ensure no distributed training environment variables are set
        for var in ["WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(var, None)
        print("Cleared distributed environment variables for single GPU training")
    
    # For Modal single GPU mode, disable DeepSpeed to avoid device placement issues
    use_deepspeed_for_training = False  # Disable DeepSpeed for Modal to simplify device management
    
    # Call the training function
    trainer, model, tokenizer = train(
        model_name=model_name,
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=500,
        max_length=max_length,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_deepspeed=use_deepspeed_for_training,
        deepspeed_config_path=deepspeed_config_path if use_deepspeed_for_training else None,
        train_sample_size=train_sample_size,
        val_sample_size=500,
        seed=42,
        run_name=run_name,
        wandb_project="olmo-finetune-modal"
    )
    
    # Commit volumes to persist checkpoints
    model_volume.commit()
    
    print(f"Training completed! Model saved to: {output_dir}")
    return output_dir


def train_olmo_model(
    model_name: str = "allenai/OLMo-2-1124-7B",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    use_lora: bool = True,
    use_4bit: bool = False,
    train_sample_size: int = None,
    run_name: str = None,
    gpu_type: str = "A100",
    gpu_count: int = 2,
):
    """
    Train OLMo model on Modal with configurable GPU settings.
    
    This function executes the training job using the global Modal function.
    Note: GPU configuration is passed as parameters to the function since
    Modal functions must be defined at global scope with fixed configurations.
    """
    # Validate GPU configuration
    gpu_config = get_gpu_config(gpu_type, gpu_count)
    print(f"Using GPU config: {gpu_config}")
    
    # Execute the training directly - the GPU config is passed as parameters
    # The actual GPU allocation is handled by the function's decorator
    return train_olmo_model_impl.remote(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        use_lora=use_lora,
        use_4bit=use_4bit,
        train_sample_size=train_sample_size,
        run_name=run_name,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )


@app.function(
    image=image,
    gpu=get_gpu_config("A100", 1),  # Single GPU for inference
    volumes={MODEL_DIR: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def test_model_inference(
    checkpoint_path: str,
    prompt: str = "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
    max_new_tokens: int = 256,
):
    """Test inference with a trained model checkpoint."""
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login
    
    # Login to HuggingFace if needed
    if "HUGGINGFACE_TOKEN" in os.environ:
        login(token=os.environ["HUGGINGFACE_TOKEN"])
    
    # Load model and tokenizer from checkpoint
    print(f"Loading model from: {checkpoint_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    print(f"Generating response for prompt: {prompt[:100]}...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response:\n{response}")
    
    return response


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
)
def list_checkpoints():
    """List all available model checkpoints."""
    import os
    
    checkpoints = []
    output_base = OUTPUT_DIR
    
    if os.path.exists(output_base):
        for run_dir in os.listdir(output_base):
            run_path = os.path.join(output_base, run_dir)
            if os.path.isdir(run_path):
                # Check for final model and checkpoints
                final_model = os.path.join(run_path, "final_model")
                if os.path.exists(final_model):
                    checkpoints.append(final_model)
                
                # List intermediate checkpoints
                for item in os.listdir(run_path):
                    if item.startswith("checkpoint-"):
                        checkpoint_path = os.path.join(run_path, item)
                        checkpoints.append(checkpoint_path)
    
    print("Available checkpoints:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")
    
    return checkpoints


@app.function(
    image=image,
    schedule=modal.Period(days=7),  # Run weekly cleanup
    volumes={MODEL_DIR: model_volume},
)
def cleanup_old_checkpoints(keep_last_n: int = 5):
    """Clean up old checkpoints to save storage space."""
    import os
    import shutil
    from datetime import datetime, timedelta
    
    output_base = OUTPUT_DIR
    if not os.path.exists(output_base):
        return
    
    # Get all run directories with timestamps
    runs = []
    for run_dir in os.listdir(output_base):
        run_path = os.path.join(output_base, run_dir)
        if os.path.isdir(run_path):
            # Get creation time
            mtime = os.path.getmtime(run_path)
            runs.append((mtime, run_path))
    
    # Sort by modification time (newest first)
    runs.sort(reverse=True)
    
    # Keep only the last N runs
    for i, (mtime, run_path) in enumerate(runs):
        if i >= keep_last_n:
            print(f"Removing old checkpoint: {run_path}")
            shutil.rmtree(run_path)
    
    # Commit volume changes
    model_volume.commit()
    
    print(f"Cleanup completed. Kept {min(len(runs), keep_last_n)} most recent runs.")


@app.function(
    image=image,
    timeout=300,
)
def show_gpu_recommendations(use_case: str = "training"):
    """Show GPU recommendations for different use cases."""
    recommendations = get_gpu_recommendations(use_case)
    
    print(f"\nGPU Recommendations for {use_case.title()}:")
    print("=" * 50)
    
    for tier, config in recommendations.items():
        gpu_config = get_gpu_config(config["type"], config["count"])
        print(f"\n{tier.title()}: {gpu_config}")
        print(f"  Description: {config['description']}")
        print(f"  Command: --gpu_type {config['type']} --gpu_count {config['count']}")
    
    print(f"\nAvailable GPU Types:")
    print("T4, L4, A10, A100, A100-40GB, A100-80GB, L40S, H100, H100i, H200, B200")
    
    print(f"\nGPU Count Limits:")
    print("- A10: Maximum 4 GPUs per container")
    print("- All others: Maximum 8 GPUs per container")
    
    return recommendations


@app.local_entrypoint()
def main(
    action: str = "train",
    model_name: str = "allenai/OLMo-2-1124-7B",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    use_lora: bool = True,
    use_4bit: bool = False,
    train_sample_size: int = None,
    run_name: str = None,
    checkpoint_path: str = None,
    prompt: str = None,
    gpu_type: str = "A100",
    gpu_count: int = 2,
):
    """
    Main entry point for Modal app.
    
    Actions:
    - train: Start a training job
    - test: Test inference with a checkpoint
    - list: List available checkpoints
    - cleanup: Clean up old checkpoints
    - recommendations: Show GPU recommendations for different use cases
    """
    
    if action == "train":
        print(f"Starting training job on Modal with {gpu_type}:{gpu_count} GPU configuration...")
        result = train_olmo_model(
            model_name=model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            use_lora=use_lora,
            use_4bit=use_4bit,
            train_sample_size=train_sample_size,
            run_name=run_name,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )
        print(f"Training completed! Output directory: {result}")
        
    elif action == "test":
        if not checkpoint_path:
            print("Error: checkpoint_path is required for testing")
            return
        
        if not prompt:
            prompt = "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n"
        
        print("Testing model inference...")
        response = test_model_inference.remote(
            checkpoint_path=checkpoint_path,
            prompt=prompt
        )
        print(f"Generated response:\n{response}")
        
    elif action == "list":
        print("Listing available checkpoints...")
        checkpoints = list_checkpoints.remote()
        for checkpoint in checkpoints:
            print(f"  - {checkpoint}")
            
    elif action == "cleanup":
        print("Cleaning up old checkpoints...")
        cleanup_old_checkpoints.remote()
        print("Cleanup completed.")
        
    elif action == "recommendations":
        print("Showing GPU recommendations...")
        show_gpu_recommendations.remote("training")
        show_gpu_recommendations.remote("inference")
        show_gpu_recommendations.remote("development")
        
    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, test, list, cleanup, recommendations")


if __name__ == "__main__":
    # Example usage:
    # modal run modal_app.py --action train --num-epochs 3 --batch-size 4
    # modal run modal_app.py --action test --checkpoint-path /vol/outputs/run_20240101_120000/final_model
    # modal run modal_app.py --action list
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Modal OLMo Fine-tuning App")
    parser.add_argument("--action", type=str, default="train", choices=["train", "test", "list", "cleanup", "recommendations"])
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-1124-7B")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true", default=False)
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--gpu_type", type=str, default="A100", 
                       choices=["T4", "L4", "A10", "A100", "A100-40GB", "A100-80GB", "L40S", "H100", "H100i", "H200", "B200"],
                       help="GPU type to use for training")
    parser.add_argument("--gpu_count", type=int, default=2, 
                       help="Number of GPUs to use (1-8 for most types, 1-4 for A10)")
    
    args = parser.parse_args()
    
    main(
        action=args.action,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        train_sample_size=args.train_sample_size,
        run_name=args.run_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
    )
