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
    .pip_install_from_requirements("requirements.txt")
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
    # Add local files last to avoid rebuilding on every change
    .add_local_file("train.py", "/root/app/train.py")
    .add_local_file("data_utils.py", "/root/app/data_utils.py")
    .add_local_file("model_utils.py", "/root/app/model_utils.py")
    .add_local_file("configs/deepspeed_config.json", "/root/app/configs/deepspeed_config.json")
    .add_local_file("configs/deepspeed_config_single_gpu.json", "/root/app/configs/deepspeed_config_single_gpu.json")
)

# Create volumes for model storage and checkpoints
model_volume = modal.Volume.from_name("olmo-model-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("olmo-datasets", create_if_missing=True)

# Mount points
MODEL_DIR = "/vol/models"
DATASET_DIR = "/vol/datasets"
OUTPUT_DIR = "/vol/outputs"

# GPU configuration - using dual A100s for better performance
gpu_config = "A100:2"  # Using 2x A100 GPUs for distributed training


@app.function(
    image=image,
    gpu=gpu_config,
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
    from .train import train
    
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
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Run training
    print(f"Starting training with the following configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Length: {max_length}")
    print(f"  Use LoRA: {use_lora}")
    print(f"  Use 4-bit: {use_4bit}")
    print(f"  Output Directory: {output_dir}")
    
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
        use_deepspeed=True,
        deepspeed_config_path="/root/app/configs/deepspeed_config.json",
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


@app.function(
    image=image,
    gpu=gpu_config,
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
):
    """
    Main entry point for Modal app.
    
    Actions:
    - train: Start a training job
    - test: Test inference with a checkpoint
    - list: List available checkpoints
    - cleanup: Clean up old checkpoints
    """
    
    if action == "train":
        print("Starting training job on Modal...")
        result = train_olmo_model.remote(
            model_name=model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            use_lora=use_lora,
            use_4bit=use_4bit,
            train_sample_size=train_sample_size,
            run_name=run_name,
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
        
    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, test, list, cleanup")


if __name__ == "__main__":
    # Example usage:
    # modal run modal_app.py --action train --num-epochs 3 --batch-size 4
    # modal run modal_app.py --action test --checkpoint-path /vol/outputs/run_20240101_120000/final_model
    # modal run modal_app.py --action list
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Modal OLMo Fine-tuning App")
    parser.add_argument("--action", type=str, default="train", choices=["train", "test", "list", "cleanup"])
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
    )
