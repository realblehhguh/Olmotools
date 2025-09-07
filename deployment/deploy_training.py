#!/usr/bin/env python3
"""
Fire-and-forget deployment script for Modal OLMo training.
Submits the training job to Modal and exits immediately.
"""

import argparse
import asyncio
import sys
from datetime import datetime
import json
import os

# Import Modal app
import modal
from core.modal_app import app, train_olmo_model


async def deploy_training(
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
    """Submit training job to Modal and return immediately."""
    
    # Generate run name if not provided
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"training_{timestamp}"
    
    print(f"🚀 Deploying training job: {run_name}")
    print(f"Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Max Length: {max_length}")
    print(f"  - LoRA: {use_lora}")
    print(f"  - 4-bit: {use_4bit}")
    print(f"  - GPUs: 2x A100")
    
    # Deploy the app and run the training function
    try:
        # Deploy the app first
        print("Deploying app to Modal...")
        # Use remote() with detach=True for fire-and-forget
        handle = await train_olmo_model.remote.aio(
            model_name=model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            use_lora=use_lora,
            use_4bit=use_4bit,
            train_sample_size=train_sample_size,
            run_name=run_name,
            return_exceptions=True
        )
        
        # Get the function call ID for tracking
        call_id = str(handle) if handle else "unknown"
        
        print(f"✅ Training job submitted successfully!")
        print(f"📋 Job ID: {call_id}")
        print(f"🏷️ Run Name: {run_name}")
        
        # Save deployment info for Discord bot
        deployment_info = {
            "job_id": call_id,
            "run_name": run_name,
            "wandb_run_name": run_name,
            "wandb_project": "olmo-finetune-modal",
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_name": model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "use_lora": use_lora,
                "use_4bit": use_4bit,
                "train_sample_size": train_sample_size,
            }
        }
        
        # Save to file for Discord bot to read
        os.makedirs("deployments", exist_ok=True)
        deployment_file = f"deployments/{run_name}.json"
        with open(deployment_file, "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"💾 Deployment info saved to: {deployment_file}")
        print("\n📊 Monitor your training:")
        print(f"  - Modal Dashboard: https://modal.com/apps")
        print(f"  - W&B Dashboard: https://wandb.ai/your-entity/olmo-finetune-modal")
        print(f"  - Discord Bot: Run discord_bot.py to get DM updates")
        print("\n🎯 Training is now running in the cloud!")
        print("You can safely close this terminal. The training will continue on Modal.")
        
        return call_id, run_name


def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy OLMo training to Modal (fire-and-forget)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="allenai/OLMo-2-1124-7B",
        help="Model name or path"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=2048,
        help="Maximum sequence length"
    )
    
    # Model configuration
    parser.add_argument(
        "--use-lora", 
        action="store_true", 
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--no-lora", 
        dest="use_lora", 
        action="store_false",
        help="Disable LoRA"
    )
    parser.add_argument(
        "--use-4bit", 
        action="store_true", 
        default=False,
        help="Use 4-bit quantization"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--train-sample-size", 
        type=int, 
        default=None,
        help="Number of training samples (None for full dataset)"
    )
    
    # Run configuration
    parser.add_argument(
        "--run-name", 
        type=str, 
        default=None,
        help="Name for this training run"
    )
    
    # Quick presets
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Quick test with minimal settings (100 samples, 1 epoch)"
    )
    parser.add_argument(
        "--full-training", 
        action="store_true",
        help="Full training with default settings"
    )
    
    args = parser.parse_args()
    
    # Apply presets
    if args.quick_test:
        args.train_sample_size = 100
        args.num_epochs = 1
        if not args.run_name:
            args.run_name = "quick_test"
        print("🧪 Using quick test settings: 100 samples, 1 epoch")
    elif args.full_training:
        args.train_sample_size = None
        args.num_epochs = 3
        if not args.run_name:
            args.run_name = "full_training"
        print("🎯 Using full training settings")
    
    # Run deployment
    try:
        call_id, run_name = asyncio.run(deploy_training(
            model_name=args.model_name,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            use_lora=args.use_lora,
            use_4bit=args.use_4bit,
            train_sample_size=args.train_sample_size,
            run_name=args.run_name,
        ))
        
        # Success - exit cleanly
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
