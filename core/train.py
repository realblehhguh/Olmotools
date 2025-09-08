"""
Training script for fine-tuning OLMo-2-1124-7B with DeepSpeed.
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Optional

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling
)
from transformers.trainer_callback import TrainerCallback
import wandb

# Import modules - handle both relative and absolute imports for Modal compatibility
try:
    # Try relative imports first (when run as module)
    from .model_utils import (
        load_olmo_model_and_tokenizer,
        create_training_arguments,
        prepare_model_for_training,
        save_model_checkpoint
    )
    from .data_utils import create_data_module
except ImportError:
    # Fall back to absolute imports (when run directly in Modal)
    from model_utils import (
        load_olmo_model_and_tokenizer,
        create_training_arguments,
        prepare_model_for_training,
        save_model_checkpoint
    )
    from data_utils import create_data_module

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SaveCheckpointCallback(TrainerCallback):
    """Custom callback to save checkpoints with additional metadata."""
    
    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Save additional training metadata with checkpoint."""
        if state.is_world_process_zero:
            # Save training state
            checkpoint_path = os.path.join(
                args.output_dir,
                f"checkpoint-{state.global_step}"
            )
            
            state_dict = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
            }
            
            with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
                json.dump(state_dict, f, indent=2)
            
            logger.info(f"Saved training state to {checkpoint_path}")


def setup_wandb(project_name: str = "olmo-finetune", run_name: Optional[str] = None):
    """Initialize Weights & Biases logging."""
    if "WANDB_API_KEY" in os.environ:
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model": "allenai/OLMo-2-1124-7B",
                "dataset": "QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered",
                "training_method": "DeepSpeed"
            }
        )
        logger.info("Weights & Biases initialized")
    else:
        logger.warning("WANDB_API_KEY not found. Skipping W&B initialization.")


def train(
    model_name: str = "allenai/OLMo-2-1124-7B",
    output_dir: str = "./outputs",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    max_length: int = 2048,
    use_lora: bool = True,
    use_4bit: bool = False,
    use_deepspeed: bool = True,
    deepspeed_config_path: Optional[str] = "configs/deepspeed_config.json",
    train_sample_size: Optional[int] = None,
    val_sample_size: Optional[int] = 500,
    seed: int = 42,
    run_name: Optional[str] = None,
    wandb_project: str = "olmo-finetune",
    resume_from_checkpoint: Optional[str] = None
):
    """Main training function."""
    
    # Set seed for reproducibility
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if available
    setup_wandb(project_name=wandb_project, run_name=run_name)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_olmo_model_and_tokenizer(
        model_name=model_name,
        use_lora=use_lora,
        use_4bit=use_4bit,
        torch_dtype=torch.float16 if not use_4bit else torch.float16,
        use_deepspeed=use_deepspeed
    )
    
    # Prepare model for training
    model = prepare_model_for_training(model)
    
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    # Load and prepare datasets
    logger.info("Loading and preparing datasets...")
    train_dataset, val_dataset, data_collator = create_data_module(
        tokenizer=tokenizer,
        max_length=max_length,
        train_sample_size=train_sample_size,
        val_sample_size=val_sample_size
    )
    
    # Create training arguments
    logger.info("Creating training arguments...")
    training_args = create_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        deepspeed_config=deepspeed_config_path if use_deepspeed else None,
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        run_name=run_name
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[SaveCheckpointCallback()]
    )
    
    # Train model
    logger.info("Starting training...")
    logger.info(f"Total training samples: {len(train_dataset)}")
    logger.info(f"Total validation samples: {len(val_dataset)}")
    logger.info(f"Number of epochs: {num_train_epochs}")
    logger.info(f"Total batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    
    # Resume from checkpoint if specified
    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint}")
    
    # Start training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Save training metrics
    metrics = train_result.metrics
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final model saved to: {output_dir}/final_model")
    
    # Close W&B run if initialized
    if wandb.run is not None:
        wandb.finish()
    
    return trainer, model, tokenizer


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Fine-tune OLMo-2-1124-7B with DeepSpeed")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-1124-7B",
                        help="Model name or path")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--use_4bit", action="store_true", default=False,
                        help="Use 4-bit quantization")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # Dataset arguments
    parser.add_argument("--train_sample_size", type=int, default=None,
                        help="Number of training samples (None for full dataset)")
    parser.add_argument("--val_sample_size", type=int, default=500,
                        help="Number of validation samples")
    
    # DeepSpeed arguments
    parser.add_argument("--use_deepspeed", action="store_true", default=True,
                        help="Use DeepSpeed for training")
    parser.add_argument("--deepspeed_config", type=str, default="configs/deepspeed_config.json",
                        help="Path to DeepSpeed config file")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for the training run")
    parser.add_argument("--wandb_project", type=str, default="olmo-finetune",
                        help="W&B project name")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Run training
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        use_deepspeed=args.use_deepspeed,
        deepspeed_config_path=args.deepspeed_config,
        train_sample_size=args.train_sample_size,
        val_sample_size=args.val_sample_size,
        seed=args.seed,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
