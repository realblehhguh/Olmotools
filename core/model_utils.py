"""
Model utilities for loading and configuring OLMo-2-1124-7B model.
"""

import torch
import time
import os
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_network_error(func, max_retries=3, delay=10):
    """Retry function on network errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except (OSError, ConnectionError, TimeoutError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            wait_time = delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Network error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return None


def load_olmo_model_and_tokenizer(
    model_name: str = "allenai/OLMo-2-1124-7B",
    use_lora: bool = True,
    use_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    use_deepspeed: bool = False
) -> tuple:
    """Load OLMo model and tokenizer with optional quantization and LoRA."""
    
    logger.info(f"Loading model: {model_name}")
    
    # Check for distributed training environment
    is_distributed = (
        use_deepspeed or 
        os.environ.get("WORLD_SIZE", "1") != "1" or
        os.environ.get("LOCAL_RANK") is not None or
        os.environ.get("RANK") is not None
    )
    
    if is_distributed:
        logger.info("Distributed training detected - disabling device_map to avoid conflicts")
        device_map = None
    
    # Quantization config if using 4-bit or 8-bit
    bnb_config = None
    if use_4bit:
        if is_distributed:
            logger.warning("4-bit quantization may not work well with distributed training")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype
        )
        logger.info("Using 4-bit quantization")
    elif load_in_8bit:
        if is_distributed:
            logger.warning("8-bit quantization may not work well with distributed training")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        logger.info("Using 8-bit quantization")
    
    # Load tokenizer with retry logic
    logger.info("Loading tokenizer...")
    tokenizer = retry_on_network_error(
        lambda: AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        ),
        max_retries=3,
        delay=10
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with retry logic
    logger.info("Loading model...")
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    
    # Only add device_map if not using distributed training
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    
    model = retry_on_network_error(
        lambda: AutoModelForCausalLM.from_pretrained(**model_kwargs),
        max_retries=3,
        delay=10
    )
    
    # Prepare model for training if using quantization
    if use_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if requested
    if use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # Enable gradient checkpointing after LoRA
        model.gradient_checkpointing_enable()
    else:
        # Only enable gradient checkpointing for non-LoRA models
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """Create LoRA configuration for parameter-efficient fine-tuning."""
    
    if target_modules is None:
        # Default target modules for OLMo model
        # These may need adjustment based on the actual OLMo architecture
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    return lora_config


def create_training_arguments(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    deepspeed_config: Optional[str] = None,
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = True,
    report_to: str = "wandb",
    run_name: Optional[str] = None
) -> TrainingArguments:
    """Create training arguments for fine-tuning."""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        deepspeed=deepspeed_config,
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch",
        adam_beta2=0.999,
        max_grad_norm=1.0,
        weight_decay=0.01,
        report_to=report_to,
        run_name=run_name,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    return training_args


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    use_gradient_checkpointing: bool = True
) -> AutoModelForCausalLM:
    """Prepare model for training with optimizations."""
    
    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Incompatible with gradient checkpointing
    
    # Ensure model is in training mode
    model.train()
    
    return model


def save_model_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    checkpoint_name: str = "checkpoint"
) -> None:
    """Save model checkpoint."""
    
    checkpoint_path = f"{output_dir}/{checkpoint_name}"
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    
    # Save model
    model.save_pretrained(checkpoint_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_path)
    
    logger.info(f"Checkpoint saved successfully")
