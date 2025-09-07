"""
Data utilities for loading and preprocessing the WizardLM unfiltered dataset.
"""

import json
import torch
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WizardLMDataProcessor:
    """Process WizardLM dataset for fine-tuning."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        dataset_name: str = "QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        
    def load_dataset(self, split: str = "train", sample_size: Optional[int] = None) -> Dataset:
        """Load the WizardLM unfiltered dataset from HuggingFace."""
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        # Load dataset from HuggingFace
        dataset = load_dataset(self.dataset_name, split=split)
        
        # Sample if requested (useful for testing)
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))
            logger.info(f"Sampled {sample_size} examples from dataset")
        
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        return dataset
    
    def format_instruction(self, example: Dict[str, Any]) -> str:
        """Format a single example into instruction format."""
        # Handle different possible formats in the dataset
        if "instruction" in example and "output" in example:
            # Standard alpaca format
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        elif "text" in example:
            # Direct text format
            prompt = example["text"]
        else:
            # Fallback for other formats
            prompt = str(example)
        
        return prompt
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples for training."""
        # Format all examples
        formatted_texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            example = {key: examples[key][i] for key in examples.keys()}
            formatted_texts.append(self.format_instruction(example))
        
        # Tokenize with padding and truncation
        model_inputs = self.tokenizer(
            formatted_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # Copy input_ids to labels for language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def prepare_dataset(
        self,
        split: str = "train",
        sample_size: Optional[int] = None,
        num_proc: int = 4
    ) -> Dataset:
        """Load and prepare dataset for training."""
        # Load raw dataset
        dataset = self.load_dataset(split=split, sample_size=sample_size)
        
        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        logger.info(f"Dataset prepared with {len(tokenized_dataset)} tokenized examples")
        return tokenized_dataset


class DataCollatorForSupervisedDataset:
    """Data collator for supervised fine-tuning."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of instances."""
        input_ids = []
        labels = []
        attention_mask = []
        
        for instance in instances:
            input_ids.append(torch.tensor(instance["input_ids"]))
            labels.append(torch.tensor(instance["labels"]))
            attention_mask.append(torch.tensor(instance["attention_mask"]))
        
        # Stack tensors
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        
        # Set padding tokens in labels to -100 (ignored in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


def create_data_module(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    train_sample_size: Optional[int] = None,
    val_sample_size: Optional[int] = None
) -> tuple:
    """Create train and validation datasets with data processor."""
    
    processor = WizardLMDataProcessor(
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Prepare training dataset
    train_dataset = processor.prepare_dataset(
        split="train",
        sample_size=train_sample_size
    )
    
    # For validation, we'll use a small portion of the training set
    # since the dataset doesn't have a separate validation split
    val_dataset = processor.prepare_dataset(
        split="train",
        sample_size=val_sample_size or 500  # Small validation set
    )
    
    # Create data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return train_dataset, val_dataset, data_collator
