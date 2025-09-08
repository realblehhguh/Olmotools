"""
HuggingFace Hub utilities for pushing trained models.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json

try:
    from huggingface_hub import HfApi, login, create_repo, Repository
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def check_huggingface_availability():
    """Check if HuggingFace Hub is available."""
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace Hub is not available. Please install it with: "
            "pip install huggingface_hub"
        )


def authenticate_huggingface(token: Optional[str] = None) -> bool:
    """
    Authenticate with HuggingFace Hub.
    
    Args:
        token: HuggingFace token. If None, will try to use environment variable.
        
    Returns:
        bool: True if authentication successful, False otherwise.
    """
    check_huggingface_availability()
    
    if token is None:
        token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    
    if not token:
        logger.error("No HuggingFace token provided. Set HUGGINGFACE_TOKEN environment variable or provide token.")
        return False
    
    try:
        login(token=token, add_to_git_credential=True)
        logger.info("Successfully authenticated with HuggingFace Hub")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with HuggingFace Hub: {str(e)}")
        return False


def create_model_card(
    model_name: str,
    base_model: str,
    training_config: Dict[str, Any],
    description: Optional[str] = None,
    use_lora: bool = True
) -> str:
    """
    Create a model card for the trained model.
    
    Args:
        model_name: Name of the fine-tuned model
        base_model: Base model that was fine-tuned
        training_config: Training configuration dictionary
        description: Optional description of the model
        use_lora: Whether LoRA was used for training
        
    Returns:
        str: Model card content in markdown format
    """
    
    # Extract key training parameters
    epochs = training_config.get('num_epochs', 'N/A')
    batch_size = training_config.get('batch_size', 'N/A')
    learning_rate = training_config.get('learning_rate', 'N/A')
    max_length = training_config.get('max_length', 'N/A')
    
    model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- fine-tuned
- olmo
- modal
{"- lora" if use_lora else "- full-fine-tuning"}
- text-generation
language:
- en
pipeline_tag: text-generation
---

# {model_name}

{description or f"This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model})."}

## Model Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: {"LoRA (Low-Rank Adaptation)" if use_lora else "Full Fine-tuning"}
- **Training Framework**: Modal + DeepSpeed
- **Language**: English

## Training Details

### Training Configuration
- **Epochs**: {epochs}
- **Batch Size**: {batch_size}
- **Learning Rate**: {learning_rate}
- **Max Sequence Length**: {max_length}
- **Optimization**: {"LoRA with rank adaptation" if use_lora else "Full parameter fine-tuning"}

### Training Data
- **Dataset**: QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered
- **Data Type**: Instruction-following conversations

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Infrastructure

This model was trained using:
- **Platform**: Modal (modal.com)
- **GPU**: {training_config.get('gpu_type', 'A100')} x {training_config.get('gpu_count', 2)}
- **Framework**: PyTorch + DeepSpeed + Transformers

## Limitations and Bias

This model inherits the limitations and biases of the base model and training data. Please use responsibly and be aware of potential biases in generated content.

## Citation

If you use this model, please cite the original OLMo paper:

```bibtex
@article{{olmo2024,
  title={{OLMo: Accelerating the Science of Language Models}},
  author={{Groeneveld, Dirk and others}},
  journal={{arXiv preprint arXiv:2402.00838}},
  year={{2024}}
}}
```
"""
    
    return model_card


def push_model_to_huggingface(
    model_path: str,
    repo_name: str,
    token: Optional[str] = None,
    private: bool = False,
    description: Optional[str] = None,
    base_model: str = "allenai/OLMo-2-1124-7B",
    training_config: Optional[Dict[str, Any]] = None,
    use_lora: bool = True,
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Push a trained model to HuggingFace Hub.
    
    Args:
        model_path: Local path to the saved model
        repo_name: Name of the repository on HuggingFace Hub (username/model-name)
        token: HuggingFace token
        private: Whether to create a private repository
        description: Description for the model
        base_model: Base model that was fine-tuned
        training_config: Training configuration dictionary
        use_lora: Whether LoRA was used
        commit_message: Custom commit message
        
    Returns:
        Dict containing push result information
    """
    check_huggingface_availability()
    
    result = {
        'success': False,
        'repo_url': None,
        'error': None,
        'repo_name': repo_name
    }
    
    try:
        # Authenticate
        if not authenticate_huggingface(token):
            result['error'] = "Failed to authenticate with HuggingFace Hub"
            return result
        
        # Validate model path
        model_path = Path(model_path)
        if not model_path.exists():
            result['error'] = f"Model path does not exist: {model_path}"
            return result
        
        # Check if model files exist
        required_files = ['config.json']
        missing_files = []
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            result['error'] = f"Missing required model files: {missing_files}"
            return result
        
        # Initialize HF API
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.repo_info(repo_id=repo_name, token=token)
            logger.info(f"Repository {repo_name} already exists")
        except RepositoryNotFoundError:
            logger.info(f"Creating repository {repo_name}")
            api.create_repo(
                repo_id=repo_name,
                token=token,
                private=private,
                repo_type="model"
            )
        
        # Create model card
        if training_config:
            model_card_content = create_model_card(
                model_name=repo_name,
                base_model=base_model,
                training_config=training_config,
                description=description,
                use_lora=use_lora
            )
            
            # Save model card to model directory
            with open(model_path / "README.md", "w", encoding="utf-8") as f:
                f.write(model_card_content)
        
        # Upload all files in the model directory
        logger.info(f"Uploading model files to {repo_name}")
        
        # Get list of files to upload
        files_to_upload = []
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_path)
                files_to_upload.append(str(relative_path))
        
        logger.info(f"Files to upload: {files_to_upload}")
        
        # Upload files
        commit_msg = commit_message or f"Upload fine-tuned model based on {base_model}"
        
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=token,
            commit_message=commit_msg
        )
        
        # Construct repository URL
        repo_url = f"https://huggingface.co/{repo_name}"
        
        result.update({
            'success': True,
            'repo_url': repo_url,
            'files_uploaded': files_to_upload,
            'message': f"Successfully pushed model to {repo_url}"
        })
        
        logger.info(f"Successfully pushed model to HuggingFace Hub: {repo_url}")
        
    except Exception as e:
        error_msg = f"Failed to push model to HuggingFace Hub: {str(e)}"
        logger.error(error_msg)
        result['error'] = error_msg
    
    return result


def validate_repo_name(repo_name: str, username: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate HuggingFace repository name format.
    
    Args:
        repo_name: Repository name to validate
        username: Optional username to prepend if not in repo_name
        
    Returns:
        Dict with validation result
    """
    result = {
        'valid': False,
        'formatted_name': None,
        'error': None
    }
    
    if not repo_name:
        result['error'] = "Repository name cannot be empty"
        return result
    
    # Check if repo_name already includes username
    if '/' in repo_name:
        parts = repo_name.split('/')
        if len(parts) != 2:
            result['error'] = "Repository name format should be 'username/model-name'"
            return result
        username, model_name = parts
    else:
        model_name = repo_name
        if not username:
            result['error'] = "Username required when repository name doesn't include it"
            return result
    
    # Validate characters (HuggingFace allows alphanumeric, hyphens, underscores, dots)
    import re
    if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
        result['error'] = "Model name can only contain letters, numbers, dots, hyphens, and underscores"
        return result
    
    if not re.match(r'^[a-zA-Z0-9._-]+$', username):
        result['error'] = "Username can only contain letters, numbers, dots, hyphens, and underscores"
        return result
    
    formatted_name = f"{username}/{model_name}"
    result.update({
        'valid': True,
        'formatted_name': formatted_name
    })
    
    return result


def get_user_info(token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get user information from HuggingFace Hub.
    
    Args:
        token: HuggingFace token
        
    Returns:
        Dict with user information or None if failed
    """
    check_huggingface_availability()
    
    try:
        if not authenticate_huggingface(token):
            return None
        
        api = HfApi()
        user_info = api.whoami(token=token)
        return user_info
    except Exception as e:
        logger.error(f"Failed to get user info: {str(e)}")
        return None
