# HuggingFace Hub Integration

This document explains how to use the HuggingFace Hub integration feature in the OLMo Training WebUI.

## Overview

The WebUI now supports automatically pushing trained models to HuggingFace Hub after training completes. This feature allows you to:

- Automatically upload your fine-tuned models to HuggingFace Hub
- Generate comprehensive model cards with training details
- Share your models publicly or keep them private
- Make models immediately available for inference and download

## Setup

### 1. Get a HuggingFace Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Write" permissions
3. Copy the token (it starts with `hf_`)

### 2. Configure in WebUI

1. Open the training deployment form
2. Check "Push model to HuggingFace Hub after training"
3. Fill in the required fields:
   - **Repository Name**: Format should be `username/model-name`
   - **HuggingFace Token**: Paste your token from step 1
   - **Private Repository**: Check if you want a private repo (requires HuggingFace Pro)
   - **Model Description**: Optional description for the model card

## Features

### Automatic Model Card Generation

The system automatically generates a comprehensive model card including:

- Training configuration (epochs, batch size, learning rate, etc.)
- Base model information
- Training method (LoRA or full fine-tuning)
- GPU configuration used
- Usage examples
- Training infrastructure details
- Proper citations

### Security

- Tokens are handled securely and not stored permanently
- Support for environment variables (`HUGGINGFACE_TOKEN`)
- Input validation for repository names

### Error Handling

- Comprehensive error messages for common issues
- Validation of repository name format
- Authentication error handling
- Network error retry logic

## Usage Examples

### Basic Usage

1. Configure your training parameters as usual
2. Enable HuggingFace integration
3. Set repository name: `myusername/olmo-finetuned-chat`
4. Provide your HuggingFace token
5. Deploy training

### Private Repository

1. Enable HuggingFace integration
2. Check "Private Repository"
3. Ensure you have HuggingFace Pro subscription
4. Configure other settings as normal

### Custom Description

Add a custom description to provide more context about your model:

```
This model is fine-tuned on conversational data for improved chat capabilities. 
It demonstrates better instruction following and maintains the reasoning 
capabilities of the base OLMo model.
```

## Environment Variables

You can set the HuggingFace token as an environment variable:

```bash
export HUGGINGFACE_TOKEN=hf_your_token_here
```

This way, you don't need to enter the token in the WebUI each time.

## Repository Name Format

Repository names must follow the format: `username/model-name`

Valid examples:
- `alice/olmo-chat-v1`
- `research-lab/olmo-instruction-tuned`
- `myorg/olmo-domain-specific`

Invalid examples:
- `just-model-name` (missing username)
- `user/model/extra` (too many slashes)
- `user/model name` (spaces not allowed)

## Model Card Example

The generated model card will look like this:

```markdown
---
license: apache-2.0
base_model: allenai/OLMo-2-1124-7B
tags:
- fine-tuned
- olmo
- modal
- lora
- text-generation
language:
- en
pipeline_tag: text-generation
---

# myuser/olmo-finetuned

This model is a fine-tuned version of [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B).

## Model Details

- **Base Model**: allenai/OLMo-2-1124-7B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: Modal + DeepSpeed
- **Language**: English

## Training Details

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 2e-05
- **Max Sequence Length**: 2048
- **Optimization**: LoRA with rank adaptation

...
```

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Check that your token is valid and has write permissions
   - Ensure the token starts with `hf_`

2. **Repository Already Exists**
   - The system will update existing repositories
   - Make sure you have write access to the repository

3. **Invalid Repository Name**
   - Use format: `username/model-name`
   - Only use letters, numbers, hyphens, underscores, and dots

4. **Network Errors**
   - The system will retry failed uploads automatically
   - Check your internet connection

### Getting Help

If you encounter issues:

1. Check the deployment logs in the WebUI
2. Verify your HuggingFace token permissions
3. Ensure the repository name format is correct
4. Check that `huggingface_hub` is installed in your environment

## Integration with Training Pipeline

The HuggingFace push happens automatically after training completes:

1. Model trains successfully
2. Model is saved locally
3. If HuggingFace push is enabled:
   - Model card is generated
   - Files are uploaded to HuggingFace Hub
   - Repository URL is provided in results

The local model is always saved regardless of HuggingFace push success/failure.

## Best Practices

1. **Use descriptive repository names** that indicate the model's purpose
2. **Provide meaningful descriptions** to help others understand your model
3. **Test with private repositories first** before making models public
4. **Keep your tokens secure** and don't share them
5. **Use environment variables** for tokens in production environments

## API Integration

The HuggingFace integration also works with the API endpoints:

```bash
curl -X POST http://your-webui/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your-api-key",
    "model_name": "allenai/OLMo-2-1124-7B",
    "push_to_hf": true,
    "hf_repo_name": "myuser/my-model",
    "hf_token": "hf_your_token",
    "hf_private": false,
    "hf_description": "My fine-tuned model"
  }'
