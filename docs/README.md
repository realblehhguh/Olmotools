# OLMo-2-1124-7B Fine-tuning with DeepSpeed on Modal Labs

This project implements fine-tuning of the Allen AI OLMo-2-1124-7B model using the uncensored WizardLM dataset, leveraging DeepSpeed for efficient distributed training on Modal Labs infrastructure.

## Overview

This implementation follows Eric Hartford's approach to creating uncensored models by:
1. Using the QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered dataset (refusals already removed)
2. Fine-tuning with DeepSpeed for memory-efficient training
3. Running on Modal Labs for scalable GPU compute

## Features

- **Model**: allenai/OLMo-2-1124-7B (7 billion parameter language model)
- **Dataset**: QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered
- **Training Method**: DeepSpeed with ZeRO Stage 2 optimization
- **Parameter-Efficient Fine-tuning**: LoRA support for reduced memory usage
- **Infrastructure**: Modal Labs with A100 GPUs
- **Monitoring**: Weights & Biases integration for training metrics
- **Checkpointing**: Automatic checkpoint saving with resume capability

## Project Structure

```
modal_olmo_finetune/
├── modal_app.py           # Main Modal Labs application
├── train.py               # Training script with DeepSpeed integration
├── data_utils.py          # Dataset loading and preprocessing
├── model_utils.py         # Model configuration and utilities
├── configs/
│   └── deepspeed_config.json  # DeepSpeed configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Prerequisites

1. **Modal Labs Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install with `pip install modal`
3. **Modal Authentication**: Run `modal token new`
4. **HuggingFace Token** (optional): For accessing gated models
5. **Weights & Biases API Key** (optional): For training metrics

## Setup

### 1. Clone or Create Project

```bash
# Navigate to your project directory
cd modal_olmo_finetune
```

### 2. Set up Modal Secrets

```bash
# Set HuggingFace token (if needed)
modal secret create huggingface-token HUGGINGFACE_TOKEN=<your-token>

# Set W&B API key (optional)
modal secret create wandb-secret WANDB_API_KEY=<your-api-key>
```

### 3. Deploy to Modal

```bash
# Deploy the app to Modal
modal deploy modal_app.py
```

## Usage

### Training

Start a fine-tuning job with default parameters:

```bash
modal run modal_app.py --action train
```

Customize training parameters:

```bash
modal run modal_app.py \
  --action train \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --max_length 2048 \
  --use_lora \
  --run_name "uncensored_olmo_v1"
```

Train with a sample of the dataset (for testing):

```bash
modal run modal_app.py \
  --action train \
  --train_sample_size 1000 \
  --num_epochs 1 \
  --run_name "test_run"
```

### List Available Checkpoints

```bash
modal run modal_app.py --action list
```

### Test Model Inference

Test a trained model with a custom prompt:

```bash
modal run modal_app.py \
  --action test \
  --checkpoint_path "/vol/outputs/run_20240101_120000/final_model" \
  --prompt "### Instruction:\nWrite a Python function to calculate fibonacci numbers.\n\n### Response:\n"
```

### Clean Up Old Checkpoints

```bash
modal run modal_app.py --action cleanup
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | allenai/OLMo-2-1124-7B | Base model to fine-tune |
| `num_epochs` | 3 | Number of training epochs |
| `batch_size` | 4 | Batch size per GPU |
| `learning_rate` | 2e-5 | Learning rate |
| `max_length` | 2048 | Maximum sequence length |
| `use_lora` | True | Use LoRA for parameter-efficient training |
| `use_4bit` | False | Use 4-bit quantization |
| `train_sample_size` | None | Number of training samples (None = full dataset) |
| `run_name` | None | Custom name for the training run |

## DeepSpeed Configuration

The project uses DeepSpeed ZeRO Stage 2 optimization with:
- Mixed precision training (FP16)
- CPU offloading for optimizer and parameters
- Gradient clipping (1.0)
- AdamW optimizer
- WarmupDecayLR scheduler

Modify `configs/deepspeed_config.json` to adjust these settings.

## Memory Requirements

- **Full Fine-tuning**: Requires A100 80GB or multiple GPUs
- **LoRA Fine-tuning**: Can run on A100 40GB
- **4-bit + LoRA**: Can run on smaller GPUs (V100 32GB)

## Expected Training Time

On Modal Labs with A100 GPU:
- Full dataset (~70k examples): 20-30 hours for 3 epochs
- With LoRA: 10-15 hours for 3 epochs
- Sample of 1000 examples: 30-60 minutes

## Monitoring Training

If W&B is configured, monitor training at:
```
https://wandb.ai/<your-username>/olmo-finetune-modal
```

Metrics tracked:
- Training/validation loss
- Learning rate schedule
- GPU memory usage
- Training speed (samples/sec)

## Output Structure

```
/vol/outputs/
├── run_20240101_120000/
│   ├── final_model/           # Final trained model
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files...
│   ├── checkpoint-500/        # Intermediate checkpoints
│   ├── checkpoint-1000/
│   └── training_metrics.json  # Final training metrics
```

## Troubleshooting

### Out of Memory Errors

1. Enable gradient checkpointing (already enabled by default)
2. Reduce batch size
3. Enable 4-bit quantization with `--use_4bit`
4. Reduce max_length parameter

### Slow Training

1. Ensure you're using A100 GPUs
2. Check DeepSpeed configuration
3. Consider using multiple GPUs (modify gpu_config in modal_app.py)

### Dataset Loading Issues

1. Check HuggingFace token is set correctly
2. Verify internet connectivity in Modal environment
3. Try with smaller sample size first

## Advanced Usage

### Multi-GPU Training

Modify `modal_app.py`:

```python
# Change GPU configuration
gpu_config = modal.gpu.A100(count=4)  # Use 4 A100 GPUs
```

### Custom Dataset

Modify `data_utils.py` to load your custom dataset:

```python
dataset_name = "your-username/your-dataset"
```

### Different Base Model

```bash
modal run modal_app.py \
  --action train \
  --model_name "meta-llama/Llama-2-7b-hf"
```

## Citation

This implementation is based on Eric Hartford's uncensored models approach:
- Blog: https://erichartford.com/uncensored-models
- Dataset: https://huggingface.co/datasets/QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered

## License

This project is provided as-is for educational and research purposes. Please ensure you comply with:
- OLMo model license from Allen AI
- WizardLM dataset license
- Modal Labs terms of service

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Modal Labs documentation
3. Check HuggingFace Transformers documentation
4. Review DeepSpeed documentation

## Next Steps

After training:
1. Test the model thoroughly with various prompts
2. Consider quantization for deployment (GGUF, GPTQ)
3. Deploy for inference using Modal, vLLM, or other serving frameworks
4. Fine-tune further with domain-specific data if needed
