# GPU Configuration Guide

This guide explains how to use the enhanced GPU configuration features in the Modal OLMo fine-tuning application.

## Overview

The application now supports all GPU types available on Modal Labs, with configurable GPU counts and intelligent validation. You can specify both the GPU type and the number of GPUs for your training jobs.

## Available GPU Types

Based on Modal's latest documentation, the following GPU types are supported:

### Basic GPUs
- **T4**: Basic GPU, good for inference and light training
- **L4**: Mid-range GPU with good price/performance, 48GB GPU RAM

### Training GPUs
- **A10**: Good for training, supports up to 4 GPUs, up to 96GB GPU RAM
- **A100**: High-end training GPU, supports up to 8 GPUs
- **A100-40GB**: A100 with 40GB memory
- **A100-80GB**: A100 with 80GB memory
- **L40S**: High-end GPU for training, supports up to 8 GPUs

### Latest High-End GPUs
- **H100**: Latest high-end GPU for training, supports up to 8 GPUs
- **H100i**: H100 inference optimized, supports up to 8 GPUs
- **H200**: Latest flagship GPU, supports up to 8 GPUs
- **B200**: Most powerful GPU available (Blackwell architecture), supports up to 8 GPUs

## GPU Count Limits

- **A10**: Maximum 4 GPUs per container (up to 96GB GPU RAM)
- **All others**: Maximum 8 GPUs per container (up to 1,536GB CPU RAM)

## Usage Examples

### Basic Training Commands

```bash
# Default configuration (A100:2)
modal run modal_app.py --action train

# Single GPU training with T4
modal run modal_app.py --action train --gpu_type T4 --gpu_count 1

# High-performance training with 4 H100 GPUs
modal run modal_app.py --action train --gpu_type H100 --gpu_count 4

# Maximum performance with 8 B200 GPUs
modal run modal_app.py --action train --gpu_type B200 --gpu_count 8

# Budget-friendly training with L4
modal run modal_app.py --action train --gpu_type L4 --gpu_count 1
```

### Complete Training Example

```bash
modal run modal_app.py \
  --action train \
  --gpu_type H100 \
  --gpu_count 4 \
  --model_name allenai/OLMo-2-1124-7B \
  --num_epochs 5 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --run_name my_experiment
```

## GPU Recommendations

Use the recommendations action to see suggested configurations:

```bash
modal run modal_app.py --action recommendations
```

This will show recommendations for:
- **Training**: Different tiers from budget to maximum performance
- **Inference**: Optimized for inference workloads
- **Development**: For development and testing

### Training Recommendations

| Tier | GPU Config | Description |
|------|------------|-------------|
| Budget | L4:1 | Cost-effective for small models |
| Balanced | A100:2 | Good balance of performance and cost |
| Performance | H100:4 | High performance for large models |
| Maximum | B200:8 | Maximum performance for largest models |

### Inference Recommendations

| Tier | GPU Config | Description |
|------|------------|-------------|
| Budget | T4:1 | Basic inference workloads |
| Balanced | L4:1 | Good performance for most inference tasks |
| Performance | A100:1 | High-throughput inference |
| Maximum | H100i:1 | Optimized for inference workloads |

## Configuration Validation

The application automatically validates your GPU configuration:

- **GPU Type Validation**: Ensures you're using a supported GPU type
- **GPU Count Validation**: Enforces the maximum GPU limits for each type
- **Case Handling**: Accepts various case formats (e.g., "h100", "H100", "H100i")
- **Alias Support**: Supports common aliases (e.g., "A100-40" → "A100-40GB")

## DeepSpeed Configuration

The application automatically selects the appropriate DeepSpeed configuration:

- **Single GPU** (`gpu_count = 1`): Uses `deepspeed_config_single_gpu.json`
- **Multi-GPU** (`gpu_count > 1`): Uses `deepspeed_config.json`

## Error Handling

Common validation errors and solutions:

### Invalid GPU Type
```
ValueError: Invalid GPU type: INVALID. Valid types: A10, A100, A100-40GB, A100-80GB, B200, H100, H100i, H200, L40S, L4, T4
```
**Solution**: Use one of the supported GPU types listed above.

### GPU Count Exceeded
```
ValueError: A10 GPUs support maximum 4 GPUs per container (up to 96GB GPU RAM)
```
**Solution**: Reduce the GPU count to the maximum allowed for that GPU type.

### Minimum GPU Count
```
ValueError: At least 1 GPU required
```
**Solution**: Specify a GPU count of 1 or higher.

## Performance Considerations

### Choosing the Right GPU

1. **For Development/Testing**: Start with T4 or L4 (single GPU)
2. **For Small Models**: L4 or A100 (1-2 GPUs)
3. **For Large Models**: H100 or H200 (4-8 GPUs)
4. **For Maximum Performance**: B200 (8 GPUs)

### Memory Considerations

- **T4**: ~16GB GPU memory
- **L4**: ~48GB GPU memory
- **A10**: ~24GB GPU memory (up to 96GB total with 4 GPUs)
- **A100**: ~40GB or 80GB GPU memory
- **H100/H200/B200**: ~80GB+ GPU memory

### Cost Optimization

- Use fewer, more powerful GPUs rather than many smaller ones
- Consider L4 for cost-effective training of smaller models
- Use T4 for inference and development work
- Reserve H100/H200/B200 for production training of large models

## Monitoring and Logging

The application logs detailed GPU information during training:

```
GPU Configuration: H100:4
GPU Available: NVIDIA H100 80GB HBM3
GPU Memory: 81.92 GB
Number of GPUs: 4
```

## Troubleshooting

### Common Issues

1. **GPU Not Available**: Check Modal's current GPU availability
2. **Out of Memory**: Reduce batch size or use fewer/smaller GPUs
3. **Slow Training**: Consider using more powerful GPUs or increasing GPU count

### Getting Help

- Check Modal's pricing page for current GPU availability
- Review Modal's documentation for the latest GPU specifications
- Use the recommendations action to see suggested configurations

## Advanced Usage

### Custom GPU Configurations

You can create custom training functions with specific GPU configurations:

```python
from modal_app import create_training_function

# Create a function with specific GPU config
training_func = create_training_function("H100", 8)
result = training_func.remote(model_name="my-model", num_epochs=10)
```

### Programmatic Access

```python
from modal_app import get_gpu_config, get_gpu_recommendations

# Validate configuration
gpu_config = get_gpu_config("B200", 8)  # Returns "B200:8"

# Get recommendations
recs = get_gpu_recommendations("training")
print(recs["performance"])  # {'type': 'H100', 'count': 4, 'description': '...'}
```

## Future Enhancements

Planned improvements include:

- Automatic GPU selection based on model size
- Cost estimation for different GPU configurations
- Performance benchmarking across GPU types
- Dynamic scaling based on training progress

---

For the latest GPU availability and pricing, check [Modal's documentation](https://modal.com/docs/guide/gpu).
