# Modal OLMo Training - Fix Guide

## Issues Identified and Fixed

### 1. W&B Integration Issues
- **Problem**: W&B runs not appearing in dashboard
- **Causes**: 
  - Secret name mismatch between Modal app and training script
  - Modal secrets not properly configured as environment variables

### 2. Modal Deployment Issues
- **Problem**: Training not running on Modal
- **Causes**:
  - Incorrect use of `spawn()` with `async with app.run()`
  - Deployment script using incorrect async patterns

## Solutions Applied

### 1. Fixed Modal App (`modal_app.py`)
- Removed unsupported `required=False` parameter from secrets
- Ensured proper environment variable mapping

### 2. Created New Deployment Script (`deploy_modal.py`)
- Simplified deployment logic
- Proper use of `with app.run()` and `.remote()`
- Better error handling and logging

### 3. Created Test Script (`test_modal_deployment.py`)
- Verifies secret configuration
- Tests W&B initialization
- Checks environment variables

## How to Deploy Training Now

### Step 1: Verify Secrets
First, ensure your Modal secrets are properly configured:

```bash
modal secret list
```

You should see:
- `wandb-secret` - Contains your W&B API key
- `huggingface-token` - Contains your HuggingFace token

If missing, create them:

```bash
# Create W&B secret
modal secret create wandb-secret WANDB_API_KEY=your_wandb_api_key_here

# Create HuggingFace secret (optional, for gated models)
modal secret create huggingface-token HUGGINGFACE_TOKEN=your_hf_token_here
```

### Step 2: Test Secret Configuration
Run the test script to verify everything is set up correctly:

```bash
cd modal_olmo_finetune
python test_modal_deployment.py
```

You should see:
- ✅ WANDB_API_KEY is set
- ✅ HUGGINGFACE_TOKEN is set (if configured)
- ✅ W&B initialization successful

### Step 3: Deploy Training

#### Option 1: Use the New Deployment Script (Recommended)
```bash
# Quick test (100 samples, 1 epoch)
python deploy_modal.py --quick-test

# Full training
python deploy_modal.py --full-training

# Custom configuration
python deploy_modal.py \
  --model-name "allenai/OLMo-2-1124-7B" \
  --num-epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --use-lora \
  --run-name "my_training_run"
```

#### Option 2: Use Modal CLI Directly
```bash
# Deploy and run
modal run modal_app.py --action train --num_epochs 3 --batch_size 4
```

### Step 4: Monitor Training

1. **Modal Dashboard**: https://modal.com/apps
   - Check function logs
   - Monitor GPU usage
   - View runtime status

2. **W&B Dashboard**: https://wandb.ai/your-entity/olmo-finetune-modal
   - View training metrics
   - Monitor loss curves
   - Check system metrics

3. **Check Deployment Info**:
   ```bash
   cat deployments/full_training.json
   ```

## Troubleshooting

### Issue: Training doesn't appear in W&B
1. Check if W&B secret is configured:
   ```bash
   modal secret list
   ```

2. Verify the secret contains `WANDB_API_KEY`:
   ```bash
   python test_modal_deployment.py
   ```

3. Check Modal function logs for W&B initialization messages

### Issue: Training fails to start
1. Check Modal dashboard for error logs
2. Verify GPU availability in your Modal account
3. Ensure all required files are present:
   - `train.py`
   - `model_utils.py`
   - `data_utils.py`
   - `configs/deepspeed_config.json`

### Issue: Out of memory errors
1. Reduce batch size:
   ```bash
   python deploy_modal.py --batch-size 2
   ```

2. Enable gradient checkpointing (already enabled by default)

3. Use LoRA (enabled by default):
   ```bash
   python deploy_modal.py --use-lora
   ```

## Quick Commands Reference

```bash
# Test secrets
python test_modal_deployment.py

# Quick test run
python deploy_modal.py --quick-test

# Full training
python deploy_modal.py --full-training

# List checkpoints
modal run modal_app.py --action list

# Test inference
modal run modal_app.py --action test --checkpoint_path /vol/outputs/run_xxx/final_model
```

## Files Modified

1. **modal_app.py**: Fixed secret handling
2. **deploy_training.py**: Fixed async/await issues (deprecated)
3. **deploy_modal.py**: New simplified deployment script
4. **test_modal_deployment.py**: New test script for verification

## Next Steps

1. Run `python test_modal_deployment.py` to verify setup
2. Run `python deploy_modal.py --quick-test` for a test run
3. If successful, run `python deploy_modal.py --full-training` for full training
4. Monitor progress on W&B dashboard

## Important Notes

- The training will run on 2x A100 GPUs by default
- Full training may take several hours
- Checkpoints are saved to Modal volumes
- W&B logging requires the `wandb-secret` to be configured
