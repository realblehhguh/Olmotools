# OLMo Fine-tuning with Modal + Discord Bot Setup Guide

This guide will help you set up the complete system for fine-tuning OLMo-2-1124-7B on Modal with Discord notifications.

## System Overview

- **Dual GPU Training**: Uses 2x NVIDIA A100 GPUs for distributed training
- **Fire-and-Forget Deployment**: Submit jobs to Modal cloud and close your terminal
- **Discord Bot Monitoring**: Real-time training updates via Discord DMs
- **W&B Integration**: Full metrics tracking and visualization

## Prerequisites

1. **Modal Account**: Sign up at https://modal.com
2. **Discord Bot**: Create a bot at https://discord.com/developers/applications
3. **Weights & Biases Account**: Sign up at https://wandb.ai
4. **HuggingFace Account**: Sign up at https://huggingface.co

## Step 1: Install Dependencies

### For Training Deployment
```bash
pip install modal
```

### For Discord Bot (Local)
```bash
pip install -r bot_requirements.txt
```

## Step 2: Configure Modal

1. **Authenticate with Modal**:
```bash
modal token new
```

2. **Create Modal Secrets**:
```bash
# Create HuggingFace secret
modal secret create huggingface-token

# Create W&B secret
modal secret create wandb-secret
```

Enter your tokens when prompted.

## Step 3: Set Up Discord Bot

### 3.1 Create Discord Application
1. Go to https://discord.com/developers/applications
2. Click "New Application" and name it (e.g., "OLMo Training Bot")
3. Go to "Bot" section
4. Click "Reset Token" and copy the token
5. Enable these Intents:
   - MESSAGE CONTENT INTENT
   - DIRECT MESSAGES

### 3.2 Get Your Discord User ID
1. Enable Developer Mode in Discord (Settings → Advanced → Developer Mode)
2. Right-click your username and select "Copy User ID"

### 3.3 Invite Bot to Your Server (Optional)
1. Go to OAuth2 → URL Generator
2. Select scopes: `bot`
3. Select permissions: `Send Messages`, `Read Message History`
4. Copy the URL and open it to invite the bot

## Step 4: Configure Environment

1. **Copy the template**:
```bash
cp .env.template .env
```

2. **Edit `.env` file** with your credentials:
```env
# Discord Bot Configuration
DISCORD_BOT_TOKEN=your-actual-bot-token
DISCORD_USER_ID=your-actual-discord-id

# Weights & Biases Configuration
WANDB_API_KEY=your-wandb-api-key
WANDB_ENTITY=your-wandb-username
WANDB_PROJECT=olmo-finetune-modal

# HuggingFace Configuration
HUGGINGFACE_TOKEN=your-hf-token
```

## Step 5: Deploy Training

### Quick Test Run (100 samples, 1 epoch)
```bash
python deploy_training.py --quick-test
```

### Full Training Run
```bash
python deploy_training.py \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --run-name "my_experiment"
```

### Custom Configuration
```bash
python deploy_training.py \
    --model-name "allenai/OLMo-2-1124-7B" \
    --num-epochs 5 \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --max-length 2048 \
    --train-sample-size 10000 \
    --use-lora \
    --run-name "custom_run"
```

## Step 6: Start Discord Bot

In a separate terminal, run:
```bash
python discord_bot.py
```

The bot will:
- Send you a DM when it starts
- Monitor all deployed training runs
- Send updates on training progress
- Notify you when training completes

### Discord Bot Commands

Send these commands via DM to the bot:
- `/status` - Get status of all active runs
- `/metrics [run_name]` - Get detailed metrics for a run
- `/list` - List all tracked runs
- `/stop [run_name]` - Stop monitoring a specific run

## Step 7: Monitor Training

### Via Discord
You'll receive DMs with:
- 🚀 Training start notification
- 📊 Progress updates each epoch
- ✅ Completion notification
- ❌ Error alerts

### Via W&B Dashboard
1. Go to https://wandb.ai/your-entity/olmo-finetune-modal
2. Click on your run to see:
   - Real-time loss curves
   - GPU utilization
   - Learning rate schedules
   - System metrics

### Via Modal Dashboard
1. Go to https://modal.com/apps
2. Find your app "olmo-finetune-deepspeed"
3. View logs, GPU usage, and costs

## Workflow Example

1. **Deploy training**:
```bash
python deploy_training.py --quick-test --run-name "test_run"
```
Output:
```
🚀 Deploying training job: test_run
✅ Training job submitted successfully!
📋 Job ID: fn-abc123...
🏷️ Run Name: test_run
```

2. **Start Discord bot** (in another terminal):
```bash
python discord_bot.py
```

3. **Receive Discord DM**:
```
🚀 Training Started
Run: test_run
Model: OLMo-2-1124-7B
Epochs: 1
Batch Size: 4
GPUs: 2x A100
```

4. **Get updates** every epoch:
```
📊 Training Update: test_run
🏃 Status: RUNNING
Progress: [████████░░░░░░░░░░░░] 40.0%
Epoch: 0.40
Loss: 2.3456
LR: 1.50e-05
Runtime: 0h 12m
```

5. **Completion notification**:
```
✅ Training Update: test_run
Status: FINISHED
Progress: [████████████████████] 100.0%
Final Loss: 1.2345
Runtime: 0h 45m
```

## Troubleshooting

### Modal Issues
- **Authentication Error**: Run `modal token new` again
- **Secret Not Found**: Create secrets with `modal secret create`
- **GPU Not Available**: Check your Modal account limits

### Discord Bot Issues
- **Bot Not Sending DMs**: Ensure DM permissions are enabled
- **Commands Not Working**: Check MESSAGE CONTENT INTENT is enabled
- **Connection Error**: Verify DISCORD_BOT_TOKEN is correct

### Training Issues
- **OOM Errors**: Reduce batch size or max_length
- **Slow Training**: Normal for large models, 2x A100 helps
- **Network Timeouts**: Already handled with retry logic

## Cost Estimation

Modal pricing (approximate):
- A100 GPU: ~$3.09/hour per GPU
- With 2x A100: ~$6.18/hour
- Storage: ~$0.02/GB/month
- Quick test (~1 hour): ~$6
- Full training (~8 hours): ~$50

## Advanced Features

### Resume Training
If training fails, you can resume from checkpoint:
```python
# In modal_app.py, add resume_from_checkpoint parameter
```

### Multi-Run Management
Deploy multiple experiments:
```bash
for lr in 1e-5 2e-5 3e-5; do
    python deploy_training.py --learning-rate $lr --run-name "lr_${lr}"
done
```

### Custom Datasets
Modify `data_utils.py` to use your own dataset:
```python
dataset = load_dataset("your-dataset")
```

## Support

- Modal Documentation: https://modal.com/docs
- Discord.py Documentation: https://discordpy.readthedocs.io
- W&B Documentation: https://docs.wandb.ai
- OLMo Model Card: https://huggingface.co/allenai/OLMo-2-1124-7B

## Next Steps

1. ✅ Test with quick run first
2. 📊 Monitor via Discord and W&B
3. 🎯 Adjust hyperparameters based on results
4. 🚀 Deploy full training runs
5. 💾 Download fine-tuned models from Modal volumes

Happy Training! 🎉
