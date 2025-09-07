# OLMo Training Deployer - Web UI Guide

A beautiful web interface for deploying OLMo training jobs to Modal through Render, eliminating the need for terminal access.

## 🚀 Overview

The Training Deployer provides a user-friendly web interface to:
- Configure training parameters through an intuitive UI
- Deploy training jobs to Modal with one click
- Monitor deployment status and history
- View training progress in real-time
- Access W&B metrics and Modal logs

## 📋 Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
   - Get your Modal token from the dashboard
   - Note your `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`

2. **Render Account**: Sign up at [render.com](https://render.com)
   - You'll deploy the web UI here

3. **W&B Account** (Optional): For training metrics
   - Get your API key from [wandb.ai/settings](https://wandb.ai/settings)

## 🔧 Setup Instructions

### Step 1: Prepare Your Repository

Ensure your repository contains all the necessary files:
```
modal_olmo_finetune/
├── training_deployer.py       # Flask web application
├── deployer_requirements.txt  # Python dependencies
├── render-training.yaml       # Render configuration
├── deploy_modal.py           # Modal deployment script
├── modal_app.py              # Modal application
├── templates/                # HTML templates
│   ├── base.html
│   ├── index.html
│   └── status.html
├── static/                   # CSS and JavaScript
│   ├── css/style.css
│   └── js/app.js
└── [other training files]    # Your existing Modal training code
```

### Step 2: Deploy to Render

1. **Create a New Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub/GitLab repository
   - Select the repository containing the training deployer

2. **Configure the Service**:
   - **Name**: `olmo-training-deployer` (or your preference)
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `modal_olmo_finetune` (if not in root)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r deployer_requirements.txt`
   - **Start Command**: `python training_deployer.py`

3. **Set Environment Variables** in Render Dashboard:

   **Required Variables**:
   ```
   MODAL_TOKEN_ID=your_modal_token_id
   MODAL_TOKEN_SECRET=your_modal_token_secret
   DEPLOY_API_KEY=generate_a_secure_key_here
   FLASK_SECRET_KEY=generate_another_secure_key
   ```

   **Optional Variables**:
   ```
   WANDB_API_KEY=your_wandb_api_key
   WANDB_ENTITY=your_wandb_username
   WANDB_PROJECT=olmo-finetune-modal
   ```

4. **Deploy**:
   - Click "Create Web Service"
   - Render will build and deploy your application
   - Wait for the build to complete (usually 2-5 minutes)

### Step 3: Access Your Deployment UI

Once deployed, you'll get a URL like:
```
https://olmo-training-deployer.onrender.com
```

Visit this URL to access your training deployment interface!

## 🎯 Using the Web UI

### Quick Start

1. **Open the Deployment UI** in your browser
2. **Enter your API Key** (the `DEPLOY_API_KEY` you set in Render)
3. **Choose a Preset**:
   - 🧪 **Quick Test**: 100 samples, 1 epoch (for testing)
   - 🚀 **Full Training**: Full dataset, 3 epochs
   - ⚙️ **Custom**: Configure all parameters manually

4. **Click "Deploy Training"**
5. **Monitor Progress** on the status page

### Configuration Options

#### Model Settings
- **Model Name**: Choose from OLMo variants
- **Run Name**: Custom name for your training run
- **Use LoRA**: Enable parameter-efficient fine-tuning (recommended)
- **4-bit Quantization**: Reduce memory usage

#### Training Parameters
- **Epochs**: Number of training iterations (1-10)
- **Batch Size**: Samples per GPU (2, 4, or 8)
- **Learning Rate**: Training speed (default: 2e-5)
- **Max Sequence Length**: Token limit (512-4096)
- **Sample Size**: Number of training samples (empty = full dataset)

### Monitoring Deployments

The status page shows:
- **Active Deployments**: Currently running jobs
- **Deployment History**: Past training runs
- **Status Indicators**:
  - 🔵 Running (auto-refreshes)
  - ✅ Completed
  - ❌ Failed (with error details)

Click on any deployment to view:
- Configuration details
- Timeline (created, started, ended)
- Error messages (if failed)
- Output directory (if completed)
- Links to Modal and W&B dashboards

## 🔐 Security

### API Key Protection
- The `DEPLOY_API_KEY` prevents unauthorized deployments
- Share this key only with authorized users
- Regenerate if compromised

### Best Practices
1. Use strong, unique API keys
2. Enable HTTPS in production (Render provides this)
3. Restrict CORS origins in production
4. Regularly update dependencies

## 📊 Monitoring Training

### Modal Dashboard
- View logs: [modal.com/apps](https://modal.com/apps)
- Monitor GPU usage
- Check function execution status

### W&B Dashboard
- Track metrics: [wandb.ai](https://wandb.ai)
- View loss curves
- Compare runs
- Download models

## 🛠️ Troubleshooting

### Deployment Issues

**Problem**: "Invalid API key"
- **Solution**: Check that you're using the correct `DEPLOY_API_KEY`

**Problem**: "Modal authentication failed"
- **Solution**: Verify `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in Render

**Problem**: Training doesn't start
- **Solution**: 
  1. Check Modal dashboard for errors
  2. Verify Modal credits/GPU availability
  3. Check deployment logs in Render

### Web UI Issues

**Problem**: Page not loading
- **Solution**: Check Render service status and logs

**Problem**: Form data lost on refresh
- **Solution**: The UI auto-saves form data to localStorage

**Problem**: Status not updating
- **Solution**: The page auto-refreshes for running jobs every 10 seconds

## 📝 API Reference

### Endpoints

#### `POST /deploy`
Deploy a training job programmatically:

```bash
curl -X POST https://your-app.onrender.com/deploy \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_deploy_api_key" \
  -d '{
    "model_name": "allenai/OLMo-2-1124-7B",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "use_lora": true,
    "run_name": "api_deployment"
  }'
```

#### `GET /api/status/{deployment_id}`
Get deployment status as JSON:

```bash
curl https://your-app.onrender.com/api/status/abc123def456
```

#### `GET /api/deployments`
List all deployments:

```bash
curl https://your-app.onrender.com/api/deployments
```

## 🔄 Updating the Deployer

To update your deployment:

1. Push changes to your repository
2. Render will automatically redeploy if auto-deploy is enabled
3. Or manually trigger a deploy from Render dashboard

## 💰 Cost Considerations

- **Render**: $7/month for starter plan (web service)
- **Modal**: Pay-per-use for GPU compute
- **Storage**: 1GB included with Render for deployment history

## 🤝 Support

- **Modal Issues**: Check [Modal documentation](https://modal.com/docs)
- **Render Issues**: Check [Render documentation](https://render.com/docs)
- **Training Issues**: Review the main README.md

## 📚 Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Render Documentation](https://render.com/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [W&B Documentation](https://docs.wandb.ai/)

---

**Note**: This deployer runs on Render and triggers training jobs on Modal. The actual training happens on Modal's infrastructure with GPU resources, while Render hosts the lightweight web interface.
