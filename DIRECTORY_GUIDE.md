# Modal OLMo Finetune - Directory Navigation Guide

This guide explains the organized directory structure for easy navigation and understanding of the project components.

## 📁 Directory Structure

```
modal_olmo_finetune/
├── 📋 DIRECTORY_GUIDE.md          # This navigation guide
├── 🔧 .env.template               # Environment variables template
├── 📦 requirements.txt            # Main Python dependencies
├── 🚫 .gitignore                  # Git ignore rules
│
├── 🧠 core/                       # Core training components
│   ├── train.py                   # Main training script
│   ├── data_utils.py              # Data processing utilities
│   ├── model_utils.py             # Model handling utilities
│   └── modal_app.py               # Modal application definition
│
├── 🚀 deployment/                 # Deployment scripts
│   ├── deploy_modal.py            # Simplified Modal deployment
│   └── deploy_training.py         # Original deployment script
│
├── 🤖 discord_bot/               # Discord bot components
│   ├── discord_bot.py             # Basic Discord bot
│   ├── discord_bot_with_health.py # Discord bot with health endpoint
│   └── bot_requirements.txt       # Discord bot dependencies
│
├── 🌐 web_ui/                     # Web interface for deployments
│   ├── training_deployer.py       # Flask web application
│   ├── deployer_requirements.txt  # Web UI dependencies
│   ├── templates/                 # HTML templates
│   │   ├── base.html              # Base template
│   │   ├── index.html             # Main deployment form
│   │   └── status.html            # Status monitoring page
│   └── static/                    # Static assets
│       ├── css/
│       │   └── style.css          # Custom styles
│       └── js/
│           └── app.js             # JavaScript functionality
│
├── 📚 docs/                       # Documentation
│   ├── README.md                  # Main project documentation
│   ├── SETUP_GUIDE.md             # Setup instructions
│   ├── RENDER_DEPLOYMENT.md       # Render deployment guide
│   ├── TRAINING_DEPLOYER_GUIDE.md # Web UI deployment guide
│   ├── FIX_GUIDE.md               # Troubleshooting guide
│   └── UPTIMEROBOT_SETUP.md       # Uptime monitoring setup
│
├── 🔧 scripts/                    # Utility scripts
│   ├── quickstart.sh              # Quick setup script (Linux/Mac)
│   ├── setup_render.sh            # Render setup script (Linux/Mac)
│   └── setup_render.ps1           # Render setup script (Windows)
│
├── 🧪 tests/                      # Test files
│   ├── test_modal_deployment.py   # Modal deployment tests
│   ├── test_simple.py             # Basic connectivity tests
│   ├── test_fixes.py              # Fix verification tests
│   └── test_local.py              # Local testing utilities
│
├── ⚙️ configs/                    # Configuration files
│   ├── deepspeed_config.json      # DeepSpeed config (multi-GPU)
│   └── deepspeed_config_single_gpu.json # DeepSpeed config (single GPU)
│
├── 🎛️ render_configs/            # Render deployment configurations
│   ├── render.yaml                # Discord bot Render config
│   ├── render-training.yaml       # Web UI Render config
│   └── runtime.txt                # Python runtime specification
│
└── 📊 deployments/               # Deployment tracking
    ├── .gitkeep                   # Keep directory in git
    └── full_training.json         # Example deployment record
```

## 🎯 Quick Navigation

### For Training
- **Start here**: `core/train.py` - Main training script
- **Deploy to Modal**: `deployment/deploy_modal.py`
- **Configuration**: `configs/deepspeed_config.json`

### For Web UI Deployment
- **Web Interface**: `web_ui/training_deployer.py`
- **Setup Guide**: `docs/TRAINING_DEPLOYER_GUIDE.md`
- **Render Config**: `render_configs/render-training.yaml`

### For Discord Bot
- **Bot Code**: `discord_bot/discord_bot_with_health.py`
- **Setup Guide**: `docs/UPTIMEROBOT_SETUP.md`
- **Render Config**: `render_configs/render.yaml`

### For Documentation
- **Main README**: `docs/README.md`
- **Setup Instructions**: `docs/SETUP_GUIDE.md`
- **Troubleshooting**: `docs/FIX_GUIDE.md`

### For Testing
- **Test Modal**: `tests/test_modal_deployment.py`
- **Test Locally**: `tests/test_local.py`

## 🚀 Common Workflows

### 1. First Time Setup
```bash
# 1. Read the main documentation
cat docs/README.md

# 2. Follow setup guide
cat docs/SETUP_GUIDE.md

# 3. Configure environment
cp .env.template .env
# Edit .env with your credentials

# 4. Test Modal connection
python tests/test_modal_deployment.py
```

### 2. Deploy Training via Command Line
```bash
# Quick test
python deployment/deploy_modal.py --quick-test

# Full training
python deployment/deploy_modal.py --full-training
```

### 3. Deploy Web UI to Render
```bash
# 1. Read the web UI guide
cat docs/TRAINING_DEPLOYER_GUIDE.md

# 2. Use the Render configuration
# Upload render_configs/render-training.yaml to Render

# 3. Set environment variables in Render dashboard
```

### 4. Deploy Discord Bot to Render
```bash
# 1. Read the bot setup guide
cat docs/UPTIMEROBOT_SETUP.md

# 2. Use the Render configuration
# Upload render_configs/render.yaml to Render
```

## 🔍 File Purposes

### Core Components
- **`core/train.py`**: Main training logic with DeepSpeed integration
- **`core/modal_app.py`**: Modal application with GPU functions
- **`core/data_utils.py`**: Dataset loading and preprocessing
- **`core/model_utils.py`**: Model loading and LoRA setup

### Deployment Options
- **`deployment/deploy_modal.py`**: Simplified, reliable deployment script
- **`deployment/deploy_training.py`**: Original deployment script (legacy)

### Web Interface
- **`web_ui/training_deployer.py`**: Flask app for web-based deployments
- **`web_ui/templates/`**: HTML templates for the web interface
- **`web_ui/static/`**: CSS and JavaScript for the web interface

### Monitoring & Bots
- **`discord_bot/discord_bot_with_health.py`**: Discord bot with health endpoint for monitoring
- **`discord_bot/discord_bot.py`**: Basic Discord bot for notifications

## 🛠️ Development Tips

1. **Working on Core Training**: Focus on `core/` directory
2. **Adding Features**: Update relevant directory and documentation
3. **Testing Changes**: Use files in `tests/` directory
4. **Deployment Issues**: Check `docs/FIX_GUIDE.md`
5. **New Configurations**: Add to `configs/` directory

## 📝 Maintenance

- **Documentation**: Keep `docs/` updated with changes
- **Dependencies**: Update `requirements.txt` and `*_requirements.txt` files
- **Configurations**: Version control changes in `configs/` and `render_configs/`
- **Tests**: Add new tests to `tests/` directory

---

This organized structure makes it easy to:
- 🎯 Find specific functionality quickly
- 📚 Understand project components
- 🚀 Deploy different parts independently
- 🔧 Maintain and update the codebase
- 📖 Onboard new developers

For detailed information about any component, refer to the documentation in the `docs/` directory.
