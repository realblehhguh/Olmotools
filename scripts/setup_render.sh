#!/bin/bash

# Setup script for deploying Discord bot to Render
# This script helps prepare your repository for Render deployment

echo "🚀 OLMo Discord Bot - Render Setup Script"
echo "========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "discord_bot.py" ]; then
    echo "❌ Error: discord_bot.py not found."
    echo "Please run this script from the modal_olmo_finetune directory."
    exit 1
fi

# Create deployments directory if it doesn't exist
if [ ! -d "deployments" ]; then
    echo "📁 Creating deployments directory..."
    mkdir -p deployments
    echo "✅ Created deployments/"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Environment files
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Project specific
deployments/completed_*.json
wandb/
*.ckpt
*.pt
*.pth
outputs/
checkpoints/
EOF
    echo "✅ Created .gitignore"
fi

# Check if repository is initialized
if [ ! -d ".git" ]; then
    echo ""
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
fi

# Check for required files
echo ""
echo "📋 Checking required files..."

required_files=("discord_bot.py" "bot_requirements.txt" "render.yaml")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "❌ Missing required files. Please ensure all files are present."
    exit 1
fi

# Create a minimal runtime.txt for Python version
if [ ! -f "runtime.txt" ]; then
    echo ""
    echo "📝 Creating runtime.txt..."
    echo "python-3.10.12" > runtime.txt
    echo "✅ Created runtime.txt"
fi

# Prompt for GitHub repository
echo ""
echo "🔗 GitHub Setup"
echo "---------------"
echo "Have you created a GitHub repository for this project? (y/n)"
read -r has_repo

if [ "$has_repo" = "y" ] || [ "$has_repo" = "Y" ]; then
    echo "Enter your GitHub repository URL:"
    echo "Example: https://github.com/username/olmo-discord-bot.git"
    read -r repo_url
    
    if [ ! -z "$repo_url" ]; then
        # Check if remote already exists
        if git remote | grep -q "origin"; then
            echo "Remote 'origin' already exists. Updating URL..."
            git remote set-url origin "$repo_url"
        else
            git remote add origin "$repo_url"
        fi
        echo "✅ GitHub remote configured"
    fi
else
    echo ""
    echo "📌 To create a GitHub repository:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository (e.g., 'olmo-discord-bot')"
    echo "3. Don't initialize with README or .gitignore"
    echo "4. Run this script again and provide the repository URL"
fi

# Stage and commit files
echo ""
echo "📦 Preparing files for deployment..."

# Add files to git
git add discord_bot.py
git add bot_requirements.txt
git add render.yaml
git add runtime.txt
git add deployments/.gitkeep 2>/dev/null || touch deployments/.gitkeep && git add deployments/.gitkeep

# Check if there are changes to commit
if ! git diff --cached --quiet; then
    echo "Creating initial commit..."
    git commit -m "Initial commit for Discord bot deployment to Render"
    echo "✅ Files committed"
else
    echo "✅ No changes to commit"
fi

# Push to GitHub if remote is configured
if git remote | grep -q "origin"; then
    echo ""
    echo "📤 Push to GitHub? (y/n)"
    read -r should_push
    
    if [ "$should_push" = "y" ] || [ "$should_push" = "Y" ]; then
        echo "Pushing to GitHub..."
        git branch -M main 2>/dev/null || true
        git push -u origin main
        echo "✅ Pushed to GitHub"
    fi
fi

# Display next steps
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. If you haven't already, push your code to GitHub:"
echo "   git push -u origin main"
echo ""
echo "2. Go to https://dashboard.render.com"
echo ""
echo "3. Click 'New +' → 'Blueprint'"
echo ""
echo "4. Connect your GitHub repository"
echo ""
echo "5. Set these environment variables in Render:"
echo "   - DISCORD_BOT_TOKEN"
echo "   - DISCORD_USER_ID"
echo "   - WANDB_API_KEY"
echo "   - WANDB_ENTITY"
echo ""
echo "6. Click 'Create Service'"
echo ""
echo "📚 For detailed instructions, see RENDER_DEPLOYMENT.md"
echo ""
echo "💡 Tips:"
echo "   - Use 'git status' to check your repository status"
echo "   - Use 'git push' to update your deployed bot"
echo "   - Check Render logs if the bot doesn't start"
echo ""
echo "Good luck with your deployment! 🚀"
