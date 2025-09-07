# Setup script for deploying Discord bot to Render (PowerShell version)
# This script helps prepare your repository for Render deployment on Windows

Write-Host "OLMo Discord Bot - Render Setup Script (PowerShell)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "Git is not installed. Please install git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "discord_bot.py")) {
    Write-Host "Error: discord_bot.py not found." -ForegroundColor Red
    Write-Host "Please run this script from the modal_olmo_finetune directory." -ForegroundColor Yellow
    Set-Location -Path $PSScriptRoot -ErrorAction SilentlyContinue
    if (-not (Test-Path "discord_bot.py")) {
        exit 1
    }
}

# Create deployments directory if it doesn't exist
if (-not (Test-Path "deployments")) {
    Write-Host "Creating deployments directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path "deployments" -Force | Out-Null
    Write-Host "Created deployments/" -ForegroundColor Green
}

# Create .gitignore if it doesn't exist
if (-not (Test-Path ".gitignore")) {
    Write-Host "Creating .gitignore..." -ForegroundColor Green
    
    $gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
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
"@
    
    Set-Content -Path ".gitignore" -Value $gitignoreContent
    Write-Host "Created .gitignore" -ForegroundColor Green
}

# Check if repository is initialized
if (-not (Test-Path ".git")) {
    Write-Host ""
    Write-Host "Initializing Git repository..." -ForegroundColor Green
    git init
    Write-Host "Git repository initialized" -ForegroundColor Green
}

# Check for required files
Write-Host ""
Write-Host "Checking required files..." -ForegroundColor Yellow

$requiredFiles = @("discord_bot.py", "bot_requirements.txt", "render.yaml")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  $file exists" -ForegroundColor Green
    } else {
        Write-Host "  $file is missing" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing required files. Please ensure all files are present." -ForegroundColor Red
    exit 1
}

# Create a minimal runtime.txt for Python version
if (-not (Test-Path "runtime.txt")) {
    Write-Host ""
    Write-Host "Creating runtime.txt..." -ForegroundColor Green
    Set-Content -Path "runtime.txt" -Value "python-3.10.12"
    Write-Host "Created runtime.txt" -ForegroundColor Green
}

# Prompt for GitHub repository
Write-Host ""
Write-Host "GitHub Setup" -ForegroundColor Cyan
Write-Host "---------------" -ForegroundColor Cyan

$hasRepo = Read-Host "Have you created a GitHub repository for this project? (y/n)"

if ($hasRepo -eq "y" -or $hasRepo -eq "Y") {
    Write-Host "Enter your GitHub repository URL:" -ForegroundColor Yellow
    Write-Host "Example: https://github.com/username/olmo-discord-bot.git" -ForegroundColor Gray
    $repoUrl = Read-Host "Repository URL"
    
    if ($repoUrl) {
        # Check if remote already exists
        $remotes = git remote
        if ($remotes -contains "origin") {
            Write-Host "Remote origin already exists. Updating URL..." -ForegroundColor Yellow
            git remote set-url origin $repoUrl
        } else {
            git remote add origin $repoUrl
        }
        Write-Host "GitHub remote configured" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "To create a GitHub repository:" -ForegroundColor Yellow
    Write-Host "1. Go to https://github.com/new" -ForegroundColor White
    Write-Host "2. Create a new repository (example: olmo-discord-bot)" -ForegroundColor White
    Write-Host "3. Do not initialize with README or .gitignore" -ForegroundColor White
    Write-Host "4. Run this script again and provide the repository URL" -ForegroundColor White
}

# Stage and commit files
Write-Host ""
Write-Host "Preparing files for deployment..." -ForegroundColor Green

# Create .gitkeep in deployments if needed
if (-not (Test-Path "deployments/.gitkeep")) {
    New-Item -ItemType File -Path "deployments/.gitkeep" -Force | Out-Null
}

# Add files to git
git add discord_bot.py 2>$null
git add bot_requirements.txt 2>$null
git add render.yaml 2>$null
git add runtime.txt 2>$null
git add deployments/.gitkeep 2>$null

# Check if there are changes to commit
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "Creating initial commit..." -ForegroundColor Green
    git commit -m "Initial commit for Discord bot deployment to Render"
    Write-Host "Files committed" -ForegroundColor Green
} else {
    Write-Host "No changes to commit" -ForegroundColor Green
}

# Push to GitHub if remote is configured
$remotes = git remote
if ($remotes -contains "origin") {
    Write-Host ""
    $shouldPush = Read-Host "Push to GitHub? (y/n)"
    
    if ($shouldPush -eq "y" -or $shouldPush -eq "Y") {
        Write-Host "Pushing to GitHub..." -ForegroundColor Green
        
        # Ensure we're on main branch
        $currentBranch = git branch --show-current
        if ($currentBranch -ne "main") {
            git branch -M main
        }
        
        try {
            git push -u origin main
            Write-Host "Pushed to GitHub" -ForegroundColor Green
        } catch {
            Write-Host "Push failed. You may need to push manually:" -ForegroundColor Yellow
            Write-Host "   git push -u origin main" -ForegroundColor White
        }
    }
}

# Display next steps
Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. If you haven't already pushed your code to GitHub:" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Go to https://dashboard.render.com" -ForegroundColor White
Write-Host ""
Write-Host "3. Click New + then Blueprint" -ForegroundColor White
Write-Host ""
Write-Host "4. Connect your GitHub repository" -ForegroundColor White
Write-Host ""
Write-Host "5. Set these environment variables in Render:" -ForegroundColor White
Write-Host "   - DISCORD_BOT_TOKEN" -ForegroundColor Yellow
Write-Host "   - DISCORD_USER_ID" -ForegroundColor Yellow
Write-Host "   - WANDB_API_KEY" -ForegroundColor Yellow
Write-Host "   - WANDB_ENTITY" -ForegroundColor Yellow
Write-Host ""
Write-Host "6. Click Create Service" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions see RENDER_DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tips:" -ForegroundColor Magenta
Write-Host "   - Use git status to check your repository status" -ForegroundColor White
Write-Host "   - Use git push to update your deployed bot" -ForegroundColor White
Write-Host "   - Check Render logs if the bot doesn't start" -ForegroundColor White
Write-Host ""
Write-Host "Good luck with your deployment!" -ForegroundColor Green

# Pause at the end so the window doesn't close immediately
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
