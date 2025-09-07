#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Helper script to deploy the OLMo Training WebUI to Render.

.DESCRIPTION
    This PowerShell script automates the deployment process and guides users through setup.
    It checks prerequisites, gathers credentials, and provides step-by-step instructions.

.EXAMPLE
    .\deploy_webui.ps1
#>

param(
    [switch]$SkipGitCheck,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Definition -Detailed
    exit 0
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * $Title.Length) -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Generate-SecureKey {
    param([int]$Length = 32)
    
    $chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    $key = ""
    for ($i = 0; $i -lt $Length; $i++) {
        $key += $chars[(Get-Random -Maximum $chars.Length)]
    }
    return $key
}

function Test-GitRepository {
    if ($SkipGitCheck) {
        Write-Warning "Skipping git checks as requested"
        return $true
    }

    try {
        $gitStatus = git status 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Not in a git repository. Please initialize git first:"
            Write-Host "   git init"
            Write-Host "   git add ."
            Write-Host "   git commit -m 'Initial commit'"
            return $false
        }

        $gitStatusPorcelain = git status --porcelain 2>&1
        if ($gitStatusPorcelain) {
            Write-Warning "You have uncommitted changes. Please commit them first:"
            Write-Host "   git add ."
            Write-Host "   git commit -m 'Update for WebUI deployment'"
            return $false
        }

        return $true
    }
    catch {
        Write-Error "Git is not installed or not in PATH. Please install git first."
        return $false
    }
}

function Get-GitRemoteUrl {
    try {
        $remoteUrl = git remote get-url origin 2>&1
        if ($LASTEXITCODE -eq 0) {
            return $remoteUrl.Trim()
        }
        return $null
    }
    catch {
        return $null
    }
}

function New-RenderConfig {
    $config = @{
        services = @(
            @{
                type = "web"
                name = "olmo-training-webui"
                runtime = "python"
                region = "oregon"
                plan = "starter"
                buildCommand = "cd modal_olmo_finetune; pip install --upgrade pip; pip install -r web_ui/deployer_requirements.txt"
                startCommand = "cd modal_olmo_finetune/web_ui; python training_deployer.py"
                healthCheckPath = "/health"
                envVars = @(
                    @{ key = "MODAL_TOKEN_ID"; sync = $false }
                    @{ key = "MODAL_TOKEN_SECRET"; sync = $false }
                    @{ key = "DEPLOY_API_KEY"; generateValue = $true; sync = $false }
                    @{ key = "FLASK_SECRET_KEY"; generateValue = $true; sync = $false }
                    @{ key = "WANDB_API_KEY"; sync = $false }
                    @{ key = "WANDB_ENTITY"; value = "your-wandb-entity" }
                    @{ key = "WANDB_PROJECT"; value = "olmo-finetune-modal" }
                    @{ key = "FLASK_ENV"; value = "production" }
                    @{ key = "PYTHON_VERSION"; value = "3.10" }
                    @{ key = "TZ"; value = "America/Los_Angeles" }
                    @{ key = "LOG_LEVEL"; value = "INFO" }
                    @{ key = "ALLOWED_ORIGINS"; value = "*" }
                )
                autoDeploy = $true
                scaling = @{
                    minInstances = 1
                    maxInstances = 1
                }
            }
        )
        disks = @(
            @{
                name = "olmo-webui-storage"
                mountPath = "/opt/render/project/src/modal_olmo_finetune/web_ui"
                sizeGB = 1
            }
        )
    }

    $configPath = "modal_olmo_finetune/render_configs/webui-render.yaml"
    
    # Ensure directory exists
    $configDir = Split-Path $configPath -Parent
    if (!(Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    # Convert to YAML format (simplified)
    $yamlContent = @'
services:
  - type: web
    name: olmo-training-webui
    runtime: python
    region: oregon
    plan: starter
    buildCommand: cd modal_olmo_finetune && pip install --upgrade pip && pip install -r web_ui/deployer_requirements.txt
    startCommand: cd modal_olmo_finetune/web_ui && python training_deployer.py
    healthCheckPath: /health
    envVars:
      - key: MODAL_TOKEN_ID
        sync: false
      - key: MODAL_TOKEN_SECRET
        sync: false
      - key: DEPLOY_API_KEY
        generateValue: true
        sync: false
      - key: FLASK_SECRET_KEY
        generateValue: true
        sync: false
      - key: WANDB_API_KEY
        sync: false
      - key: WANDB_ENTITY
        value: your-wandb-entity
      - key: WANDB_PROJECT
        value: olmo-finetune-modal
      - key: FLASK_ENV
        value: production
      - key: PYTHON_VERSION
        value: "3.10"
      - key: TZ
        value: America/Los_Angeles
      - key: LOG_LEVEL
        value: INFO
      - key: ALLOWED_ORIGINS
        value: "*"
    autoDeploy: true
    scaling:
      minInstances: 1
      maxInstances: 1

disks:
  - name: olmo-webui-storage
    mountPath: /opt/render/project/src/modal_olmo_finetune/web_ui
    sizeGB: 1
'@

    Set-Content -Path $configPath -Value $yamlContent -Encoding UTF8
    return $configPath
}

function Get-ModalCredentials {
    Write-Header "Modal Credentials Setup"
    Write-Host "You need your Modal credentials to deploy the WebUI."
    Write-Host "Get them from: https://modal.com/settings/tokens"
    Write-Host ""

    do {
        $modalTokenId = Read-Host "Enter your MODAL_TOKEN_ID"
    } while ([string]::IsNullOrWhiteSpace($modalTokenId))

    do {
        $modalTokenSecret = Read-Host "Enter your MODAL_TOKEN_SECRET" -AsSecureString
        $modalTokenSecretPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($modalTokenSecret))
    } while ([string]::IsNullOrWhiteSpace($modalTokenSecretPlain))

    return @{
        MODAL_TOKEN_ID = $modalTokenId.Trim()
        MODAL_TOKEN_SECRET = $modalTokenSecretPlain.Trim()
    }
}

function Get-WandbCredentials {
    Write-Header "Weights and Biases Setup (Optional)"
    Write-Host "W and B integration is optional but recommended for tracking training metrics."
    Write-Host "Get your API key from: https://wandb.ai/settings"
    Write-Host ""

    $useWandb = Read-Host "Do you want to set up W and B integration? (y/N)"
    
    if ($useWandb -match "^[yY]") {
        $wandbApiKey = Read-Host "Enter your WANDB_API_KEY"
        $wandbEntity = Read-Host "Enter your W and B username/entity"
        
        if (![string]::IsNullOrWhiteSpace($wandbApiKey) -and ![string]::IsNullOrWhiteSpace($wandbEntity)) {
            return @{
                WANDB_API_KEY = $wandbApiKey.Trim()
                WANDB_ENTITY = $wandbEntity.Trim()
            }
        }
    }
    
    return @{}
}

function New-EnvFile {
    param([hashtable]$Credentials)
    
    $deployApiKey = Generate-SecureKey
    $flaskSecretKey = Generate-SecureKey
    
    $envContent = @"
# Modal Credentials (Required)
MODAL_TOKEN_ID=$($Credentials.MODAL_TOKEN_ID)
MODAL_TOKEN_SECRET=$($Credentials.MODAL_TOKEN_SECRET)

# Deployment Security
DEPLOY_API_KEY=$deployApiKey
FLASK_SECRET_KEY=$flaskSecretKey

# W and B Integration (Optional)
WANDB_API_KEY=$($Credentials.WANDB_API_KEY)
WANDB_ENTITY=$($Credentials.WANDB_ENTITY)
WANDB_PROJECT=olmo-finetune-modal

# Application Settings
FLASK_ENV=development
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*
"@

    $envPath = "modal_olmo_finetune/.env"
    Set-Content -Path $envPath -Value $envContent -Encoding UTF8
    
    return @{
        Path = $envPath
        DeployApiKey = $deployApiKey
    }
}

function Show-DeploymentInstructions {
    param(
        [string]$GitUrl,
        [string]$DeployApiKey
    )
    
    Write-Header "Render Deployment Instructions"
    Write-Host ""
    Write-Host "1. Go to https://dashboard.render.com"
    Write-Host "2. Click 'New +' -> 'Web Service'"
    Write-Host "3. Connect your GitHub/GitLab repository"
    
    if ($GitUrl) {
        Write-Host "4. Select repository: $GitUrl"
    } else {
        Write-Host "4. Select your repository containing this project"
    }
    
    Write-Host "5. Configure the service:"
    Write-Host "   - Name: olmo-training-webui"
    Write-Host "   - Region: Oregon (or closest to you)"
    Write-Host "   - Branch: main"
    Write-Host "   - Root Directory: (leave empty)"
    Write-Host "   - Runtime: Python 3"
    Write-Host "   - Build Command: cd modal_olmo_finetune && pip install -r web_ui/deployer_requirements.txt"
    Write-Host "   - Start Command: cd modal_olmo_finetune/web_ui && python training_deployer.py"
    Write-Host ""
    Write-Host "6. Set Environment Variables in Render Dashboard:"
    Write-Host "   REQUIRED:" -ForegroundColor Yellow
    Write-Host "   - MODAL_TOKEN_ID: (your Modal token ID)"
    Write-Host "   - MODAL_TOKEN_SECRET: (your Modal token secret)"
    Write-Host "   - DEPLOY_API_KEY: $DeployApiKey" -ForegroundColor Green
    Write-Host "   - FLASK_SECRET_KEY: (generate a secure key)"
    Write-Host ""
    Write-Host "   OPTIONAL (for W and B):" -ForegroundColor Yellow
    Write-Host "   - WANDB_API_KEY: (your W and B API key)"
    Write-Host "   - WANDB_ENTITY: (your W and B username)"
    Write-Host ""
    Write-Host "7. Click 'Create Web Service'"
    Write-Host "8. Wait for deployment to complete (2-5 minutes)"
    Write-Host "9. Access your WebUI at the provided Render URL"
    Write-Host ""
    Write-Host "Save your DEPLOY_API_KEY - you'll need it to use the WebUI!" -ForegroundColor Magenta
    Write-Host "   DEPLOY_API_KEY: $DeployApiKey" -ForegroundColor Green
}

function Test-LocalSetup {
    Write-Header "Testing Local Setup"
    
    $requiredFiles = @(
        "modal_olmo_finetune/web_ui/training_deployer.py",
        "modal_olmo_finetune/web_ui/deployer_requirements.txt",
        "modal_olmo_finetune/web_ui/templates/index.html",
        "modal_olmo_finetune/core/modal_app.py"
    )
    
    $missingFiles = @()
    foreach ($file in $requiredFiles) {
        if (!(Test-Path $file)) {
            $missingFiles += $file
        }
    }
    
    if ($missingFiles.Count -gt 0) {
        Write-Error "Missing required files:"
        foreach ($file in $missingFiles) {
            Write-Host "   - $file" -ForegroundColor Red
        }
        return $false
    }
    
    Write-Success "All required files present"
    return $true
}

function Main {
    Write-Host "OLMo Training WebUI Deployment Helper" -ForegroundColor Cyan
    Write-Host ("=" * 50) -ForegroundColor Cyan
    Write-Host "This script will help you deploy the WebUI to Render."
    Write-Host ""
    
    # Check if we're in the right directory
    if (!(Test-Path "modal_olmo_finetune")) {
        Write-Error "Please run this script from the project root directory"
        Write-Host "   (the directory containing modal_olmo_finetune/)"
        exit 1
    }
    
    # Test local setup
    if (!(Test-LocalSetup)) {
        Write-Error "Local setup test failed. Please fix the issues above."
        exit 1
    }
    
    # Check git status
    if (!(Test-GitRepository)) {
        exit 1
    }
    
    $gitUrl = Get-GitRemoteUrl
    
    # Get credentials
    $modalCreds = Get-ModalCredentials
    $wandbCreds = Get-WandbCredentials
    
    $allCreds = $modalCreds + $wandbCreds
    
    # Create environment file
    $envResult = New-EnvFile -Credentials $allCreds
    Write-Success "Created environment file: $($envResult.Path)"
    
    # Create Render configuration
    $renderConfigPath = New-RenderConfig
    Write-Success "Created Render configuration: $renderConfigPath"
    
    # Show deployment instructions
    Show-DeploymentInstructions -GitUrl $gitUrl -DeployApiKey $envResult.DeployApiKey
    
    Write-Host ""
    Write-Success "Deployment setup complete!"
    Write-Host ""
    Write-Info "Next steps:"
    Write-Host "1. Push your changes to your git repository"
    Write-Host "2. Follow the deployment instructions above"
    Write-Host "3. Test your deployed WebUI"
    Write-Host ""
    Write-Host "For troubleshooting, check the deployment logs in Render dashboard."
}

# Call the main function
Main
