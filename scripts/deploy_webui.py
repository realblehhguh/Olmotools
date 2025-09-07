#!/usr/bin/env python3
"""
Helper script to deploy the OLMo Training WebUI to Render.
This script automates the deployment process and guides users through setup.
"""

import os
import sys
import json
import subprocess
import secrets
import string
import argparse
from pathlib import Path
from typing import Dict, Optional

def generate_secure_key(length: int = 32) -> str:
    """Generate a secure random key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def check_git_status() -> bool:
    """Check if we're in a git repository and if there are uncommitted changes."""
    try:
        # Check if we're in a git repo
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Not in a git repository. Please initialize git first:")
            print("   git init")
            print("   git add .")
            print("   git commit -m 'Initial commit'")
            return False
        
        # Check for uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes. Please commit them first:")
            print("   git add .")
            print("   git commit -m 'Update for WebUI deployment'")
            return False
        
        return True
    except FileNotFoundError:
        print("❌ Git is not installed. Please install git first.")
        return False

def get_git_remote_url() -> Optional[str]:
    """Get the git remote URL."""
    try:
        result = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None

def create_render_yaml() -> str:
    """Create or update the render.yaml file for WebUI deployment."""
    render_config = {
        "services": [
            {
                "type": "web",
                "name": "olmo-training-webui",
                "runtime": "python",
                "region": "oregon",
                "plan": "starter",
                "buildCommand": "cd modal_olmo_finetune && pip install --upgrade pip && pip install -r web_ui/deployer_requirements.txt",
                "startCommand": "cd modal_olmo_finetune/web_ui && python training_deployer.py",
                "healthCheckPath": "/health",
                "envVars": [
                    {
                        "key": "MODAL_TOKEN_ID",
                        "sync": False
                    },
                    {
                        "key": "MODAL_TOKEN_SECRET", 
                        "sync": False
                    },
                    {
                        "key": "DEPLOY_API_KEY",
                        "generateValue": True,
                        "sync": False
                    },
                    {
                        "key": "FLASK_SECRET_KEY",
                        "generateValue": True,
                        "sync": False
                    },
                    {
                        "key": "WANDB_API_KEY",
                        "sync": False
                    },
                    {
                        "key": "WANDB_ENTITY",
                        "value": "your-wandb-entity"
                    },
                    {
                        "key": "WANDB_PROJECT",
                        "value": "olmo-finetune-modal"
                    },
                    {
                        "key": "FLASK_ENV",
                        "value": "production"
                    },
                    {
                        "key": "PYTHON_VERSION",
                        "value": "3.10"
                    },
                    {
                        "key": "TZ",
                        "value": "America/Los_Angeles"
                    },
                    {
                        "key": "LOG_LEVEL",
                        "value": "INFO"
                    },
                    {
                        "key": "ALLOWED_ORIGINS",
                        "value": "*"
                    }
                ],
                "autoDeploy": True,
                "scaling": {
                    "minInstances": 1,
                    "maxInstances": 1
                }
            }
        ],
        "disks": [
            {
                "name": "olmo-webui-storage",
                "mountPath": "/opt/render/project/src/modal_olmo_finetune/web_ui",
                "sizeGB": 1
            }
        ]
    }
    
    # Write to render_configs directory
    config_path = Path("modal_olmo_finetune/render_configs/webui-render.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(render_config, f, default_flow_style=False, indent=2)
    
    return str(config_path)

def get_modal_credentials() -> Dict[str, str]:
    """Get Modal credentials from user."""
    print("\n🔑 Modal Credentials Setup")
    print("=" * 50)
    print("You need your Modal credentials to deploy the WebUI.")
    print("Get them from: https://modal.com/settings/tokens")
    print()
    
    modal_token_id = input("Enter your MODAL_TOKEN_ID: ").strip()
    modal_token_secret = input("Enter your MODAL_TOKEN_SECRET: ").strip()
    
    if not modal_token_id or not modal_token_secret:
        print("❌ Modal credentials are required!")
        sys.exit(1)
    
    return {
        "MODAL_TOKEN_ID": modal_token_id,
        "MODAL_TOKEN_SECRET": modal_token_secret
    }

def get_wandb_credentials() -> Dict[str, str]:
    """Get W&B credentials from user (optional)."""
    print("\n📊 Weights & Biases Setup (Optional)")
    print("=" * 50)
    print("W&B integration is optional but recommended for tracking training metrics.")
    print("Get your API key from: https://wandb.ai/settings")
    print()
    
    use_wandb = input("Do you want to set up W&B integration? (y/N): ").strip().lower()
    
    if use_wandb in ['y', 'yes']:
        wandb_api_key = input("Enter your WANDB_API_KEY: ").strip()
        wandb_entity = input("Enter your W&B username/entity: ").strip()
        
        if wandb_api_key and wandb_entity:
            return {
                "WANDB_API_KEY": wandb_api_key,
                "WANDB_ENTITY": wandb_entity
            }
    
    return {}

def create_env_template(credentials: Dict[str, str]) -> str:
    """Create a .env template file with the credentials."""
    env_content = [
        "# Modal Credentials (Required)",
        f"MODAL_TOKEN_ID={credentials.get('MODAL_TOKEN_ID', 'your_modal_token_id')}",
        f"MODAL_TOKEN_SECRET={credentials.get('MODAL_TOKEN_SECRET', 'your_modal_token_secret')}",
        "",
        "# Deployment Security",
        f"DEPLOY_API_KEY={generate_secure_key()}",
        f"FLASK_SECRET_KEY={generate_secure_key()}",
        "",
        "# W&B Integration (Optional)",
        f"WANDB_API_KEY={credentials.get('WANDB_API_KEY', '')}",
        f"WANDB_ENTITY={credentials.get('WANDB_ENTITY', 'your-wandb-entity')}",
        "WANDB_PROJECT=olmo-finetune-modal",
        "",
        "# Application Settings",
        "FLASK_ENV=development",
        "LOG_LEVEL=INFO",
        "ALLOWED_ORIGINS=*"
    ]
    
    env_path = Path("modal_olmo_finetune/.env")
    with open(env_path, 'w') as f:
        f.write('\n'.join(env_content))
    
    return str(env_path)

def print_deployment_instructions(git_url: Optional[str], deploy_api_key: str):
    """Print instructions for completing the deployment on Render."""
    print("\n🚀 Render Deployment Instructions")
    print("=" * 50)
    print()
    print("1. Go to https://dashboard.render.com")
    print("2. Click 'New +' → 'Web Service'")
    print("3. Connect your GitHub/GitLab repository")
    
    if git_url:
        print(f"4. Select repository: {git_url}")
    else:
        print("4. Select your repository containing this project")
    
    print("5. Configure the service:")
    print("   - Name: olmo-training-webui")
    print("   - Region: Oregon (or closest to you)")
    print("   - Branch: main")
    print("   - Root Directory: (leave empty)")
    print("   - Runtime: Python 3")
    print("   - Build Command: cd modal_olmo_finetune && pip install -r web_ui/deployer_requirements.txt")
    print("   - Start Command: cd modal_olmo_finetune/web_ui && python training_deployer.py")
    print()
    print("6. Set Environment Variables in Render Dashboard:")
    print("   REQUIRED:")
    print("   - MODAL_TOKEN_ID: (your Modal token ID)")
    print("   - MODAL_TOKEN_SECRET: (your Modal token secret)")
    print(f"   - DEPLOY_API_KEY: {deploy_api_key}")
    print("   - FLASK_SECRET_KEY: (generate a secure key)")
    print()
    print("   OPTIONAL (for W&B):")
    print("   - WANDB_API_KEY: (your W&B API key)")
    print("   - WANDB_ENTITY: (your W&B username)")
    print()
    print("7. Click 'Create Web Service'")
    print("8. Wait for deployment to complete (2-5 minutes)")
    print("9. Access your WebUI at the provided Render URL")
    print()
    print("📝 Save your DEPLOY_API_KEY - you'll need it to use the WebUI!")
    print(f"   DEPLOY_API_KEY: {deploy_api_key}")

def test_local_setup():
    """Test the local setup before deployment."""
    print("\n🧪 Testing Local Setup")
    print("=" * 30)
    
    # Check if required files exist
    required_files = [
        "modal_olmo_finetune/web_ui/training_deployer.py",
        "modal_olmo_finetune/web_ui/deployer_requirements.txt",
        "modal_olmo_finetune/web_ui/templates/index.html",
        "modal_olmo_finetune/core/modal_app.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    
    # Test import (optional - dependencies may not be installed locally)
    try:
        sys.path.append("modal_olmo_finetune/web_ui")
        import training_deployer
        print("✅ WebUI imports successfully")
    except ImportError as e:
        print(f"⚠️  Import warning: {e}")
        print("   This is normal if dependencies aren't installed locally.")
        print("   Dependencies will be installed during Render deployment.")
    
    return True

def main():
    """Main deployment helper function."""
    print("🌐 OLMo Training WebUI Deployment Helper")
    print("=" * 50)
    print("This script will help you deploy the WebUI to Render.")
    print()
    
    # Check if we're in the right directory
    if not Path("modal_olmo_finetune").exists():
        print("❌ Please run this script from the project root directory")
        print("   (the directory containing modal_olmo_finetune/)")
        sys.exit(1)
    
    # Test local setup
    if not test_local_setup():
        print("\n❌ Local setup test failed. Please fix the issues above.")
        sys.exit(1)
    
    # Check git status
    if not check_git_status():
        sys.exit(1)
    
    git_url = get_git_remote_url()
    
    # Get credentials
    modal_creds = get_modal_credentials()
    wandb_creds = get_wandb_credentials()
    
    all_creds = {**modal_creds, **wandb_creds}
    
    # Create environment file
    env_path = create_env_template(all_creds)
    print(f"\n✅ Created environment file: {env_path}")
    
    # Create Render configuration
    try:
        import yaml
    except ImportError:
        print("Installing PyYAML...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=True)
        import yaml
    
    render_config_path = create_render_yaml()
    print(f"✅ Created Render configuration: {render_config_path}")
    
    # Generate API key for instructions
    deploy_api_key = generate_secure_key()
    
    # Print deployment instructions
    print_deployment_instructions(git_url, deploy_api_key)
    
    print("\n🎉 Setup Complete!")
    print("Follow the instructions above to complete deployment on Render.")
    print()
    print("💡 Pro Tips:")
    print("- Save the DEPLOY_API_KEY in a secure location")
    print("- Test the WebUI locally first with: cd modal_olmo_finetune/web_ui && python training_deployer.py")
    print("- Monitor deployment logs in the Render dashboard")
    print("- The WebUI will be available at your Render service URL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helper script to deploy the OLMo Training WebUI to Render",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_webui.py                    # Interactive deployment setup
  python deploy_webui.py --skip-git-check  # Skip git repository checks
  python deploy_webui.py --help            # Show this help message

This script will guide you through setting up your OLMo Training WebUI
for deployment on Render. It will:
  1. Check your local setup
  2. Gather your Modal and W&B credentials
  3. Create configuration files
  4. Provide step-by-step deployment instructions
        """
    )
    
    parser.add_argument(
        "--skip-git-check",
        action="store_true",
        help="Skip git repository and commit status checks"
    )
    
    args = parser.parse_args()
    
    try:
