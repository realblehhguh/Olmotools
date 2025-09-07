# Deploying Discord Bot to Render

This guide explains how to deploy the OLMo training Discord bot to Render for 24/7 cloud hosting.

## Why Render?

- **Always Online**: Bot runs 24/7 without keeping your computer on
- **Auto-restart**: Automatically restarts if it crashes
- **Easy Deployment**: Git-based deployment with automatic updates
- **Affordable**: Starting at $7/month for worker services
- **Built-in Logging**: View logs directly in Render dashboard

## Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Render Account**: Sign up at https://render.com
3. **Discord Bot Token**: From Discord Developer Portal
4. **W&B API Key**: From your Weights & Biases account

## Step-by-Step Deployment

### 1. Prepare Your Repository

1. **Create a GitHub repository** for your project
2. **Push your code** including:
   - `discord_bot.py`
   - `bot_requirements.txt`
   - `render.yaml`
   - `deployments/` folder (create empty if needed)

```bash
git init
git add .
git commit -m "Initial commit for Discord bot"
git remote add origin https://github.com/YOUR_USERNAME/olmo-discord-bot.git
git push -u origin main
```

### 2. Connect to Render

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub account if not already connected
4. Select your repository
5. Render will detect the `render.yaml` file automatically

### 3. Configure Environment Variables

In the Render dashboard, set these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `DISCORD_BOT_TOKEN` | Your Discord bot token | `MTIz...abc` |
| `DISCORD_USER_ID` | Your Discord user ID | `123456789012345678` |
| `WANDB_API_KEY` | Your W&B API key | `local-abc123...` |
| `WANDB_ENTITY` | Your W&B username/org | `your-username` |

**How to set environment variables:**
1. Go to your service in Render dashboard
2. Click **"Environment"** tab
3. Add each variable with its value
4. Click **"Save Changes"**

### 4. Deploy the Bot

1. Click **"Create Service"** or **"Deploy"**
2. Render will:
   - Clone your repository
   - Install dependencies from `bot_requirements.txt`
   - Start the Discord bot
3. Check the **"Logs"** tab to verify the bot started successfully

### 5. Verify Bot is Running

You should see in the logs:
```
✅ BotName#1234 has connected to Discord!
🤖 OLMo Training Bot Online!
```

And receive a DM from the bot:
```
🤖 OLMo Training Bot Online!
I'll monitor your W&B runs and send you updates.
Commands: /status, /metrics, /list, /stop
```

## Managing Deployments

### View Logs
1. Go to Render dashboard
2. Click on your service
3. Click **"Logs"** tab

### Restart Bot
1. Go to your service
2. Click **"Manual Deploy"** → **"Deploy"**

### Update Bot Code
1. Make changes locally
2. Push to GitHub:
```bash
git add .
git commit -m "Update bot features"
git push
```
3. Render will automatically redeploy (if autoDeploy is enabled)

## Cost Breakdown

### Render Pricing (as of 2024)
- **Starter Plan**: $7/month
  - 512MB RAM
  - 0.5 CPU
  - Perfect for Discord bots
  
- **Standard Plan**: $25/month
  - 2GB RAM
  - 1 CPU
  - For heavier workloads

### Optional Add-ons
- **Persistent Disk**: $0.25/GB/month
- **PostgreSQL Database**: Free tier available

## Monitoring and Maintenance

### Health Monitoring
The bot will automatically restart if it crashes. Monitor health via:
- Render dashboard metrics
- Discord bot status (should show as online)
- W&B run tracking

### Updating Dependencies
To update bot dependencies:
1. Update `bot_requirements.txt`
2. Push to GitHub
3. Render will rebuild with new dependencies

### Scaling (if needed)
For multiple training runs or users:
1. Upgrade to Standard plan
2. Modify `render.yaml`:
```yaml
scaling:
  minInstances: 1
  maxInstances: 3
```

## Troubleshooting

### Bot Not Starting
- Check logs in Render dashboard
- Verify all environment variables are set
- Ensure `discord_bot.py` has no syntax errors

### Bot Not Sending Messages
- Verify `DISCORD_USER_ID` is correct
- Check bot has DM permissions
- Ensure MESSAGE CONTENT INTENT is enabled in Discord

### W&B Connection Issues
- Verify `WANDB_API_KEY` is correct
- Check `WANDB_ENTITY` matches your username
- Ensure W&B project exists

### Deployment Tracking Issues
- Check persistent disk is mounted
- Verify `deployments/` folder exists
- Check file permissions

## Alternative Deployment Options

If Render doesn't suit your needs:

### Free Alternatives
- **Replit**: Free tier with limitations
- **Railway**: Free trial credits
- **Fly.io**: Free tier available

### Paid Alternatives
- **Heroku**: $7/month for Eco dynos
- **DigitalOcean App Platform**: $5/month
- **AWS EC2**: Pay-per-use

## Security Best Practices

1. **Never commit secrets** to GitHub
2. **Use environment variables** for all sensitive data
3. **Rotate tokens regularly**
4. **Set up 2FA** on all accounts
5. **Monitor bot activity** for unusual behavior

## Support

- Render Documentation: https://render.com/docs
- Discord.py Help: https://discordpy.readthedocs.io
- W&B Support: https://docs.wandb.ai

## Quick Reference

### Render CLI Commands
```bash
# Install Render CLI (optional)
brew install render/render/render

# Deploy from CLI
render deploy

# View logs
render logs --tail
```

### Useful Links
- [Render Dashboard](https://dashboard.render.com)
- [Render Status Page](https://status.render.com)
- [Render Community](https://community.render.com)

## Next Steps

1. ✅ Deploy bot to Render
2. 📊 Monitor first training run
3. 🔧 Adjust bot settings as needed
4. 🚀 Scale if handling multiple runs
5. 💾 Consider database for long-term storage

---

**Note**: Remember to deploy your training jobs to Modal separately using `deploy_training.py`. The Render deployment only handles the Discord bot monitoring component.
