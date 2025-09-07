# Setting up UptimeRobot for Discord Bot Monitoring

This guide explains how to monitor your Discord bot deployed on Render using UptimeRobot's free tier.

## Why UptimeRobot?

- **Free Monitoring**: 50 monitors on free plan
- **5-Minute Checks**: Pings your bot every 5 minutes
- **Email/SMS Alerts**: Get notified when bot goes down
- **Uptime Statistics**: Track bot availability over time
- **Prevents Idling**: Keeps free-tier services active

## Prerequisites

1. Discord bot deployed on Render (using `discord_bot_with_health.py`)
2. Render web service URL (e.g., `https://olmo-discord-bot.onrender.com`)
3. UptimeRobot account (free)

## Step 1: Deploy Bot with Health Check

1. **Ensure you're using the health check version**:
   - Use `discord_bot_with_health.py` instead of `discord_bot.py`
   - Update `render.yaml` to use web service type
   - Deploy to Render

2. **Get your Render URL**:
   - Go to Render dashboard
   - Click on your service
   - Copy the URL (e.g., `https://olmo-discord-bot.onrender.com`)

## Step 2: Create UptimeRobot Account

1. Go to https://uptimerobot.com
2. Click "Register for FREE"
3. Verify your email
4. Log in to dashboard

## Step 3: Add Health Check Monitor

1. **Click "Add New Monitor"**

2. **Configure Monitor Settings**:
   ```
   Monitor Type: HTTP(s)
   Friendly Name: OLMo Discord Bot Health
   URL: https://your-bot.onrender.com/health
   Monitoring Interval: 5 minutes
   Monitor Timeout: 30 seconds
   ```

3. **Advanced Settings** (Optional):
   ```
   HTTP Method: GET
   Alert Contacts: Your email (selected by default)
   Custom HTTP Headers: (leave empty)
   Keyword: OK (optional - checks response contains "OK")
   ```

4. **Click "Create Monitor"**

## Step 4: Add Status Page Monitor

For more detailed monitoring, add a second monitor for the `/status` endpoint:

1. **Click "Add New Monitor"**

2. **Configure**:
   ```
   Monitor Type: HTTP(s)
   Friendly Name: OLMo Bot Status
   URL: https://your-bot.onrender.com/status
   Monitoring Interval: 5 minutes
   ```

3. **Keyword Monitoring**:
   ```
   Keyword Type: Exists
   Keyword: "discord_connected":true
   ```

This checks that the bot is connected to Discord.

## Step 5: Configure Alerts

1. **Go to "My Settings"** → **"Alert Contacts"**

2. **Add Alert Methods**:
   - Email (already configured)
   - SMS (requires phone verification)
   - Webhook (send to another Discord channel)
   - Slack/Telegram/etc.

3. **Alert Settings**:
   - Send alert when down for: 0 minutes (immediate)
   - Send reminder every: 0 (only once)

## Step 6: Create Public Status Page (Optional)

1. **Go to "Status Pages"**
2. **Click "Add New Status Page"**
3. **Configure**:
   ```
   Page Name: OLMo Training Bot Status
   Custom Domain: (optional)
   Monitors: Select both health monitors
   ```
4. **Share the public URL** with your team

## Health Check Endpoints

Your bot provides these endpoints:

### `/health` - Simple Health Check
- **Success (200)**: `OK - Discord bot is running`
- **Error (503)**: `ERROR - Bot status: [status]`
- Used for basic up/down monitoring

### `/status` - Detailed Status
Returns JSON with:
```json
{
  "status": "running",
  "discord_connected": true,
  "uptime_seconds": 3600,
  "uptime_formatted": "1:00:00",
  "active_training_runs": 2,
  "runs": ["experiment_1", "experiment_2"],
  "wandb_project": "your-entity/olmo-finetune-modal",
  "health": "healthy"
}
```

## Monitoring Dashboard

After setup, you'll see:

1. **Response Time Graph**: Track latency
2. **Uptime Percentage**: Daily/Weekly/Monthly statistics
3. **Incident History**: When bot was down
4. **Response Time**: Average response times

## Alert Examples

You'll receive alerts like:

**Down Alert**:
```
Monitor is DOWN: OLMo Discord Bot Health
URL: https://olmo-discord-bot.onrender.com/health
Reason: Connection timeout
Time: 2024-01-15 10:30:45 PST
```

**Up Alert**:
```
Monitor is UP: OLMo Discord Bot Health
URL: https://olmo-discord-bot.onrender.com/health
Downtime: 2 minutes
Time: 2024-01-15 10:32:45 PST
```

## Best Practices

1. **Multiple Endpoints**: Monitor both `/health` and `/status`
2. **Keyword Monitoring**: Check for specific response content
3. **Reasonable Intervals**: 5 minutes is good for free tier
4. **Multiple Alert Channels**: Email + SMS for critical alerts
5. **Status Page**: Share with team for transparency

## Troubleshooting

### Bot Shows as Down but Is Running
- Check Render logs for errors
- Verify health check port matches Render's PORT env
- Ensure `discord_bot_with_health.py` is being used
- Check firewall/security settings

### Too Many False Alerts
- Increase monitor timeout (30-60 seconds)
- Check Render service isn't sleeping
- Verify network stability

### Health Check Failing
- Test locally: `curl https://your-bot.onrender.com/health`
- Check Discord bot token is valid
- Verify all environment variables are set

## Integration with Discord

You can also send UptimeRobot alerts to Discord:

1. **Create Discord Webhook**:
   - Go to Discord Server Settings
   - Integrations → Webhooks → New Webhook
   - Copy webhook URL

2. **Add to UptimeRobot**:
   - Alert Contacts → Add Alert Contact
   - Type: Webhook
   - URL: Your Discord webhook URL
   - Method: POST

3. **Webhook will post**:
   - Bot down/up notifications
   - Response time warnings
   - SSL certificate expiry alerts

## Cost

**UptimeRobot Free Tier**:
- 50 monitors
- 5-minute intervals
- 2-month log retention
- Unlimited alert contacts

**Render Costs**:
- Web service (required for health endpoint): $7/month
- Persistent disk (optional): $0.25/GB/month

## Advanced Monitoring

For production use, consider:

1. **Custom Metrics**: Add training-specific metrics to `/status`
2. **Performance Monitoring**: Track response times
3. **Error Rate Monitoring**: Count failed W&B connections
4. **Resource Monitoring**: Track memory/CPU usage
5. **Multi-Region Monitoring**: Use UptimeRobot's multiple locations

## Summary

With this setup:
- ✅ Bot health monitored every 5 minutes
- ✅ Instant alerts when bot goes down
- ✅ Prevents Render from spinning down service
- ✅ Public status page for transparency
- ✅ Historical uptime data
- ✅ Free monitoring solution

Your Discord bot will now be monitored 24/7, ensuring you're immediately notified of any issues!
