#!/usr/bin/env python3
"""
Discord bot that monitors W&B training runs and sends DM updates.
"""

import discord
from discord.ext import commands, tasks
import wandb
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bot configuration
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_USER_ID = int(os.getenv("DISCORD_USER_ID", "0"))
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "iamhappyandfree-personalcompany")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "olmo-finetune-modal")

# Initialize W&B
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True

bot = commands.Bot(command_prefix='/', intents=intents)

# Track active runs
active_runs: Dict[str, dict] = {}
last_metrics: Dict[str, dict] = {}


class TrainingMonitor:
    """Monitor W&B training runs and track progress."""
    
    def __init__(self, run_name: str, project: str = WANDB_PROJECT, entity: str = WANDB_ENTITY):
        self.run_name = run_name
        self.project = project
        self.entity = entity
        self.api = wandb.Api()
        self.run = None
        self.last_step = 0
        self.last_epoch = 0
        
    def get_run(self):
        """Get or refresh the W&B run."""
        try:
            if not self.run:
                runs = self.api.runs(f"{self.entity}/{self.project}", 
                                    filters={"display_name": self.run_name})
                for run in runs:
                    if run.name == self.run_name or run.display_name == self.run_name:
                        self.run = run
                        return self.run
            else:
                # Refresh run data
                self.run.load(force=True)
            return self.run
        except Exception as e:
            print(f"Error getting run {self.run_name}: {e}")
            return None
    
    def get_metrics(self) -> Optional[dict]:
        """Get latest metrics from the run."""
        run = self.get_run()
        if not run:
            return None
        
        try:
            # Get summary metrics
            summary = run.summary._json_dict
            
            # Get history for latest metrics
            history = run.history()
            if not history.empty:
                latest = history.iloc[-1].to_dict()
                
                # Extract key metrics
                metrics = {
                    "state": run.state,
                    "step": latest.get("_step", 0),
                    "epoch": latest.get("epoch", 0),
                    "loss": latest.get("loss", None),
                    "eval_loss": latest.get("eval_loss", None),
                    "learning_rate": latest.get("learning_rate", None),
                    "runtime": run.summary.get("_runtime", 0),
                    "progress": self._calculate_progress(run, latest),
                }
                
                # Add GPU metrics if available
                if "gpu_memory_allocated" in latest:
                    metrics["gpu_memory"] = latest["gpu_memory_allocated"]
                
                return metrics
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return None
    
    def _calculate_progress(self, run, latest) -> float:
        """Calculate training progress percentage."""
        try:
            # Try to get total steps from config
            config = run.config
            num_epochs = config.get("num_train_epochs", 3)
            current_epoch = latest.get("epoch", 0)
            
            if num_epochs > 0:
                return (current_epoch / num_epochs) * 100
            return 0
        except:
            return 0
    
    def format_status(self, metrics: dict) -> str:
        """Format metrics into a status message."""
        if not metrics:
            return "⏳ Waiting for metrics..."
        
        status = []
        
        # State emoji
        state_emoji = {
            "running": "🏃",
            "finished": "✅",
            "failed": "❌",
            "crashed": "💥",
        }.get(metrics["state"], "❓")
        
        status.append(f"{state_emoji} **Status**: {metrics['state'].upper()}")
        
        # Progress bar
        progress = metrics.get("progress", 0)
        bar_length = 20
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        status.append(f"**Progress**: [{bar}] {progress:.1f}%")
        
        # Metrics
        if metrics.get("epoch"):
            status.append(f"**Epoch**: {metrics['epoch']:.2f}")
        
        if metrics.get("loss"):
            status.append(f"**Loss**: {metrics['loss']:.4f}")
        
        if metrics.get("eval_loss"):
            status.append(f"**Eval Loss**: {metrics['eval_loss']:.4f}")
        
        if metrics.get("learning_rate"):
            status.append(f"**LR**: {metrics['learning_rate']:.2e}")
        
        if metrics.get("gpu_memory"):
            status.append(f"**GPU Memory**: {metrics['gpu_memory']:.1f}%")
        
        # Runtime
        runtime = metrics.get("runtime", 0)
        if runtime > 0:
            hours = runtime // 3600
            minutes = (runtime % 3600) // 60
            status.append(f"**Runtime**: {hours}h {minutes}m")
        
        return "\n".join(status)


@bot.event
async def on_ready():
    """Bot startup event."""
    print(f'✅ {bot.user} has connected to Discord!')
    
    # Start monitoring task
    monitor_runs.start()
    
    # Send startup message to user
    user = await bot.fetch_user(DISCORD_USER_ID)
    if user:
        await user.send("🤖 **OLMo Training Bot Online!**\n"
                       "I'll monitor your W&B runs and send you updates.\n"
                       "Commands: `/status`, `/metrics`, `/list`, `/stop`")


@tasks.loop(seconds=30)
async def monitor_runs():
    """Periodically check W&B runs and send updates."""
    if not DISCORD_USER_ID:
        return
    
    user = await bot.fetch_user(DISCORD_USER_ID)
    if not user:
        return
    
    # Check deployment files for new runs
    deployment_files = glob.glob("deployments/*.json")
    for file_path in deployment_files:
        try:
            with open(file_path, "r") as f:
                deployment = json.load(f)
            
            run_name = deployment["run_name"]
            
            # Skip if already monitoring
            if run_name in active_runs:
                continue
            
            # Start monitoring new run
            active_runs[run_name] = deployment
            monitor = TrainingMonitor(run_name)
            
            # Send initial notification
            embed = discord.Embed(
                title="🚀 Training Started",
                description=f"**Run**: {run_name}",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            config = deployment["config"]
            embed.add_field(name="Model", value=config["model_name"].split("/")[-1], inline=True)
            embed.add_field(name="Epochs", value=config["num_epochs"], inline=True)
            embed.add_field(name="Batch Size", value=config["batch_size"], inline=True)
            embed.add_field(name="GPUs", value="2x A100", inline=True)
            
            await user.send(embed=embed)
            
        except Exception as e:
            print(f"Error loading deployment {file_path}: {e}")
    
    # Monitor active runs
    for run_name, deployment in list(active_runs.items()):
        try:
            monitor = TrainingMonitor(run_name)
            metrics = monitor.get_metrics()
            
            if not metrics:
                continue
            
            # Check for significant updates
            should_update = False
            last = last_metrics.get(run_name, {})
            
            # Send update on state change
            if metrics["state"] != last.get("state"):
                should_update = True
            
            # Send update every epoch
            if metrics.get("epoch", 0) > last.get("epoch", 0):
                should_update = True
            
            # Send update on completion
            if metrics["state"] in ["finished", "failed", "crashed"]:
                should_update = True
            
            if should_update:
                # Create status embed
                color = {
                    "running": discord.Color.blue(),
                    "finished": discord.Color.green(),
                    "failed": discord.Color.red(),
                    "crashed": discord.Color.red(),
                }.get(metrics["state"], discord.Color.gray())
                
                embed = discord.Embed(
                    title=f"📊 Training Update: {run_name}",
                    description=monitor.format_status(metrics),
                    color=color,
                    timestamp=datetime.now()
                )
                
                # Add W&B link
                wandb_url = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_name}"
                embed.add_field(
                    name="Links",
                    value=f"[W&B Dashboard]({wandb_url})",
                    inline=False
                )
                
                await user.send(embed=embed)
                
                # Update last metrics
                last_metrics[run_name] = metrics
                
                # Remove from active if completed
                if metrics["state"] in ["finished", "failed", "crashed"]:
                    del active_runs[run_name]
                    
                    # Archive deployment file
                    os.rename(f"deployments/{run_name}.json", 
                             f"deployments/completed_{run_name}.json")
                    
        except Exception as e:
            print(f"Error monitoring run {run_name}: {e}")


@bot.command(name='status')
async def status(ctx):
    """Get status of all active training runs."""
    if not active_runs:
        await ctx.send("No active training runs.")
        return
    
    embed = discord.Embed(
        title="🏃 Active Training Runs",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    
    for run_name in active_runs:
        monitor = TrainingMonitor(run_name)
        metrics = monitor.get_metrics()
        
        if metrics:
            status_text = f"State: {metrics['state']}\n"
            status_text += f"Progress: {metrics.get('progress', 0):.1f}%\n"
            if metrics.get('loss'):
                status_text += f"Loss: {metrics['loss']:.4f}"
            
            embed.add_field(
                name=run_name,
                value=status_text,
                inline=False
            )
    
    await ctx.send(embed=embed)


@bot.command(name='metrics')
async def metrics(ctx, run_name: str = None):
    """Get detailed metrics for a specific run."""
    if not run_name and active_runs:
        run_name = list(active_runs.keys())[0]
    
    if not run_name:
        await ctx.send("No active runs. Specify a run name.")
        return
    
    monitor = TrainingMonitor(run_name)
    metrics = monitor.get_metrics()
    
    if not metrics:
        await ctx.send(f"Could not get metrics for run: {run_name}")
        return
    
    embed = discord.Embed(
        title=f"📊 Metrics: {run_name}",
        description=monitor.format_status(metrics),
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    
    await ctx.send(embed=embed)


@bot.command(name='list')
async def list_runs(ctx):
    """List all tracked runs."""
    deployments = glob.glob("deployments/*.json")
    
    if not deployments:
        await ctx.send("No training runs found.")
        return
    
    embed = discord.Embed(
        title="📋 All Training Runs",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    
    for file_path in deployments:
        try:
            with open(file_path, "r") as f:
                deployment = json.load(f)
            
            run_name = deployment["run_name"]
            start_time = deployment.get("start_time", "Unknown")
            
            status = "✅ Completed" if "completed_" in file_path else "🏃 Active"
            
            embed.add_field(
                name=run_name,
                value=f"Status: {status}\nStarted: {start_time}",
                inline=False
            )
        except:
            continue
    
    await ctx.send(embed=embed)


@bot.command(name='stop')
async def stop_monitoring(ctx, run_name: str = None):
    """Stop monitoring a specific run."""
    if not run_name:
        await ctx.send("Please specify a run name to stop monitoring.")
        return
    
    if run_name in active_runs:
        del active_runs[run_name]
        await ctx.send(f"Stopped monitoring: {run_name}")
    else:
        await ctx.send(f"Run not found: {run_name}")


def main():
    """Main entry point for Discord bot."""
    if not DISCORD_TOKEN:
        print("❌ DISCORD_BOT_TOKEN not found in environment variables!")
        print("Please create a .env file with:")
        print("  DISCORD_BOT_TOKEN=your-bot-token")
        print("  DISCORD_USER_ID=your-discord-user-id")
        print("  WANDB_API_KEY=your-wandb-api-key")
        return
    
    if not DISCORD_USER_ID:
        print("❌ DISCORD_USER_ID not found in environment variables!")
        return
    
    print("🤖 Starting Discord bot...")
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
