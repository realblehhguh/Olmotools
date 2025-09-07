#!/usr/bin/env python3
"""
Web UI for deploying OLMo training jobs to Modal via Render.
Provides a user-friendly interface for configuring and launching training.
"""

import os
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import threading
import queue
import uuid

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_cors import CORS
import modal

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app, origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','))

# Configuration
DEPLOY_API_KEY = os.environ.get('DEPLOY_API_KEY', 'change-me-in-production')
DEPLOYMENT_HISTORY_FILE = 'deployment_history.json'
MAX_HISTORY_ITEMS = 50

# Queue for deployment status updates
status_queue = queue.Queue()

# Active deployments tracking
active_deployments = {}


def load_deployment_history() -> List[Dict]:
    """Load deployment history from file."""
    if os.path.exists(DEPLOYMENT_HISTORY_FILE):
        try:
            with open(DEPLOYMENT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_deployment_history(history: List[Dict]):
    """Save deployment history to file."""
    # Keep only the last MAX_HISTORY_ITEMS
    history = history[-MAX_HISTORY_ITEMS:]
    with open(DEPLOYMENT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def verify_api_key(api_key: str) -> bool:
    """Verify the deployment API key."""
    return api_key == DEPLOY_API_KEY


def run_deployment(deployment_id: str, config: Dict):
    """Run the deployment in a background thread."""
    try:
        # Update status
        active_deployments[deployment_id]['status'] = 'running'
        active_deployments[deployment_id]['start_time'] = datetime.now().isoformat()
        
        # Import deployment function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from deployment.deploy_modal import deploy_training
        
        # Run deployment
        result = deploy_training(
            model_name=config.get('model_name', 'allenai/OLMo-2-1124-7B'),
            num_epochs=config.get('num_epochs', 3),
            batch_size=config.get('batch_size', 4),
            learning_rate=config.get('learning_rate', 2e-5),
            max_length=config.get('max_length', 2048),
            use_lora=config.get('use_lora', True),
            use_4bit=config.get('use_4bit', False),
            train_sample_size=config.get('train_sample_size'),
            run_name=config.get('run_name', f"web_deploy_{deployment_id[:8]}")
        )
        
        # Update status on success
        active_deployments[deployment_id]['status'] = 'completed'
        active_deployments[deployment_id]['result'] = result
        active_deployments[deployment_id]['end_time'] = datetime.now().isoformat()
        
        # Add to history
        history = load_deployment_history()
        history.append(active_deployments[deployment_id])
        save_deployment_history(history)
        
    except Exception as e:
        # Update status on error
        active_deployments[deployment_id]['status'] = 'failed'
        active_deployments[deployment_id]['error'] = str(e)
        active_deployments[deployment_id]['traceback'] = traceback.format_exc()
        active_deployments[deployment_id]['end_time'] = datetime.now().isoformat()
        
        # Add to history
        history = load_deployment_history()
        history.append(active_deployments[deployment_id])
        save_deployment_history(history)


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/deploy', methods=['POST'])
def deploy():
    """Deploy training job endpoint."""
    try:
        # Get form data or JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Verify API key
        api_key = data.get('api_key') or request.headers.get('X-API-Key')
        if not verify_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Parse configuration
        config = {
            'model_name': data.get('model_name', 'allenai/OLMo-2-1124-7B'),
            'num_epochs': int(data.get('num_epochs', 3)),
            'batch_size': int(data.get('batch_size', 4)),
            'learning_rate': float(data.get('learning_rate', 2e-5)),
            'max_length': int(data.get('max_length', 2048)),
            'use_lora': data.get('use_lora', 'true').lower() == 'true',
            'use_4bit': data.get('use_4bit', 'false').lower() == 'true',
            'train_sample_size': int(data.get('train_sample_size')) if data.get('train_sample_size') else None,
            'run_name': data.get('run_name', ''),
            'preset': data.get('preset', 'custom')
        }
        
        # Apply presets
        if config['preset'] == 'quick_test':
            config['train_sample_size'] = 100
            config['num_epochs'] = 1
            config['run_name'] = config['run_name'] or 'quick_test'
        elif config['preset'] == 'full_training':
            config['train_sample_size'] = None
            config['num_epochs'] = 3
            config['run_name'] = config['run_name'] or 'full_training'
        
        # Generate deployment ID
        deployment_id = str(uuid.uuid4())
        
        # Create deployment record
        deployment = {
            'id': deployment_id,
            'config': config,
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'user': data.get('user', 'web_ui')
        }
        
        # Add to active deployments
        active_deployments[deployment_id] = deployment
        
        # Start deployment in background thread
        thread = threading.Thread(
            target=run_deployment, 
            args=(deployment_id, config)
        )
        thread.daemon = True
        thread.start()
        
        # Return response
        if request.is_json:
            return jsonify({
                'success': True,
                'deployment_id': deployment_id,
                'message': 'Deployment started successfully'
            })
        else:
            flash('Deployment started successfully!', 'success')
            return redirect(url_for('status', deployment_id=deployment_id))
            
    except Exception as e:
        error_msg = f'Deployment failed: {str(e)}'
        if request.is_json:
            return jsonify({'error': error_msg}), 500
        else:
            flash(error_msg, 'error')
            return redirect(url_for('index'))


@app.route('/status')
@app.route('/status/<deployment_id>')
def status(deployment_id=None):
    """Status page for deployments."""
    if deployment_id:
        # Get specific deployment
        deployment = active_deployments.get(deployment_id)
        if not deployment:
            # Check history
            history = load_deployment_history()
            for item in history:
                if item['id'] == deployment_id:
                    deployment = item
                    break
        
        if deployment:
            return render_template('status.html', deployment=deployment)
        else:
            flash('Deployment not found', 'error')
            return redirect(url_for('status'))
    else:
        # Get all deployments
        history = load_deployment_history()
        return render_template('status.html', 
                             active=list(active_deployments.values()),
                             history=history[::-1])  # Reverse to show newest first


@app.route('/api/status/<deployment_id>')
def api_status(deployment_id):
    """API endpoint for deployment status."""
    deployment = active_deployments.get(deployment_id)
    if not deployment:
        # Check history
        history = load_deployment_history()
        for item in history:
            if item['id'] == deployment_id:
                deployment = item
                break
    
    if deployment:
        return jsonify(deployment)
    else:
        return jsonify({'error': 'Deployment not found'}), 404


@app.route('/api/deployments')
def api_deployments():
    """API endpoint for all deployments."""
    history = load_deployment_history()
    return jsonify({
        'active': list(active_deployments.values()),
        'history': history[::-1]
    })


@app.route('/health')
def health():
    """Health check endpoint for Render."""
    return jsonify({'status': 'healthy'}), 200


@app.route('/logs/<deployment_id>')
def logs(deployment_id):
    """View logs for a deployment."""
    # For now, redirect to Modal dashboard
    # In future, could implement log streaming
    return redirect('https://modal.com/apps')


if __name__ == '__main__':
    # Check for required environment variables
    required_vars = ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your Render dashboard or .env file")
        sys.exit(1)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
