// OLMo Training Deployer - JavaScript functionality

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg max-w-sm z-50 fade-in`;
    
    const bgColor = type === 'success' ? 'bg-green-500' : 
                    type === 'error' ? 'bg-red-500' : 
                    type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500';
    
    notification.classList.add(bgColor, 'text-white');
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-${type === 'success' ? 'check-circle' : 
                            type === 'error' ? 'exclamation-circle' : 
                            type === 'warning' ? 'exclamation-triangle' : 
                            'info-circle'} mr-2"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" 
                    class="ml-4 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Form validation
function validateDeploymentForm() {
    const form = document.getElementById('deploymentForm');
    if (!form) return true;
    
    const apiKey = form.querySelector('#api_key').value;
    if (!apiKey) {
        showNotification('Please enter your API key', 'error');
        return false;
    }
    
    const epochs = parseInt(form.querySelector('#num_epochs').value);
    if (epochs < 1 || epochs > 10) {
        showNotification('Epochs must be between 1 and 10', 'error');
        return false;
    }
    
    const learningRate = parseFloat(form.querySelector('#learning_rate').value);
    if (learningRate <= 0 || learningRate > 0.001) {
        showNotification('Learning rate must be between 0 and 0.001', 'error');
        return false;
    }
    
    return true;
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(err => {
        showNotification('Failed to copy', 'error');
    });
}

// Status polling for active deployments
let statusPollingInterval = null;

function startStatusPolling(deploymentId) {
    if (statusPollingInterval) {
        clearInterval(statusPollingInterval);
    }
    
    statusPollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${deploymentId}`);
            const data = await response.json();
            
            if (data.status !== 'running' && data.status !== 'queued') {
                clearInterval(statusPollingInterval);
                // Refresh the page to show final status
                window.location.reload();
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 5000); // Poll every 5 seconds
}

// API deployment function
async function deployViaAPI(config) {
    try {
        const response = await fetch('/deploy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': config.api_key
            },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Deployment started successfully!', 'success');
            // Redirect to status page
            window.location.href = `/status/${data.deployment_id}`;
        } else {
            showNotification(data.error || 'Deployment failed', 'error');
        }
    } catch (error) {
        showNotification('Network error: ' + error.message, 'error');
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+Enter to submit form
    if (e.ctrlKey && e.key === 'Enter') {
        const form = document.getElementById('deploymentForm');
        if (form && validateDeploymentForm()) {
            form.submit();
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        const modal = document.getElementById('loadingModal');
        if (modal && !modal.classList.contains('hidden')) {
            modal.classList.add('hidden');
        }
    }
});

// Initialize tooltips
function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(element => {
        const tooltipText = element.getAttribute('data-tooltip');
        const tooltip = document.createElement('span');
        tooltip.className = 'tooltiptext';
        tooltip.textContent = tooltipText;
        element.classList.add('tooltip');
        element.appendChild(tooltip);
    });
}

// Handle preset configuration
function applyPresetConfig(preset) {
    const configs = {
        quick_test: {
            num_epochs: 1,
            batch_size: 4,
            train_sample_size: 100,
            run_name_prefix: 'quick_test'
        },
        full_training: {
            num_epochs: 3,
            batch_size: 4,
            train_sample_size: '',
            run_name_prefix: 'full_training'
        }
    };
    
    if (configs[preset]) {
        const config = configs[preset];
        const form = document.getElementById('deploymentForm');
        
        Object.keys(config).forEach(key => {
            if (key === 'run_name_prefix') {
                const runNameInput = form.querySelector('#run_name');
                if (runNameInput) {
                    runNameInput.value = `${config[key]}_${Date.now()}`;
                }
            } else {
                const input = form.querySelector(`#${key}`);
                if (input) {
                    input.value = config[key];
                    // Update visual feedback for range inputs
                    if (input.type === 'range') {
                        const valueDisplay = document.getElementById(`${key}_value`);
                        if (valueDisplay) {
                            valueDisplay.textContent = config[key];
                        }
                    }
                }
            }
        });
        
        showNotification(`Applied ${preset.replace('_', ' ')} preset`, 'success');
    }
}

// Real-time form updates
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips
    initTooltips();
    
    // Handle range input updates
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        input.addEventListener('input', (e) => {
            const valueDisplay = document.getElementById(`${e.target.id}_value`);
            if (valueDisplay) {
                valueDisplay.textContent = e.target.value;
            }
        });
    });
    
    // Handle checkbox state persistence
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        // Load saved state from localStorage
        const savedState = localStorage.getItem(`checkbox_${checkbox.id}`);
        if (savedState !== null) {
            checkbox.checked = savedState === 'true';
        }
        
        // Save state on change
        checkbox.addEventListener('change', (e) => {
            localStorage.setItem(`checkbox_${e.target.id}`, e.target.checked);
        });
    });
    
    // Auto-save form data
    const form = document.getElementById('deploymentForm');
    if (form) {
        // Load saved form data
        const savedFormData = localStorage.getItem('deploymentFormData');
        if (savedFormData) {
            const data = JSON.parse(savedFormData);
            Object.keys(data).forEach(key => {
                const input = form.querySelector(`[name="${key}"]`);
                if (input && input.type !== 'password') {
                    input.value = data[key];
                }
            });
        }
        
        // Save form data on change (except passwords)
        form.addEventListener('change', () => {
            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                if (key !== 'api_key') {
                    data[key] = value;
                }
            });
            localStorage.setItem('deploymentFormData', JSON.stringify(data));
        });
    }
    
    // Handle status page auto-refresh
    const deploymentStatus = document.querySelector('[data-deployment-status]');
    if (deploymentStatus) {
        const status = deploymentStatus.getAttribute('data-deployment-status');
        const deploymentId = deploymentStatus.getAttribute('data-deployment-id');
        
        if (status === 'running' || status === 'queued') {
            startStatusPolling(deploymentId);
        }
    }
});

// Export functions for use in HTML
window.showNotification = showNotification;
window.copyToClipboard = copyToClipboard;
window.applyPresetConfig = applyPresetConfig;
window.deployViaAPI = deployViaAPI;
window.validateDeploymentForm = validateDeploymentForm;
