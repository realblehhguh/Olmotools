"""
Local test script to validate the setup before deploying to Modal.
This tests data loading and model initialization locally.
"""

import sys
import torch
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test if we can load and process the dataset."""
    logger.info("Testing data loading...")
    
    try:
        from core.data_utils import WizardLMDataProcessor
        from transformers import AutoTokenizer
        
        # Load a small tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create data processor
        processor = WizardLMDataProcessor(
            tokenizer=tokenizer,
            max_length=512,
            dataset_name="QuixiAI/WizardLM_alpaca_evol_instruct_70k_unfiltered"
        )
        
        # Load a small sample
        dataset = processor.load_dataset(split="train", sample_size=5)
        logger.info(f"✓ Successfully loaded {len(dataset)} samples")
        
        # Test formatting
        example = dataset[0]
        formatted = processor.format_instruction(example)
        logger.info(f"✓ Successfully formatted example")
        logger.info(f"  Example (first 200 chars): {formatted[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False


def test_model_config():
    """Test model configuration utilities."""
    logger.info("Testing model configuration...")
    
    try:
        from core.model_utils import create_lora_config, create_training_arguments
        
        # Test LoRA config creation
        lora_config = create_lora_config()
        logger.info(f"✓ Created LoRA config with r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        # Test training arguments
        training_args = create_training_arguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2
        )
        logger.info(f"✓ Created training arguments")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model configuration failed: {e}")
        return False


def test_deepspeed_config():
    """Test DeepSpeed configuration loading."""
    logger.info("Testing DeepSpeed configuration...")
    
    try:
        import json
        from pathlib import Path
        
        config_path = Path("configs/deepspeed_config.json")
        if not config_path.exists():
            logger.error(f"✗ DeepSpeed config not found at {config_path}")
            return False
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Check key configurations
        assert "zero_optimization" in config
        assert "fp16" in config
        assert "optimizer" in config
        
        logger.info(f"✓ DeepSpeed config loaded successfully")
        logger.info(f"  ZeRO Stage: {config['zero_optimization']['stage']}")
        logger.info(f"  FP16 Enabled: {config['fp16']['enabled']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DeepSpeed config test failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability and CUDA setup."""
    logger.info("Testing GPU availability...")
    
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA is available")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        logger.warning("⚠ CUDA not available - training will be slow on CPU")
        logger.info("  This is expected if testing locally without GPU")
        return True


def test_dependencies():
    """Test if all required dependencies are installed."""
    logger.info("Testing dependencies...")
    
    required_packages = [
        "torch",
        "transformers",
        "deepspeed",
        "datasets",
        "accelerate",
        "peft",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.error(f"✗ {package} is not installed")
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Running OLMo Fine-tuning Setup Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("GPU Availability", test_gpu_availability),
        ("DeepSpeed Config", test_deepspeed_config),
        ("Model Configuration", test_model_config),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed! Ready to deploy to Modal.")
        logger.info("\nNext steps:")
        logger.info("1. Set up Modal secrets (see README.md)")
        logger.info("2. Run: modal deploy modal_app.py")
        logger.info("3. Start training: modal run modal_app.py --action train")
    else:
        logger.info("\n❌ Some tests failed. Please fix the issues before deploying.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
