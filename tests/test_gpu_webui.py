#!/usr/bin/env python3
"""
Test script for GPU configuration in the Web UI.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'web_ui'))

from training_deployer import get_gpu_recommendations, get_gpu_types

def test_gpu_recommendations():
    """Test GPU recommendations functionality."""
    print("Testing GPU recommendations...")
    
    # Test training recommendations
    training_recs = get_gpu_recommendations("training")
    assert "budget" in training_recs
    assert "balanced" in training_recs
    assert "performance" in training_recs
    assert "maximum" in training_recs
    
    # Verify structure
    for tier, config in training_recs.items():
        assert "type" in config
        assert "count" in config
        assert "description" in config
        print(f"  {tier}: {config['type']}:{config['count']} - {config['description']}")
    
    # Test inference recommendations
    inference_recs = get_gpu_recommendations("inference")
    assert len(inference_recs) > 0
    
    # Test development recommendations
    dev_recs = get_gpu_recommendations("development")
    assert len(dev_recs) > 0
    
    print("✓ GPU recommendations test passed")


def test_gpu_types():
    """Test GPU types functionality."""
    print("\nTesting GPU types...")
    
    gpu_types = get_gpu_types()
    assert len(gpu_types) > 0
    
    # Verify structure
    for gpu in gpu_types:
        assert "value" in gpu
        assert "label" in gpu
        assert "max_count" in gpu
        print(f"  {gpu['value']}: {gpu['label']} (max: {gpu['max_count']})")
    
    # Check for specific GPU types
    gpu_values = [gpu["value"] for gpu in gpu_types]
    expected_gpus = ["T4", "L4", "A10", "A100", "H100", "B200"]
    for expected in expected_gpus:
        assert expected in gpu_values, f"Expected GPU type {expected} not found"
    
    print("✓ GPU types test passed")


def test_gpu_validation():
    """Test GPU configuration validation logic."""
    print("\nTesting GPU validation logic...")
    
    # Test A10 limit
    a10_gpu = next((gpu for gpu in get_gpu_types() if gpu["value"] == "A10"), None)
    assert a10_gpu is not None
    assert a10_gpu["max_count"] == 4
    
    # Test other GPU limits
    h100_gpu = next((gpu for gpu in get_gpu_types() if gpu["value"] == "H100"), None)
    assert h100_gpu is not None
    assert h100_gpu["max_count"] == 8
    
    print("✓ GPU validation test passed")


def main():
    """Run all tests."""
    print("Running GPU Web UI tests...\n")
    
    try:
        test_gpu_recommendations()
        test_gpu_types()
        test_gpu_validation()
        
        print("\n🎉 All tests passed! GPU Web UI integration is working correctly.")
        
        # Print summary
        print("\n" + "="*60)
        print("GPU CONFIGURATION SUMMARY")
        print("="*60)
        
        print("\nAvailable GPU Types:")
        for gpu in get_gpu_types():
            print(f"  • {gpu['value']}: {gpu['label']} (max: {gpu['max_count']} GPUs)")
        
        print("\nTraining Recommendations:")
        training_recs = get_gpu_recommendations("training")
        for tier, config in training_recs.items():
            print(f"  • {tier.title()}: {config['type']}:{config['count']} - {config['description']}")
        
        print("\nInference Recommendations:")
        inference_recs = get_gpu_recommendations("inference")
        for tier, config in inference_recs.items():
            print(f"  • {tier.title()}: {config['type']}:{config['count']} - {config['description']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
