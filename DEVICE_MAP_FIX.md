# Device Map Fix for Distributed Training

## Problem Description

The error message:
```
You can't train a model that has been loaded with `device_map='auto'` in any distributed mode. Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`.
```

This error occurs when trying to use distributed training (like DeepSpeed) with a model that was loaded using `device_map='auto'`. The `device_map='auto'` parameter tells Transformers to automatically distribute the model across available devices, but this conflicts with distributed training frameworks that need to handle device placement themselves.

## Root Cause

In the original code, the model was always loaded with `device_map="auto"` regardless of whether distributed training was being used:

```python
# Original problematic code in model_utils.py
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # This causes conflicts with distributed training
    torch_dtype=torch_dtype,
    # ... other parameters
)
```

## Solution Implemented

The fix automatically detects distributed training environments and disables `device_map` when necessary:

### 1. Enhanced Model Loading Function

Modified `load_olmo_model_and_tokenizer()` in `core/model_utils.py`:

```python
def load_olmo_model_and_tokenizer(
    model_name: str = "allenai/OLMo-2-1124-7B",
    use_lora: bool = True,
    use_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    use_deepspeed: bool = False  # New parameter
) -> tuple:
    # Check for distributed training environment
    is_distributed = (
        use_deepspeed or 
        os.environ.get("WORLD_SIZE", "1") != "1" or
        os.environ.get("LOCAL_RANK") is not None or
        os.environ.get("RANK") is not None
    )
    
    if is_distributed:
        logger.info("Distributed training detected - disabling device_map to avoid conflicts")
        device_map = None
    
    # Only add device_map if not using distributed training
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    
    if device_map is not None:
        model_kwargs["device_map"] = device_map
```

### 2. Updated Training Function

Modified `train()` function in `core/train.py` to pass the `use_deepspeed` parameter:

```python
model, tokenizer = load_olmo_model_and_tokenizer(
    model_name=model_name,
    use_lora=use_lora,
    use_4bit=use_4bit,
    torch_dtype=torch.float16 if not use_4bit else torch.float16,
    use_deepspeed=use_deepspeed  # Pass DeepSpeed flag
)
```

## How the Fix Works

The solution automatically detects distributed training in several ways:

1. **Explicit DeepSpeed Flag**: When `use_deepspeed=True` is passed
2. **Environment Variables**: Checks for distributed training environment variables:
   - `WORLD_SIZE != "1"`: Indicates multi-process training
   - `LOCAL_RANK` is set: Indicates local process rank in distributed setup
   - `RANK` is set: Indicates global process rank in distributed setup

When any of these conditions are met, the fix:
- Sets `device_map = None` to disable automatic device mapping
- Logs a message indicating distributed training was detected
- Allows the distributed training framework (DeepSpeed) to handle device placement

## Training Scenarios Supported

### 1. Single GPU Training (Non-Distributed)
```bash
python core/train.py --use_deepspeed=False
```
- Uses `device_map="auto"` for automatic device placement
- Works with single GPU setups

### 2. DeepSpeed Single GPU Training
```bash
python core/train.py --use_deepspeed=True --deepspeed_config=configs/deepspeed_config_single_gpu.json
```
- Disables `device_map` automatically
- Uses DeepSpeed for memory optimization even on single GPU

### 3. DeepSpeed Multi-GPU Training
```bash
deepspeed core/train.py --deepspeed_config=configs/deepspeed_config.json
```
- Automatically detects distributed environment via environment variables
- Disables `device_map` to prevent conflicts
- Allows DeepSpeed to handle multi-GPU coordination

### 4. Modal Deployment
```bash
modal run core/modal_app.py --action=train --gpu_count=1
modal run core/modal_app.py --action=train --gpu_count=2
```
- Automatically handles device mapping based on GPU count
- Single GPU: May use `device_map="auto"`
- Multi-GPU: Disables `device_map` for DeepSpeed

## Testing the Fix

Run the test script to verify the fix works correctly:

```bash
cd modal_olmo_finetune
python test_device_map_fix.py
```

This will test:
1. Single GPU mode (should use `device_map="auto"`)
2. DeepSpeed mode (should disable `device_map`)
3. Simulated distributed environment (should disable `device_map`)

## Benefits of This Fix

1. **Automatic Detection**: No manual intervention required
2. **Backward Compatible**: Existing single GPU training continues to work
3. **Flexible**: Supports both distributed and non-distributed training
4. **Clear Logging**: Provides informative messages about device mapping decisions
5. **Comprehensive**: Handles multiple distributed training scenarios

## Alternative Solutions (Not Recommended)

The original error message suggested these workarounds:

1. **Force Single Process**: `--num_processes=1`
   - Limits training to single GPU only
   - Doesn't utilize multi-GPU capabilities

2. **Direct Python Execution**: `python myscript.py`
   - May not work with all distributed training setups
   - Doesn't address the root cause

Our implemented solution is superior because it:
- Automatically handles the device mapping conflict
- Preserves multi-GPU training capabilities
- Requires no manual configuration changes
- Works across different deployment environments

## Files Modified

1. `core/model_utils.py`: Enhanced model loading with distributed training detection
2. `core/train.py`: Updated to pass DeepSpeed flag to model loading
3. `test_device_map_fix.py`: Test script to verify the fix
4. `DEVICE_MAP_FIX.md`: This documentation file

## Verification

To verify the fix is working:

1. Check that single GPU training still works with `device_map="auto"`
2. Confirm that DeepSpeed training no longer produces the device_map error
3. Verify that distributed training environment variables are properly detected
4. Test both Modal and local training environments

The fix ensures seamless operation across all supported training configurations while maintaining optimal performance for each scenario.
