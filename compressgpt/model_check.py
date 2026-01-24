"""
Model compatibility checker for CompressGPT.

This module provides utilities to check if a model can run on the current machine
without actually loading the model weights. It estimates memory requirements and
provides recommendations for training and inference.

Usage:
    python -m compressgpt.model_check meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig


@dataclass
class GPUInfo:
    """GPU information."""
    available: bool
    device_type: str  # "cuda", "mps", or "none"
    device_name: Optional[str]
    total_memory_gb: float
    compute_capability: Optional[Tuple[int, int]]


@dataclass
class SystemInfo:
    """System information."""
    total_ram_gb: float
    cpu_count: int


@dataclass
class ModelInfo:
    """Model information extracted from config."""
    model_id: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    intermediate_size: Optional[int]
    estimated_parameters_b: float


@dataclass
class MemoryRequirements:
    """Memory requirements for different scenarios (in GB)."""
    inference_fp32: float
    inference_fp16: float
    inference_int8: float
    inference_int4: float
    training_full_fp32: float
    training_full_fp16: float
    training_lora_fp32: float
    training_lora_fp16: float
    training_qlora_int8: float
    training_qlora_int4: float


@dataclass
class CompatibilityResult:
    """Compatibility check results."""
    can_run_inference_fp32: bool
    can_run_inference_fp16: bool
    can_run_inference_int8: bool
    can_run_inference_int4: bool
    can_run_training_full_fp32: bool
    can_run_training_full_fp16: bool
    can_run_training_lora_fp32: bool
    can_run_training_lora_fp16: bool
    can_run_training_qlora_int8: bool
    can_run_training_qlora_int4: bool
    recommended_inference_batch_size: int
    recommended_training_batch_size: int
    recommended_gradient_accumulation_steps: int
    warnings: list[str]


def get_gpu_info() -> GPUInfo:
    """Get GPU information from the system."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        
        # Get compute capability
        compute_capability = torch.cuda.get_device_capability(0)
        
        return GPUInfo(
            available=True,
            device_type="cuda",
            device_name=device_name,
            total_memory_gb=total_memory_gb,
            compute_capability=compute_capability,
        )
    elif torch.backends.mps.is_available():
        # For Apple Silicon, estimate memory from system RAM
        # MPS shares unified memory with CPU
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
            # MPS can use about 70% of system RAM
            total_memory_gb = (total_ram / (1024 ** 3)) * 0.7
        except ImportError:
            # Fallback estimate for M1/M2/M3 machines
            total_memory_gb = 16.0  # Conservative estimate
        
        return GPUInfo(
            available=True,
            device_type="mps",
            device_name="Apple Silicon (MPS)",
            total_memory_gb=total_memory_gb,
            compute_capability=None,
        )
    else:
        return GPUInfo(
            available=False,
            device_type="none",
            device_name=None,
            total_memory_gb=0.0,
            compute_capability=None,
        )


def get_system_info() -> SystemInfo:
    """Get system information."""
    try:
        import psutil
        total_ram = psutil.virtual_memory().total
        total_ram_gb = total_ram / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=True)
    except ImportError:
        # Fallback estimates
        total_ram_gb = 16.0
        cpu_count = 8
    
    return SystemInfo(
        total_ram_gb=total_ram_gb,
        cpu_count=cpu_count,
    )


def get_model_info(model_id: str) -> ModelInfo:
    """
    Get model information from HuggingFace config without downloading weights.
    
    Args:
        model_id: HuggingFace model ID or local path
    
    Returns:
        ModelInfo with extracted parameters
    """
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Failed to load model config from {model_id}: {e}")
    
    # Extract common config parameters
    hidden_size = getattr(config, "hidden_size", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    vocab_size = getattr(config, "vocab_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    
    if not all([hidden_size, num_hidden_layers, vocab_size, num_attention_heads]):
        raise ValueError(f"Model config missing required parameters: {config}")
    
    # Estimate parameters
    estimated_params = estimate_parameters_from_config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
    )
    
    return ModelInfo(
        model_id=model_id,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        estimated_parameters_b=estimated_params / 1e9,
    )


def estimate_parameters_from_config(
    hidden_size: int,
    num_hidden_layers: int,
    vocab_size: int,
    intermediate_size: Optional[int] = None,
) -> int:
    """
    Estimate total model parameters from config.
    
    For transformer models, parameters come from:
    1. Embedding layer: vocab_size * hidden_size
    2. Each transformer layer:
       - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
       - FFN: 2 * hidden_size * intermediate_size
       - Layer norms: ~4 * hidden_size (small, often ignored)
    3. Output layer: vocab_size * hidden_size (often tied with embedding)
    
    Args:
        hidden_size: Model hidden dimension
        num_hidden_layers: Number of transformer layers
        vocab_size: Vocabulary size
        intermediate_size: FFN intermediate dimension (defaults to 4*hidden_size)
    
    Returns:
        Estimated total parameters
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    
    # Embedding layer
    embedding_params = vocab_size * hidden_size
    
    # Transformer layers
    attention_params_per_layer = 4 * (hidden_size ** 2)
    ffn_params_per_layer = 2 * hidden_size * intermediate_size
    layer_norm_params_per_layer = 4 * hidden_size  # Approximate
    
    params_per_layer = (
        attention_params_per_layer +
        ffn_params_per_layer +
        layer_norm_params_per_layer
    )
    
    total_layer_params = params_per_layer * num_hidden_layers
    
    # Output layer (often tied with embedding, but count separately for safety)
    output_params = vocab_size * hidden_size
    
    total_params = embedding_params + total_layer_params + output_params
    
    return total_params


def calculate_model_memory_requirements(
    num_parameters: float,
    lora_r: int = 16,
    lora_target_modules: int = 4,
) -> MemoryRequirements:
    """
    Calculate memory requirements for different scenarios.
    
    Memory formula:
    - Model weights: num_params * bytes_per_param
    - Inference: weights + activations (~20% of weights)
    - Training full: weights + gradients + optimizer (AdamW = 2x gradients)
    - Training LoRA: base_weights + lora_weights + lora_gradients + lora_optimizer
    - Training QLoRA: quantized_base + lora_weights + lora_gradients + lora_optimizer
    
    Args:
        num_parameters: Total model parameters (in billions)
        lora_r: LoRA rank
        lora_target_modules: Number of target modules for LoRA
    
    Returns:
        MemoryRequirements for different scenarios
    """
    num_params_full = num_parameters * 1e9
    
    # Bytes per parameter for different dtypes
    bytes_fp32 = 4
    bytes_fp16 = 2
    bytes_int8 = 1
    bytes_int4 = 0.5
    
    # Activation memory (approximate as 20% of model weights)
    activation_multiplier = 1.2
    
    # LoRA parameters: r * d * num_target_modules * 2 (in and out projections)
    # Assume average d = hidden_size (rough estimate)
    lora_params = 2 * lora_r * lora_target_modules * 1000  # Conservative estimate
    
    # Inference memory
    inference_fp32 = (num_params_full * bytes_fp32 * activation_multiplier) / (1024 ** 3)
    inference_fp16 = (num_params_full * bytes_fp16 * activation_multiplier) / (1024 ** 3)
    inference_int8 = (num_params_full * bytes_int8 * activation_multiplier) / (1024 ** 3)
    inference_int4 = (num_params_full * bytes_int4 * activation_multiplier) / (1024 ** 3)
    
    # Training full fine-tuning: weights + gradients + optimizer (AdamW = 2x params for states)
    # Total = 1x weights + 1x gradients + 2x optimizer = 4x weights
    training_full_fp32 = (num_params_full * bytes_fp32 * 4) / (1024 ** 3)
    training_full_fp16 = (num_params_full * bytes_fp16 * 4) / (1024 ** 3)
    
    # Training LoRA: base_weights (frozen) + lora_weights + lora_gradients + lora_optimizer
    # LoRA adds minimal parameters, so mainly base_weights cost
    training_lora_fp32 = (num_params_full * bytes_fp32 + lora_params * bytes_fp32 * 4) / (1024 ** 3)
    training_lora_fp16 = (num_params_full * bytes_fp16 + lora_params * bytes_fp16 * 4) / (1024 ** 3)
    
    # Training QLoRA: quantized_base + lora_weights + lora_gradients + lora_optimizer
    training_qlora_int8 = (num_params_full * bytes_int8 + lora_params * bytes_fp16 * 4) / (1024 ** 3)
    training_qlora_int4 = (num_params_full * bytes_int4 + lora_params * bytes_fp16 * 4) / (1024 ** 3)
    
    return MemoryRequirements(
        inference_fp32=inference_fp32,
        inference_fp16=inference_fp16,
        inference_int8=inference_int8,
        inference_int4=inference_int4,
        training_full_fp32=training_full_fp32,
        training_full_fp16=training_full_fp16,
        training_lora_fp32=training_lora_fp32,
        training_lora_fp16=training_lora_fp16,
        training_qlora_int8=training_qlora_int8,
        training_qlora_int4=training_qlora_int4,
    )


def check_model_compatibility(
    model_id: str,
    lora_r: int = 16,
    safety_margin: float = 0.8,
) -> Tuple[GPUInfo, SystemInfo, ModelInfo, MemoryRequirements, CompatibilityResult]:
    """
    Check if a model can run on the current system.
    
    Args:
        model_id: HuggingFace model ID
        lora_r: LoRA rank for estimation
        safety_margin: Use only this fraction of available memory (default 0.8 = 80%)
    
    Returns:
        Tuple of (gpu_info, system_info, model_info, memory_reqs, compatibility)
    """
    # Get system information
    gpu_info = get_gpu_info()
    system_info = get_system_info()
    model_info = get_model_info(model_id)
    
    # Calculate memory requirements
    memory_reqs = calculate_model_memory_requirements(
        num_parameters=model_info.estimated_parameters_b,
        lora_r=lora_r,
    )
    
    # Available memory with safety margin
    available_memory_gb = gpu_info.total_memory_gb * safety_margin if gpu_info.available else 0
    
    # Check compatibility for different scenarios
    warnings = []
    
    if not gpu_info.available:
        warnings.append("No GPU detected. Training will be extremely slow on CPU.")
    
    if gpu_info.device_type == "mps":
        warnings.append("Apple Silicon (MPS) detected. Some features may have limited support.")
        warnings.append("⚠️  CRITICAL: BitsAndBytes quantization (QLoRA) is NOT supported on MPS.")
        warnings.append("Only LoRA training (without quantization) will work on Apple Silicon.")
        warnings.append("For QLoRA training, you need a CUDA GPU.")
    
    # Compatibility checks
    can_run_inference_fp32 = memory_reqs.inference_fp32 <= available_memory_gb
    can_run_inference_fp16 = memory_reqs.inference_fp16 <= available_memory_gb
    can_run_inference_int8 = memory_reqs.inference_int8 <= available_memory_gb
    can_run_inference_int4 = memory_reqs.inference_int4 <= available_memory_gb
    
    can_run_training_full_fp32 = memory_reqs.training_full_fp32 <= available_memory_gb
    can_run_training_full_fp16 = memory_reqs.training_full_fp16 <= available_memory_gb
    can_run_training_lora_fp32 = memory_reqs.training_lora_fp32 <= available_memory_gb
    can_run_training_lora_fp16 = memory_reqs.training_lora_fp16 <= available_memory_gb
    
    # QLoRA is NOT supported on MPS, even if memory is sufficient
    if gpu_info.device_type == "mps":
        can_run_training_qlora_int8 = False
        can_run_training_qlora_int4 = False
    else:
        can_run_training_qlora_int8 = memory_reqs.training_qlora_int8 <= available_memory_gb
        can_run_training_qlora_int4 = memory_reqs.training_qlora_int4 <= available_memory_gb
    
    # Recommendations for batch size
    # Estimate: each sample uses ~1/1000 of model memory for inference
    if can_run_inference_fp16:
        available_for_batch = available_memory_gb - memory_reqs.inference_fp16
        recommended_inference_batch_size = max(1, int(available_for_batch / (memory_reqs.inference_fp16 / 100)))
    else:
        recommended_inference_batch_size = 1
    
    if can_run_training_lora_fp16:
        available_for_batch = available_memory_gb - memory_reqs.training_lora_fp16
        recommended_training_batch_size = max(1, int(available_for_batch / (memory_reqs.training_lora_fp16 / 50)))
    elif can_run_training_qlora_int4:
        available_for_batch = available_memory_gb - memory_reqs.training_qlora_int4
        recommended_training_batch_size = max(1, int(available_for_batch / (memory_reqs.training_qlora_int4 / 50)))
    else:
        recommended_training_batch_size = 1
    
    # Cap batch sizes at reasonable values
    recommended_inference_batch_size = min(recommended_inference_batch_size, 32)
    recommended_training_batch_size = min(recommended_training_batch_size, 8)
    
    # Gradient accumulation steps
    # If batch size is small, use gradient accumulation to reach effective batch size of 32
    target_effective_batch_size = 32
    if recommended_training_batch_size > 0:
        recommended_gradient_accumulation_steps = max(
            1,
            target_effective_batch_size // recommended_training_batch_size
        )
    else:
        recommended_gradient_accumulation_steps = 1
    
    compatibility = CompatibilityResult(
        can_run_inference_fp32=can_run_inference_fp32,
        can_run_inference_fp16=can_run_inference_fp16,
        can_run_inference_int8=can_run_inference_int8,
        can_run_inference_int4=can_run_inference_int4,
        can_run_training_full_fp32=can_run_training_full_fp32,
        can_run_training_full_fp16=can_run_training_full_fp16,
        can_run_training_lora_fp32=can_run_training_lora_fp32,
        can_run_training_lora_fp16=can_run_training_lora_fp16,
        can_run_training_qlora_int8=can_run_training_qlora_int8,
        can_run_training_qlora_int4=can_run_training_qlora_int4,
        recommended_inference_batch_size=recommended_inference_batch_size,
        recommended_training_batch_size=recommended_training_batch_size,
        recommended_gradient_accumulation_steps=recommended_gradient_accumulation_steps,
        warnings=warnings,
    )
    
    return gpu_info, system_info, model_info, memory_reqs, compatibility


def print_compatibility_report(
    gpu_info: GPUInfo,
    system_info: SystemInfo,
    model_info: ModelInfo,
    memory_reqs: MemoryRequirements,
    compatibility: CompatibilityResult,
):
    """Print a formatted compatibility report."""
    print("\n" + "=" * 80)
    print("CompressGPT Model Compatibility Report")
    print("=" * 80)
    
    # Model information
    print("\n📊 Model Information:")
    print(f"  Model ID: {model_info.model_id}")
    print(f"  Parameters: {model_info.estimated_parameters_b:.2f}B")
    print(f"  Hidden Size: {model_info.hidden_size}")
    print(f"  Layers: {model_info.num_hidden_layers}")
    print(f"  Vocab Size: {model_info.vocab_size}")
    
    # System information
    print("\n💻 System Information:")
    print(f"  GPU Available: {gpu_info.available}")
    if gpu_info.available:
        print(f"  GPU Type: {gpu_info.device_type.upper()}")
        print(f"  GPU Name: {gpu_info.device_name}")
        print(f"  GPU Memory: {gpu_info.total_memory_gb:.2f} GB")
        if gpu_info.compute_capability:
            print(f"  Compute Capability: {gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}")
    print(f"  System RAM: {system_info.total_ram_gb:.2f} GB")
    print(f"  CPU Cores: {system_info.cpu_count}")
    
    # Memory requirements
    print("\n💾 Memory Requirements:")
    print("  Inference:")
    print(f"    FP32: {memory_reqs.inference_fp32:.2f} GB")
    print(f"    FP16: {memory_reqs.inference_fp16:.2f} GB")
    print(f"    INT8: {memory_reqs.inference_int8:.2f} GB")
    print(f"    INT4: {memory_reqs.inference_int4:.2f} GB")
    print("  Training Full Fine-tuning:")
    print(f"    FP32: {memory_reqs.training_full_fp32:.2f} GB")
    print(f"    FP16: {memory_reqs.training_full_fp16:.2f} GB")
    print("  Training LoRA:")
    print(f"    FP32: {memory_reqs.training_lora_fp32:.2f} GB")
    print(f"    FP16: {memory_reqs.training_lora_fp16:.2f} GB")
    print("  Training QLoRA:")
    print(f"    INT8: {memory_reqs.training_qlora_int8:.2f} GB")
    print(f"    INT4: {memory_reqs.training_qlora_int4:.2f} GB")
    
    # Compatibility status
    print("\n✅ Compatibility Status:")
    print("  Inference:")
    print(f"    FP32: {'✓ Yes' if compatibility.can_run_inference_fp32 else '✗ No'}")
    print(f"    FP16: {'✓ Yes' if compatibility.can_run_inference_fp16 else '✗ No'}")
    print(f"    INT8: {'✓ Yes' if compatibility.can_run_inference_int8 else '✗ No'}")
    print(f"    INT4: {'✓ Yes' if compatibility.can_run_inference_int4 else '✗ No'}")
    print("  Training Full Fine-tuning:")
    print(f"    FP32: {'✓ Yes' if compatibility.can_run_training_full_fp32 else '✗ No'}")
    print(f"    FP16: {'✓ Yes' if compatibility.can_run_training_full_fp16 else '✗ No'}")
    print("  Training LoRA:")
    print(f"    FP32: {'✓ Yes' if compatibility.can_run_training_lora_fp32 else '✗ No'}")
    print(f"    FP16: {'✓ Yes' if compatibility.can_run_training_lora_fp16 else '✗ No'}")
    print("  Training QLoRA:")
    print(f"    INT8: {'✓ Yes' if compatibility.can_run_training_qlora_int8 else '✗ No'}")
    print(f"    INT4: {'✓ Yes' if compatibility.can_run_training_qlora_int4 else '✗ No'}")
    
    # Recommendations
    print("\n🎯 Recommendations:")
    print(f"  Inference Batch Size: {compatibility.recommended_inference_batch_size}")
    print(f"  Training Batch Size: {compatibility.recommended_training_batch_size}")
    print(f"  Gradient Accumulation Steps: {compatibility.recommended_gradient_accumulation_steps}")
    print(f"  Effective Training Batch Size: {compatibility.recommended_training_batch_size * compatibility.recommended_gradient_accumulation_steps}")
    
    # Warnings
    if compatibility.warnings:
        print("\n⚠️  Warnings:")
        for warning in compatibility.warnings:
            print(f"  - {warning}")
    
    # Summary recommendation
    print("\n📝 Summary:")
    if gpu_info.device_type == "mps":
        if compatibility.can_run_training_lora_fp16:
            print("  ✓ Your Apple Silicon Mac can run LoRA training with FP16 precision.")
            print("  ⚠️  QLoRA (quantization) is NOT supported on MPS - use --stages ft,merge")
        elif compatibility.can_run_training_lora_fp32:
            print("  ✓ Your Apple Silicon Mac can run LoRA training with FP32 precision.")
            print("  ⚠️  QLoRA (quantization) is NOT supported on MPS - use --stages ft,merge")
        elif compatibility.can_run_inference_fp16:
            print("  ✓ Your Apple Silicon Mac can run inference with FP16.")
            print("  ⚠️  Training may not be feasible. Consider using a cloud GPU.")
        else:
            print("  ✗ Your Apple Silicon Mac may not have enough memory for this model.")
            print("  💡 Consider using a smaller model or cloud GPU service.")
    elif compatibility.can_run_training_lora_fp16:
        print("  ✓ Your system can run LoRA training with FP16 precision.")
    elif compatibility.can_run_training_qlora_int4:
        print("  ✓ Your system can run QLoRA training with 4-bit quantization.")
        print("  💡 Use --qlora_bits 4 for memory-efficient training.")
    elif compatibility.can_run_inference_int4:
        print("  ✓ Your system can run inference with 4-bit quantization.")
        print("  ⚠️  Training may not be feasible. Consider using a cloud GPU.")
    else:
        print("  ✗ Your system may not have enough memory for this model.")
        print("  💡 Consider using a smaller model or cloud GPU service.")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check if a model can run on your system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m compressgpt.model_check meta-llama/Llama-3.2-1B-Instruct
  python -m compressgpt.model_check meta-llama/Llama-3.2-3B-Instruct --lora_r 32
  python -m compressgpt.model_check mistralai/Mistral-7B-v0.1 --json
        """
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank for estimation (default: 16)"
    )
    parser.add_argument(
        "--safety_margin",
        type=float,
        default=0.8,
        help="Use only this fraction of available memory (default: 0.8)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        # Check compatibility
        gpu_info, system_info, model_info, memory_reqs, compatibility = check_model_compatibility(
            model_id=args.model_id,
            lora_r=args.lora_r,
            safety_margin=args.safety_margin,
        )
        
        if args.json:
            # Output as JSON
            result = {
                "model_info": {
                    "model_id": model_info.model_id,
                    "parameters_b": model_info.estimated_parameters_b,
                    "hidden_size": model_info.hidden_size,
                    "num_layers": model_info.num_hidden_layers,
                    "vocab_size": model_info.vocab_size,
                },
                "system_info": {
                    "gpu_available": gpu_info.available,
                    "gpu_type": gpu_info.device_type,
                    "gpu_name": gpu_info.device_name,
                    "gpu_memory_gb": gpu_info.total_memory_gb,
                    "system_ram_gb": system_info.total_ram_gb,
                    "cpu_cores": system_info.cpu_count,
                },
                "memory_requirements": {
                    "inference_fp16_gb": memory_reqs.inference_fp16,
                    "inference_int4_gb": memory_reqs.inference_int4,
                    "training_lora_fp16_gb": memory_reqs.training_lora_fp16,
                    "training_qlora_int4_gb": memory_reqs.training_qlora_int4,
                },
                "compatibility": {
                    "can_run_inference": compatibility.can_run_inference_fp16,
                    "can_run_training_lora": compatibility.can_run_training_lora_fp16,
                    "can_run_training_qlora": compatibility.can_run_training_qlora_int4,
                    "recommended_inference_batch_size": compatibility.recommended_inference_batch_size,
                    "recommended_training_batch_size": compatibility.recommended_training_batch_size,
                    "recommended_gradient_accumulation_steps": compatibility.recommended_gradient_accumulation_steps,
                },
                "warnings": compatibility.warnings,
            }
            print(json.dumps(result, indent=2))
        else:
            # Print formatted report
            print_compatibility_report(
                gpu_info=gpu_info,
                system_info=system_info,
                model_info=model_info,
                memory_reqs=memory_reqs,
                compatibility=compatibility,
            )
        
        # Exit with appropriate code
        if not any([
            compatibility.can_run_inference_int4,
            compatibility.can_run_training_qlora_int4,
        ]):
            sys.exit(1)  # Model cannot run
        else:
            sys.exit(0)  # Model can run
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
