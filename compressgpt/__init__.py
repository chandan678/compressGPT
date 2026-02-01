"""
compressGPT - LLM Compression and Optimization Library

This library automates LLM compression and optimization, providing tools for
building datasets, fine-tuning, and creating the smallest runnable models
that preserve target accuracy.
"""

from compressgpt.create_dataset import DatasetBuilder
from compressgpt.compute_metrics import ComputeMetrics
from compressgpt.model_runner import ModelRunner

# Lazy imports for optional dependencies (peft required for trainer)
def __getattr__(name):
    """Lazy import for modules with optional dependencies."""
    if name == "CompressTrainer":
        from compressgpt.trainer import CompressTrainer
        return CompressTrainer
    elif name in ("LoraConfig", "QLoraConfig", "TrainingConfig", 
                  "PipelineConfig", "QuantizationConfig", "DeploymentConfig"):
        from compressgpt import config as cfg
        return getattr(cfg, name)
    raise AttributeError(f"module 'compressgpt' has no attribute '{name}'")

__version__ = "0.2.0"
__all__ = [
    "DatasetBuilder",
    "ComputeMetrics",
    "ModelRunner",
    "CompressTrainer",
    "LoraConfig",
    "QLoraConfig",
    "TrainingConfig",
    "PipelineConfig",
    "QuantizationConfig",
    "DeploymentConfig",
]
