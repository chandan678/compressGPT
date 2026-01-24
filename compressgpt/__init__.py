"""
compressGPT - LLM Compression and Optimization Library

This library automates LLM compression and optimization, providing tools for
building datasets, fine-tuning, and creating the smallest runnable models
that preserve target accuracy.
"""

from compressgpt.create_dataset import DatasetBuilder
from compressgpt.compute_metrics import ComputeMetrics
from compressgpt.model_runner import ModelRunner

__version__ = "0.1.0"
__all__ = ["DatasetBuilder", "ComputeMetrics", "ModelRunner"]
