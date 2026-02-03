"""
Vendored GGUF conversion code from llama.cpp.

This module contains code from the llama.cpp project for converting
HuggingFace models to GGUF format with quantization support.

Source: https://github.com/ggml-org/llama.cpp
License: MIT
Version: gguf-py 0.17.1
"""

from pathlib import Path

# Make the vendored gguf package importable
VENDOR_DIR = Path(__file__).parent
GGUF_DIR = VENDOR_DIR / "gguf"

__version__ = "0.17.1"
__source__ = "https://github.com/ggml-org/llama.cpp"
