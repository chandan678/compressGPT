"""GGUF conversion utilities using the bundled llama.cpp converter."""

import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)

# Supported quantization types that work with pure Python (no llama.cpp binary needed)
SUPPORTED_QUANT_TYPES = Literal["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"]


def convert_to_gguf(
    model_path: str | Path,
    output_path: Optional[str | Path] = None,
    quantization: SUPPORTED_QUANT_TYPES = "q8_0",
    model_name: Optional[str] = None,
    use_temp_file: bool = False,
    verbose: bool = False,
) -> Path:
    """Convert a HuggingFace model directory to GGUF."""
    model_path = Path(model_path)
    
    # Validate model path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"No config.json found in model directory: {model_path}")
    
    # Validate quantization
    valid_quants = ["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"]
    if quantization not in valid_quants:
        raise ValueError(f"Invalid quantization '{quantization}'. Must be one of: {valid_quants}")
    
    # Determine output path
    if output_path is None:
        output_path = model_path / f"model-{quantization}.gguf"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting model: {model_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Quantization: {quantization}")
    
    # Import the vendored converter
    vendor_dir = Path(__file__).parent / "gguf_vendor"
    
    # Add vendor directory to path for imports
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    
    try:
        # Import gguf from vendor
        import gguf
        import convert_hf_to_gguf
        from convert_hf_to_gguf import ModelBase, get_model_architecture, ModelType
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        # Map quantization string to gguf file type
        ftype_map = {
            "f32": gguf.LlamaFileType.ALL_F32,
            "f16": gguf.LlamaFileType.MOSTLY_F16,
            "bf16": gguf.LlamaFileType.MOSTLY_BF16,
            "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
            "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
            "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
            "auto": gguf.LlamaFileType.GUESSED,
        }
        
        output_type = ftype_map[quantization]
        
        # Run conversion
        with torch.inference_mode():
            hparams = ModelBase.load_hparams(model_path, is_mistral_format=False)
            model_architecture = get_model_architecture(hparams, ModelType.TEXT)
            logger.info(f"Detected architecture: {model_architecture}")
            
            try:
                model_class = ModelBase.from_model_architecture(model_architecture, model_type=ModelType.TEXT)
            except NotImplementedError:
                raise RuntimeError(f"Model architecture '{model_architecture}' is not supported for GGUF conversion")
            
            model_instance = model_class(
                dir_model=model_path,
                ftype=output_type,
                fname_out=output_path,
                is_big_endian=False,
                use_temp_file=use_temp_file,
                eager=False,  # Use lazy evaluation
                metadata_override=None,
                model_name=model_name,
                split_max_tensors=0,
                split_max_size=0,
                dry_run=False,
                small_first_shard=False,
                remote_hf_model_id=None,
            )
            
            logger.info("Writing GGUF file...")
            model_instance.write()
            
        logger.info(f"✅ Model successfully converted to: {output_path}")
        return output_path
        
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import GGUF converter. This may indicate missing dependencies. "
            f"Try: pip install sentencepiece protobuf\n"
            f"Error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"GGUF conversion failed: {e}") from e


def get_supported_architectures() -> list[str]:
    """Return supported model architectures for GGUF conversion."""
    vendor_dir = Path(__file__).parent / "gguf_vendor"
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    
    try:
        import convert_hf_to_gguf
        from convert_hf_to_gguf import ModelBase, ModelType
        return list(ModelBase._model_classes[ModelType.TEXT].keys())
    except ImportError:
        return []


def check_model_supported(model_path: str | Path) -> tuple[bool, str]:
    """Check if a model directory is supported for GGUF conversion."""
    model_path = Path(model_path)
    
    if not (model_path / "config.json").exists():
        return False, "No config.json found"
    
    vendor_dir = Path(__file__).parent / "gguf_vendor"
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    
    try:
        import convert_hf_to_gguf
        from convert_hf_to_gguf import ModelBase, get_model_architecture, ModelType
        
        hparams = ModelBase.load_hparams(model_path, is_mistral_format=False)
        architecture = get_model_architecture(hparams, ModelType.TEXT)
        
        try:
            ModelBase.from_model_architecture(architecture, model_type=ModelType.TEXT)
            return True, architecture
        except NotImplementedError:
            return False, f"Architecture '{architecture}' not supported"
            
    except Exception as e:
        return False, str(e)
