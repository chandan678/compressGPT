"""
Test cases for vendored GGUF code from llama.cpp.

These tests verify the vendored gguf-py library works correctly and should
pass after every update from llama.cpp. They test:
1. Import paths work correctly
2. Key classes and functions exist
3. Model architecture detection works
4. Quantization types are available
5. Converter API is functional

Run after updating vendored code:
    pytest tests/test_gguf_vendor.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGgufVendorImports:
    """Test that vendored gguf modules can be imported."""

    def test_gguf_package_importable(self):
        """Test gguf package can be imported from vendor."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        import gguf
        assert gguf is not None
        assert hasattr(gguf, '__file__')
        assert 'gguf_vendor' in gguf.__file__

    def test_gguf_constants_importable(self):
        """Test gguf constants module exists with key enums."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf import constants
        
        # Key enums that must exist
        assert hasattr(constants, 'GGMLQuantizationType')
        assert hasattr(constants, 'GGUFValueType')

    def test_gguf_writer_importable(self):
        """Test GGUFWriter class exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf import GGUFWriter
        assert GGUFWriter is not None

    def test_gguf_reader_importable(self):
        """Test GGUFReader class exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf import GGUFReader
        assert GGUFReader is not None

    def test_convert_script_importable(self):
        """Test convert_hf_to_gguf.py can be imported."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        import convert_hf_to_gguf
        assert convert_hf_to_gguf is not None

    def test_model_base_class_exists(self):
        """Test ModelBase class exists in converter."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import ModelBase
        assert ModelBase is not None
        assert hasattr(ModelBase, '_model_classes')
        assert hasattr(ModelBase, 'from_model_architecture')

    def test_model_type_enum_exists(self):
        """Test ModelType enum exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import ModelType
        assert hasattr(ModelType, 'TEXT')
        assert hasattr(ModelType, 'MMPROJ')


class TestGgufQuantizationTypes:
    """Test quantization types are available."""

    def test_llama_file_type_enum_exists(self):
        """Test LlamaFileType enum exists with required types."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        import gguf
        
        assert hasattr(gguf, 'LlamaFileType')
        ftype = gguf.LlamaFileType
        
        # Required quantization types for compressgpt
        assert hasattr(ftype, 'ALL_F32')
        assert hasattr(ftype, 'MOSTLY_F16')
        assert hasattr(ftype, 'MOSTLY_BF16')
        assert hasattr(ftype, 'MOSTLY_Q8_0')

    def test_quants_module_exists(self):
        """Test quants module with quantization functions."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf import quants
        assert quants is not None
        
        # Should have quantize function
        assert hasattr(quants, 'quantize')


class TestModelArchitectureSupport:
    """Test model architecture detection and support."""

    def test_model_classes_registered(self):
        """Test that model classes are registered."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import ModelBase, ModelType
        
        text_models = ModelBase._model_classes[ModelType.TEXT]
        assert len(text_models) > 0, "No text models registered"

    def test_minimum_architecture_count(self):
        """Test at least 50 architectures are supported."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import ModelBase, ModelType
        
        text_models = ModelBase._model_classes[ModelType.TEXT]
        # Should have at least 50 model architectures
        assert len(text_models) >= 50, f"Expected >= 50 models, got {len(text_models)}"

    def test_common_architectures_supported(self):
        """Test common model architectures are supported."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import ModelBase, ModelType
        
        text_models = ModelBase._model_classes[ModelType.TEXT]
        
        # Common architectures that should always be supported
        expected_archs = [
            'LlamaForCausalLM',
            'MistralForCausalLM', 
            'Qwen2ForCausalLM',
            'PhiForCausalLM',
            'GPT2LMHeadModel',
        ]
        
        for arch in expected_archs:
            assert arch in text_models, f"Architecture {arch} not supported"

    def test_from_model_architecture_works(self):
        """Test from_model_architecture returns a class."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import ModelBase, ModelType
        
        model_class = ModelBase.from_model_architecture(
            'LlamaForCausalLM', 
            model_type=ModelType.TEXT
        )
        assert model_class is not None
        assert issubclass(model_class, ModelBase)

    def test_unsupported_architecture_raises(self):
        """Test unsupported architecture raises NotImplementedError."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import ModelBase, ModelType
        
        with pytest.raises(NotImplementedError):
            ModelBase.from_model_architecture(
                'NonExistentModelArchitecture',
                model_type=ModelType.TEXT
            )


class TestGetModelArchitecture:
    """Test get_model_architecture function."""

    def test_function_exists(self):
        """Test get_model_architecture function exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import get_model_architecture
        assert callable(get_model_architecture)

    def test_extracts_architecture_from_hparams(self):
        """Test architecture extraction from config."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from convert_hf_to_gguf import get_model_architecture, ModelType
        
        # Mock hparams dict like from config.json
        hparams = {
            'architectures': ['LlamaForCausalLM'],
            'model_type': 'llama'
        }
        
        arch = get_model_architecture(hparams, ModelType.TEXT)
        assert arch == 'LlamaForCausalLM'


class TestGgufConverterWrapper:
    """Test the gguf_converter.py wrapper module."""

    def test_converter_module_importable(self):
        """Test gguf_converter module can be imported."""
        from compressgpt.gguf_converter import (
            convert_to_gguf,
            check_model_supported,
            get_supported_architectures,
        )
        
        assert callable(convert_to_gguf)
        assert callable(check_model_supported)
        assert callable(get_supported_architectures)

    def test_get_supported_architectures_returns_list(self):
        """Test get_supported_architectures returns architecture list."""
        from compressgpt.gguf_converter import get_supported_architectures
        
        archs = get_supported_architectures()
        assert isinstance(archs, list)
        assert len(archs) > 50

    def test_check_model_supported_returns_tuple(self):
        """Test check_model_supported returns (bool, str) tuple."""
        from compressgpt.gguf_converter import check_model_supported
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # No config.json - should return (False, error)
            is_supported, message = check_model_supported(tmpdir)
            assert isinstance(is_supported, bool)
            assert isinstance(message, str)
            assert is_supported is False
            assert "config.json" in message

    def test_convert_to_gguf_validates_model_path(self):
        """Test convert_to_gguf raises on invalid path."""
        from compressgpt.gguf_converter import convert_to_gguf
        
        with pytest.raises(FileNotFoundError):
            convert_to_gguf("/nonexistent/path")

    def test_convert_to_gguf_validates_quantization(self):
        """Test convert_to_gguf raises on invalid quantization."""
        from compressgpt.gguf_converter import convert_to_gguf
        import tempfile
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal config.json
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
            
            with pytest.raises(ValueError, match="Invalid quantization"):
                convert_to_gguf(tmpdir, quantization="invalid_quant")


class TestVendorVersionTracking:
    """Test version tracking files exist."""

    def test_version_file_exists(self):
        """Test VERSION file exists in vendor directory."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        version_file = vendor_dir / "VERSION"
        
        assert version_file.exists(), "VERSION file missing from gguf_vendor"
        
        content = version_file.read_text()
        assert "llama.cpp" in content.lower() or "gguf" in content.lower()

    def test_license_file_exists(self):
        """Test LICENSE file exists in vendor directory."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        license_file = vendor_dir / "LICENSE"
        
        assert license_file.exists(), "LICENSE file missing from gguf_vendor"
        
        content = license_file.read_text()
        assert "MIT" in content

    def test_init_file_exists(self):
        """Test __init__.py exists in vendor directory."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        init_file = vendor_dir / "__init__.py"
        
        assert init_file.exists(), "__init__.py missing from gguf_vendor"


class TestVendorDirectoryStructure:
    """Test vendor directory has expected structure."""

    def test_gguf_subdirectory_exists(self):
        """Test gguf/ subdirectory exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        gguf_dir = vendor_dir / "gguf"
        
        assert gguf_dir.exists()
        assert gguf_dir.is_dir()

    def test_convert_script_exists(self):
        """Test convert_hf_to_gguf.py exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        convert_script = vendor_dir / "convert_hf_to_gguf.py"
        
        assert convert_script.exists()

    def test_required_gguf_modules_exist(self):
        """Test required gguf modules exist."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        gguf_dir = vendor_dir / "gguf"
        
        required_files = [
            "__init__.py",
            "constants.py",
            "gguf_writer.py",
            "gguf_reader.py",
            "quants.py",
            "tensor_mapping.py",
            "vocab.py",
            "lazy.py",
            "utility.py",
        ]
        
        for filename in required_files:
            filepath = gguf_dir / filename
            assert filepath.exists(), f"Required file {filename} missing from gguf/"


class TestTensorMappingSupport:
    """Test tensor mapping for model conversion."""

    def test_tensor_mapping_module_exists(self):
        """Test tensor_mapping module exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf import tensor_mapping
        assert tensor_mapping is not None

    def test_tensor_names_class_exists(self):
        """Test TensorNameMap class exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf.tensor_mapping import TensorNameMap
        assert TensorNameMap is not None


class TestVocabSupport:
    """Test vocabulary handling support."""

    def test_vocab_module_exists(self):
        """Test vocab module exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf import vocab
        assert vocab is not None

    def test_vocab_types_exist(self):
        """Test vocab types are defined."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf.vocab import MistralTokenizerType, MistralVocab
        assert MistralTokenizerType is not None
        assert MistralVocab is not None


class TestQuantizationFunctionality:
    """Test quantization works correctly."""

    def test_quantize_function_exists(self):
        """Test quantize function exists in quants module."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf.quants import quantize
        assert callable(quantize)

    def test_q8_0_quantization_type_exists(self):
        """Test Q8_0 quantization type exists."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        from gguf.constants import GGMLQuantizationType
        assert hasattr(GGMLQuantizationType, 'Q8_0')

    def test_basic_quantization_works(self):
        """Test basic quantization with numpy array."""
        vendor_dir = Path(__file__).parent.parent / "compressgpt" / "gguf_vendor"
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        
        import numpy as np
        from gguf.quants import quantize
        from gguf.constants import GGMLQuantizationType
        
        # Create a small test array (must be multiple of 32 for Q8_0)
        data = np.random.randn(32).astype(np.float32)
        
        # This should not raise
        quantized = quantize(data, GGMLQuantizationType.Q8_0)
        assert quantized is not None
        assert len(quantized) > 0
