"""
Test cases for utils.py module.

Run with: pytest tests/test_utils.py -v
"""

import sys
import pytest
import torch
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from compressgpt.utils import (
    validate_response_template,
    setup_data_collator,
    clear_gpu_memory,
    save_metrics,
    format_metrics_table
)


class TestValidateResponseTemplate:
    """Tests for validate_response_template function."""
    
    def test_valid_response_template(self):
        """Test that valid response template passes."""
        # Should not raise
        validate_response_template("Answer:")
    
    def test_special_tokens_raise(self):
        """Test that response template with special tokens raises."""
        with pytest.raises(ValueError, match="special token pattern"):
            validate_response_template("<|start_header_id|>")
    
    def test_empty_template_raises(self):
        """Test that empty response template raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_response_template("")
    
    def test_allow_special_tokens(self):
        """Test that special tokens are allowed when flag is set."""
        # Should not raise
        validate_response_template("<|special|>", allow_special_tokens=True)


class TestSetupDataCollator:
    """Tests for setup_data_collator function."""
    
    @patch.dict('sys.modules', {'trl': MagicMock()})
    def test_returns_collator(self):
        """Test that function returns a data collator when trl is available."""
        import sys
        mock_trl = sys.modules['trl']
        mock_collator = MagicMock()
        mock_trl.DataCollatorForCompletionOnlyLM = MagicMock(return_value=mock_collator)
        
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        # Need to reimport to pick up the mocked trl
        from importlib import reload
        import compressgpt.utils
        reload(compressgpt.utils)
        
        collator = compressgpt.utils.setup_data_collator(tokenizer, "Answer:")
        
        assert collator is not None


class TestClearGpuMemory:
    """Tests for clear_gpu_memory function."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_clears_cuda_memory(self, mock_gc, mock_empty_cache, mock_cuda_available):
        """Test that CUDA memory is cleared when available."""
        mock_cuda_available.return_value = True
        
        clear_gpu_memory()
        
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('torch.mps.empty_cache')
    @patch('gc.collect')
    def test_clears_mps_memory(self, mock_gc, mock_mps_empty, mock_mps_available, mock_cuda_available):
        """Test that MPS memory is cleared when available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        
        clear_gpu_memory()
        
        mock_mps_empty.assert_called_once()
        mock_gc.assert_called_once()
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('gc.collect')
    def test_cpu_only_calls_gc(self, mock_gc, mock_mps_available, mock_cuda_available):
        """Test that only garbage collection runs on CPU."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        
        clear_gpu_memory()
        
        mock_gc.assert_called_once()


class TestSaveMetrics:
    """Tests for save_metrics function."""
    
    def test_saves_metrics_to_file(self):
        """Test that metrics are saved correctly to JSON."""
        metrics = {"accuracy": 0.95, "f1_macro": 0.92}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            save_metrics(metrics, temp_path)
            
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded == metrics
        finally:
            import os
            os.unlink(temp_path)
    
    def test_saves_with_sorted_keys(self):
        """Test that keys are sorted in saved JSON."""
        metrics = {"z_metric": 1, "a_metric": 2, "m_metric": 3}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            save_metrics(metrics, temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Check that keys appear in sorted order
            a_pos = content.index('a_metric')
            m_pos = content.index('m_metric')
            z_pos = content.index('z_metric')
            assert a_pos < m_pos < z_pos
        finally:
            import os
            os.unlink(temp_path)


class TestFormatMetricsTable:
    """Tests for format_metrics_table function."""
    
    def test_formats_basic_metrics(self):
        """Test basic metrics formatting."""
        metrics = {"accuracy": 0.95, "f1_macro": 0.92}
        
        result = format_metrics_table(metrics)
        
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "0.9500" in result
        assert "0.9200" in result
    
    def test_includes_stage_name(self):
        """Test that stage name appears in table."""
        metrics = {"accuracy": 0.95}
        
        result = format_metrics_table(metrics, stage_name="FT Stage")
        
        assert "FT Stage" in result
    
    def test_handles_non_numeric_values(self):
        """Test formatting of non-numeric values."""
        metrics = {"status": "completed", "accuracy": 0.95}
        
        result = format_metrics_table(metrics)
        
        assert "completed" in result


# Note: ComputeMetrics tests have been moved to test_compute_metrics.py
# See test_compute_metrics.py for comprehensive tests including:
# - TestComputeMetricsInit
# - TestInputTypeDetection  
# - TestLabelsToTokenIds
# - TestCompute
# - TestAsTrainerCallback
# - TestGetPreprocessLogits
# - TestConfusionMatrix
# - TestPrintReport


class TestSetupDataCollatorModelMode:
    """Tests for setup_data_collator model_mode functionality.
    
    This tests the fix for context-sensitive tokenization issues where
    "Answer:" tokenizes differently standalone vs. after a space.
    For instruct models, we now use the assistant header instead.
    """
    
    @patch('trl.DataCollatorForCompletionOnlyLM')
    def test_base_model_uses_response_template(self, mock_collator_class):
        """Test that base model mode uses the provided response template."""
        tokenizer = Mock()
        tokenizer.chat_template = None
        tokenizer.pad_token_id = 0
        
        setup_data_collator(
            tokenizer, 
            "Answer:",
            model_mode="base"
        )
        
        # For base models, response_template should be used directly
        mock_collator_class.assert_called_once()
        call_kwargs = mock_collator_class.call_args[1]
        assert call_kwargs['response_template'] == "Answer:"
    
    @patch('trl.DataCollatorForCompletionOnlyLM')
    def test_instruct_model_detects_llama3_header(self, mock_collator_class):
        """Test that instruct model detects Llama 3 assistant header."""
        tokenizer = Mock()
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>"
        tokenizer.pad_token_id = 0
        
        setup_data_collator(
            tokenizer, 
            "Answer:",  # This should be ignored for instruct
            model_mode="instruct"
        )
        
        # For Llama 3 instruct, should use assistant header
        mock_collator_class.assert_called_once()
        call_kwargs = mock_collator_class.call_args[1]
        assert "<|start_header_id|>assistant<|end_header_id|>" in call_kwargs['response_template']
    
    @patch('trl.DataCollatorForCompletionOnlyLM')
    def test_instruct_model_detects_chatml_header(self, mock_collator_class):
        """Test that instruct model detects ChatML assistant header."""
        tokenizer = Mock()
        tokenizer.chat_template = "<|im_start|>user\n{content}<|im_end|>"
        tokenizer.pad_token_id = 0
        
        setup_data_collator(
            tokenizer, 
            "Answer:",
            model_mode="instruct"
        )
        
        mock_collator_class.assert_called_once()
        call_kwargs = mock_collator_class.call_args[1]
        assert "<|im_start|>assistant" in call_kwargs['response_template']
    
    @patch('trl.DataCollatorForCompletionOnlyLM')
    def test_instruct_model_detects_llama2_header(self, mock_collator_class):
        """Test that instruct model detects Llama 2 [INST] header."""
        tokenizer = Mock()
        tokenizer.chat_template = "[INST] {content} [/INST]"
        tokenizer.pad_token_id = 0
        
        setup_data_collator(
            tokenizer, 
            "Answer:",
            model_mode="instruct"
        )
        
        mock_collator_class.assert_called_once()
        call_kwargs = mock_collator_class.call_args[1]
        assert "[/INST]" in call_kwargs['response_template']
    
    @patch('trl.DataCollatorForCompletionOnlyLM')
    def test_instruct_model_fallback_when_no_known_header(self, mock_collator_class):
        """Test that instruct model falls back when no known header detected."""
        import warnings
        
        tokenizer = Mock()
        tokenizer.chat_template = "some_unknown_template"
        tokenizer.pad_token_id = 0
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            setup_data_collator(
                tokenizer, 
                "Answer:",
                model_mode="instruct"
            )
            
            # Should warn about fallback
            assert len(w) >= 1
            assert "Could not detect assistant header" in str(w[0].message)
        
        # Falls back to stripped response template
        mock_collator_class.assert_called_once()
        call_kwargs = mock_collator_class.call_args[1]
        assert call_kwargs['response_template'] == "Answer:"
    
    @patch('trl.DataCollatorForCompletionOnlyLM')
    def test_default_model_mode_is_base(self, mock_collator_class):
        """Test that default model_mode is 'base'."""
        tokenizer = Mock()
        tokenizer.chat_template = "<|start_header_id|>assistant"  # Has instruct markers
        tokenizer.pad_token_id = 0
        
        # Default (no model_mode specified) should use base behavior
        setup_data_collator(tokenizer, "Answer:")
        
        # Should use provided template, not detect assistant header
        mock_collator_class.assert_called_once()
        call_kwargs = mock_collator_class.call_args[1]
        assert call_kwargs['response_template'] == "Answer:"


class TestLoadMetricsRemoved:
    """Test that load_metrics function was removed."""
    
    def test_load_metrics_does_not_exist(self):
        """Test that load_metrics function no longer exists."""
        import compressgpt.utils as utils
        
        assert not hasattr(utils, 'load_metrics')
