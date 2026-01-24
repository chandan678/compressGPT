"""
Test cases for utils.py module.

Run with: pytest tests/test_utils_fixed.py -v
"""

import pytest
import torch
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from compressgpt.utils import (
    validate_label_tokens,
    validate_response_template,
    setup_data_collator,
    clear_gpu_memory,
    save_metrics,
    format_metrics_table
)


class TestValidateLabelTokens:
    """Tests for validate_label_tokens function."""
    
    def test_valid_single_tokens(self):
        """Test that single-token labels pass validation."""
        tokenizer = Mock()
        # Mock encode to return single token (with leading space)
        tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens: 
            [100] if text == " yes" else [200])
        tokenizer.decode = Mock(side_effect=lambda ids: " yes" if ids == [100] else " no")
        
        labels = ["yes", "no"]
        
        # Should not raise
        result = validate_label_tokens(tokenizer, labels)
        assert result == {"yes": 100, "no": 200}
    
    def test_multi_token_label_raises(self):
        """Test that multi-token labels raise ValueError."""
        tokenizer = Mock()
        # "maybe" encodes to 3 tokens
        tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens:
            [100] if text == " yes" else [150, 151, 152])
        tokenizer.decode = Mock(return_value="token")
        
        labels = ["yes", "maybe"]
        
        with pytest.raises(ValueError, match="Label validation failed"):
            validate_label_tokens(tokenizer, labels)
    
    def test_empty_label_list_passes(self):
        """Test that empty label list returns empty dict."""
        tokenizer = Mock()
        
        result = validate_label_tokens(tokenizer, [])
        assert result == {}


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
    
    def test_returns_collator(self):
        """Test that function returns a data collator."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        collator = setup_data_collator(tokenizer, "Answer:")
        
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


class TestLoadMetricsRemoved:
    """Test that load_metrics function was removed."""
    
    def test_load_metrics_does_not_exist(self):
        """Test that load_metrics function no longer exists."""
        import compressgpt.utils as utils
        
        assert not hasattr(utils, 'load_metrics')
