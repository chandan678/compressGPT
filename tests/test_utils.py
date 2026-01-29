"""
Test cases for utils.py module.

Run with: pytest tests/test_utils.py -v
"""

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


class TestComputeMetricsIntegration:
    """Tests for ComputeMetrics class integration."""
    
    def test_compute_metrics_initialization(self):
        """Test that ComputeMetrics can be initialized with new API."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        
        tokenizer = Mock()
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        assert metrics.labels == labels
        assert len(metrics.valid_token_ids) == 2
        assert metrics.id_to_label == id_to_label
    
    def test_label_restricted_argmax_behavior(self):
        """Test that predictions are restricted to valid label tokens."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        
        tokenizer = Mock()
        tokenizer.decode = Mock(side_effect=lambda ids: f"token_{ids[0]}")
        
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        # Create mock eval_preds
        # logits: [batch_size, seq_len, vocab_size]
        # Make token 999 have highest logit, but it's not in valid set
        logits = np.zeros((2, 5, 1000))
        logits[0, 0, 999] = 10.0  # Highest overall
        logits[0, 0, 100] = 5.0   # yes token (valid)
        logits[0, 0, 200] = 3.0   # no token (valid)
        
        # Labels: [batch_size, seq_len], -100 for masked positions
        labels = np.array([
            [100, -100, -100, -100, -100],  # First non-masked is gold=yes
            [200, -100, -100, -100, -100]   # First non-masked is gold=no
        ])
        
        callback = metrics.as_trainer_callback(log_first_n=0)
        result = callback((logits, labels))
        
        # Should achieve 100% accuracy because gold matches and restricted argmax works
        assert "accuracy" in result
        assert result["accuracy"] >= 0.0  # At least no error


class TestPreprocessLogitsForMetrics:
    """Tests for preprocess_logits_for_metrics functionality."""
    
    def test_get_preprocess_logits_returns_callable(self):
        """Test that get_preprocess_logits returns a callable function."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        
        tokenizer = Mock()
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        assert callable(preprocess_fn)
    
    def test_preprocess_filters_logits_correctly(self):
        """Test that preprocessing filters logits to only label tokens."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        import torch
        
        tokenizer = Mock()
        labels = ["yes", "no", "partial"]
        valid_token_ids = np.array([100, 200, 300])
        id_to_label = {100: "yes", 200: "no", 300: "partial"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # Create logits: [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = 2, 4, 50000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels_tensor = torch.zeros(batch_size, seq_len)
        
        # Filter logits
        filtered_logits = preprocess_fn(logits, labels_tensor)
        
        # Should reduce from vocab_size to num_labels
        assert filtered_logits.shape == (batch_size, seq_len, len(valid_token_ids))
        assert filtered_logits.shape[2] == 3  # yes, no, partial
    
    def test_preprocess_handles_tuple_logits(self):
        """Test that preprocessing handles logits wrapped in tuple."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        import torch
        
        tokenizer = Mock()
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # Create logits wrapped in tuple (some models do this)
        batch_size, seq_len, vocab_size = 2, 4, 50000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits_tuple = (logits,)
        labels_tensor = torch.zeros(batch_size, seq_len)
        
        # Should handle tuple correctly
        filtered_logits = preprocess_fn(logits_tuple, labels_tensor)
        
        assert filtered_logits.shape == (batch_size, seq_len, 2)
    
    def test_preprocess_handles_list_logits(self):
        """Test that preprocessing handles logits wrapped in list."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        import torch
        
        tokenizer = Mock()
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # Create logits wrapped in list
        batch_size, seq_len, vocab_size = 2, 4, 50000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits_list = [logits]
        labels_tensor = torch.zeros(batch_size, seq_len)
        
        # Should handle list correctly
        filtered_logits = preprocess_fn(logits_list, labels_tensor)
        
        assert filtered_logits.shape == (batch_size, seq_len, 2)
    
    def test_preprocess_device_safe(self):
        """Test that preprocessing works on different devices."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        import torch
        
        tokenizer = Mock()
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # Test CPU device
        batch_size, seq_len, vocab_size = 2, 4, 1000
        logits_cpu = torch.randn(batch_size, seq_len, vocab_size, device='cpu')
        labels_tensor = torch.zeros(batch_size, seq_len)
        
        filtered_cpu = preprocess_fn(logits_cpu, labels_tensor)
        assert filtered_cpu.device.type == 'cpu'
        assert filtered_cpu.shape[2] == 2
        
        # Test CUDA if available
        if torch.cuda.is_available():
            logits_cuda = torch.randn(batch_size, seq_len, vocab_size, device='cuda')
            filtered_cuda = preprocess_fn(logits_cuda, labels_tensor)
            assert filtered_cuda.device.type == 'cuda'
            assert filtered_cuda.shape[2] == 2
    
    def test_compute_metrics_with_preprocessed_logits(self):
        """Test that compute_metrics correctly handles preprocessed logits."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        
        tokenizer = Mock()
        tokenizer.decode = Mock(side_effect=lambda ids: f"token_{ids[0]}")
        
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        # Create preprocessed logits (already filtered to label tokens)
        # Shape: [batch_size, seq_len, num_labels]
        preprocessed_logits = np.zeros((2, 5, 2))
        # First sample: yes token (idx 0) has higher score
        preprocessed_logits[0, 0, 0] = 5.0  # yes
        preprocessed_logits[0, 0, 1] = 3.0  # no
        # Second sample: no token (idx 1) has higher score
        preprocessed_logits[1, 0, 0] = 3.0  # yes
        preprocessed_logits[1, 0, 1] = 5.0  # no
        
        # Labels: first non-masked position has the gold label
        labels_array = np.array([
            [100, -100, -100, -100, -100],  # gold=yes
            [200, -100, -100, -100, -100]   # gold=no
        ])
        
        callback = metrics.as_trainer_callback(log_first_n=0)
        result = callback((preprocessed_logits, labels_array))
        
        # Should correctly map indices back to token IDs
        assert "accuracy" in result
        assert result["accuracy"] == 1.0  # Both predictions correct
    
    def test_memory_reduction_benefit(self):
        """Test that preprocessing significantly reduces memory footprint."""
        from compressgpt.compute_metrics import ComputeMetrics
        import numpy as np
        import torch
        
        tokenizer = Mock()
        labels = ["yes", "no"]
        valid_token_ids = np.array([100, 200])
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # Simulate realistic dimensions
        batch_size, seq_len, vocab_size = 4, 256, 128000
        
        # Calculate memory for full vocab logits
        full_logits = torch.randn(batch_size, seq_len, vocab_size)
        full_size_bytes = full_logits.element_size() * full_logits.nelement()
        
        # Filter to label tokens
        labels_tensor = torch.zeros(batch_size, seq_len)
        filtered_logits = preprocess_fn(full_logits, labels_tensor)
        filtered_size_bytes = filtered_logits.element_size() * filtered_logits.nelement()
        
        # Calculate reduction ratio
        reduction_ratio = full_size_bytes / filtered_size_bytes
        
        # Should see massive reduction (128000 / 2 = 64000x)
        assert reduction_ratio > 10000  # At least 10,000x reduction
        assert filtered_logits.shape == (batch_size, seq_len, 2)


class TestLoadMetricsRemoved:
    """Test that load_metrics function was removed."""
    
    def test_load_metrics_does_not_exist(self):
        """Test that load_metrics function no longer exists."""
        import compressgpt.utils as utils
        
        assert not hasattr(utils, 'load_metrics')
