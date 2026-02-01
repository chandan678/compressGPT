"""
Test cases for compute_metrics.py module.

Tests cover:
- ComputeMetrics initialization
- Input type detection (token IDs vs label strings)
- Metric computation (accuracy, F1, precision, recall)
- Label-restricted argmax in as_trainer_callback
- Position offset fix (logits[pos-1] predicts token at pos)
- Preprocess logits for memory reduction
- Confusion matrix generation

Run with: pytest tests/test_compute_metrics.py -v
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock


class TestComputeMetricsInit:
    """Tests for ComputeMetrics initialization."""
    
    def test_basic_initialization(self):
        """Test basic initialization with required arguments."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        labels = ["yes", "no"]
        valid_token_ids = [100, 200]
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
        )
        
        assert metrics.labels == labels
        assert len(metrics.valid_token_ids) == 2
        assert metrics.id_to_label == id_to_label
        assert metrics.tokenizer is None
    
    def test_initialization_with_tokenizer(self):
        """Test initialization with optional tokenizer."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        tokenizer = Mock()
        labels = ["yes", "no"]
        valid_token_ids = [100, 200]
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
            tokenizer=tokenizer,
        )
        
        assert metrics.tokenizer is tokenizer
    
    def test_valid_token_ids_converted_to_numpy(self):
        """Test that valid_token_ids is converted to numpy array."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        labels = ["yes", "no", "partial"]
        valid_token_ids = [100, 200, 300]
        id_to_label = {100: "yes", 200: "no", 300: "partial"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
        )
        
        assert isinstance(metrics.valid_token_ids, np.ndarray)
        assert list(metrics.valid_token_ids) == [100, 200, 300]
    
    def test_label_token_ids_reverse_mapping_created(self):
        """Test that label_token_ids mapping is created from id_to_label."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        labels = ["yes", "no"]
        valid_token_ids = [100, 200]
        id_to_label = {100: "yes", 200: "no"}
        
        metrics = ComputeMetrics(
            labels=labels,
            valid_token_ids=valid_token_ids,
            id_to_label=id_to_label,
        )
        
        assert metrics.label_token_ids == {"yes": 100, "no": 200}


class TestInputTypeDetection:
    """Tests for input type detection."""
    
    def test_detect_token_ids_input(self):
        """Test detection of token ID input."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        assert metrics._detect_input_type([100, 200, 100]) == "token_ids"
    
    def test_detect_labels_input(self):
        """Test detection of label string input."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        assert metrics._detect_input_type(["yes", "no", "yes"]) == "labels"
    
    def test_detect_empty_input(self):
        """Test detection of empty input."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        assert metrics._detect_input_type([]) == "empty"
    
    def test_detect_unknown_input(self):
        """Test detection of unknown input type."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        assert metrics._detect_input_type([1.5, 2.5]) == "unknown"


class TestLabelsToTokenIds:
    """Tests for label string to token ID conversion."""
    
    def test_converts_valid_labels(self):
        """Test conversion of valid labels to token IDs."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no", "partial"],
            valid_token_ids=[100, 200, 300],
            id_to_label={100: "yes", 200: "no", 300: "partial"},
        )
        
        result = metrics._labels_to_token_ids(["yes", "no", "partial", "yes"])
        assert result == [100, 200, 300, 100]
    
    def test_unknown_label_returns_negative_one(self):
        """Test that unknown labels return -1."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        result = metrics._labels_to_token_ids(["yes", "unknown", "no"])
        assert result == [100, -1, 200]


class TestCompute:
    """Tests for main compute method."""
    
    def test_compute_with_token_ids(self):
        """Test compute with token ID inputs."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        predictions = [100, 100, 200, 200]
        gold_labels = [100, 200, 200, 100]  # 2 correct, 2 wrong
        
        result = metrics.compute(predictions, gold_labels)
        
        assert result["accuracy"] == 0.5
        assert "f1_macro" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "f1_yes" in result
        assert "f1_no" in result
    
    def test_compute_with_label_strings(self):
        """Test compute with label string inputs."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        predictions = ["yes", "yes", "no", "no"]
        gold_labels = ["yes", "no", "no", "yes"]
        
        result = metrics.compute(predictions, gold_labels)
        
        assert result["accuracy"] == 0.5
    
    def test_compute_mixed_inputs(self):
        """Test compute with mixed input types (labels for preds, IDs for gold)."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        predictions = ["yes", "no"]
        gold_labels = [100, 200]
        
        result = metrics.compute(predictions, gold_labels)
        
        assert result["accuracy"] == 1.0
    
    def test_compute_perfect_accuracy(self):
        """Test compute with 100% accuracy."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        predictions = [100, 200, 100, 200]
        gold_labels = [100, 200, 100, 200]
        
        result = metrics.compute(predictions, gold_labels)
        
        assert result["accuracy"] == 1.0
        assert result["f1_macro"] == 1.0
    
    def test_compute_length_mismatch_raises(self):
        """Test that mismatched prediction/gold lengths raise error."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        with pytest.raises(ValueError, match="Length mismatch"):
            metrics.compute([100, 200], [100])


class TestAsTrainerCallback:
    """Tests for as_trainer_callback method (HuggingFace Trainer integration)."""
    
    def test_returns_callable(self):
        """Test that as_trainer_callback returns a callable."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        callback = metrics.as_trainer_callback()
        assert callable(callback)
    
    def test_label_restricted_argmax(self):
        """Test that predictions are restricted to valid label tokens only."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="token")
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=np.array([100, 200]),
            id_to_label={100: "yes", 200: "no"},
            tokenizer=tokenizer,
        )
        
        # Create logits where token 999 has highest value but is NOT in valid set
        # Token 100 (yes) has highest value among valid tokens
        logits = np.zeros((2, 5, 1000))
        logits[0, 1, 999] = 10.0  # Highest overall (invalid)
        logits[0, 1, 100] = 5.0   # yes token - highest valid
        logits[0, 1, 200] = 3.0   # no token
        logits[1, 1, 999] = 10.0  # Highest overall (invalid)
        logits[1, 1, 100] = 3.0   # yes token
        logits[1, 1, 200] = 5.0   # no token - highest valid
        
        # Labels at position 2 (predicting from logits[pos-1] = logits[1])
        labels = np.array([
            [-100, -100, 100, -100, -100],  # gold=yes
            [-100, -100, 200, -100, -100],  # gold=no
        ])
        
        callback = metrics.as_trainer_callback(log_first_n=0)
        result = callback((logits, labels))
        
        # Should get 100% accuracy because restricted argmax selects correct valid tokens
        assert result["accuracy"] == 1.0
    
    def test_position_offset_fix(self):
        """
        Test the critical position offset fix: logits[pos-1] predicts token at pos.
        
        In causal LM:
        - labels[i, pos] contains the gold token ID at position pos
        - logits[i, pos-1] contains the distribution to predict token at pos
        
        This test verifies the fix: we look at logits[pos-1], not logits[pos].
        """
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=np.array([100, 200]),
            id_to_label={100: "yes", 200: "no"},
        )
        
        # logits[pos=1] should predict token at pos=2
        # logits[pos=2] is a distractor with wrong prediction
        logits = np.zeros((1, 5, 1000))
        logits[0, 1, 100] = 10.0  # Correct: logits[1] predicts yes at pos 2
        logits[0, 1, 200] = 1.0
        logits[0, 2, 200] = 10.0  # Wrong: if we looked at logits[2], we'd get no
        logits[0, 2, 100] = 1.0
        
        labels = np.array([
            [-100, -100, 100, -100, -100],  # gold=yes at pos=2
        ])
        
        callback = metrics.as_trainer_callback(log_first_n=0)
        result = callback((logits, labels))
        
        # With correct offset (pos-1), we get yes→yes = correct
        # With wrong offset (pos), we would get no→yes = wrong
        assert result["accuracy"] == 1.0
    
    def test_handles_preprocessed_logits(self):
        """Test callback works with pre-filtered logits from preprocess_logits_for_metrics."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=np.array([100, 200]),
            id_to_label={100: "yes", 200: "no"},
        )
        
        # Pre-filtered logits: [batch, seq, num_labels] instead of [batch, seq, vocab]
        # Index 0 = yes (token 100), Index 1 = no (token 200)
        logits = np.zeros((2, 5, 2))  # Only 2 columns for 2 labels
        logits[0, 1, 0] = 5.0  # yes has highest logit
        logits[0, 1, 1] = 3.0
        logits[1, 1, 0] = 3.0
        logits[1, 1, 1] = 5.0  # no has highest logit
        
        labels = np.array([
            [-100, -100, 100, -100, -100],  # gold=yes
            [-100, -100, 200, -100, -100],  # gold=no
        ])
        
        callback = metrics.as_trainer_callback(log_first_n=0)
        result = callback((logits, labels))
        
        assert result["accuracy"] == 1.0
    
    def test_skips_samples_with_no_labels(self):
        """Test that samples with all -100 labels are skipped."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=np.array([100, 200]),
            id_to_label={100: "yes", 200: "no"},
        )
        
        logits = np.zeros((3, 5, 1000))
        logits[0, 1, 100] = 5.0  # Sample 0: predicts yes
        logits[1, 1, 100] = 5.0  # Sample 1: would predict yes but all masked
        logits[2, 1, 200] = 5.0  # Sample 2: predicts no
        
        labels = np.array([
            [-100, -100, 100, -100, -100],  # Valid, gold=yes, pred=yes ✓
            [-100, -100, -100, -100, -100], # All masked - should skip
            [-100, -100, 200, -100, -100],  # Valid, gold=no, pred=no ✓
        ])
        
        callback = metrics.as_trainer_callback(log_first_n=0)
        result = callback((logits, labels))
        
        # Only 2 valid samples (sample 1 skipped), both correct
        assert result["accuracy"] == 1.0
    
    def test_skips_gold_labels_outside_valid_set(self):
        """Test that samples with gold labels outside valid set are skipped."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=np.array([100, 200]),
            id_to_label={100: "yes", 200: "no"},
        )
        
        logits = np.zeros((2, 5, 1000))
        logits[:, 1, 100] = 5.0
        
        labels = np.array([
            [-100, -100, 100, -100, -100],  # Valid gold=yes
            [-100, -100, 999, -100, -100],  # Invalid gold=999 (not in valid set)
        ])
        
        callback = metrics.as_trainer_callback(log_first_n=0)
        result = callback((logits, labels))
        
        # Only 1 valid sample
        assert result["accuracy"] == 1.0


class TestGetPreprocessLogits:
    """Tests for get_preprocess_logits method (memory optimization)."""
    
    def test_returns_callable(self):
        """Test that get_preprocess_logits returns a callable."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        assert callable(preprocess_fn)
    
    def test_filters_to_label_tokens_only(self):
        """Test that logits are filtered to label tokens only."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no", "partial"],
            valid_token_ids=[100, 200, 300],
            id_to_label={100: "yes", 200: "no", 300: "partial"},
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # Full vocab logits
        batch_size, seq_len, vocab_size = 4, 10, 32000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.zeros(batch_size, seq_len)
        
        filtered = preprocess_fn(logits, labels)
        
        # Should reduce to 3 label tokens
        assert filtered.shape == (batch_size, seq_len, 3)
    
    def test_preserves_correct_logit_values(self):
        """Test that filtered logits preserve correct values."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        logits = torch.zeros(1, 3, 1000)
        logits[0, 0, 100] = 5.0  # yes
        logits[0, 0, 200] = 3.0  # no
        logits[0, 1, 100] = 2.0
        logits[0, 1, 200] = 7.0
        labels = torch.zeros(1, 3)
        
        filtered = preprocess_fn(logits, labels)
        
        assert filtered[0, 0, 0].item() == 5.0  # yes at index 0
        assert filtered[0, 0, 1].item() == 3.0  # no at index 1
        assert filtered[0, 1, 0].item() == 2.0
        assert filtered[0, 1, 1].item() == 7.0
    
    def test_handles_tuple_logits(self):
        """Test that tuple-wrapped logits are handled."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        logits = torch.randn(2, 5, 1000)
        logits_tuple = (logits,)
        labels = torch.zeros(2, 5)
        
        filtered = preprocess_fn(logits_tuple, labels)
        
        assert filtered.shape == (2, 5, 2)
    
    def test_handles_list_logits(self):
        """Test that list-wrapped logits are handled."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        logits = torch.randn(2, 5, 1000)
        logits_list = [logits]
        labels = torch.zeros(2, 5)
        
        filtered = preprocess_fn(logits_list, labels)
        
        assert filtered.shape == (2, 5, 2)
    
    def test_device_safe(self):
        """Test that filtering works on different devices."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # CPU test
        logits_cpu = torch.randn(2, 5, 1000)
        labels_cpu = torch.zeros(2, 5)
        filtered_cpu = preprocess_fn(logits_cpu, labels_cpu)
        
        assert filtered_cpu.device == logits_cpu.device
        assert filtered_cpu.shape == (2, 5, 2)
    
    def test_memory_reduction_benefit(self):
        """Test that filtering provides significant memory reduction."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no", "partial"],
            valid_token_ids=[100, 200, 300],
            id_to_label={100: "yes", 200: "no", 300: "partial"},
        )
        
        preprocess_fn = metrics.get_preprocess_logits()
        
        # Realistic dimensions
        batch_size, seq_len, vocab_size = 8, 512, 128256
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.zeros(batch_size, seq_len)
        
        filtered = preprocess_fn(logits, labels)
        
        # Memory reduction: 128256 -> 3 tokens
        reduction_factor = vocab_size / 3
        assert reduction_factor > 40000  # Massive reduction
        assert filtered.shape[-1] == 3


class TestConfusionMatrix:
    """Tests for get_confusion_matrix method."""
    
    def test_basic_confusion_matrix(self):
        """Test basic confusion matrix generation."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        predictions = [100, 100, 200, 200]
        gold_labels = [100, 200, 200, 100]
        
        result = metrics.get_confusion_matrix(predictions, gold_labels)
        
        assert "matrix" in result
        assert "labels" in result
        assert result["labels"] == ["yes", "no"]
        assert result["matrix"].shape == (2, 2)
    
    def test_confusion_matrix_with_label_strings(self):
        """Test confusion matrix with label string inputs."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        predictions = ["yes", "yes", "no", "no"]
        gold_labels = ["yes", "no", "no", "yes"]
        
        result = metrics.get_confusion_matrix(predictions, gold_labels)
        
        # Should have 1 TP yes, 1 TP no, 1 FP yes (pred yes, gold no), 1 FN yes (pred no, gold yes)
        assert result["matrix"].shape == (2, 2)


class TestPrintReport:
    """Tests for print_report method."""
    
    def test_print_report_runs_without_error(self, capsys):
        """Test that print_report executes without error."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        predictions = [100, 200, 100, 200]
        gold_labels = [100, 200, 100, 100]
        
        metrics.print_report(predictions, gold_labels)
        
        captured = capsys.readouterr()
        assert "Classification Report" in captured.out
        assert "Accuracy" in captured.out
        assert "F1" in captured.out


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_sample(self):
        """Test metrics with single sample."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        result = metrics.compute([100], [100])
        assert result["accuracy"] == 1.0
    
    def test_all_same_prediction(self):
        """Test when all predictions are the same."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no"],
            valid_token_ids=[100, 200],
            id_to_label={100: "yes", 200: "no"},
        )
        
        result = metrics.compute([100, 100, 100], [100, 200, 100])
        assert result["accuracy"] == pytest.approx(2/3)
    
    def test_three_class_classification(self):
        """Test with three classes."""
        from compressgpt.compute_metrics import ComputeMetrics
        
        metrics = ComputeMetrics(
            labels=["yes", "no", "partial"],
            valid_token_ids=[100, 200, 300],
            id_to_label={100: "yes", 200: "no", 300: "partial"},
        )
        
        predictions = [100, 200, 300, 100, 200, 300]
        gold_labels = [100, 200, 300, 100, 200, 300]
        
        result = metrics.compute(predictions, gold_labels)
        
        assert result["accuracy"] == 1.0
        assert "f1_yes" in result
        assert "f1_no" in result
        assert "f1_partial" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
