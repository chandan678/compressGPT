"""
Tests for ModelRunner

Tests cover:
- Initialization with metadata
- Label extraction from noisy text
- Clean and map to token ID
- Single and batch inference (mocked model)
- Logits for labels
"""

import pytest
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import Mock, MagicMock, patch
import torch

# Mock peft and trl at module level (optional dependencies for trainer)
# This allows importing model_runner without requiring peft/trl
if 'peft' not in sys.modules:
    mock_peft = MagicMock()
    mock_peft.__spec__ = ModuleSpec("peft", None)
    sys.modules['peft'] = mock_peft

if 'trl' not in sys.modules:
    mock_trl = MagicMock()
    mock_trl.__spec__ = ModuleSpec("trl", None)
    sys.modules['trl'] = mock_trl

from compressgpt.model_runner import ModelRunner
class TokenizerOutput:
    """Mimics BatchEncoding that can be unpacked with ** and moved to device."""
    def __init__(self, batch_size):
        self.input_ids = torch.zeros(batch_size, 10, dtype=torch.long)
        self.attention_mask = torch.ones(batch_size, 10, dtype=torch.long)
        self._data = {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
        }
    
    def to(self, device):
        return self
    
    def keys(self):
        return self._data.keys()
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __iter__(self):
        return iter(self._data)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with label token mappings."""
    # Token ID mappings (with space prefix as in real usage)
    token_map = {
        " yes": 3763,
        " no": 645,
        "yes": 9891,
        "no": 789,
        " Yes": 3904,
        " partial": 7276,
    }
    
    # Reverse map for decode
    id_to_text = {v: k for k, v in token_map.items()}
    id_to_text[12345] = " garbage"
    id_to_text[99999] = " YES!!"
    
    class MockTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"
        
        def encode(self, text, add_special_tokens=True):
            if text in token_map:
                return [token_map[text]]
            return [0]
        
        def decode(self, token_ids):
            if isinstance(token_ids, list) and len(token_ids) == 1:
                return id_to_text.get(token_ids[0], "<unk>")
            return "<unk>"
        
        def __call__(self, texts, return_tensors=None, padding=False, truncation=False):
            if isinstance(texts, str):
                texts = [texts]
            return TokenizerOutput(len(texts))
    
    return MockTokenizer()


@pytest.fixture
def sample_metadata():
    """Create sample metadata as returned by DatasetBuilder.get_metadata()."""
    return {
        "model_id": "test-model",
        "model_mode": "base",
        "response_trigger": "Answer:",
        "labels": ["yes", "no", "partial"],
        "label_token_ids": {
            "yes": 3763,
            "no": 645,
            "partial": 7276,
        },
        "id_to_label": {
            3763: "yes",
            645: "no",
            7276: "partial",
        },
        "label_counts": {"yes": 50, "no": 30, "partial": 20},
        "num_samples": 100,
    }


class MockModelOutput:
    """Mimics model output with logits."""
    def __init__(self, logits):
        self.logits = logits


class MockModel:
    """Mock model for inference tests."""
    device = "cpu"
    
    def eval(self):
        pass
    
    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1
        seq_len = 10
        vocab_size = 32000
        
        # Create logits with highest value at token 3763 (yes)
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, -1, 3763] = 10.0  # "yes" token has highest logit
        logits[:, -1, 645] = 5.0   # "no" token
        logits[:, -1, 7276] = 2.0  # "partial" token
        
        return MockModelOutput(logits)


@pytest.fixture
def mock_model():
    """Create a mock model for inference tests."""
    return MockModel()


class TestModelRunnerInit:
    """Tests for ModelRunner initialization."""
    
    def test_init_with_loaded_model(self, mock_model, mock_tokenizer, sample_metadata):
        """Test initialization with pre-loaded model."""
        runner = ModelRunner(
            model=mock_model,
            tokenizer=mock_tokenizer,
            metadata=sample_metadata,
        )
        
        assert runner.model == mock_model
        assert runner.tokenizer == mock_tokenizer
        assert runner.labels == ["yes", "no", "partial"]
        assert runner.label_token_ids == {"yes": 3763, "no": 645, "partial": 7276}
        assert runner.valid_token_ids == {3763, 645, 7276}
    
    def test_init_without_tokenizer_raises(self, mock_model, sample_metadata):
        """Test that init without tokenizer raises error when model is loaded."""
        with pytest.raises(ValueError, match="tokenizer must be provided"):
            ModelRunner(
                model=mock_model,
                tokenizer=None,
                metadata=sample_metadata,
            )
    
    def test_init_without_metadata(self, mock_model, mock_tokenizer):
        """Test initialization without metadata (for raw inference)."""
        runner = ModelRunner(
            model=mock_model,
            tokenizer=mock_tokenizer,
            metadata=None,
        )
        
        assert runner.labels is None
        assert runner.label_token_ids is None
    
    def test_init_with_invalid_metadata_raises(self, mock_model, mock_tokenizer):
        """Test that metadata without label_token_ids raises error."""
        bad_metadata = {"labels": ["yes", "no"]}  # Missing label_token_ids
        
        with pytest.raises(ValueError, match="label_token_ids"):
            ModelRunner(
                model=mock_model,
                tokenizer=mock_tokenizer,
                metadata=bad_metadata,
            )


class TestLabelExtraction:
    """Tests for label extraction from noisy text."""
    
    def test_extract_clean_label(self, mock_model, mock_tokenizer, sample_metadata):
        """Test extraction of clean labels."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        assert runner._extract_label_from_text("yes") == "yes"
        assert runner._extract_label_from_text("no") == "no"
        assert runner._extract_label_from_text("partial") == "partial"
    
    def test_extract_label_with_spaces(self, mock_model, mock_tokenizer, sample_metadata):
        """Test extraction of labels with leading/trailing spaces."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        assert runner._extract_label_from_text(" yes") == "yes"
        assert runner._extract_label_from_text("no ") == "no"
        assert runner._extract_label_from_text("  partial  ") == "partial"
    
    def test_extract_label_case_insensitive(self, mock_model, mock_tokenizer, sample_metadata):
        """Test extraction handles different cases."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        assert runner._extract_label_from_text("YES") == "yes"
        assert runner._extract_label_from_text("No") == "no"
        assert runner._extract_label_from_text("PARTIAL") == "partial"
        assert runner._extract_label_from_text("YeS") == "yes"
    
    def test_extract_label_with_punctuation(self, mock_model, mock_tokenizer, sample_metadata):
        """Test extraction from labels with punctuation."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        assert runner._extract_label_from_text("Yes!") == "yes"
        assert runner._extract_label_from_text("no.") == "no"
        assert runner._extract_label_from_text("yes?") == "yes"
    
    def test_extract_label_from_sentence(self, mock_model, mock_tokenizer, sample_metadata):
        """Test extraction from sentences containing labels."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        assert runner._extract_label_from_text("The answer is yes") == "yes"
        assert runner._extract_label_from_text("I think no because...") == "no"
        assert runner._extract_label_from_text("This is partial match") == "partial"
    
    def test_extract_label_not_found(self, mock_model, mock_tokenizer, sample_metadata):
        """Test extraction returns None when no label found."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        assert runner._extract_label_from_text("garbage") is None
        assert runner._extract_label_from_text("maybe") is None
        assert runner._extract_label_from_text("") is None


class TestCleanAndMapToTokenId:
    """Tests for _clean_and_map_to_token_id method."""
    
    def test_direct_match(self, mock_model, mock_tokenizer, sample_metadata):
        """Test direct token ID match (fast path)."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        # Token 3763 is directly in id_to_label
        token_id, raw_decoded, label = runner._clean_and_map_to_token_id(3763)
        
        assert token_id == 3763
        assert label == "yes"
    
    def test_cleaned_match(self, mock_model, mock_tokenizer, sample_metadata):
        """Test token that needs cleaning (slow path)."""
        # Make tokenizer return " YES!!" for token 99999
        mock_tokenizer.decode = lambda ids: " YES!!" if ids == [99999] else "<unk>"
        
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        token_id, raw_decoded, label = runner._clean_and_map_to_token_id(99999)
        
        assert token_id == 3763  # Mapped to "yes" token
        assert label == "yes"
        assert raw_decoded == " YES!!"
    
    def test_unknown_token(self, mock_model, mock_tokenizer, sample_metadata):
        """Test unrecognized token returns UNKNOWN_TOKEN_ID."""
        mock_tokenizer.decode = lambda ids: " garbage" if ids == [12345] else "<unk>"
        
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        token_id, raw_decoded, label = runner._clean_and_map_to_token_id(12345)
        
        assert token_id == runner.UNKNOWN_TOKEN_ID  # -1
        assert label is None
        assert raw_decoded == " garbage"


class TestRunSingle:
    """Tests for run_single method."""
    
    def test_run_single_prediction(self, mock_model, mock_tokenizer, sample_metadata):
        """Test single prompt inference."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        token_id, label = runner.run_single("Test prompt: Answer:")
        
        assert token_id == 3763  # Mock model returns "yes" as highest
        assert label == "yes"
    
    def test_run_single_unknown_prediction(self, sample_metadata):
        """Test single prompt with unknown prediction."""
        # Create a model that returns unknown token
        class UnknownTokenModel:
            device = "cpu"
            def eval(self): pass
            def __call__(self, **kwargs):
                logits = torch.zeros(1, 10, 32000)
                logits[:, -1, 12345] = 10.0  # Unknown token highest
                return MockModelOutput(logits)
        
        # Create tokenizer that decodes unknown token
        class UnknownTokenizer:
            pad_token = "<pad>"
            eos_token = "</s>"
            def decode(self, ids):
                return "garbage" if ids == [12345] else "<unk>"
            def __call__(self, texts, **kwargs):
                return TokenizerOutput(1)
        
        runner = ModelRunner(UnknownTokenModel(), UnknownTokenizer(), sample_metadata)
        
        token_id, label = runner.run_single("Test prompt:")
        
        assert token_id == -1  # UNKNOWN_TOKEN_ID
        assert label == "garbage"


class TestRun:
    """Tests for batch run method."""
    
    def test_run_requires_metadata(self, mock_model, mock_tokenizer):
        """Test that run() requires metadata."""
        runner = ModelRunner(mock_model, mock_tokenizer, metadata=None)
        
        with pytest.raises(ValueError, match="metadata is required"):
            runner.run(Mock())
    
    def test_run_batch_predictions(self, mock_model, mock_tokenizer, sample_metadata):
        """Test batch inference returns correct token IDs."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata, batch_size=2)
        
        # Create mock dataset
        dataset = {
            "prompt": ["Prompt 1", "Prompt 2", "Prompt 3"],
            "response": ["yes", "no", "yes"],
        }
        
        predictions, gold_labels = runner.run(dataset, show_progress=False)
        
        assert len(predictions) == 3
        assert len(gold_labels) == 3
        
        # All predictions should be 3763 (yes) based on mock model
        assert predictions == [3763, 3763, 3763]
        
        # Gold labels mapped correctly
        assert gold_labels == [3763, 645, 3763]  # yes, no, yes
    
    def test_run_with_invalid_gold_label_raises(self, mock_model, mock_tokenizer, sample_metadata):
        """Test that invalid gold label raises error."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        dataset = {
            "prompt": ["Prompt 1"],
            "response": ["invalid_label"],  # Not in valid labels
        }
        
        with pytest.raises(ValueError, match="not found in label_token_ids"):
            runner.run(dataset, show_progress=False)


class TestGetLogitsForLabels:
    """Tests for get_logits_for_labels method."""
    
    def test_get_logits_returns_all_labels(self, mock_model, mock_tokenizer, sample_metadata):
        """Test that logits are returned for all valid labels."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        logits = runner.get_logits_for_labels("Test prompt:")
        
        assert "yes" in logits
        assert "no" in logits
        assert "partial" in logits
        
        # Based on mock model setup
        assert logits["yes"] == 10.0
        assert logits["no"] == 5.0
        assert logits["partial"] == 2.0
    
    def test_get_logits_relative_values(self, mock_model, mock_tokenizer, sample_metadata):
        """Test that yes has highest logit (model confidence)."""
        runner = ModelRunner(mock_model, mock_tokenizer, sample_metadata)
        
        logits = runner.get_logits_for_labels("Test prompt:")
        
        # Yes should be highest
        assert logits["yes"] > logits["no"]
        assert logits["yes"] > logits["partial"]


class TestUnknownTokenId:
    """Tests for UNKNOWN_TOKEN_ID constant."""
    
    def test_unknown_token_id_is_negative(self):
        """Test UNKNOWN_TOKEN_ID is -1."""
        assert ModelRunner.UNKNOWN_TOKEN_ID == -1
