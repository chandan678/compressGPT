"""
Test cases for LabelSpace class.

Run with: pytest tests/test_label_space.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock
from compressgpt.label_space import LabelSpace


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    
    # Mock encode to return single token per label
    def mock_encode(text, add_special_tokens=False):
        label_map = {
            " yes": [100],
            " no": [200],
            " partial": [300],
            " multi token": [400, 401, 402]  # Multi-token label
        }
        return label_map.get(text, [999])
    
    tokenizer.encode = Mock(side_effect=mock_encode)
    
    # Mock decode
    def mock_decode(ids):
        id_to_text = {100: " yes", 200: " no", 300: " partial"}
        if isinstance(ids, list) and len(ids) == 1:
            return id_to_text.get(ids[0], "unknown")
        return "unknown"
    
    tokenizer.decode = Mock(side_effect=mock_decode)
    
    return tokenizer


class TestLabelSpaceInit:
    """Tests for LabelSpace initialization."""
    
    def test_basic_init(self, mock_tokenizer):
        """Test basic initialization with valid single-token labels."""
        labels = ["yes", "no"]
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        # Labels are sorted
        assert set(label_space.labels) == {"yes", "no"}
        assert label_space.label_prefix == " "
        assert len(label_space.label_token_ids) == 2
        assert label_space.label_token_ids["yes"] == 100
        assert label_space.label_token_ids["no"] == 200
    
    def test_multi_token_label_raises(self, mock_tokenizer):
        """Test that multi-token labels raise ValueError."""
        labels = ["yes", "multi token"]
        
        with pytest.raises(ValueError, match="tokenizes to 3 tokens"):
            LabelSpace(
                tokenizer=mock_tokenizer,
                labels=labels,
                label_prefix=" "
            )
    
    def test_labels_sorted(self, mock_tokenizer):
        """Test that labels are stored as provided."""
        labels = ["zebra", "apple", "banana"]
        # Mock the encoding - each gets unique token
        mock_tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens: 
            [500] if "zebra" in text else [600] if "apple" in text else [700])
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        # Labels should be stored
        assert len(label_space.labels) == 3
        assert set(label_space.labels) == {"zebra", "apple", "banana"}
    
    def test_empty_labels_list(self, mock_tokenizer):
        """Test initialization with empty labels list."""
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=[],
            label_prefix=" "
        )
        
        assert len(label_space.labels) == 0
        assert len(label_space.label_token_ids) == 0
        assert len(label_space.valid_token_ids) == 0


class TestLabelSpaceAttributes:
    """Tests for LabelSpace computed attributes."""
    
    def test_valid_token_ids_is_list(self, mock_tokenizer):
        """Test that valid_token_ids is a list."""
        labels = ["yes", "no", "partial"]
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        assert isinstance(label_space.valid_token_ids, list)
        assert len(label_space.valid_token_ids) == 3
    
    def test_valid_token_ids_sorted(self, mock_tokenizer):
        """Test that valid_token_ids are sorted."""
        labels = ["yes", "no", "partial"]
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        # Should be [100, 200, 300]
        assert label_space.valid_token_ids[0] == 100
        assert label_space.valid_token_ids[1] == 200
        assert label_space.valid_token_ids[2] == 300
    
    def test_id_to_label_mapping(self, mock_tokenizer):
        """Test id_to_label reverse mapping."""
        labels = ["yes", "no"]
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        assert label_space.id_to_label[100] == "yes"
        assert label_space.id_to_label[200] == "no"


class TestLabelSpaceSerialization:
    """Tests for LabelSpace serialization."""
    
    def test_to_dict(self, mock_tokenizer):
        """Test serialization to dictionary."""
        labels = ["yes", "no"]
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        data = label_space.to_dict()
        
        assert set(data["labels"]) == {"yes", "no"}
        assert data["label_prefix"] == " "
        assert data["label_token_ids"] == {"yes": 100, "no": 200}
        assert set(data["valid_token_ids"]) == {100, 200}
    
    def test_from_dict(self, mock_tokenizer):
        """Test deserialization from dictionary."""
        data = {
            "labels": ["yes", "no"],
            "label_prefix": " ",
            "label_token_ids": {"yes": 100, "no": 200},
            "valid_token_ids": [100, 200]
        }
        
        label_space = LabelSpace.from_dict(data, mock_tokenizer)
        
        assert set(label_space.labels) == {"yes", "no"}
        assert label_space.label_prefix == " "
        assert label_space.label_token_ids == {"yes": 100, "no": 200}
        assert isinstance(label_space.valid_token_ids, list)
        assert set(label_space.valid_token_ids) == {100, 200}
    
    def test_round_trip_serialization(self, mock_tokenizer):
        """Test that to_dict -> from_dict preserves data."""
        labels = ["yes", "no", "partial"]
        
        original = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        data = original.to_dict()
        restored = LabelSpace.from_dict(data, mock_tokenizer)
        
        assert restored.labels == original.labels
        assert restored.label_prefix == original.label_prefix
        assert restored.label_token_ids == original.label_token_ids
        assert np.array_equal(restored.valid_token_ids, original.valid_token_ids)


class TestLabelSpaceEdgeCases:
    """Tests for edge cases."""
    
    def test_duplicate_labels_handled(self, mock_tokenizer):
        """Test that duplicate labels are removed."""
        labels = ["yes", "yes", "no"]
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        # Duplicates should be removed
        assert len(label_space.labels) == 2
        assert set(label_space.labels) == {"yes", "no"}
    
    def test_labels_with_different_prefix(self, mock_tokenizer):
        """Test using a different label prefix."""
        # Mock different encoding with different prefix
        mock_tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens: 
            [500] if text == "->yes" else [600])
        
        labels = ["yes", "no"]
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix="->"
        )
        
        assert label_space.label_prefix == "->"
        assert label_space.label_token_ids["yes"] == 500
    
    def test_unicode_labels(self, mock_tokenizer):
        """Test labels with unicode characters."""
        mock_tokenizer.encode = Mock(side_effect=lambda text, add_special_tokens: 
            [700] if "是" in text else [800])
        
        labels = ["是", "否"]  # Chinese yes/no
        
        label_space = LabelSpace(
            tokenizer=mock_tokenizer,
            labels=labels,
            label_prefix=" "
        )
        
        assert len(label_space.labels) == 2
        assert "是" in label_space.label_token_ids
        assert "否" in label_space.label_token_ids


class TestLabelSpaceValidation:
    """Tests for label validation logic."""
    
    def test_validates_each_label_independently(self, mock_tokenizer):
        """Test that each label is validated independently."""
        # First label is valid, second is multi-token
        labels = ["yes", "multi token"]
        
        with pytest.raises(ValueError) as exc_info:
            LabelSpace(
                tokenizer=mock_tokenizer,
                labels=labels,
                label_prefix=" "
            )
        
        assert "multi token" in str(exc_info.value)
    
    def test_empty_label_raises(self, mock_tokenizer):
        """Test that empty label string raises error."""
        mock_tokenizer.encode = Mock(return_value=[])
        
        labels = ["yes", ""]
        
        with pytest.raises(ValueError):
            LabelSpace(
                tokenizer=mock_tokenizer,
                labels=labels,
                label_prefix=" "
            )
