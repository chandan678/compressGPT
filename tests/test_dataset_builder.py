"""
Test cases for DatasetBuilder class (updated for new API).

Run with: pytest tests/test_dataset_builder.py -v
"""

import os
import json
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from compressgpt.create_dataset import DatasetBuilder
from compressgpt.label_space import LabelSpace


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a temporary CSV file for testing."""
    data = {
        "elected_name": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown", "Charlie Davis"],
        "partner_name": ["Jon Smyth", "Janet Doe", "Robert Williams", "Alicia Brown", "Charles Davis"],
        "labeled_result": ["yes", "partial", "no", "YES", "No"]
    }
    df = pd.DataFrame(data)
    
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def csv_with_nan_path(tmp_path):
    """Create a CSV file with NaN values for testing."""
    data = {
        "elected_name": ["John Smith", "Jane Doe", None, "Alice Brown", "Charlie Davis"],
        "partner_name": ["Jon Smyth", None, "Robert Williams", "Alicia Brown", "Charles Davis"],
        "labeled_result": ["yes", "partial", "no", None, "yes"]
    }
    df = pd.DataFrame(data)
    
    csv_path = tmp_path / "nan_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.chat_template = None  # Base model by default
    
    # Mock encode to return single token per label
    def mock_encode(text, add_special_tokens=False):
        label_map = {" yes": [100], " no": [200], " partial": [300]}
        return label_map.get(text, [999])
    
    tokenizer.encode = Mock(side_effect=mock_encode)
    tokenizer.decode = Mock(side_effect=lambda ids: " yes" if ids == [100] else " no")
    
    # Mock apply_chat_template
    def mock_apply_template(messages, tokenize=False, add_generation_prompt=False):
        if len(messages) == 1:
            return f"<|user|>{messages[0]['content']}<|assistant|>"
        else:
            return f"<|user|>{messages[0]['content']}<|assistant|>{messages[1]['content']}"
    
    tokenizer.apply_chat_template = Mock(side_effect=mock_apply_template)
    
    return tokenizer


@pytest.fixture
def prompt_template():
    """Standard prompt template for name matching."""
    return "Decide if two names belong to the same person.\nReturn: yes, no, or partial.\n\nName 1: {name1}\nName 2: {name2}\nAnswer:"


@pytest.fixture
def input_column_map():
    """Standard column mapping for name matching."""
    return {"name1": "elected_name", "name2": "partner_name"}


class TestDatasetBuilderInit:
    """Tests for DatasetBuilder initialization and validation."""
    
    def test_basic_init_with_mock_tokenizer(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test basic initialization works correctly."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        assert builder.data_path == sample_csv_path
        assert builder.model_id == "mock-model"
        assert builder.prompt_template == prompt_template
        assert builder.input_column_map == input_column_map
        assert builder.label_column == "labeled_result"
        assert builder.valid_labels is None
        # response_trigger is stripped for consistent matching
        assert builder.response_trigger == "Answer:"
    
    def test_response_trigger_extraction(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that response trigger is correctly extracted and stripped."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        # response_trigger is stripped for consistent matching
        assert builder.response_trigger == "Answer:"
    
    def test_response_trigger_multiword(self, sample_csv_path, input_column_map, mock_tokenizer):
        """Test extraction of multi-word response trigger."""
        template = "Name 1: {name1}\nName 2: {name2}\nYour Answer is:"
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        # response_trigger is stripped for consistent matching
        assert builder.response_trigger == "Your Answer is:"
    
    def test_model_mode_detection_base(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test model mode detection for base model."""
        mock_tokenizer.chat_template = None
        
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            model_mode="auto"
        )
        
        assert builder.model_mode == "base"
    
    def test_model_mode_detection_instruct(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test model mode detection for instruct model."""
        mock_tokenizer.chat_template = "{% for message in messages %}..."
        
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            model_mode="auto"
        )
        
        assert builder.model_mode == "instruct"
    
    def test_pad_token_set_to_eos(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that pad_token is set to eos_token if not present."""
        assert mock_tokenizer.pad_token is None
        
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        assert mock_tokenizer.pad_token == "<eos>"
    
    def test_missing_placeholder_in_map_raises(self, sample_csv_path, mock_tokenizer):
        """Test that missing placeholder mapping raises ValueError."""
        template = "Name 1: {name1}\nName 2: {name2}\nAnswer:"
        
        with pytest.raises(ValueError, match="not found in input_column_map"):
            DatasetBuilder(
                data_path=sample_csv_path,
                model_id="mock-model",
                prompt_template=template,
                input_column_map={"name1": "elected_name"},  # Missing name2
                label_column="labeled_result",
                tokenizer=mock_tokenizer
            )
    
    def test_extra_key_in_map_raises(self, sample_csv_path, mock_tokenizer):
        """Test that extra key in mapping raises ValueError."""
        template = "Name 1: {name1}\nAnswer:"
        
        with pytest.raises(ValueError, match="not used in template"):
            DatasetBuilder(
                data_path=sample_csv_path,
                model_id="mock-model",
                prompt_template=template,
                input_column_map={"name1": "elected_name", "name2": "partner_name"},
                label_column="labeled_result",
                tokenizer=mock_tokenizer
            )
    
    def test_invalid_column_in_map_raises(self, sample_csv_path, prompt_template, mock_tokenizer):
        """Test that invalid CSV column raises ValueError."""
        with pytest.raises(ValueError, match="not found in CSV"):
            DatasetBuilder(
                data_path=sample_csv_path,
                model_id="mock-model",
                prompt_template=prompt_template,
                input_column_map={"name1": "nonexistent_column", "name2": "partner_name"},
                label_column="labeled_result",
                tokenizer=mock_tokenizer
            )
    
    def test_invalid_label_column_raises(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that invalid label column raises ValueError."""
        with pytest.raises(ValueError, match="not found in CSV"):
            DatasetBuilder(
                data_path=sample_csv_path,
                model_id="mock-model",
                prompt_template=prompt_template,
                input_column_map=input_column_map,
                label_column="nonexistent_label",
                tokenizer=mock_tokenizer
            )
    
    def test_no_response_trigger_raises(self, sample_csv_path, input_column_map, mock_tokenizer):
        """Test that template without response trigger raises ValueError."""
        template = "Name 1: {name1}\nName 2: {name2}"  # No trigger at end
        
        with pytest.raises(ValueError, match="No response trigger found"):
            DatasetBuilder(
                data_path=sample_csv_path,
                model_id="mock-model",
                prompt_template=template,
                input_column_map=input_column_map,
                label_column="labeled_result",
                tokenizer=mock_tokenizer
            )


class TestDatasetBuilderBuild:
    """Tests for DatasetBuilder.build() method."""
    
    def test_build_returns_self(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that build() returns self for chaining."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        result = builder.build()
        
        assert result is builder
    
    def test_dataset_property_after_build(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that .dataset property works after build()."""
        from datasets import Dataset
        
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        builder.build()
        
        assert isinstance(builder.dataset, Dataset)
        assert "text" in builder.dataset.column_names
        assert "gold_label" in builder.dataset.column_names
    
    def test_dataset_property_before_build_raises(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that .dataset property raises before build()."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        with pytest.raises(RuntimeError, match="not built yet"):
            _ = builder.dataset
    
    def test_metadata_property_after_build(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that .metadata property works after build()."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        builder.build()
        metadata = builder.metadata
        
        assert "model_id" in metadata
        assert "model_mode" in metadata
        assert "response_trigger" in metadata
        assert "label_space" in metadata
        assert "label_counts" in metadata
        assert metadata["model_id"] == "mock-model"
        # response_trigger is stripped for consistent matching
        assert metadata["response_trigger"] == "Answer:"
    
    def test_build_correct_row_count(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that build() produces correct number of rows."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        ).build()
        
        assert len(builder.dataset) == 5
    
    def test_build_normalizes_labels(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that labels are normalized to lowercase."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        ).build()
        
        # All gold_label values should be lowercase and stripped
        for row in builder.dataset:
            assert row["gold_label"] == row["gold_label"].lower()
            assert row["gold_label"] == row["gold_label"].strip()
    
    def test_build_substitutes_values_base_model(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that placeholder values are correctly substituted for base model."""
        mock_tokenizer.chat_template = None
        
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            is_train=True
        ).build()
        
        # First row should have "John Smith" and "Jon Smyth"
        assert "John Smith" in builder.dataset[0]["text"]
        assert "Jon Smyth" in builder.dataset[0]["text"]
        assert "{name1}" not in builder.dataset[0]["text"]
        assert "{name2}" not in builder.dataset[0]["text"]
        # Should have label for training
        assert " yes" in builder.dataset[0]["text"]
    
    def test_build_instruct_model_applies_chat_template(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that instruct model applies chat template."""
        mock_tokenizer.chat_template = "{% for message in messages %}..."
        
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            model_mode="instruct",
            is_train=True
        ).build()
        
        # Should have chat template markers
        assert "<|user|>" in builder.dataset[0]["text"]
        assert "<|assistant|>" in builder.dataset[0]["text"]
    
    def test_build_creates_text_prompt_only_for_training(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that text_prompt_only column is created for training datasets."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            is_train=True
        ).build()
        
        assert "text_prompt_only" in builder.dataset.column_names
    
    def test_build_skips_text_prompt_only_for_eval(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that text_prompt_only column is not created for eval datasets."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            is_train=False
        ).build()
        
        assert "text_prompt_only" not in builder.dataset.column_names
    
    def test_build_skips_nan_rows(self, csv_with_nan_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that rows with NaN values are skipped."""
        builder = DatasetBuilder(
            data_path=csv_with_nan_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        ).build()
        
        # Only 2 complete rows should remain
        assert len(builder.dataset) == 2
        assert builder._skipped_rows > 0
    
    def test_build_with_valid_labels_filter(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test filtering by valid_labels."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            valid_labels={"yes", "no"},  # Exclude "partial"
            tokenizer=mock_tokenizer
        ).build()
        
        # Should have 4 rows (1 partial filtered out)
        assert len(builder.dataset) == 4
        for row in builder.dataset:
            assert row["gold_label"] in {"yes", "no"}
    
    def test_build_tracks_label_counts(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that label counts are tracked correctly."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        ).build()
        
        metadata = builder.metadata
        assert metadata["label_counts"]["yes"] == 2  # "yes" and "YES"
        assert metadata["label_counts"]["no"] == 2   # "no" and "No"
        assert metadata["label_counts"]["partial"] == 1


class TestDatasetBuilderSave:
    """Tests for save() method."""
    
    def test_save_creates_file(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer, tmp_path):
        """Test saving dataset to JSONL file."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        ).build()
        
        output_path = tmp_path / "output.jsonl"
        builder.save(str(output_path))
        
        # Read and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 5
        
        first_row = json.loads(lines[0])
        assert "text" in first_row
        assert "gold_label" in first_row
    
    def test_save_without_build_raises(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer, tmp_path):
        """Test that save() raises error if build() not called."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        )
        
        output_path = tmp_path / "output.jsonl"
        
        with pytest.raises(RuntimeError, match="No data to save"):
            builder.save(str(output_path))
    
    def test_auto_save_with_output_path(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer, tmp_path):
        """Test that output_path triggers auto-save after build()."""
        output_path = tmp_path / "auto_output.jsonl"
        
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            output_path=str(output_path)
        ).build()
        
        # File should exist automatically
        assert output_path.exists()


class TestLabelSpaceIntegration:
    """Tests for LabelSpace integration in DatasetBuilder."""
    
    def test_label_space_created_after_build(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that LabelSpace is created during build()."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        ).build()
        
        assert builder._label_space is not None
        assert isinstance(builder._label_space, LabelSpace)
    
    def test_label_space_in_metadata(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that LabelSpace is serialized in metadata."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer
        ).build()
        
        metadata = builder.metadata
        label_space_dict = metadata["label_space"]
        
        assert "labels" in label_space_dict
        assert "label_token_ids" in label_space_dict
        assert "valid_token_ids" in label_space_dict
        assert len(label_space_dict["labels"]) == 3  # yes, no, partial


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_special_characters_in_values(self, tmp_path, mock_tokenizer):
        """Test handling of special characters in CSV values."""
        data = {
            "name1": ["O'Brien", "José García", 'John "Jack" Smith'],
            "name2": ["O'Brian", "Jose Garcia", "John Smith"],
            "label": ["yes", "partial", "yes"]
        }
        df = pd.DataFrame(data)
        
        csv_path = tmp_path / "special_chars.csv"
        df.to_csv(csv_path, index=False)
        
        builder = DatasetBuilder(
            data_path=str(csv_path),
            model_id="mock-model",
            prompt_template="Name 1: {n1}\nName 2: {n2}\nAnswer:",
            input_column_map={"n1": "name1", "n2": "name2"},
            label_column="label",
            tokenizer=mock_tokenizer
        ).build()
        
        assert len(builder.dataset) == 3
        assert "O'Brien" in builder.dataset[0]["text"]
        assert "José García" in builder.dataset[1]["text"]
    
    def test_whitespace_in_labels(self, tmp_path, mock_tokenizer):
        """Test that labels with whitespace are properly stripped."""
        data = {
            "text": ["hello", "world"],
            "label": ["  yes  ", "\tno\n"]
        }
        df = pd.DataFrame(data)
        
        csv_path = tmp_path / "whitespace.csv"
        df.to_csv(csv_path, index=False)
        
        builder = DatasetBuilder(
            data_path=str(csv_path),
            model_id="mock-model",
            prompt_template="Text: {t}\nAnswer:",
            input_column_map={"t": "text"},
            label_column="label",
            tokenizer=mock_tokenizer
        ).build()
        
        assert builder.dataset[0]["gold_label"] == "yes"
        assert builder.dataset[1]["gold_label"] == "no"
    
    def test_keep_fields_option(self, sample_csv_path, prompt_template, input_column_map, mock_tokenizer):
        """Test that keep_fields preserves debug columns."""
        builder = DatasetBuilder(
            data_path=sample_csv_path,
            model_id="mock-model",
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            tokenizer=mock_tokenizer,
            keep_fields=True
        ).build()
        
        assert "prompt" in builder.dataset.column_names
        assert "response" in builder.dataset.column_names
