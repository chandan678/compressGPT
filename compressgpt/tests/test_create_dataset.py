"""
Test cases for DatasetBuilder class.

Run with: pytest tests/ -v
"""

import os
import json
import pytest
import pandas as pd
from compressgpt import DatasetBuilder


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
def single_input_csv_path(tmp_path):
    """Create a CSV with single input column for testing."""
    data = {
        "text": ["I love this!", "This is terrible", "It's okay", "Amazing product"],
        "sentiment": ["positive", "negative", "neutral", "positive"]
    }
    df = pd.DataFrame(data)
    
    csv_path = tmp_path / "single_input.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


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
    
    def test_basic_init(self, sample_csv_path, prompt_template, input_column_map):
        """Test basic initialization works correctly."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        assert builder.csv_file_path == sample_csv_path
        assert builder.prompt_template == prompt_template
        assert builder.input_column_map == input_column_map
        assert builder.label_column == "labeled_result"
        assert builder.valid_labels is None
    
    def test_response_template_extraction(self, sample_csv_path, prompt_template, input_column_map):
        """Test that response template is correctly extracted."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        assert builder.get_response_template() == "Answer:"
    
    def test_response_template_multiword(self, sample_csv_path, input_column_map):
        """Test extraction of multi-word response template."""
        template = "Name 1: {name1}\nName 2: {name2}\nThe answer is:"
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        assert builder.get_response_template() == "The answer is:"
    
    def test_missing_placeholder_in_map(self, sample_csv_path, prompt_template):
        """Test error when template placeholder missing from input_column_map."""
        with pytest.raises(ValueError) as exc_info:
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=prompt_template,
                input_column_map={"name1": "elected_name"},  # Missing name2
                label_column="labeled_result"
            )
        assert "name2" in str(exc_info.value)
    
    def test_extra_key_in_map(self, sample_csv_path, prompt_template, input_column_map):
        """Test error when input_column_map has extra keys."""
        input_column_map["extra_key"] = "some_column"
        with pytest.raises(ValueError) as exc_info:
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=prompt_template,
                input_column_map=input_column_map,
                label_column="labeled_result"
            )
        assert "extra_key" in str(exc_info.value)
    
    def test_missing_csv_column(self, sample_csv_path, prompt_template):
        """Test error when mapped CSV column doesn't exist."""
        with pytest.raises(ValueError) as exc_info:
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=prompt_template,
                input_column_map={"name1": "nonexistent_column", "name2": "partner_name"},
                label_column="labeled_result"
            )
        assert "nonexistent_column" in str(exc_info.value)
    
    def test_missing_label_column(self, sample_csv_path, prompt_template, input_column_map):
        """Test error when label column doesn't exist."""
        with pytest.raises(ValueError) as exc_info:
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=prompt_template,
                input_column_map=input_column_map,
                label_column="nonexistent_label"
            )
        assert "nonexistent_label" in str(exc_info.value)
    
    def test_no_response_template(self, sample_csv_path, input_column_map):
        """Test error when template has no text after last placeholder."""
        template = "Name 1: {name1}\nName 2: {name2}"
        with pytest.raises(ValueError) as exc_info:
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=template,
                input_column_map=input_column_map,
                label_column="labeled_result"
            )
        assert "response template" in str(exc_info.value).lower()


class TestDatasetBuilderBuild:
    """Tests for DatasetBuilder.build() method."""
    
    def test_build_creates_dataset(self, sample_csv_path, prompt_template, input_column_map):
        """Test that build() creates a valid dataset."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        assert dataset is not None
        assert len(dataset) == 5
        assert "prompt" in dataset.column_names
        assert "response" in dataset.column_names
    
    def test_build_normalizes_labels(self, sample_csv_path, prompt_template, input_column_map):
        """Test that labels are normalized to lowercase."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        # All responses should be lowercase
        for item in dataset:
            assert item["response"] == item["response"].lower()
    
    def test_build_with_valid_labels_filter(self, sample_csv_path, prompt_template, input_column_map):
        """Test filtering by valid_labels."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            valid_labels={"yes", "no"}  # Excludes "partial"
        )
        
        dataset = builder.build()
        
        # Should exclude rows with "partial" label
        assert len(dataset) == 4
        for item in dataset:
            assert item["response"] in {"yes", "no"}
    
    def test_build_skips_nan_rows(self, csv_with_nan_path, prompt_template, input_column_map):
        """Test that rows with NaN values are skipped."""
        builder = DatasetBuilder(
            csv_file_path=csv_with_nan_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        # 4 rows have NaN values, only 1 should remain
        assert len(dataset) == 1
    
    def test_build_substitutes_values_correctly(self, sample_csv_path, prompt_template, input_column_map):
        """Test that values are correctly substituted into prompts."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        # Check first prompt has correct substitutions
        first_prompt = dataset[0]["prompt"]
        assert "John Smith" in first_prompt
        assert "Jon Smyth" in first_prompt
        assert "{name1}" not in first_prompt
        assert "{name2}" not in first_prompt


class TestDatasetBuilderSave:
    """Tests for DatasetBuilder.save_jsonl() method."""
    
    def test_save_jsonl(self, sample_csv_path, prompt_template, input_column_map, tmp_path):
        """Test saving dataset to JSONL file."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        builder.build()
        
        output_path = str(tmp_path / "output.jsonl")
        builder.save_jsonl(output_path)
        
        # Verify file exists and has correct content
        assert os.path.exists(output_path)
        
        with open(output_path, "r") as f:
            lines = f.readlines()
        
        assert len(lines) == 5
        
        # Parse first line and verify structure
        first_item = json.loads(lines[0])
        assert "prompt" in first_item
        assert "response" in first_item
    
    def test_save_before_build_raises_error(self, sample_csv_path, prompt_template, input_column_map, tmp_path):
        """Test that save_jsonl raises error if build() not called."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        output_path = str(tmp_path / "output.jsonl")
        
        with pytest.raises(RuntimeError):
            builder.save_jsonl(output_path)


class TestSingleInputColumn:
    """Tests for single input column scenarios."""
    
    def test_single_input_column(self, single_input_csv_path):
        """Test DatasetBuilder with single input column."""
        template = "Classify the sentiment of this text:\n{text}\nSentiment:"
        
        builder = DatasetBuilder(
            csv_file_path=single_input_csv_path,
            prompt_template=template,
            input_column_map={"text": "text"},
            label_column="sentiment"
        )
        
        dataset = builder.build()
        
        assert len(dataset) == 4
        assert "I love this!" in dataset[0]["prompt"]
        assert dataset[0]["response"] == "positive"
