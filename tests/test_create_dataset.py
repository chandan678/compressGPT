"""
Test cases for DatasetBuilder class.

Run with: pytest tests/ -v
"""

import os
import json
import pytest
import pandas as pd
from create_dataset import DatasetBuilder


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
        template = "Name 1: {name1}\nName 2: {name2}\nYour Answer is:"
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        assert builder.get_response_template() == "Your Answer is:"
    
    def test_missing_placeholder_in_map_raises(self, sample_csv_path):
        """Test that missing placeholder mapping raises ValueError."""
        template = "Name 1: {name1}\nName 2: {name2}\nAnswer:"
        
        with pytest.raises(ValueError, match="not found in input_column_map"):
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=template,
                input_column_map={"name1": "elected_name"},  # Missing name2
                label_column="labeled_result"
            )
    
    def test_extra_key_in_map_raises(self, sample_csv_path):
        """Test that extra key in mapping raises ValueError."""
        template = "Name 1: {name1}\nAnswer:"
        
        with pytest.raises(ValueError, match="not used in template"):
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=template,
                input_column_map={"name1": "elected_name", "name2": "partner_name"},
                label_column="labeled_result"
            )
    
    def test_invalid_column_in_map_raises(self, sample_csv_path, prompt_template):
        """Test that invalid CSV column raises ValueError."""
        with pytest.raises(ValueError, match="not found in CSV"):
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=prompt_template,
                input_column_map={"name1": "nonexistent_column", "name2": "partner_name"},
                label_column="labeled_result"
            )
    
    def test_invalid_label_column_raises(self, sample_csv_path, prompt_template, input_column_map):
        """Test that invalid label column raises ValueError."""
        with pytest.raises(ValueError, match="Label column.*not found"):
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=prompt_template,
                input_column_map=input_column_map,
                label_column="nonexistent_label"
            )
    
    def test_no_response_template_raises(self, sample_csv_path, input_column_map):
        """Test that template without response trigger raises ValueError."""
        template = "Name 1: {name1}\nName 2: {name2}"  # No Answer: at end
        
        with pytest.raises(ValueError, match="No response template found"):
            DatasetBuilder(
                csv_file_path=sample_csv_path,
                prompt_template=template,
                input_column_map=input_column_map,
                label_column="labeled_result"
            )


class TestDatasetBuilderBuild:
    """Tests for DatasetBuilder.build() method."""
    
    def test_build_returns_dataset(self, sample_csv_path, prompt_template, input_column_map):
        """Test that build() returns a HuggingFace Dataset."""
        from datasets import Dataset
        
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        assert isinstance(dataset, Dataset)
        assert "prompt" in dataset.column_names
        assert "response" in dataset.column_names
    
    def test_build_correct_row_count(self, sample_csv_path, prompt_template, input_column_map):
        """Test that build() produces correct number of rows."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        assert len(dataset) == 5
    
    def test_build_normalizes_labels(self, sample_csv_path, prompt_template, input_column_map):
        """Test that labels are normalized to lowercase."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        # All labels should be lowercase
        for row in dataset:
            assert row["response"] == row["response"].lower()
            assert row["response"] == row["response"].strip()
    
    def test_build_substitutes_values(self, sample_csv_path, prompt_template, input_column_map):
        """Test that placeholder values are correctly substituted."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        # First row should have "John Smith" and "Jon Smyth"
        assert "John Smith" in dataset[0]["prompt"]
        assert "Jon Smyth" in dataset[0]["prompt"]
        assert "{name1}" not in dataset[0]["prompt"]
        assert "{name2}" not in dataset[0]["prompt"]
    
    def test_build_skips_nan_rows(self, csv_with_nan_path, prompt_template, input_column_map):
        """Test that rows with NaN values are skipped."""
        builder = DatasetBuilder(
            csv_file_path=csv_with_nan_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        dataset = builder.build()
        
        # 4 rows have NaN (row 1 has NaN in partner_name, row 2 in elected_name, row 3 in label)
        # Only 2 rows should remain: row 0 and row 4
        assert len(dataset) == 2
        assert builder._skipped_rows > 0
    
    def test_build_with_valid_labels_filter(self, sample_csv_path, prompt_template, input_column_map):
        """Test filtering by valid_labels."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result",
            valid_labels={"yes", "no"}  # Exclude "partial"
        )
        
        dataset = builder.build()
        
        # Should have 4 rows (1 partial filtered out)
        assert len(dataset) == 4
        for row in dataset:
            assert row["response"] in {"yes", "no"}
    
    def test_build_tracks_label_counts(self, sample_csv_path, prompt_template, input_column_map):
        """Test that label counts are tracked correctly."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        builder.build()
        
        assert builder._label_counts["yes"] == 2  # "yes" and "YES"
        assert builder._label_counts["no"] == 2   # "no" and "No"
        assert builder._label_counts["partial"] == 1


class TestDatasetBuilderSingleInput:
    """Tests for single input column scenarios."""
    
    def test_single_input_build(self, single_input_csv_path):
        """Test building dataset with single input column."""
        builder = DatasetBuilder(
            csv_file_path=single_input_csv_path,
            prompt_template="Classify the sentiment of this text:\n{text}\nSentiment:",
            input_column_map={"text": "text"},
            label_column="sentiment"
        )
        
        dataset = builder.build()
        
        assert len(dataset) == 4
        assert "I love this!" in dataset[0]["prompt"]
        assert dataset[0]["response"] == "positive"


class TestDatasetBuilderSaveJsonl:
    """Tests for save_jsonl() method."""
    
    def test_save_jsonl(self, sample_csv_path, prompt_template, input_column_map, tmp_path):
        """Test saving dataset to JSONL file."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        builder.build()
        
        output_path = tmp_path / "output.jsonl"
        builder.save_jsonl(str(output_path))
        
        # Read and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 5
        
        first_row = json.loads(lines[0])
        assert "prompt" in first_row
        assert "response" in first_row
    
    def test_save_jsonl_without_build_raises(self, sample_csv_path, prompt_template, input_column_map, tmp_path):
        """Test that save_jsonl() raises error if build() not called."""
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=prompt_template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        output_path = tmp_path / "output.jsonl"
        
        with pytest.raises(RuntimeError, match="Call build\\(\\) first"):
            builder.save_jsonl(str(output_path))


class TestDatasetBuilderEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_prompt_ends_with_newline_answer(self, sample_csv_path, input_column_map):
        """Test prompt template with newline before Answer:."""
        template = "Name 1: {name1}\nName 2: {name2}\n\nAnswer:"
        builder = DatasetBuilder(
            csv_file_path=sample_csv_path,
            prompt_template=template,
            input_column_map=input_column_map,
            label_column="labeled_result"
        )
        
        assert builder.get_response_template() == "Answer:"
    
    def test_special_characters_in_values(self, tmp_path):
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
            csv_file_path=str(csv_path),
            prompt_template="Name 1: {n1}\nName 2: {n2}\nAnswer:",
            input_column_map={"n1": "name1", "n2": "name2"},
            label_column="label"
        )
        
        dataset = builder.build()
        
        assert len(dataset) == 3
        assert "O'Brien" in dataset[0]["prompt"]
        assert "José García" in dataset[1]["prompt"]
    
    def test_whitespace_in_labels(self, tmp_path):
        """Test that labels with whitespace are properly stripped."""
        data = {
            "text": ["hello", "world"],
            "label": ["  yes  ", "\tno\n"]
        }
        df = pd.DataFrame(data)
        
        csv_path = tmp_path / "whitespace.csv"
        df.to_csv(csv_path, index=False)
        
        builder = DatasetBuilder(
            csv_file_path=str(csv_path),
            prompt_template="Text: {t}\nAnswer:",
            input_column_map={"t": "text"},
            label_column="label"
        )
        
        dataset = builder.build()
        
        assert dataset[0]["response"] == "yes"
        assert dataset[1]["response"] == "no"
