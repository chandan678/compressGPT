"""
Dataset Builder for compressGPT SFT Training

This module provides the DatasetBuilder class that converts tabular CSV data
into prompt-response pairs suitable for Supervised Fine-Tuning (SFT).

Example usage:
    builder = DatasetBuilder(
        csv_file_path="data.csv",
        prompt_template="Do these names match?\\nName 1: {name1}\\nName 2: {name2}\\nAnswer:",
        input_column_map={"name1": "elected_name", "name2": "partner_name"},
        label_column="labeled_result",
        valid_labels={"yes", "no", "partial"}  # optional
    )
    dataset = builder.build()
    builder.save_jsonl("output.jsonl")
"""

import re
import json
import pandas as pd
from datasets import Dataset
from typing import Optional


class DatasetBuilder:
    """
    Builds SFT-ready datasets from CSV files using prompt templates.
    
    The prompt template uses {placeholder} syntax for input variables.
    The text after the last placeholder (e.g., "Answer:") becomes the
    response_template for DataCollatorForCompletionOnlyLM.
    
    Attributes:
        csv_file_path: Path to the input CSV file
        prompt_template: Template string with {placeholder} for inputs, ending with response trigger
        input_column_map: Dict mapping placeholder names to CSV column names
        label_column: CSV column name containing the labels/responses
        valid_labels: Optional set of allowed label values (filters out others)
        response_template: Auto-extracted trigger string for completion (e.g., "Answer:")
    """
    
    def __init__(
        self,
        csv_file_path: str,
        prompt_template: str,
        input_column_map: dict[str, str],
        label_column: str,
        valid_labels: Optional[set[str]] = None
    ):
        """
        Initialize the DatasetBuilder.
        
        Args:
            csv_file_path: Path to CSV file
            prompt_template: Prompt string with {placeholder} syntax, ending with response trigger
                Example: "Compare names:\\nName 1: {name1}\\nName 2: {name2}\\nAnswer:"
            input_column_map: Maps template placeholders to CSV columns
                Example: {"name1": "elected_name", "name2": "partner_name"}
            label_column: CSV column name for the response/label
            valid_labels: Optional set of valid labels to filter by (e.g., {"yes", "no", "partial"})
        
        Raises:
            ValueError: If template placeholders don't match input_column_map keys
            ValueError: If label_column not found in CSV
        """
        self.csv_file_path = csv_file_path
        self.prompt_template = prompt_template
        self.input_column_map = input_column_map
        self.label_column = label_column
        self.valid_labels = valid_labels
        
        # Load dataframe
        self.dataframe = pd.read_csv(csv_file_path)
        
        # Extract placeholders from template
        self._template_placeholders = set(re.findall(r'\{(\w+)\}', prompt_template))
        
        # Extract response template (text after last placeholder)
        self.response_template = self._extract_response_template()
        
        # Validate inputs
        self._validate()
        
        # Storage for built dataset
        self._dataset: Optional[Dataset] = None
        self._formatted_data: list[dict] = []
        self._skipped_rows = 0
        self._label_counts: dict[str, int] = {}
    
    def _extract_response_template(self) -> str:
        """
        Extract the response template from the prompt.
        This is the text after the last {placeholder} that triggers completion.
        
        Returns:
            The response template string (e.g., "Answer:")
        """
        # Find the position after the last placeholder
        last_placeholder_end = 0
        for match in re.finditer(r'\{(\w+)\}', self.prompt_template):
            last_placeholder_end = match.end()
        
        if last_placeholder_end == 0:
            raise ValueError("No placeholders found in prompt_template")
        
        response_template = self.prompt_template[last_placeholder_end:].strip()
        
        if not response_template:
            raise ValueError(
                "No response template found after last placeholder. "
                "Prompt should end with a trigger like 'Answer:' after the last {placeholder}"
            )
        
        return response_template
    
    def _validate(self) -> None:
        """Validate that template placeholders match input_column_map and columns exist."""
        # Check all template placeholders have mappings
        map_keys = set(self.input_column_map.keys())
        if self._template_placeholders != map_keys:
            missing_in_map = self._template_placeholders - map_keys
            extra_in_map = map_keys - self._template_placeholders
            
            error_parts = []
            if missing_in_map:
                error_parts.append(f"Placeholders {missing_in_map} not found in input_column_map")
            if extra_in_map:
                error_parts.append(f"Keys {extra_in_map} in input_column_map not used in template")
            
            raise ValueError(". ".join(error_parts))
        
        # Check all mapped columns exist in CSV
        csv_columns = set(self.dataframe.columns)
        for placeholder, csv_col in self.input_column_map.items():
            if csv_col not in csv_columns:
                raise ValueError(f"Column '{csv_col}' (mapped from '{placeholder}') not found in CSV. Available: {csv_columns}")
        
        # Check label column exists
        if self.label_column not in csv_columns:
            raise ValueError(f"Label column '{self.label_column}' not found in CSV. Available: {csv_columns}")
    
    def build(self) -> Dataset:
        """
        Build the SFT dataset from CSV data.
        
        Processes each row:
        1. Checks for NaN values in required columns (skips row if found)
        2. Normalizes label (lowercase, strip whitespace)
        3. Filters by valid_labels if specified
        4. Substitutes values into prompt template
        5. Creates prompt-response pair
        
        Returns:
            HuggingFace Dataset with 'prompt' and 'response' columns
        """
        self._formatted_data = []
        self._skipped_rows = 0
        self._label_counts = {}
        skipped_nan = 0
        skipped_invalid_label = 0
        
        # Get list of required columns
        required_columns = list(self.input_column_map.values()) + [self.label_column]
        
        for idx, row in self.dataframe.iterrows():
            # Check for NaN in required columns
            has_nan = False
            for col in required_columns:
                if pd.isna(row[col]):
                    has_nan = True
                    break
            
            if has_nan:
                skipped_nan += 1
                continue
            
            # Normalize label
            label = str(row[self.label_column]).lower().strip()
            
            # Filter by valid labels if specified
            if self.valid_labels is not None and label not in self.valid_labels:
                skipped_invalid_label += 1
                continue
            
            # Build prompt by substituting values
            prompt = self.prompt_template
            for placeholder, csv_col in self.input_column_map.items():
                value = str(row[csv_col])
                prompt = prompt.replace(f"{{{placeholder}}}", value)
            
            # Add to formatted data
            self._formatted_data.append({
                "prompt": prompt,
                "response": label
            })
            
            # Track label counts
            self._label_counts[label] = self._label_counts.get(label, 0) + 1
        
        self._skipped_rows = skipped_nan + skipped_invalid_label
        
        # Print summary
        self._print_summary(skipped_nan, skipped_invalid_label)
        
        # Create HuggingFace Dataset
        self._dataset = Dataset.from_list(self._formatted_data)
        
        return self._dataset
    
    def _print_summary(self, skipped_nan: int, skipped_invalid_label: int) -> None:
        """Print a summary of the built dataset."""
        print("\n" + "=" * 60)
        print("📊 Dataset Build Summary")
        print("=" * 60)
        print(f"Source file: {self.csv_file_path}")
        print(f"Total rows in CSV: {len(self.dataframe)}")
        print("-" * 60)
        
        if skipped_nan > 0:
            print(f"⚠️  Skipped {skipped_nan} rows due to missing (NaN) values")
        if skipped_invalid_label > 0:
            print(f"⚠️  Skipped {skipped_invalid_label} rows due to invalid labels")
        
        print(f"✅ Final dataset size: {len(self._formatted_data)} rows")
        print("-" * 60)
        print("📋 Label distribution:")
        for label, count in sorted(self._label_counts.items()):
            pct = (count / len(self._formatted_data) * 100) if self._formatted_data else 0
            print(f"   '{label}': {count} ({pct:.1f}%)")
        print("-" * 60)
        print(f"🔧 Response template for DataCollatorForCompletionOnlyLM:")
        print(f"   response_template=\"{self.response_template}\"")
        print("=" * 60 + "\n")
    
    def get_response_template(self) -> str:
        """
        Get the response template for DataCollatorForCompletionOnlyLM.
        
        Use this with the collator:
            collator = DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                response_template=builder.get_response_template()
            )
        
        Returns:
            The response template string (e.g., "Answer:")
        """
        return self.response_template
    
    def get_metadata(self, tokenizer=None) -> dict:
        """
        Get metadata about the dataset for use with ModelRunner and ComputeMetrics.
        
        This provides all the information needed to run inference and compute
        metrics dynamically based on the labels in the dataset.
        
        Args:
            tokenizer: Optional tokenizer to encode labels to token IDs.
                       If provided, label_token_ids will be populated.
        
        Returns:
            Dictionary containing:
                - labels: List of unique label strings
                - label_counts: Dict of label -> count
                - label_token_ids: Dict of label -> token_id (if tokenizer provided)
                - id_to_label: Dict of token_id -> label (if tokenizer provided)
                - response_template: The trigger string for completion
                - num_classes: Number of unique classes
        
        Raises:
            RuntimeError: If build() has not been called yet
        
        Example:
            metadata = builder.get_metadata(tokenizer)
            runner = ModelRunner(model, tokenizer, metadata)
            metrics = ComputeMetrics(metadata, tokenizer)
        """
        if not self._label_counts:
            raise RuntimeError("No metadata available. Call build() first.")
        
        labels = sorted(self._label_counts.keys())
        
        metadata = {
            "labels": labels,
            "label_counts": self._label_counts.copy(),
            "response_template": self.response_template,
            "num_classes": len(labels),
            "label_token_ids": {},
            "id_to_label": {},
        }
        
        # Build token ID mappings if tokenizer provided
        if tokenizer is not None:
            for label in labels:
                # Encode with leading space (common for LLM token generation)
                token_id = tokenizer.encode(f" {label}", add_special_tokens=False)[0]
                metadata["label_token_ids"][label] = token_id
                metadata["id_to_label"][token_id] = label
        
        return metadata
    
    def save_jsonl(self, output_path: str) -> None:
        """
        Save the dataset to a JSONL file.
        
        Args:
            output_path: Path for the output JSONL file
        
        Raises:
            RuntimeError: If build() has not been called yet
        """
        if not self._formatted_data:
            raise RuntimeError("No data to save. Call build() first.")
        
        with open(output_path, "w") as f:
            for item in self._formatted_data:
                f.write(json.dumps(item) + "\n")
        
        print(f"✅ Saved {len(self._formatted_data)} rows to '{output_path}'")
