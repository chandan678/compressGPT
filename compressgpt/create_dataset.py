"""
Dataset Builder for compressGPT - SFT Training

This module provides the DatasetBuilder class that converts tabular CSV data
into model-ready training datasets with automatic template formatting, metadata
generation, and label validation.

Example usage:
    builder = DatasetBuilder(
        data_path="data.csv",
        model_id="meta-llama/Llama-3.2-1B",
        prompt_template="Do these match?\\nName 1: {name1}\\nName 2: {name2}\\nAnswer:",
        input_column_map={"name1": "elected_name", "name2": "partner_name"},
        label_column="labeled_result",
        valid_labels={"yes", "no"},
        is_train=True
    )
    builder.build()
    
    # Access dataset and metadata
    dataset = builder.dataset
    metadata = builder.metadata
    
    # Optional: save to disk
    builder.save("output.jsonl")
"""

import re
import json
import os
import logging
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Optional, Callable

from .label_space import LabelSpace


logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Model-aware dataset builder for classification tasks.
    
    Automatically handles:
    - Model detection (base vs instruct)
    - Tokenizer loading and label validation
    - Chat template application (for instruct models)
    - LabelSpace creation with single-token validation
    - Train vs eval formatting (with/without gold labels)
    - Metadata storage for training
    
    The prompt template uses {placeholder} syntax for input variables.
    The text after the last placeholder (e.g., "Answer:") becomes the
    response_trigger for DataCollatorForCompletionOnlyLM.
    
    Attributes:
        data_path: Path to the input CSV file
        model_id: HuggingFace model ID or local path
        model_mode: "base", "instruct", or "auto" (auto-detected)
        tokenizer: Loaded tokenizer from model_id
        label_space: LabelSpace instance with token mappings
        dataset: HuggingFace Dataset (available after build())
        metadata: Complete metadata dict (available after build())
    """
    
    def __init__(
        self,
        data_path: str,
        model_id: str,
        prompt_template: str,
        input_column_map: dict[str, str],
        label_column: str,
        valid_labels: Optional[set[str]] = None,
        *,
        is_train: bool = True,
        model_mode: str = "auto",
        tokenizer=None,
        format_fn: Optional[Callable] = None,
        keep_fields: bool = False,
        output_path: Optional[str] = None,
        label_prefix: str = " ",
        response_trigger: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize model-aware DatasetBuilder.
        
        Args:
            data_path: Path to CSV file
            model_id: HuggingFace model ID or local path (e.g., "meta-llama/Llama-3.2-1B")
            prompt_template: Prompt string with {placeholder} syntax, ending with response trigger
                Example: "Compare names:\\nName 1: {name1}\\nName 2: {name2}\\nAnswer:"
            input_column_map: Maps template placeholders to CSV columns
                Example: {"name1": "elected_name", "name2": "partner_name"}
            label_column: CSV column name for the response/label
            valid_labels: Optional set of valid labels to filter by (e.g., {"yes", "no"})
            is_train: If True, create training format (text with labels). If False, eval format.
            model_mode: "auto" (detect), "base", or "instruct"
            tokenizer: Optional tokenizer override (otherwise loaded from model_id)
            format_fn: Optional custom formatting function(row_dict) -> text_string
            keep_fields: If True, keep original prompt/response columns for debugging
            output_path: If provided, auto-save dataset after build()
            label_prefix: Prefix for label tokenization (default " " for proper spacing)
            response_trigger: Optional explicit response trigger (e.g., "Answer:"). If not provided,
                auto-extracts from prompt_template (text after last {placeholder})
            hf_token: HuggingFace API token for gated/private models
        
        Raises:
            ValueError: If template placeholders don't match input_column_map keys
            ValueError: If label_column not found in CSV
            ValueError: If labels don't tokenize to single tokens
        """
        self.data_path = data_path
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.input_column_map = input_column_map
        self.label_column = label_column
        self.valid_labels = valid_labels
        self.is_train = is_train
        self.model_mode = model_mode
        self.format_fn = format_fn
        self.keep_fields = keep_fields
        self.output_path = output_path
        self.label_prefix = label_prefix
        self.hf_token = hf_token
        
        # Load CSV data
        logger.info(f"📂 Loading data from: {data_path}")
        self.dataframe = pd.read_csv(data_path)
        logger.info(f"  {len(self.dataframe)} rows loaded")
        
        # Load tokenizer
        if tokenizer is None:
            logger.info(f"📥 Loading tokenizer from: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.hf_token)
        else:
            self.tokenizer = tokenizer
            logger.info(f"  Using provided tokenizer")
        
        # Set padding token if not present (required for batch training)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"  Set pad_token to eos_token")
        
        # Detect model mode
        self.model_mode = self._detect_model_mode() if model_mode == "auto" else model_mode
        logger.info(f"🔧 Model mode: {self.model_mode}")
        
        # Extract placeholders from template
        self._template_placeholders = set(re.findall(r'\{(\w+)\}', prompt_template))
        
        # Set or extract response trigger
        if response_trigger is not None:
            self.response_trigger = response_trigger
            logger.info(f"🎯 Response trigger (provided): {self.response_trigger!r}")
        else:
            # Auto-extract from template (text after last placeholder)
            self.response_trigger = self._extract_response_trigger()
            logger.info(f"🎯 Response trigger (extracted): {self.response_trigger!r}")
        
        # Validate inputs
        self._validate()
        
        # Storage for built dataset (populated by build())
        self._dataset: Optional[Dataset] = None
        self._formatted_data: list[dict] = []
        self._label_space: Optional[LabelSpace] = None
        self._metadata: Optional[dict] = None
        self._skipped_rows = 0
        self._label_counts: dict[str, int] = {}
    
    def _detect_model_mode(self) -> str:
        """
        Detect if model is base or instruct by checking for chat template.
        
        Returns:
            "instruct" if tokenizer has chat_template, else "base"
        """
        has_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and 
            self.tokenizer.chat_template is not None
        )
        return "instruct" if has_chat_template else "base"
    
    def _extract_response_trigger(self) -> str:
        """
        Extract the response trigger from the prompt template.
        This is the text after the last {placeholder} that triggers completion.
        
        Returns:
            The response trigger string (e.g., "Answer:")
        """
        # Find the position after the last placeholder
        last_placeholder_end = 0
        for match in re.finditer(r'\{(\w+)\}', self.prompt_template):
            last_placeholder_end = match.end()
        
        if last_placeholder_end == 0:
            raise ValueError("No placeholders found in prompt_template")
        
        # Only strip trailing whitespace, preserve leading space for tokenization consistency
        response_trigger = self.prompt_template[last_placeholder_end:].strip()
        
        if not response_trigger:
            raise ValueError(
                "No response trigger found after last placeholder. "
                "Prompt should end with a trigger like 'Answer:' after the last {placeholder}"
            )
        
        return response_trigger
    
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
    
    def build(self) -> "DatasetBuilder":
        """
        Build the model-aware dataset.
        
        Process flow:
        1. Extract labels and create LabelSpace (validates single-token constraint)
        2. Build prompts from template
        3. Format text column based on model_mode:
           - base: prompt + trigger + label
           - instruct: chat_template(prompt) + trigger + label
        4. Create train or eval dataset based on is_train flag
        5. Optionally auto-save if output_path provided
        
        Returns:
            self (DatasetBuilder instance for chaining)
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔨 Building dataset")
        logger.info("=" * 60)
        
        self._formatted_data = []
        self._label_counts = {}
        skipped_nan = 0
        skipped_invalid_label = 0
        
        # Get list of required columns
        required_columns = list(self.input_column_map.values()) + [self.label_column]
        
        # First pass: collect all labels for LabelSpace creation
        collected_labels = set()
        
        for idx, row in self.dataframe.iterrows():
            # Check for NaN
            has_nan = any(pd.isna(row[col]) for col in required_columns)
            if has_nan:
                skipped_nan += 1
                continue
            
            # Normalize label
            label = str(row[self.label_column]).lower().strip()
            
            # Filter by valid labels if specified
            if self.valid_labels is not None and label not in self.valid_labels:
                skipped_invalid_label += 1
                continue
            
            collected_labels.add(label)
            self._label_counts[label] = self._label_counts.get(label, 0) + 1
        
        # Create LabelSpace (validates single-token constraint)
        logger.info(f"🏷️  Creating LabelSpace with {len(collected_labels)} labels")
        self._label_space = LabelSpace(
            tokenizer=self.tokenizer,
            labels=sorted(collected_labels),
            label_prefix=self.label_prefix
        )
        
        # Second pass: build formatted dataset
        for idx, row in self.dataframe.iterrows():
            # Check for NaN
            has_nan = any(pd.isna(row[col]) for col in required_columns)
            if has_nan:
                continue
            
            # Normalize label
            label = str(row[self.label_column]).lower().strip()
            
            # Filter by valid labels
            if self.valid_labels is not None and label not in self.valid_labels:
                continue
            
            # Build prompt by substituting values
            prompt = self.prompt_template
            for placeholder, csv_col in self.input_column_map.items():
                value = str(row[csv_col])
                prompt = prompt.replace(f"{{{placeholder}}}", value)
            
            # Format text based on is_train and model_mode
            if self.format_fn:
                # Custom formatting function
                text = self.format_fn({"prompt": prompt, "label": label})
            else:
                text = self._format_text(prompt, label if self.is_train else None)
            
            # Build row data
            row_data = {
                "text": text,
                "gold_label": label,  # Always include for eval
            }
            
            if self.keep_fields:
                row_data["prompt"] = prompt
                row_data["response"] = label
            
            if self.is_train:
                # For training, also create prompt-only version for potential eval
                row_data["text_prompt_only"] = self._format_text(prompt, None)
            
            self._formatted_data.append(row_data)
        
        self._skipped_rows = skipped_nan + skipped_invalid_label
        
        # Create HuggingFace Dataset
        self._dataset = Dataset.from_list(self._formatted_data)
        
        # Build metadata
        self._metadata = self._build_metadata()
        
        # Print summary
        self._print_summary(skipped_nan, skipped_invalid_label)
        
        # Auto-save if output_path provided
        if self.output_path:
            self.save(self.output_path)
        
        return self
    
    def _format_text(self, prompt: str, label: Optional[str]) -> str:
        """
        Format text for training or eval based on model_mode.
        
        Args:
            prompt: Filled-in prompt with values substituted
            label: Label string (if training) or None (if eval/prompt-only)
        
        Returns:
            Formatted text string ready for SFTTrainer
        """
        if self.model_mode == "base":
            # Base model: simple concatenation
            if label is None:
                # Eval format: prompt + trigger (no label)
                return f"{prompt}"
            else:
                # Train format: prompt + trigger + label
                # Trigger is already in prompt, so just add label with prefix
                return f"{prompt}{self.label_prefix}{label}"
        
        elif self.model_mode == "instruct":
            # Instruct model: use chat template
            if label is None:
                # Eval format: user message only
                messages = [
                    {"role": "user", "content": prompt}
                ]
            else:
                # Train format: user + assistant
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"{self.label_prefix}{label}"}
                ]
            
            # Apply chat template
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(label is None)  # Add prompt for eval
            )
        
        else:
            raise ValueError(f"Unknown model_mode: {self.model_mode}")
    
    def _build_metadata(self) -> dict:
        """
        Build complete metadata dict for training.
        
        Returns:
            Metadata dictionary with model info, LabelSpace, and counts
        """
        return {
            "model_id": self.model_id,
            "model_mode": self.model_mode,
            "response_trigger": self.response_trigger,
            "label_space": self._label_space.to_dict(),
            "label_counts": self._label_counts.copy(),
            "num_samples": len(self._formatted_data),
            "is_train": self.is_train,
        }
    
    @property
    def dataset(self) -> Dataset:
        """
        Get the built HuggingFace Dataset.
        
        Returns:
            HuggingFace Dataset instance
        
        Raises:
            RuntimeError: If build() has not been called yet
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not built yet. Call build() first.")
        return self._dataset
    
    @property
    def metadata(self) -> dict:
        """
        Get the complete metadata dict.
        
        Returns:
            Metadata dictionary with model info, LabelSpace, counts
        
        Raises:
            RuntimeError: If build() has not been called yet
        """
        if self._metadata is None:
            raise RuntimeError("Metadata not available. Call build() first.")
        return self._metadata
    
    def get_metadata(self) -> dict:
        """
        Get metadata (kept for compatibility).
        
        Returns:
            Complete metadata dictionary
        """
        return self.metadata
    
    def _print_summary(self, skipped_nan: int, skipped_invalid_label: int) -> None:
        """Print a summary of the built dataset."""
        print("\n" + "=" * 60)
        print("📊 Dataset Build Summary")
        print("=" * 60)
        print(f"Source: {self.data_path}")
        print(f"Model: {self.model_id} ({self.model_mode})")
        print(f"Total rows in CSV: {len(self.dataframe)}")
        print("-" * 60)
        
        if skipped_nan > 0:
            print(f"⚠️  Skipped {skipped_nan} rows due to missing (NaN) values")
        if skipped_invalid_label > 0:
            print(f"⚠️  Skipped {skipped_invalid_label} rows due to invalid labels")
        
        print(f"✅ Final dataset size: {len(self._formatted_data)} rows")
        print(f"   Mode: {'Training' if self.is_train else 'Evaluation'}")
        print("-" * 60)
        print("📋 Label distribution:")
        for label, count in sorted(self._label_counts.items()):
            pct = (count / len(self._formatted_data) * 100) if self._formatted_data else 0
            print(f"   '{label}': {count} ({pct:.1f}%)")
        print("-" * 60)
        print(f"🔧 Response trigger for DataCollator: {self.response_trigger!r}")
        print(f"🏷️  LabelSpace: {len(self._label_space.labels)} labels, single-token validated")
        print(f"   Token IDs: {self._label_space.label_token_ids}")
        print("=" * 60 + "\n")
    
    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save the dataset to JSONL format.
        
        Args:
            output_path: Path for output file (uses self.output_path if None)
        
        Returns:
            Path where dataset was saved
        
        Raises:
            RuntimeError: If build() has not been called yet
            ValueError: If no output_path provided
        """
        if not self._formatted_data:
            raise RuntimeError("No data to save. Call build() first.")
        
        path = output_path or self.output_path
        if path is None:
            raise ValueError("No output_path provided")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Save as JSONL
        with open(path, "w") as f:
            for item in self._formatted_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"✅ Saved {len(self._formatted_data)} rows to '{path}'")
        return path
